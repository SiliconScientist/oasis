
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Callable
import numpy as np

# =============================================================================
# Data Structures
# =============================================================================

class GraphSample:
    """
    Represents a single graph sample for GNN training/inference.
    
    Attributes:
        x: Tensor [num_nodes, feature_dim] node features
        edge_index: Tensor [2, num_edges] COO format, undirected (both directions included)
        edge_type: Tensor [num_edges] relation type (0: SS, 1: AA, 2: SA_bond)
        node_type: Tensor [num_nodes] node type (0: S_bound, 1: S_unbound, 2: A_bound, 3: A_unbound)
        num_nodes: int
        y: Optional[Tensor [1]] target scalar
        atoms: ASE Atoms object (for feature building)
    """
    def __init__(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor, node_type: torch.Tensor, y: Optional[torch.Tensor] = None, atoms=None, sd_coupling:      Optional[torch.Tensor] = None, d_filling_n: Optional[torch.Tensor] = None, e_conductivity_n: Optional[torch.Tensor] = None, d_filling_mult: Optional[torch.Tensor] = None,):
        """
        Args:
            x: Node features tensor
            edge_index: Edge indices in COO format
            edge_type: Edge types
            node_type: Node types
            y: Optional target tensor
            atoms: ASE Atoms object
        """
        self.x = x
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.node_type = node_type
        self.num_nodes = x.size(0)
        self.y = y
        self.atoms = atoms
        self.sd_coupling = sd_coupling
        self.d_filling_n = d_filling_n
        self.e_conductivity_n = e_conductivity_n
        self.d_filling_mult = d_filling_mult
    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_type = self.edge_type.to(device)
        self.node_type = self.node_type.to(device)
        if self.y is not None:
            self.y = self.y.to(device)
        # Note: 'atoms' (ASE object) stays on CPU as it is not a tensor
        return self
    def _assign_node_types(self, atoms) -> torch.Tensor:
        """Assign node types based on atoms and binding info."""
        from ghit import GHIT
        ghit = GHIT(atoms)
        ghit.find_guest_atoms(guest_elements=['C', 'O', 'H', 'N', 'F', 'S', 'K'])
        guest_atoms = set(ghit.get_guest_atoms())
        host_atoms = set(range(len(atoms))) - guest_atoms
        
        ghit.find_guest_binding_atoms(distance_gap_tolerance=0.5, check_saturation=True)
        ghit.find_host_binding_atoms()
        
        guest_binding = set(ghit.get_guest_binding_atoms())
        host_binding = set()
        for hb_list in ghit.get_host_binding_atoms():
            host_binding.update(hb_list)
        
        node_types = []
        for i in range(len(atoms)):
            if i in host_atoms:
                if i in host_binding:
                    node_types.append(0)  # S_bound
                else:
                    node_types.append(1)  # S_unbound
            else:
                if i in guest_binding:
                    node_types.append(2)  # A_bound
                else:
                    node_types.append(3)  # A_unbound
        return torch.tensor(node_types, dtype=torch.long)
    
    def _build_edges(self, edge_list: List[Tuple[int, int, str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert edge list to COO format, add both directions, map edge types."""
        edge_type_map = {'SS': 0, 'AA': 1, 'SA_bond': 2}
        edges = []
        types = []
        for i, j, et in edge_list:
            edges.append([i, j])
            edges.append([j, i])  # Undirected
            types.extend([edge_type_map[et], edge_type_map[et]])
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
            edge_type = torch.tensor(types, dtype=torch.long)
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long)
            edge_type = torch.empty(0, dtype=torch.long)
        return edge_index, edge_type

class FeatureBuilder(nn.Module):
    """
    Builds raw features for each node.
    Now: [Z embedding | metal electronic features (4)]
    """
    def __init__(self, embedding_dim: int = 16):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.z_embed = nn.Embedding(100, embedding_dim)

        self.extra_dim = 4
        self.output_dim = embedding_dim + self.extra_dim

    def build_features(self, sample: GraphSample) -> torch.Tensor:
        atoms = sample.atoms
        device = next(self.z_embed.parameters()).device

        z = torch.tensor([atom.number for atom in atoms], dtype=torch.long, device=device)
        z_feat = self.z_embed(z)

        num_nodes = len(atoms)

        # Default zeros (adsorbates automatically get 0s)
        sd = torch.zeros(num_nodes, 1, device=device)
        df = torch.zeros(num_nodes, 1, device=device)
        cond = torch.zeros(num_nodes, 1, device=device)
        mult = torch.zeros(num_nodes, 1, device=device)

        # Fill if metal features provided
        if sample.sd_coupling is not None:
            sd = sample.sd_coupling.to(device).view(-1,1)
            df = sample.d_filling_n.to(device).view(-1,1)
            cond = sample.e_conductivity_n.to(device).view(-1,1)
            mult = sample.d_filling_mult.to(device).view(-1,1)

        physics = torch.cat([sd, df, cond, mult], dim=1)

        return torch.cat([z_feat, physics], dim=1)

class TypeSpecificEncoder(nn.Module):
    """
    Encodes raw features per node type to shared embedding dim.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_types: int = 4):
        super().__init__()
        self.output_dim = output_dim
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            ) for _ in range(num_types)
        ])
    
    def forward(self, raw_features: torch.Tensor, node_type: torch.Tensor) -> torch.Tensor:
        """
        Args:
            raw_features: [num_nodes, input_dim]
            node_type: [num_nodes]
        Returns:
            encoded: [num_nodes, output_dim]
        """
        num_nodes = raw_features.size(0)
        encoded = torch.zeros(num_nodes, self.output_dim, device=raw_features.device, dtype=raw_features.dtype)
        for i in range(num_nodes):
            t = node_type[i].item()
            encoded[i] = self.encoders[t](raw_features[i])
        return encoded

class MultiRelationGNNLayer(nn.Module):
    """
    Multi-relation GNN layer with separate weights per relation.
    """
    def __init__(self, input_dim: int, output_dim: int, num_relations: int = 3):
        super().__init__()
        self.num_relations = num_relations
        self.linear = nn.Linear(input_dim, output_dim)
        self.relation_weights = nn.Parameter(torch.randn(num_relations, input_dim, output_dim))
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, input_dim]
            edge_index: [2, num_edges]
            edge_type: [num_edges]
        Returns:
            out: [num_nodes, output_dim]
        """
        # Aggregate messages per relation
        out = torch.zeros_like(x)
        for r in range(self.num_relations):
            mask = (edge_type == r)
            if mask.any():
                ei_r = edge_index[:, mask]
                src, dst = ei_r[0], ei_r[1]
                msg = x[src] @ self.relation_weights[r]  # [num_edges_r, output_dim]
                out.index_add_(0, dst, msg)
        
        # Normalize by degree
        deg = torch.zeros(x.size(0), device=x.device)
        deg.index_add_(0, edge_index[1], torch.ones(edge_index.size(1), device=x.device))
        deg = deg.clamp(min=1)
        out = out / deg.unsqueeze(-1)
        
        # Residual + norm
        out = self.norm(self.linear(x) + out)
        return out

class GNNModel(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2, pooling: str = 'mean', pool_mask: Optional[torch.Tensor] = None):
        super().__init__()

        self.feature_builder = FeatureBuilder(feature_dim)
        input_dim = self.feature_builder.output_dim  # UPDATED

        self.encoder = TypeSpecificEncoder(input_dim, hidden_dim, hidden_dim)
        self.gnn_layers = nn.ModuleList([MultiRelationGNNLayer(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.pooling = pooling
        if pooling == 'attention':
            self.attention = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.pool_mask = pool_mask
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, sample: GraphSample) -> torch.Tensor:
        raw_features = self.feature_builder.build_features(sample)
        x = self.encoder(raw_features, sample.node_type)

        for layer in self.gnn_layers:
            x = layer(x, sample.edge_index, sample.edge_type)

        mask = self.pool_mask if self.pool_mask is not None else torch.ones(x.size(0), dtype=torch.bool, device=x.device)

        if self.pooling == 'mean':
            pooled = x[mask].mean(dim=0)
        else:
            weights = self.attention(x[mask])
            pooled = (x[mask] * weights).sum(dim=0) / weights.sum()

        return self.regressor(pooled)

        