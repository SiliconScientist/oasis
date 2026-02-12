import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Callable
import numpy as np
from pymatgen.core.periodic_table import Element
# =============================================================================
# Data Structures
# =============================================================================

class GraphSample:
    """
    Represents a single graph sample for GNN training/inference.
    """
    def __init__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        node_type: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        atoms=None,

        # NEW (optional)
        group: Optional[torch.Tensor] = None,
        period: Optional[torch.Tensor] = None,
    ):
        self.x = x
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.node_type = node_type
        self.num_nodes = x.size(0)
        self.y = y
        self.atoms = atoms

        self.group = group
        self.period = period
    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_type = self.edge_type.to(device)
        self.node_type = self.node_type.to(device)

        if self.y is not None:
            self.y = self.y.to(device)

        if self.group is not None:
            self.group = self.group.to(device)
        if self.period is not None:
            self.period = self.period.to(device)
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

# =============================================================================
# Feature Builders
# =============================================================================
class FeatureBuilder:
    """
    Builds raw features for each node type.

    Now: [group embedding | period embedding]
    """
    def __init__(self, group_dim: int = 8, period_dim: int = 8):
        self.group_dim = group_dim
        self.period_dim = period_dim

        self.group_embed = nn.Embedding(19, group_dim)   # groups 1–18
        self.period_embed = nn.Embedding(8, period_dim) # periods 1–7

        self.embedding_dim = group_dim + period_dim

    def build_features(self, atoms, node_type: torch.Tensor) -> torch.Tensor:
        """
        Args:
            atoms: ASE Atoms
            node_type: Tensor [num_nodes] (unused here, kept for compatibility)
        Returns:
            raw_features: Tensor [num_nodes, group_dim + period_dim]
        """
        groups = []
        periods = []

        for atom in atoms:
            el = Element.from_Z(atom.number)
            groups.append(el.group)
            periods.append(el.row)

        group = torch.tensor(groups, dtype=torch.long)
        period = torch.tensor(periods, dtype=torch.long)

        group_feat = self.group_embed(group)
        period_feat = self.period_embed(period)

        return torch.cat([group_feat, period_feat], dim=1)

# =============================================================================
# Model Modules
# =============================================================================

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
    """
    Full GNN model for regression.
    """
    def __init__(self, feature_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2, pooling: str = 'mean', pool_mask: Optional[torch.Tensor] = None):
        super().__init__()
        self.feature_builder = FeatureBuilder(feature_dim)
        self.encoder = TypeSpecificEncoder(feature_dim, hidden_dim, hidden_dim)
        self.gnn_layers = nn.ModuleList([MultiRelationGNNLayer(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.pooling = pooling
        if pooling == 'attention':
            self.attention = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.pool_mask = pool_mask
        self.regressor = nn.Linear(hidden_dim, 1)
    
    def forward(self, sample: GraphSample) -> torch.Tensor:
        """
        Args:
            sample: GraphSample
        Returns:
            pred: [1]
        """
        # Features
        raw_features = self.feature_builder.build_features(sample.atoms, sample.node_type)
        x = self.encoder(raw_features, sample.node_type)
        
        # GNN
        for layer in self.gnn_layers:
            x = layer(x, sample.edge_index, sample.edge_type)
        
        # Pooling
        if self.pool_mask is not None:
            mask = self.pool_mask
        else:
            mask = torch.ones(x.size(0), dtype=torch.bool, device=x.device)
        
        if self.pooling == 'mean':
            pooled = x[mask].mean(dim=0)
        elif self.pooling == 'attention':
            weights = self.attention(x[mask])
            pooled = (x[mask] * weights).sum(dim=0) / weights.sum()
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        return self.regressor(pooled)

# =============================================================================
# Training Example
# =============================================================================

def training_example():
    """
    Minimal training loop with synthetic data.
    """
    # Create synthetic sample
    # Assume atoms is some ASE Atoms, edge_list from make_edge_list
    # For demo, create dummy
    num_nodes = 10
    edge_index = torch.randint(0, num_nodes, (2, 20))
    edge_type = torch.randint(0, 3, (20,))
    node_type = torch.randint(0, 4, (num_nodes,))
    y = torch.randn(1)
    
    # Dummy GraphSample (without atoms for simplicity)
    class DummySample:
        def __init__(self):
            self.edge_index = edge_index
            self.edge_type = edge_type
            self.node_type = node_type
            self.num_nodes = num_nodes
            self.y = y
            self.atoms = None  # Would need real atoms
    
    sample = DummySample()
    
    # Model
    model = GNNModel(feature_dim=16, hidden_dim=32, output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Training loop
    for epoch in range(10):
        pred = model(sample)
        loss = criterion(pred, sample.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}: Loss {loss.item():.4f}")

# =============================================================================
# How to Use
# =============================================================================
"""
1. Prepare data: Use make_edge_list(atoms) to get edge_list, create GraphSample(atoms, edge_list, y)
2. Initialize model: model = GNNModel(feature_dim=16, hidden_dim=32, output_dim=1, pooling='mean')
3. Forward: pred = model(sample)
4. Train: Use training_example() as template.

To extend features: Edit FeatureBuilder.build_features to return different tensors per node_type.
To add edge types: Update num_relations in GNN layers and edge_type_map in GraphSample.
To change pooling mask: Set pool_mask in GNNModel to a boolean tensor selecting nodes.
"""

# if __name__ == "__main__":
#     training_example()</content>
# <parameter name="filePath">/Users/matt/Dropbox/CurrentResearch/GCNN/gnn_framework.py