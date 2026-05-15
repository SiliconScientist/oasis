from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


def _global_mean_pool(x: Tensor, batch: Tensor, n_graphs: int) -> Tensor:
    """Mean-pool node features into per-graph vectors."""
    out = torch.zeros(n_graphs, x.shape[1], dtype=x.dtype, device=x.device)
    idx = batch.unsqueeze(1).expand_as(x)
    out.scatter_add_(0, idx, x)
    count = torch.zeros(n_graphs, 1, dtype=x.dtype, device=x.device)
    count.scatter_add_(
        0,
        batch.unsqueeze(1),
        torch.ones(batch.shape[0], 1, dtype=x.dtype, device=x.device),
    )
    return out / count.clamp(min=1)


class SchNetInteraction(nn.Module):
    """Single SchNet interaction block.

    Computes a continuous-filter convolution: per-edge distance filters
    (produced from RBF features via a two-layer MLP with SiLU) modulate
    neighbour embeddings, which are then scatter-summed to each atom and
    mixed back with a residual linear.
    """

    def __init__(self, hidden_dim: int, n_rbf: int) -> None:
        super().__init__()
        # Maps RBF features → per-edge filter vector (the cfconv filter network).
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        h: Tensor,           # (n_nodes, hidden_dim)
        edge_index: Tensor,  # (2, n_edges) long
        rbf: Tensor,         # (n_edges, n_rbf)
    ) -> Tensor:             # (n_nodes, hidden_dim)
        edge_src, edge_dst = edge_index[0], edge_index[1]
        n_nodes, hidden_dim = h.shape

        W = self.filter_net(rbf)              # (n_edges, hidden_dim)
        messages = h[edge_src] * W            # (n_edges, hidden_dim)

        agg = torch.zeros(n_nodes, hidden_dim, dtype=h.dtype, device=h.device)
        agg.scatter_add_(0, edge_dst.unsqueeze(1).expand_as(messages), messages)

        return h + self.output_linear(agg)    # residual connection


class SchNetEncoder(nn.Module):
    """SchNet-style encoder: atom-type embeddings + continuous-filter interactions.

    Accepts integer atomic numbers rather than raw float node features,
    giving each element a learned dense embedding. Per-edge Gaussian RBF
    expansion of interatomic distances drives the filter network in each
    interaction block, so gating decisions are physically informed by
    actual bond lengths.

    Args:
        hidden_dim:     Width of atom embeddings and interaction layers.
        out_features:   Number of output logits (= n_experts for MoE gating).
        n_layers:       Number of SchNet interaction blocks.
        n_rbf:          Number of Gaussian RBF basis functions.
        r_max:          Distance cutoff (Å); RBF centers span [0, r_max].
        max_atomic_num: Largest atomic number accepted (inclusive).
    """

    def __init__(
        self,
        hidden_dim: int,
        out_features: int,
        n_layers: int,
        n_rbf: int = 20,
        r_max: float = 6.0,
        max_atomic_num: int = 100,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_atomic_num + 1, hidden_dim)

        centers = torch.linspace(0.0, r_max, n_rbf)
        width = torch.tensor(r_max / (n_rbf - 1) if n_rbf > 1 else float(r_max))
        self.register_buffer("_rbf_centers", centers)
        self.register_buffer("_rbf_width", width)

        self.interactions = nn.ModuleList(
            [SchNetInteraction(hidden_dim, n_rbf) for _ in range(n_layers)]
        )
        self.output_proj = nn.Linear(hidden_dim, out_features)

    def _rbf_expand(self, distances: Tensor) -> Tensor:
        """Expand (n_edges,) distances → (n_edges, n_rbf) Gaussian features."""
        diff = distances.unsqueeze(1) - self._rbf_centers  # (n_edges, n_rbf)
        return torch.exp(-0.5 * (diff / self._rbf_width) ** 2)

    def forward(
        self,
        atomic_numbers: Tensor,   # (total_nodes,) long
        edge_index: Tensor,        # (2, total_edges) long
        edge_distances: Tensor,    # (total_edges,) float32
        batch_vector: Tensor,      # (total_nodes,) long
    ) -> Tensor:                   # (n_graphs, out_features)
        n_graphs = int(batch_vector.max().item()) + 1

        h = self.embedding(atomic_numbers)       # (total_nodes, hidden_dim)
        rbf = self._rbf_expand(edge_distances)   # (total_edges, n_rbf)

        for interaction in self.interactions:
            h = interaction(h, edge_index, rbf)

        pooled = _global_mean_pool(h, batch_vector, n_graphs)  # (n_graphs, hidden_dim)
        return self.output_proj(pooled)                         # (n_graphs, out_features)
