from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from oasis.learning_curve.families.gnn_gate import collate_graphs
from oasis.sweep import GraphRecord


def _scatter_mean(src: Tensor, idx: Tensor, n_bins: int) -> Tensor:
    """Mean-aggregate src rows into n_bins buckets indexed by idx."""
    hidden = src.shape[1]
    out = torch.zeros(n_bins, hidden, dtype=src.dtype, device=src.device)
    if src.shape[0] == 0:
        return out
    out.scatter_add_(0, idx.unsqueeze(1).expand_as(src), src)
    count = torch.zeros(n_bins, 1, dtype=src.dtype, device=src.device)
    count.scatter_add_(
        0,
        idx.unsqueeze(1),
        torch.ones(idx.shape[0], 1, dtype=src.dtype, device=src.device),
    )
    return out / count.clamp(min=1)


def _global_mean_pool(x: Tensor, batch: Tensor, n_graphs: int) -> Tensor:
    """Mean-pool node features into per-graph vectors."""
    out = torch.zeros(n_graphs, x.shape[1], dtype=x.dtype, device=x.device)
    out.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x)
    count = torch.zeros(n_graphs, 1, dtype=x.dtype, device=x.device)
    count.scatter_add_(
        0,
        batch.unsqueeze(1),
        torch.ones(batch.shape[0], 1, dtype=x.dtype, device=x.device),
    )
    return out / count.clamp(min=1)


class ProbeGnnEncoder(nn.Module):
    """Message-passing GNN for direct adsorption energy regression.

    Accepts probe-augmented float node features (column 0: atomic number;
    columns 1..: per-atom MLIP probe energies, zero for non-binding atoms),
    runs ``n_layers`` of scatter-mean message-passing steps, global-mean-pools
    the resulting node embeddings, and projects to a scalar energy per graph.

    Args:
        in_features: Width of each node feature vector (1 + n_mlips after
            probe augmentation).
        hidden_dim:  Width of all hidden layers.
        n_layers:    Number of message-passing iterations.

    Forward signature::

        forward(node_features, edge_index, batch_vector) -> Tensor

    Returns:
        ``(n_graphs, 1)`` float32 tensor of predicted adsorption energies.
    """

    def __init__(self, in_features: int, hidden_dim: int, n_layers: int) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.input_proj = nn.Linear(in_features, hidden_dim)
        self.mp_linears = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        node_features: Tensor,  # (total_nodes, in_features) float32
        edge_index: Tensor,     # (2, total_edges) long
        batch_vector: Tensor,   # (total_nodes,) long
    ) -> Tensor:                # (n_graphs, 1) float32
        n_nodes = node_features.shape[0]
        n_graphs = int(batch_vector.max().item()) + 1

        h = self.input_proj(node_features)

        edge_src = edge_index[0]
        edge_dst = edge_index[1]

        for layer in self.mp_linears:
            agg = _scatter_mean(h[edge_src], edge_dst, n_nodes)
            h = F.relu(layer(h + agg))

        pooled = _global_mean_pool(h, batch_vector, n_graphs)
        return self.output_proj(pooled)
