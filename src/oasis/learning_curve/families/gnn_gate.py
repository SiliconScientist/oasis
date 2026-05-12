from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from oasis.sweep import GraphRecord


def collate_graphs(graphs: Sequence[GraphRecord]) -> tuple[Tensor, Tensor, Tensor]:
    """Batch a sequence of GraphRecords into sparse tensors.

    Returns:
        node_features: (total_nodes, n_features) float32
        edge_index:    (2, total_edges) long, with per-graph node offsets applied
        batch_vector:  (total_nodes,) long, graph index for each node
    """
    all_node_features: list[Tensor] = []
    all_edge_index: list[Tensor] = []
    all_batch: list[Tensor] = []
    node_offset = 0

    for graph_idx, graph in enumerate(graphs):
        n = graph.n_nodes
        all_node_features.append(
            torch.tensor(graph.node_features, dtype=torch.float32)
        )
        ei = torch.tensor(graph.edge_index, dtype=torch.long) + node_offset
        all_edge_index.append(ei)
        all_batch.append(torch.full((n,), graph_idx, dtype=torch.long))
        node_offset += n

    node_features = torch.cat(all_node_features, dim=0)
    edge_index = torch.cat(all_edge_index, dim=1) if all_edge_index else torch.zeros((2, 0), dtype=torch.long)
    batch_vector = torch.cat(all_batch, dim=0)

    return node_features, edge_index, batch_vector


def _scatter_mean_nodes(
    src_features: Tensor,
    dst_indices: Tensor,
    n_nodes: int,
) -> Tensor:
    """Mean-aggregate src_features into destination node buckets."""
    hidden = src_features.shape[1]
    out = torch.zeros(n_nodes, hidden, dtype=src_features.dtype, device=src_features.device)
    if src_features.shape[0] == 0:
        return out
    idx = dst_indices.unsqueeze(1).expand_as(src_features)
    out.scatter_add_(0, idx, src_features)
    count = torch.zeros(n_nodes, 1, dtype=src_features.dtype, device=src_features.device)
    count.scatter_add_(0, dst_indices.unsqueeze(1), torch.ones(dst_indices.shape[0], 1, dtype=src_features.dtype, device=src_features.device))
    return out / count.clamp(min=1)


def _global_mean_pool(x: Tensor, batch: Tensor, n_graphs: int) -> Tensor:
    """Mean-pool node features into per-graph vectors."""
    out = torch.zeros(n_graphs, x.shape[1], dtype=x.dtype, device=x.device)
    idx = batch.unsqueeze(1).expand_as(x)
    out.scatter_add_(0, idx, x)
    count = torch.zeros(n_graphs, 1, dtype=x.dtype, device=x.device)
    count.scatter_add_(0, batch.unsqueeze(1), torch.ones(batch.shape[0], 1, dtype=x.dtype, device=x.device))
    return out / count.clamp(min=1)


class GnnEncoder(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        out_features: int,
        n_layers: int,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.input_proj = nn.Linear(in_features, hidden_dim)
        self.mp_linears = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.output_proj = nn.Linear(hidden_dim, out_features)

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        batch_vector: Tensor,
    ) -> Tensor:
        n_nodes = node_features.shape[0]
        n_graphs = int(batch_vector.max().item()) + 1

        h = self.input_proj(node_features)  # (n_nodes, hidden_dim)

        edge_src = edge_index[0]
        edge_dst = edge_index[1]

        for layer in self.mp_linears:
            agg = _scatter_mean_nodes(h[edge_src], edge_dst, n_nodes)
            h = F.relu(layer(h + agg))

        pooled = _global_mean_pool(h, batch_vector, n_graphs)  # (n_graphs, hidden_dim)
        return self.output_proj(pooled)  # (n_graphs, out_features)
