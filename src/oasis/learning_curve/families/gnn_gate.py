from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
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
