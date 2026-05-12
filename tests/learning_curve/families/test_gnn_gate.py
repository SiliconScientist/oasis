from __future__ import annotations

import unittest

import numpy as np
import torch

from oasis.learning_curve.families.gnn_gate import collate_graphs
from oasis.sweep import GraphRecord


def _make_graph(sample_id: int, n_nodes: int, n_features: int, n_edges: int) -> GraphRecord:
    rng = np.random.default_rng(sample_id)
    node_features = rng.random((n_nodes, n_features)).astype(np.float32)
    if n_edges > 0:
        src = rng.integers(0, n_nodes, size=n_edges)
        dst = rng.integers(0, n_nodes, size=n_edges)
        edge_index = np.stack([src, dst], axis=0)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    return GraphRecord(
        sample_id=sample_id,
        node_features=node_features,
        edge_index=edge_index,
    )


class CollateGraphsTests(unittest.TestCase):
    def test_node_count(self) -> None:
        g0 = _make_graph(0, n_nodes=3, n_features=4, n_edges=2)
        g1 = _make_graph(1, n_nodes=5, n_features=4, n_edges=4)
        node_features, _, _ = collate_graphs([g0, g1])
        self.assertEqual(node_features.shape, (8, 4))

    def test_edge_count(self) -> None:
        g0 = _make_graph(0, n_nodes=3, n_features=4, n_edges=2)
        g1 = _make_graph(1, n_nodes=5, n_features=4, n_edges=4)
        _, edge_index, _ = collate_graphs([g0, g1])
        self.assertEqual(edge_index.shape, (2, 6))

    def test_edge_offsets_applied(self) -> None:
        g0 = _make_graph(0, n_nodes=3, n_features=4, n_edges=2)
        g1 = _make_graph(1, n_nodes=5, n_features=4, n_edges=4)
        _, edge_index, _ = collate_graphs([g0, g1])
        # g0 has 3 nodes; g1 edges must all be >= 3
        g1_edges = edge_index[:, 2:]
        self.assertTrue((g1_edges >= 3).all().item())

    def test_batch_vector_shape(self) -> None:
        g0 = _make_graph(0, n_nodes=3, n_features=4, n_edges=2)
        g1 = _make_graph(1, n_nodes=5, n_features=4, n_edges=4)
        _, _, batch_vector = collate_graphs([g0, g1])
        self.assertEqual(batch_vector.shape, (8,))

    def test_batch_vector_values(self) -> None:
        g0 = _make_graph(0, n_nodes=3, n_features=4, n_edges=2)
        g1 = _make_graph(1, n_nodes=5, n_features=4, n_edges=4)
        _, _, batch_vector = collate_graphs([g0, g1])
        expected = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)
        self.assertTrue(torch.equal(batch_vector, expected))

    def test_single_graph(self) -> None:
        g = _make_graph(0, n_nodes=4, n_features=2, n_edges=3)
        node_features, edge_index, batch_vector = collate_graphs([g])
        self.assertEqual(node_features.shape, (4, 2))
        self.assertEqual(edge_index.shape, (2, 3))
        self.assertTrue((batch_vector == 0).all().item())

    def test_empty_edges(self) -> None:
        g = _make_graph(0, n_nodes=3, n_features=2, n_edges=0)
        node_features, edge_index, batch_vector = collate_graphs([g])
        self.assertEqual(node_features.shape, (3, 2))
        self.assertEqual(edge_index.shape, (2, 0))


if __name__ == "__main__":
    unittest.main()
