from __future__ import annotations

import unittest

import numpy as np
import torch

from oasis.learning_curve.families.gnn_gate import GnnEncoder, GnnGateModel, collate_graphs
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


def _make_batched_input(
    n_graphs: int = 3,
    n_nodes_each: int = 4,
    n_features: int = 5,
    n_edges_each: int = 6,
    n_experts: int = 2,
) -> tuple[Tensor, Tensor, Tensor, int]:
    graphs = [
        _make_graph(i, n_nodes=n_nodes_each, n_features=n_features, n_edges=n_edges_each)
        for i in range(n_graphs)
    ]
    node_features, edge_index, batch_vector = collate_graphs(graphs)
    return node_features, edge_index, batch_vector, n_experts


class GnnEncoderTests(unittest.TestCase):
    def test_output_shape(self) -> None:
        node_features, edge_index, batch_vector, n_experts = _make_batched_input(
            n_graphs=3, n_nodes_each=4, n_features=5, n_experts=2
        )
        encoder = GnnEncoder(in_features=5, hidden_dim=16, out_features=2, n_layers=2)
        out = encoder(node_features, edge_index, batch_vector)
        self.assertEqual(out.shape, (3, 2))

    def test_no_nans(self) -> None:
        node_features, edge_index, batch_vector, n_experts = _make_batched_input()
        encoder = GnnEncoder(in_features=5, hidden_dim=16, out_features=2, n_layers=2)
        out = encoder(node_features, edge_index, batch_vector)
        self.assertFalse(torch.isnan(out).any().item())

    def test_gradients_flow(self) -> None:
        node_features, edge_index, batch_vector, n_experts = _make_batched_input()
        encoder = GnnEncoder(in_features=5, hidden_dim=16, out_features=2, n_layers=2)
        out = encoder(node_features, edge_index, batch_vector)
        loss = out.sum()
        loss.backward()
        grads = [p.grad for p in encoder.parameters() if p.grad is not None]
        self.assertGreater(len(grads), 0)
        self.assertTrue(any(g.abs().sum().item() > 0 for g in grads))

    def test_single_layer(self) -> None:
        node_features, edge_index, batch_vector, _ = _make_batched_input(n_graphs=2)
        encoder = GnnEncoder(in_features=5, hidden_dim=8, out_features=3, n_layers=1)
        out = encoder(node_features, edge_index, batch_vector)
        self.assertEqual(out.shape, (2, 3))

    def test_single_graph(self) -> None:
        node_features, edge_index, batch_vector, _ = _make_batched_input(n_graphs=1)
        encoder = GnnEncoder(in_features=5, hidden_dim=8, out_features=4, n_layers=2)
        out = encoder(node_features, edge_index, batch_vector)
        self.assertEqual(out.shape, (1, 4))


def _make_gnn_gate_model(
    n_experts: int = 2,
    hidden_dim: int = 8,
    n_layers: int = 1,
    in_features: int = 4,
    bias: float = 0.0,
) -> GnnGateModel:
    encoder = GnnEncoder(
        in_features=in_features,
        hidden_dim=hidden_dim,
        out_features=n_experts,
        n_layers=n_layers,
    )
    return GnnGateModel(
        state_dict=encoder.state_dict(),
        n_experts=n_experts,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        bias=bias,
    )


class GnnGateModelTests(unittest.TestCase):
    def test_predict_output_shape(self) -> None:
        n_samples = 3
        n_experts = 2
        graphs = [_make_graph(i, n_nodes=4, n_features=4, n_edges=3) for i in range(n_samples)]
        X = np.random.default_rng(0).random((n_samples, n_experts)).astype(np.float32)
        model = _make_gnn_gate_model(n_experts=n_experts, in_features=4)
        preds = model.predict(X, graphs)
        self.assertEqual(preds.shape, (n_samples,))

    def test_predict_output_finite(self) -> None:
        n_samples = 4
        n_experts = 3
        graphs = [_make_graph(i, n_nodes=5, n_features=4, n_edges=4) for i in range(n_samples)]
        X = np.random.default_rng(1).random((n_samples, n_experts)).astype(np.float32)
        model = _make_gnn_gate_model(n_experts=n_experts, in_features=4)
        preds = model.predict(X, graphs)
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_predict_bias_applied(self) -> None:
        n_samples = 2
        graphs = [_make_graph(i, n_nodes=3, n_features=4, n_edges=2) for i in range(n_samples)]
        X = np.zeros((n_samples, 2), dtype=np.float32)
        model_no_bias = _make_gnn_gate_model(n_experts=2, in_features=4, bias=0.0)
        model_with_bias = _make_gnn_gate_model(n_experts=2, in_features=4, bias=1.5)
        # Use the same state_dict so weights are identical
        model_with_bias = GnnGateModel(
            state_dict=model_no_bias.state_dict,
            n_experts=model_no_bias.n_experts,
            hidden_dim=model_no_bias.hidden_dim,
            n_layers=model_no_bias.n_layers,
            bias=1.5,
        )
        preds_no_bias = model_no_bias.predict(X, graphs)
        preds_with_bias = model_with_bias.predict(X, graphs)
        np.testing.assert_allclose(preds_with_bias, preds_no_bias + 1.5, atol=1e-5)

    def test_weights_sum_to_one_per_sample(self) -> None:
        n_samples = 5
        n_experts = 3
        graphs = [_make_graph(i, n_nodes=4, n_features=4, n_edges=3) for i in range(n_samples)]
        # X = ones so prediction equals sum(weights) + bias = 1 + bias
        X = np.ones((n_samples, n_experts), dtype=np.float32)
        model = _make_gnn_gate_model(n_experts=n_experts, in_features=4, bias=0.0)
        preds = model.predict(X, graphs)
        np.testing.assert_allclose(preds, np.ones(n_samples), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
