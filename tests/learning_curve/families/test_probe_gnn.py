from __future__ import annotations

import unittest

import numpy as np
import torch

from oasis.learning_curve.families.gnn_gate import collate_graphs
from oasis.learning_curve.families.probe_gnn import ProbeGnnEncoder
from oasis.sweep import GraphRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_probe_graph(
    sample_id: int,
    n_nodes: int,
    n_features: int,
    n_edges: int,
    *,
    rng_seed: int | None = None,
) -> GraphRecord:
    """Build a GraphRecord with float probe-style node features."""
    rng = np.random.default_rng(rng_seed if rng_seed is not None else sample_id)
    node_features = rng.random((n_nodes, n_features)).astype(np.float32)
    if n_edges > 0:
        src = rng.integers(0, n_nodes, size=n_edges)
        dst = rng.integers(0, n_nodes, size=n_edges)
        edge_index = np.stack([src, dst], axis=0).astype(np.int64)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    return GraphRecord(
        sample_id=sample_id,
        node_features=node_features,
        edge_index=edge_index,
    )


def _batched(
    n_graphs: int = 3,
    n_nodes_each: int = 4,
    n_features: int = 6,
    n_edges_each: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    graphs = [
        _make_probe_graph(i, n_nodes=n_nodes_each, n_features=n_features, n_edges=n_edges_each)
        for i in range(n_graphs)
    ]
    return collate_graphs(graphs)


def _encoder(
    in_features: int = 6,
    hidden_dim: int = 16,
    n_layers: int = 2,
) -> ProbeGnnEncoder:
    return ProbeGnnEncoder(in_features=in_features, hidden_dim=hidden_dim, n_layers=n_layers)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class ProbeGnnEncoderTests(unittest.TestCase):
    def test_output_shape_is_n_graphs_by_1(self) -> None:
        node_features, edge_index, batch_vector = _batched(n_graphs=4)
        enc = _encoder()
        out = enc(node_features, edge_index, batch_vector)
        self.assertEqual(out.shape, (4, 1))

    def test_output_shape_single_graph(self) -> None:
        node_features, edge_index, batch_vector = _batched(n_graphs=1)
        enc = _encoder()
        out = enc(node_features, edge_index, batch_vector)
        self.assertEqual(out.shape, (1, 1))

    def test_output_no_nans(self) -> None:
        node_features, edge_index, batch_vector = _batched()
        enc = _encoder()
        out = enc(node_features, edge_index, batch_vector)
        self.assertFalse(torch.isnan(out).any().item())

    def test_output_dtype_is_float32(self) -> None:
        node_features, edge_index, batch_vector = _batched()
        enc = _encoder()
        out = enc(node_features, edge_index, batch_vector)
        self.assertEqual(out.dtype, torch.float32)

    def test_gradients_flow_to_all_layers(self) -> None:
        node_features, edge_index, batch_vector = _batched()
        enc = _encoder()
        out = enc(node_features, edge_index, batch_vector)
        out.sum().backward()
        grads = [p.grad for p in enc.parameters() if p.grad is not None]
        self.assertGreater(len(grads), 0)
        self.assertTrue(any(g.abs().sum().item() > 0 for g in grads))

    def test_single_layer(self) -> None:
        node_features, edge_index, batch_vector = _batched(n_graphs=2)
        enc = _encoder(n_layers=1)
        out = enc(node_features, edge_index, batch_vector)
        self.assertEqual(out.shape, (2, 1))
        self.assertFalse(torch.isnan(out).any().item())

    def test_no_edges_produces_finite_output(self) -> None:
        graphs = [
            _make_probe_graph(i, n_nodes=3, n_features=6, n_edges=0)
            for i in range(3)
        ]
        node_features, edge_index, batch_vector = collate_graphs(graphs)
        enc = _encoder()
        out = enc(node_features, edge_index, batch_vector)
        self.assertEqual(out.shape, (3, 1))
        self.assertFalse(torch.isnan(out).any().item())

    def test_different_graphs_produce_different_predictions(self) -> None:
        # Two graphs with very different node features should give different outputs.
        rng = np.random.default_rng(0)
        g0 = GraphRecord(
            sample_id=0,
            node_features=np.zeros((3, 6), dtype=np.float32),
            edge_index=np.zeros((2, 0), dtype=np.int64),
        )
        g1 = GraphRecord(
            sample_id=1,
            node_features=np.ones((3, 6), dtype=np.float32) * 10.0,
            edge_index=np.zeros((2, 0), dtype=np.int64),
        )
        node_features, edge_index, batch_vector = collate_graphs([g0, g1])
        enc = _encoder()
        out = enc(node_features, edge_index, batch_vector)
        self.assertNotAlmostEqual(out[0].item(), out[1].item(), places=4)

    def test_output_independent_between_graphs_in_batch(self) -> None:
        # Predictions for each graph should not change when extra graphs are added.
        g0 = _make_probe_graph(0, n_nodes=4, n_features=6, n_edges=3)
        g1 = _make_probe_graph(1, n_nodes=4, n_features=6, n_edges=3)
        nf0, ei0, bv0 = collate_graphs([g0])
        nf01, ei01, bv01 = collate_graphs([g0, g1])

        torch.manual_seed(99)
        enc = _encoder()
        enc.eval()

        with torch.no_grad():
            out_solo = enc(nf0, ei0, bv0)
            out_batch = enc(nf01, ei01, bv01)

        torch.testing.assert_close(out_solo[0], out_batch[0])

    def test_varying_hidden_dims(self) -> None:
        node_features, edge_index, batch_vector = _batched(n_graphs=2)
        for hidden_dim in (8, 32, 64):
            enc = ProbeGnnEncoder(in_features=6, hidden_dim=hidden_dim, n_layers=2)
            out = enc(node_features, edge_index, batch_vector)
            self.assertEqual(out.shape, (2, 1))

    def test_n_layers_zero_not_supported_but_positive_layers_work(self) -> None:
        for n_layers in (1, 2, 3):
            node_features, edge_index, batch_vector = _batched(n_graphs=2)
            enc = ProbeGnnEncoder(in_features=6, hidden_dim=16, n_layers=n_layers)
            out = enc(node_features, edge_index, batch_vector)
            self.assertEqual(out.shape, (2, 1))
            self.assertFalse(torch.isnan(out).any().item())


if __name__ == "__main__":
    unittest.main()
