from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from oasis.config import MoETrainingConfig
from oasis.learning_curve.families.gnn_gate import (
    GnnEncoder,
    GnnGateModel,
    GnnGateTuningSpec,
    collate_graphs,
    collate_graphs_with_distances,
)
from oasis.sweep import GraphDatasetView, GraphRecord, SweepDataset, TrainValTestSweepRunnerInput
from oasis.tune import LearnedTrialTuningSpec


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


def _make_graph_with_distances(
    sample_id: int, n_nodes: int, n_edges: int, *, rng_seed: int | None = None
) -> GraphRecord:
    rng = np.random.default_rng(rng_seed if rng_seed is not None else sample_id)
    node_features = rng.random((n_nodes, 1)).astype(np.float32)
    if n_edges > 0:
        src = rng.integers(0, n_nodes, size=n_edges)
        dst = rng.integers(0, n_nodes, size=n_edges)
        edge_index = np.stack([src, dst], axis=0).astype(np.int64)
        distances = (rng.random(n_edges) * 5.0).astype(np.float32)
        edge_features = distances.reshape(-1, 1)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_features = np.zeros((0, 1), dtype=np.float32)
    return GraphRecord(
        sample_id=sample_id,
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
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


@dataclass
class _GnnMockTrial:
    """Minimal Optuna trial stub for GnnGateTuningSpec tests."""

    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 32
    n_layers: int = 1

    def __post_init__(self) -> None:
        self.params: dict[str, Any] = {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
        }

    def suggest_float(self, name: str, low: float, high: float, **kwargs: Any) -> float:
        del low, high, kwargs
        return self.params[name]

    def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
        del choices
        return self.params[name]

    def suggest_int(self, name: str, low: int, high: int) -> int:
        del low, high
        return self.params[name]


def _make_split_with_graphs(
    n_samples: int = 6,
    n_node_features: int = 4,
    n_experts: int = 2,
) -> TrainValTestSweepRunnerInput:
    rng = np.random.default_rng(42)
    graphs = [
        _make_graph(i, n_nodes=4, n_features=n_node_features, n_edges=3)
        for i in range(n_samples)
    ]
    graph_view = GraphDatasetView.from_records(graphs)
    dataset = SweepDataset(
        mlip_features=rng.random((n_samples, n_experts)).astype(np.float32),
        targets=np.array([1.5, 2.5, 3.5, 1.0, 5.5, 6.5]),
        sample_ids=np.arange(n_samples),
        graph_view=graph_view,
    )
    return TrainValTestSweepRunnerInput(
        dataset=dataset,
        sweep_size=4,
        train_idx=np.array([0, 1, 2]),
        val_idx=np.array([3]),
        test_idx=np.array([4, 5]),
    )


def _fast_spec(hidden_dims: tuple[int, ...] = (8,)) -> GnnGateTuningSpec:
    return GnnGateTuningSpec(
        training_cfg=MoETrainingConfig(epochs=2, seed=0),
        hidden_dims=hidden_dims,
    )


class GnnGateTuningSpecTests(unittest.TestCase):
    def test_is_learned_trial_tuning_spec(self) -> None:
        self.assertIsInstance(_fast_spec(), LearnedTrialTuningSpec)

    def test_build_trial_objective_returns_finite_rmse(self) -> None:
        split = _make_split_with_graphs()
        objective = _fast_spec().build_trial_objective(split)
        rmse = objective(_GnnMockTrial())
        self.assertTrue(np.isfinite(rmse))
        self.assertGreater(rmse, 0.0)

    def test_build_trial_objective_search_arch_returns_finite_rmse(self) -> None:
        # hidden_dims=() so Optuna suggest_categorical / suggest_int are exercised.
        split = _make_split_with_graphs()
        objective = _fast_spec(hidden_dims=()).build_trial_objective(split)
        rmse = objective(_GnnMockTrial(hidden_dim=32, n_layers=1))
        self.assertTrue(np.isfinite(rmse))

    def test_round_trip_predict_shape_and_finite(self) -> None:
        split = _make_split_with_graphs()
        spec = _fast_spec()
        trial = _GnnMockTrial()
        model = spec.fit_selected_model(split, trial, refit_policy="train_plus_val")
        self.assertIsInstance(model, GnnGateModel)
        preds = spec.predict(model, split.dataset_subsets().test)
        self.assertEqual(preds.shape, (2,))
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_refit_policy_train_only_returns_gnn_gate_model(self) -> None:
        split = _make_split_with_graphs()
        model = _fast_spec().fit_selected_model(
            split, _GnnMockTrial(), refit_policy="train_only"
        )
        self.assertIsInstance(model, GnnGateModel)

    def test_trial_metadata_has_expected_keys(self) -> None:
        split = _make_split_with_graphs()
        spec = _fast_spec()
        trial = _GnnMockTrial()
        model = spec.fit_selected_model(split, trial, refit_policy="train_plus_val")
        metadata = spec.trial_metadata(trial, model)
        for key in ("hidden_dim", "n_layers", "n_experts", "bias"):
            self.assertIn(key, metadata)

    def test_trial_metadata_n_experts_matches_dataset(self) -> None:
        split = _make_split_with_graphs(n_experts=3)
        spec = GnnGateTuningSpec(
            training_cfg=MoETrainingConfig(epochs=2, seed=0),
            hidden_dims=(8,),
        )
        trial = _GnnMockTrial()
        model = spec.fit_selected_model(split, trial, refit_policy="train_plus_val")
        self.assertEqual(spec.trial_metadata(trial, model)["n_experts"], 3)

    def test_rejects_single_mlip_expert(self) -> None:
        split = _make_split_with_graphs(n_experts=1)

        with self.assertRaisesRegex(
            ValueError,
            "moe requires at least 2 MLIP feature columns; got 1.",
        ):
            _fast_spec().build_trial_objective(split)


class CollateGraphsWithDistancesTests(unittest.TestCase):
    # --- return shape ---

    def test_returns_four_tensors(self) -> None:
        g0 = _make_graph_with_distances(0, n_nodes=3, n_edges=2)
        result = collate_graphs_with_distances([g0])
        self.assertEqual(len(result), 4)

    def test_edge_distances_shape_single_graph(self) -> None:
        g0 = _make_graph_with_distances(0, n_nodes=3, n_edges=5)
        _, _, _, edge_distances = collate_graphs_with_distances([g0])
        self.assertEqual(edge_distances.shape, (5,))

    def test_edge_distances_shape_multi_graph(self) -> None:
        g0 = _make_graph_with_distances(0, n_nodes=3, n_edges=2)
        g1 = _make_graph_with_distances(1, n_nodes=4, n_edges=4)
        _, _, _, edge_distances = collate_graphs_with_distances([g0, g1])
        self.assertEqual(edge_distances.shape, (6,))

    def test_edge_distances_dtype_is_float32(self) -> None:
        g0 = _make_graph_with_distances(0, n_nodes=3, n_edges=3)
        _, _, _, edge_distances = collate_graphs_with_distances([g0])
        self.assertEqual(edge_distances.dtype, torch.float32)

    def test_empty_edges(self) -> None:
        g0 = _make_graph_with_distances(0, n_nodes=3, n_edges=0)
        _, _, _, edge_distances = collate_graphs_with_distances([g0])
        self.assertEqual(edge_distances.shape, (0,))

    # --- distance values are preserved and ordered correctly ---

    def test_distances_match_source_graph_single(self) -> None:
        g0 = _make_graph_with_distances(0, n_nodes=3, n_edges=4)
        _, _, _, edge_distances = collate_graphs_with_distances([g0])
        expected = torch.tensor(g0.edge_features.squeeze(-1), dtype=torch.float32)
        torch.testing.assert_close(edge_distances, expected)

    def test_distances_first_graph_unaffected_by_second(self) -> None:
        g0 = _make_graph_with_distances(0, n_nodes=3, n_edges=3)
        g1 = _make_graph_with_distances(1, n_nodes=4, n_edges=5)
        _, _, _, edge_distances = collate_graphs_with_distances([g0, g1])
        expected_g0 = torch.tensor(g0.edge_features.squeeze(-1), dtype=torch.float32)
        torch.testing.assert_close(edge_distances[:3], expected_g0)

    def test_distances_second_graph_appended_correctly(self) -> None:
        g0 = _make_graph_with_distances(0, n_nodes=3, n_edges=3)
        g1 = _make_graph_with_distances(1, n_nodes=4, n_edges=5)
        _, _, _, edge_distances = collate_graphs_with_distances([g0, g1])
        expected_g1 = torch.tensor(g1.edge_features.squeeze(-1), dtype=torch.float32)
        torch.testing.assert_close(edge_distances[3:], expected_g1)

    # --- node_features / edge_index / batch_vector match collate_graphs ---

    def test_node_features_match_collate_graphs(self) -> None:
        g0 = _make_graph_with_distances(0, n_nodes=3, n_edges=2)
        g1 = _make_graph_with_distances(1, n_nodes=4, n_edges=3)
        nf_with, ei_with, bv_with, _ = collate_graphs_with_distances([g0, g1])
        nf_base, ei_base, bv_base = collate_graphs([g0, g1])
        torch.testing.assert_close(nf_with, nf_base)
        self.assertTrue(torch.equal(ei_with, ei_base))
        self.assertTrue(torch.equal(bv_with, bv_base))

    def test_node_offset_applied_to_edge_index(self) -> None:
        g0 = _make_graph_with_distances(0, n_nodes=3, n_edges=2)
        g1 = _make_graph_with_distances(1, n_nodes=5, n_edges=4)
        _, edge_index, _, _ = collate_graphs_with_distances([g0, g1])
        # g0 has 3 nodes; all g1 edge indices must be >= 3.
        g1_edges = edge_index[:, 2:]
        self.assertTrue((g1_edges >= 3).all().item())

    # --- 1-D edge_features (already flat) ---

    def test_accepts_1d_edge_features(self) -> None:
        rng = np.random.default_rng(7)
        node_features = rng.random((4, 1)).astype(np.float32)
        edge_index = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
        distances_1d = rng.random(3).astype(np.float32)
        g = GraphRecord(
            sample_id=99,
            node_features=node_features,
            edge_index=edge_index,
            edge_features=distances_1d,  # 1-D, not (n, 1)
        )
        _, _, _, edge_distances = collate_graphs_with_distances([g])
        self.assertEqual(edge_distances.shape, (3,))
        torch.testing.assert_close(edge_distances, torch.tensor(distances_1d))

    # --- error handling ---

    def test_raises_if_edge_features_none(self) -> None:
        g = _make_graph(0, n_nodes=3, n_features=1, n_edges=2)  # no edge_features
        with self.assertRaises(ValueError):
            collate_graphs_with_distances([g])

    def test_raises_if_any_graph_missing_edge_features(self) -> None:
        g0 = _make_graph_with_distances(0, n_nodes=3, n_edges=2)
        g1 = _make_graph(1, n_nodes=3, n_features=1, n_edges=2)
        with self.assertRaises(ValueError):
            collate_graphs_with_distances([g0, g1])

    def test_raises_if_edge_features_multi_column(self) -> None:
        rng = np.random.default_rng(0)
        node_features = rng.random((3, 1)).astype(np.float32)
        edge_index = np.array([[0, 1], [1, 2]], dtype=np.int64)
        edge_features = rng.random((2, 3)).astype(np.float32)  # (n_edges, 3) — invalid
        g = GraphRecord(
            sample_id=0,
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
        )
        with self.assertRaises(ValueError):
            collate_graphs_with_distances([g])


if __name__ == "__main__":
    unittest.main()
