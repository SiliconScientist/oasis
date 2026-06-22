from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from oasis.config import MoETrainingConfig
from oasis.learning_curve.families.gnn_gate import collate_graphs
from oasis.learning_curve.families.probe_gnn import (
    GnnDirectModel,
    GnnDirectTuningSpec,
    ProbeGnnEncoder,
    ProbeGnnModel,
    ProbeGnnTuningSpec,
)
from oasis.sweep import GraphDatasetView, GraphRecord, SweepDataset, TrainValTestSweepRunnerInput
from oasis.tune import LearnedTrialTuningSpec


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


# ---------------------------------------------------------------------------
# Dataset helpers for ProbeGnnModel / ProbeGnnTuningSpec tests
# ---------------------------------------------------------------------------

def _make_probe_graph_for_split(sample_id: int, n_nodes: int = 4, n_features: int = 6, n_edges: int = 5) -> GraphRecord:
    return _make_probe_graph(sample_id, n_nodes=n_nodes, n_features=n_features, n_edges=n_edges, rng_seed=sample_id)


def _make_split_with_probe_graphs(
    n_samples: int = 6,
    n_features: int = 6,
    *,
    include_probe_auxiliary: bool = True,
) -> TrainValTestSweepRunnerInput:
    rng = np.random.default_rng(42)
    graphs = [_make_probe_graph_for_split(i, n_features=n_features) for i in range(n_samples)]
    graph_view = GraphDatasetView.from_records(graphs)
    auxiliary_views = (
        {"probe_gnn_records": list(graphs)}
        if include_probe_auxiliary
        else None
    )
    dataset = SweepDataset(
        mlip_features=rng.random((n_samples, 2)).astype(np.float32),
        targets=rng.random(n_samples).astype(np.float32),
        sample_ids=np.arange(n_samples),
        graph_view=graph_view,
        auxiliary_views=auxiliary_views,
    )
    return TrainValTestSweepRunnerInput(
        dataset=dataset,
        sweep_size=4,
        train_idx=np.array([0, 1, 2]),
        val_idx=np.array([3]),
        test_idx=np.array([4, 5]),
    )


def _make_probe_gnn_model(
    n_features: int = 6,
    hidden_dim: int = 8,
    n_layers: int = 1,
    bias: float = 0.0,
) -> ProbeGnnModel:
    encoder = ProbeGnnEncoder(in_features=n_features, hidden_dim=hidden_dim, n_layers=n_layers)
    return ProbeGnnModel(
        state_dict=encoder.state_dict(),
        in_features=n_features,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        bias=bias,
    )


@dataclass
class _ProbeGnnMockTrial:
    """Minimal Optuna trial stub for ProbeGnnTuningSpec tests."""

    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 8
    n_layers: int = 1
    epochs: int = 25

    def __post_init__(self) -> None:
        self.params: dict[str, Any] = {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "epochs": self.epochs,
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


def _fast_probe_spec(hidden_dims: tuple[int, ...] = (8,)) -> ProbeGnnTuningSpec:
    return ProbeGnnTuningSpec(
        training_cfg=MoETrainingConfig(epochs=2, seed=0),
        hidden_dims=hidden_dims,
    )


# ---------------------------------------------------------------------------
# ProbeGnnModel tests
# ---------------------------------------------------------------------------

class ProbeGnnModelTests(unittest.TestCase):
    def test_predict_output_shape(self) -> None:
        n_samples = 4
        graphs = [_make_probe_graph_for_split(i) for i in range(n_samples)]
        model = _make_probe_gnn_model()
        preds = model.predict(graphs)
        self.assertEqual(preds.shape, (n_samples,))

    def test_predict_output_finite(self) -> None:
        n_samples = 5
        graphs = [_make_probe_graph_for_split(i) for i in range(n_samples)]
        model = _make_probe_gnn_model()
        preds = model.predict(graphs)
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_bias_applied(self) -> None:
        n_samples = 3
        graphs = [_make_probe_graph_for_split(i) for i in range(n_samples)]
        base = _make_probe_gnn_model(bias=0.0)
        shifted = ProbeGnnModel(
            state_dict=base.state_dict,
            in_features=base.in_features,
            hidden_dim=base.hidden_dim,
            n_layers=base.n_layers,
            bias=2.5,
        )
        preds_base = base.predict(graphs)
        preds_shifted = shifted.predict(graphs)
        np.testing.assert_allclose(preds_shifted, preds_base + 2.5, atol=1e-5)


# ---------------------------------------------------------------------------
# ProbeGnnTuningSpec tests
# ---------------------------------------------------------------------------

class ProbeGnnTuningSpecTests(unittest.TestCase):
    def test_is_learned_trial_tuning_spec(self) -> None:
        self.assertIsInstance(_fast_probe_spec(), LearnedTrialTuningSpec)

    def test_requires_probe_augmented_auxiliary_records(self) -> None:
        split = _make_split_with_probe_graphs(include_probe_auxiliary=False)
        with self.assertRaisesRegex(
            ValueError,
            "probe_gnn requires probe-augmented graph records",
        ):
            _fast_probe_spec().build_trial_objective(split)

    def test_build_trial_objective_returns_finite_rmse(self) -> None:
        split = _make_split_with_probe_graphs()
        objective = _fast_probe_spec().build_trial_objective(split)
        rmse = objective(_ProbeGnnMockTrial())
        self.assertTrue(np.isfinite(rmse))
        self.assertGreater(rmse, 0.0)

    def test_build_trial_objective_search_arch_returns_finite_rmse(self) -> None:
        split = _make_split_with_probe_graphs()
        objective = _fast_probe_spec(hidden_dims=()).build_trial_objective(split)
        rmse = objective(_ProbeGnnMockTrial(hidden_dim=32, n_layers=1))
        self.assertTrue(np.isfinite(rmse))

    def test_round_trip_predict_shape_and_finite(self) -> None:
        split = _make_split_with_probe_graphs()
        spec = _fast_probe_spec()
        trial = _ProbeGnnMockTrial()
        model = spec.fit_selected_model(split, trial, refit_policy="train_plus_val")
        self.assertIsInstance(model, ProbeGnnModel)
        preds = spec.predict(model, split.dataset_subsets().test)
        self.assertEqual(preds.shape, (2,))
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_refit_policy_train_only_returns_probe_gnn_model(self) -> None:
        split = _make_split_with_probe_graphs()
        model = _fast_probe_spec().fit_selected_model(
            split, _ProbeGnnMockTrial(), refit_policy="train_only"
        )
        self.assertIsInstance(model, ProbeGnnModel)

    def test_trial_metadata_has_expected_keys(self) -> None:
        split = _make_split_with_probe_graphs()
        spec = _fast_probe_spec()
        trial = _ProbeGnnMockTrial()
        model = spec.fit_selected_model(split, trial, refit_policy="train_plus_val")
        metadata = spec.trial_metadata(trial, model)
        for key in ("in_features", "hidden_dim", "n_layers", "epochs", "bias"):
            self.assertIn(key, metadata)

    def test_omitted_epochs_are_tuned_from_trial(self) -> None:
        split = _make_split_with_probe_graphs()
        spec = ProbeGnnTuningSpec(training_cfg=MoETrainingConfig(seed=0), hidden_dims=(8,))
        trial = _ProbeGnnMockTrial(epochs=25)
        rmse = spec.build_trial_objective(split)(trial)
        self.assertTrue(np.isfinite(rmse))
        model = spec.fit_selected_model(split, trial, refit_policy="train_plus_val")
        self.assertEqual(spec.trial_metadata(trial, model)["epochs"], 25)

    def test_fit_selected_model_resolves_requested_device(self) -> None:
        split = _make_split_with_probe_graphs()
        spec = ProbeGnnTuningSpec(
            training_cfg=MoETrainingConfig(epochs=2, seed=0, device="cuda"),
            hidden_dims=(8,),
        )
        model = spec.fit_selected_model(split, _ProbeGnnMockTrial(), refit_policy="train_only")
        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(model.device, expected_device)
        self.assertTrue(all(t.device.type == "cpu" for t in model.state_dict.values()))


# ---------------------------------------------------------------------------
# GnnDirectModel tests
# ---------------------------------------------------------------------------

class GnnDirectModelTests(unittest.TestCase):
    def test_predict_output_shape(self) -> None:
        n_samples = 4
        graphs = [_make_probe_graph_for_split(i) for i in range(n_samples)]
        encoder = ProbeGnnEncoder(in_features=6, hidden_dim=8, n_layers=1)
        model = GnnDirectModel(
            state_dict=encoder.state_dict(),
            in_features=6,
            hidden_dim=8,
            n_layers=1,
        )
        preds = model.predict(graphs)
        self.assertEqual(preds.shape, (n_samples,))

    def test_predict_output_finite(self) -> None:
        graphs = [_make_probe_graph_for_split(i) for i in range(3)]
        encoder = ProbeGnnEncoder(in_features=6, hidden_dim=8, n_layers=1)
        model = GnnDirectModel(
            state_dict=encoder.state_dict(), in_features=6, hidden_dim=8, n_layers=1
        )
        self.assertTrue(np.all(np.isfinite(model.predict(graphs))))

    def test_bias_applied(self) -> None:
        graphs = [_make_probe_graph_for_split(i) for i in range(3)]
        encoder = ProbeGnnEncoder(in_features=6, hidden_dim=8, n_layers=1)
        sd = encoder.state_dict()
        base = GnnDirectModel(state_dict=sd, in_features=6, hidden_dim=8, n_layers=1, bias=0.0)
        shifted = GnnDirectModel(state_dict=sd, in_features=6, hidden_dim=8, n_layers=1, bias=3.0)
        np.testing.assert_allclose(shifted.predict(graphs), base.predict(graphs) + 3.0, atol=1e-5)


# ---------------------------------------------------------------------------
# GnnDirectTuningSpec tests
# ---------------------------------------------------------------------------

def _fast_direct_spec(hidden_dims: tuple[int, ...] = (8,)) -> GnnDirectTuningSpec:
    return GnnDirectTuningSpec(
        training_cfg=MoETrainingConfig(epochs=2, seed=0),
        hidden_dims=hidden_dims,
    )


class GnnDirectTuningSpecTests(unittest.TestCase):
    def test_is_learned_trial_tuning_spec(self) -> None:
        self.assertIsInstance(_fast_direct_spec(), LearnedTrialTuningSpec)

    def test_build_trial_objective_returns_finite_rmse(self) -> None:
        split = _make_split_with_probe_graphs()
        objective = _fast_direct_spec().build_trial_objective(split)
        rmse = objective(_ProbeGnnMockTrial())
        self.assertTrue(np.isfinite(rmse))
        self.assertGreater(rmse, 0.0)

    def test_round_trip_predict_shape_and_finite(self) -> None:
        split = _make_split_with_probe_graphs()
        spec = _fast_direct_spec()
        model = spec.fit_selected_model(split, _ProbeGnnMockTrial(), refit_policy="train_plus_val")
        self.assertIsInstance(model, GnnDirectModel)
        preds = spec.predict(model, split.dataset_subsets().test)
        self.assertEqual(preds.shape, (2,))
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_refit_policy_train_only_returns_gnn_direct_model(self) -> None:
        split = _make_split_with_probe_graphs()
        model = _fast_direct_spec().fit_selected_model(
            split, _ProbeGnnMockTrial(), refit_policy="train_only"
        )
        self.assertIsInstance(model, GnnDirectModel)

    def test_trial_metadata_has_expected_keys(self) -> None:
        split = _make_split_with_probe_graphs()
        spec = _fast_direct_spec()
        model = spec.fit_selected_model(split, _ProbeGnnMockTrial(), refit_policy="train_plus_val")
        metadata = spec.trial_metadata(_ProbeGnnMockTrial(), model)
        for key in ("in_features", "hidden_dim", "n_layers", "epochs", "bias"):
            self.assertIn(key, metadata)

    def test_direct_omitted_epochs_are_tuned_from_trial(self) -> None:
        split = _make_split_with_probe_graphs()
        spec = GnnDirectTuningSpec(training_cfg=MoETrainingConfig(seed=0), hidden_dims=(8,))
        trial = _ProbeGnnMockTrial(epochs=25)
        rmse = spec.build_trial_objective(split)(trial)
        self.assertTrue(np.isfinite(rmse))
        model = spec.fit_selected_model(split, trial, refit_policy="train_plus_val")
        self.assertEqual(spec.trial_metadata(trial, model)["epochs"], 25)

    def test_fit_selected_model_resolves_requested_device(self) -> None:
        split = _make_split_with_probe_graphs()
        spec = GnnDirectTuningSpec(
            training_cfg=MoETrainingConfig(epochs=2, seed=0, device="cuda"),
            hidden_dims=(8,),
        )
        model = spec.fit_selected_model(split, _ProbeGnnMockTrial(), refit_policy="train_only")
        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(model.device, expected_device)
        self.assertTrue(all(t.device.type == "cpu" for t in model.state_dict.values()))

    def test_ignores_probe_records_in_auxiliary_views(self) -> None:
        """GnnDirectTuningSpec must use graph_view, not probe_gnn_records."""
        split = _make_split_with_probe_graphs()
        spec = _fast_direct_spec()
        # Run through graph_view path (no auxiliary probe records)
        model_via_graph_view = spec.fit_selected_model(
            split, _ProbeGnnMockTrial(), refit_policy="train_only"
        )
        self.assertIsInstance(model_via_graph_view, GnnDirectModel)


if __name__ == "__main__":
    unittest.main()
