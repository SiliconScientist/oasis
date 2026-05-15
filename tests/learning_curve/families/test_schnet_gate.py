from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from oasis.config import MoETrainingConfig
from oasis.learning_curve.families.rbf import GaussianRBF
from oasis.learning_curve.families.schnet_gate import (
    SchNetEncoder,
    SchNetGateModel,
    SchNetGateTuningSpec,
    SchNetInteraction,
)
from oasis.sweep import GraphDatasetView, GraphRecord, SweepDataset, TrainValTestSweepRunnerInput
from oasis.tune import LearnedTrialTuningSpec


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_batch(
    *,
    n_graphs: int,
    n_nodes_each: int,
    n_edges_each: int,
    n_rbf: int = 20,
    n_experts: int = 3,
    hidden_dim: int = 16,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (atomic_numbers, edge_index, edge_distances, batch_vector)."""
    rng = np.random.default_rng(seed)
    n_nodes_total = n_graphs * n_nodes_each

    atomic_numbers = torch.tensor(
        rng.integers(1, 10, size=n_nodes_total), dtype=torch.long
    )

    ei_parts: list[torch.Tensor] = []
    dist_parts: list[torch.Tensor] = []
    offset = 0
    for _ in range(n_graphs):
        src = rng.integers(0, n_nodes_each, size=n_edges_each)
        dst = rng.integers(0, n_nodes_each, size=n_edges_each)
        ei = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long) + offset
        ei_parts.append(ei)
        dist_parts.append(
            torch.tensor(rng.random(n_edges_each).astype(np.float32) * 5.0)
        )
        offset += n_nodes_each

    edge_index = torch.cat(ei_parts, dim=1)
    edge_distances = torch.cat(dist_parts)
    batch_vector = torch.repeat_interleave(
        torch.arange(n_graphs), n_nodes_each
    )
    return atomic_numbers, edge_index, edge_distances, batch_vector


def _fully_connected_graph(
    atomic_nums: list[int],
    distances: dict[tuple[int, int], float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build tensors for a fully specified small graph (single graph)."""
    n = len(atomic_nums)
    srcs, dsts, dists = [], [], []
    for (i, j), d in distances.items():
        srcs.append(i)
        dsts.append(j)
        dists.append(d)
    atomic_numbers = torch.tensor(atomic_nums, dtype=torch.long)
    edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
    edge_distances = torch.tensor(dists, dtype=torch.float32)
    batch_vector = torch.zeros(n, dtype=torch.long)
    return atomic_numbers, edge_index, edge_distances, batch_vector


# ---------------------------------------------------------------------------
# SchNetInteraction tests
# ---------------------------------------------------------------------------

class SchNetInteractionShapeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hidden_dim = 16
        self.n_rbf = 20
        self.block = SchNetInteraction(self.hidden_dim, self.n_rbf)

    def test_output_shape_matches_input(self) -> None:
        n_nodes, n_edges = 5, 8
        h = torch.randn(n_nodes, self.hidden_dim)
        edge_index = torch.randint(0, n_nodes, (2, n_edges))
        rbf = torch.randn(n_edges, self.n_rbf)
        out = self.block(h, edge_index, rbf)
        self.assertEqual(out.shape, (n_nodes, self.hidden_dim))

    def test_no_nans(self) -> None:
        n_nodes, n_edges = 6, 10
        h = torch.randn(n_nodes, self.hidden_dim)
        edge_index = torch.randint(0, n_nodes, (2, n_edges))
        rbf = torch.randn(n_edges, self.n_rbf)
        out = self.block(h, edge_index, rbf)
        self.assertFalse(torch.isnan(out).any().item())

    def test_gradients_flow(self) -> None:
        n_nodes, n_edges = 4, 6
        h = torch.randn(n_nodes, self.hidden_dim, requires_grad=True)
        edge_index = torch.randint(0, n_nodes, (2, n_edges))
        rbf = torch.randn(n_edges, self.n_rbf)
        out = self.block(h, edge_index, rbf)
        out.sum().backward()
        self.assertIsNotNone(h.grad)
        self.assertTrue(h.grad.abs().sum().item() > 0)

    def test_empty_edges_no_crash(self) -> None:
        n_nodes = 4
        h = torch.randn(n_nodes, self.hidden_dim)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        rbf = torch.zeros((0, self.n_rbf))
        out = self.block(h, edge_index, rbf)
        self.assertEqual(out.shape, (n_nodes, self.hidden_dim))
        self.assertFalse(torch.isnan(out).any().item())


# ---------------------------------------------------------------------------
# SchNetEncoder tests
# ---------------------------------------------------------------------------

class SchNetEncoderShapeTests(unittest.TestCase):
    def _make_encoder(self, **kwargs: object) -> SchNetEncoder:
        defaults = dict(hidden_dim=16, out_features=3, n_layers=2, n_rbf=8)
        defaults.update(kwargs)
        return SchNetEncoder(**defaults)  # type: ignore[arg-type]

    def test_output_shape_multi_graph(self) -> None:
        encoder = self._make_encoder(out_features=4)
        z, ei, d, bv = _make_batch(n_graphs=5, n_nodes_each=3, n_edges_each=4, n_experts=4)
        out = encoder(z, ei, d, bv)
        self.assertEqual(out.shape, (5, 4))

    def test_output_shape_single_graph(self) -> None:
        encoder = self._make_encoder(out_features=2)
        z, ei, d, bv = _make_batch(n_graphs=1, n_nodes_each=4, n_edges_each=6, n_experts=2)
        out = encoder(z, ei, d, bv)
        self.assertEqual(out.shape, (1, 2))

    def test_single_layer(self) -> None:
        encoder = self._make_encoder(n_layers=1, out_features=3)
        z, ei, d, bv = _make_batch(n_graphs=3, n_nodes_each=3, n_edges_each=4)
        out = encoder(z, ei, d, bv)
        self.assertEqual(out.shape, (3, 3))


class SchNetEncoderValueTests(unittest.TestCase):
    def _encoder(self) -> SchNetEncoder:
        return SchNetEncoder(hidden_dim=16, out_features=3, n_layers=2, n_rbf=8)

    def test_no_nans(self) -> None:
        encoder = self._encoder()
        z, ei, d, bv = _make_batch(n_graphs=3, n_nodes_each=4, n_edges_each=5)
        out = encoder(z, ei, d, bv)
        self.assertFalse(torch.isnan(out).any().item())

    def test_gradients_flow(self) -> None:
        encoder = self._encoder()
        z, ei, d, bv = _make_batch(n_graphs=2, n_nodes_each=3, n_edges_each=4)
        out = encoder(z, ei, d, bv)
        out.sum().backward()
        grads = [p.grad for p in encoder.parameters() if p.grad is not None]
        self.assertGreater(len(grads), 0)
        self.assertTrue(any(g.abs().sum().item() > 0 for g in grads))

    def test_different_atomic_numbers_differ(self) -> None:
        # Same graph topology and distances, but different atom types.
        torch.manual_seed(7)
        encoder = SchNetEncoder(hidden_dim=16, out_features=2, n_layers=1, n_rbf=8)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        distances = torch.tensor([2.0, 2.0])
        bv = torch.zeros(2, dtype=torch.long)

        z_hh = torch.tensor([1, 1], dtype=torch.long)   # H–H
        z_co = torch.tensor([6, 8], dtype=torch.long)   # C–O

        with torch.no_grad():
            out_hh = encoder(z_hh, edge_index, distances, bv)
            out_co = encoder(z_co, edge_index, distances, bv)

        self.assertFalse(torch.allclose(out_hh, out_co))

    def test_empty_edges_no_crash(self) -> None:
        encoder = self._encoder()
        z = torch.tensor([1, 6], dtype=torch.long)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        distances = torch.zeros(0)
        bv = torch.zeros(2, dtype=torch.long)
        out = encoder(z, edge_index, distances, bv)
        self.assertEqual(out.shape, (1, 3))
        self.assertFalse(torch.isnan(out).any().item())


class SchNetEncoderPermutationInvarianceTests(unittest.TestCase):
    """SchNet is invariant to permutation of atom ordering."""

    def _build_permuted(
        self,
        atomic_nums: list[int],
        edge_distances_dict: dict[tuple[int, int], float],
        perm: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply perm to atom indices; relabel all edge endpoints consistently."""
        n = len(atomic_nums)
        inv_perm = [0] * n
        for new_idx, old_idx in enumerate(perm):
            inv_perm[old_idx] = new_idx

        perm_atomic_nums = [atomic_nums[i] for i in perm]

        perm_dist_dict: dict[tuple[int, int], float] = {}
        for (src, dst), d in edge_distances_dict.items():
            new_src = inv_perm[src]
            new_dst = inv_perm[dst]
            perm_dist_dict[(new_src, new_dst)] = d

        return _fully_connected_graph(perm_atomic_nums, perm_dist_dict)

    def test_3atom_graph_permutation(self) -> None:
        torch.manual_seed(0)
        encoder = SchNetEncoder(hidden_dim=16, out_features=3, n_layers=2, n_rbf=8)
        encoder.eval()

        # H, C, O fully connected (bidirectional)
        atomic_nums = [1, 6, 8]
        distances = {
            (0, 1): 1.5, (1, 0): 1.5,
            (0, 2): 2.0, (2, 0): 2.0,
            (1, 2): 2.5, (2, 1): 2.5,
        }

        z, ei, d, bv = _fully_connected_graph(atomic_nums, distances)
        perm = [2, 0, 1]  # O, H, C
        z_p, ei_p, d_p, bv_p = self._build_permuted(atomic_nums, distances, perm)

        with torch.no_grad():
            out_orig = encoder(z, ei, d, bv)
            out_perm = encoder(z_p, ei_p, d_p, bv_p)

        torch.testing.assert_close(out_orig, out_perm, atol=1e-5, rtol=1e-4)

    def test_2atom_swap_invariance(self) -> None:
        torch.manual_seed(1)
        encoder = SchNetEncoder(hidden_dim=8, out_features=2, n_layers=1, n_rbf=8)
        encoder.eval()

        atomic_nums = [1, 6]
        distances = {(0, 1): 1.8, (1, 0): 1.8}

        z, ei, d, bv = _fully_connected_graph(atomic_nums, distances)
        z_p, ei_p, d_p, bv_p = self._build_permuted(atomic_nums, distances, perm=[1, 0])

        with torch.no_grad():
            out_orig = encoder(z, ei, d, bv)
            out_perm = encoder(z_p, ei_p, d_p, bv_p)

        torch.testing.assert_close(out_orig, out_perm, atol=1e-5, rtol=1e-4)


class SchNetEncoderRBFConsistencyTests(unittest.TestCase):
    """Internal RBF expansion must match GaussianRBF from rbf.py."""

    def test_rbf_buffers_match_gaussian_rbf(self) -> None:
        n_rbf, r_max = 16, 6.0
        encoder = SchNetEncoder(
            hidden_dim=8, out_features=2, n_layers=1, n_rbf=n_rbf, r_max=r_max
        )
        rbf = GaussianRBF(n_rbf=n_rbf, r_max=r_max)

        distances_np = np.array([0.5, 1.0, 2.0, 3.5, 5.9], dtype=np.float32)
        expected = rbf(distances_np)  # (5, n_rbf) numpy

        distances_t = torch.tensor(distances_np)
        with torch.no_grad():
            actual = encoder._rbf_expand(distances_t).numpy()

        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_rbf_buffers_are_not_parameters(self) -> None:
        encoder = SchNetEncoder(hidden_dim=8, out_features=2, n_layers=1)
        param_names = {name for name, _ in encoder.named_parameters()}
        self.assertNotIn("_rbf_centers", param_names)
        self.assertNotIn("_rbf_width", param_names)

    def test_rbf_centers_span_zero_to_r_max(self) -> None:
        r_max = 5.0
        n_rbf = 10
        encoder = SchNetEncoder(
            hidden_dim=8, out_features=2, n_layers=1, n_rbf=n_rbf, r_max=r_max
        )
        centers = encoder._rbf_centers
        self.assertAlmostEqual(centers[0].item(), 0.0, places=6)
        self.assertAlmostEqual(centers[-1].item(), r_max, places=6)
        self.assertEqual(centers.shape[0], n_rbf)


# ---------------------------------------------------------------------------
# Dataset helpers for gate framework tests
# ---------------------------------------------------------------------------

def _make_schnet_graph(sample_id: int, n_nodes: int, n_edges: int) -> GraphRecord:
    """GraphRecord with atomic-number node_features and distance edge_features."""
    rng = np.random.default_rng(sample_id)
    # node_features stores atomic numbers as float (n_atoms, 1) — same as atoms_to_graph_record
    atomic_nums = rng.integers(1, 10, size=n_nodes).astype(float)
    node_features = atomic_nums.reshape(-1, 1)
    if n_edges > 0:
        src = rng.integers(0, n_nodes, size=n_edges)
        dst = rng.integers(0, n_nodes, size=n_edges)
        edge_index = np.stack([src, dst], axis=0).astype(np.int64)
        distances = rng.uniform(0.5, 5.0, size=n_edges).astype(np.float32)
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


def _make_split_with_schnet_graphs(
    n_samples: int = 6,
    n_experts: int = 2,
) -> TrainValTestSweepRunnerInput:
    rng = np.random.default_rng(42)
    graphs = [_make_schnet_graph(i, n_nodes=4, n_edges=6) for i in range(n_samples)]
    graph_view = GraphDatasetView.from_records(graphs)
    dataset = SweepDataset(
        mlip_features=rng.random((n_samples, n_experts)).astype(np.float32),
        targets=rng.random(n_samples).astype(np.float32),
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


def _make_schnet_gate_model(
    n_experts: int = 2,
    hidden_dim: int = 8,
    n_layers: int = 1,
    n_rbf: int = 8,
    bias: float = 0.0,
) -> SchNetGateModel:
    encoder = SchNetEncoder(
        hidden_dim=hidden_dim,
        out_features=n_experts,
        n_layers=n_layers,
        n_rbf=n_rbf,
    )
    return SchNetGateModel(
        state_dict=encoder.state_dict(),
        n_experts=n_experts,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_rbf=n_rbf,
        bias=bias,
    )


@dataclass
class _SchNetMockTrial:
    """Minimal Optuna trial stub for SchNetGateTuningSpec tests."""

    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 8
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


def _fast_spec(hidden_dims: tuple[int, ...] = (8,)) -> SchNetGateTuningSpec:
    return SchNetGateTuningSpec(
        training_cfg=MoETrainingConfig(epochs=2, seed=0),
        hidden_dims=hidden_dims,
        n_rbf=8,
    )


# ---------------------------------------------------------------------------
# SchNetGateModel tests
# ---------------------------------------------------------------------------

class SchNetGateModelTests(unittest.TestCase):
    def test_predict_output_shape(self) -> None:
        n_samples, n_experts = 4, 2
        graphs = [_make_schnet_graph(i, n_nodes=3, n_edges=4) for i in range(n_samples)]
        X = np.random.default_rng(0).random((n_samples, n_experts)).astype(np.float32)
        model = _make_schnet_gate_model(n_experts=n_experts)
        preds = model.predict(X, graphs)
        self.assertEqual(preds.shape, (n_samples,))

    def test_predict_output_finite(self) -> None:
        n_samples, n_experts = 5, 3
        graphs = [_make_schnet_graph(i, n_nodes=4, n_edges=5) for i in range(n_samples)]
        X = np.random.default_rng(1).random((n_samples, n_experts)).astype(np.float32)
        model = _make_schnet_gate_model(n_experts=n_experts)
        preds = model.predict(X, graphs)
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_bias_applied(self) -> None:
        n_samples = 3
        graphs = [_make_schnet_graph(i, n_nodes=3, n_edges=3) for i in range(n_samples)]
        X = np.zeros((n_samples, 2), dtype=np.float32)
        base = _make_schnet_gate_model(n_experts=2, bias=0.0)
        shifted = SchNetGateModel(
            state_dict=base.state_dict,
            n_experts=base.n_experts,
            hidden_dim=base.hidden_dim,
            n_layers=base.n_layers,
            n_rbf=base.n_rbf,
            bias=2.5,
        )
        preds_base = base.predict(X, graphs)
        preds_shifted = shifted.predict(X, graphs)
        np.testing.assert_allclose(preds_shifted, preds_base + 2.5, atol=1e-5)

    def test_weights_sum_to_one_per_sample(self) -> None:
        n_samples, n_experts = 4, 3
        graphs = [_make_schnet_graph(i, n_nodes=3, n_edges=4) for i in range(n_samples)]
        # X = ones → prediction = sum(weights) + bias = 1.0 + 0.0
        X = np.ones((n_samples, n_experts), dtype=np.float32)
        model = _make_schnet_gate_model(n_experts=n_experts, bias=0.0)
        preds = model.predict(X, graphs)
        np.testing.assert_allclose(preds, np.ones(n_samples), atol=1e-5)


# ---------------------------------------------------------------------------
# SchNetGateTuningSpec tests
# ---------------------------------------------------------------------------

class SchNetGateTuningSpecTests(unittest.TestCase):
    def test_is_learned_trial_tuning_spec(self) -> None:
        self.assertIsInstance(_fast_spec(), LearnedTrialTuningSpec)

    def test_build_trial_objective_returns_finite_rmse(self) -> None:
        split = _make_split_with_schnet_graphs()
        objective = _fast_spec().build_trial_objective(split)
        rmse = objective(_SchNetMockTrial())
        self.assertTrue(np.isfinite(rmse))
        self.assertGreater(rmse, 0.0)

    def test_build_trial_objective_search_arch_returns_finite_rmse(self) -> None:
        # hidden_dims=() exercises the suggest_categorical / suggest_int path.
        split = _make_split_with_schnet_graphs()
        objective = _fast_spec(hidden_dims=()).build_trial_objective(split)
        rmse = objective(_SchNetMockTrial(hidden_dim=32, n_layers=1))
        self.assertTrue(np.isfinite(rmse))

    def test_round_trip_predict_shape_and_finite(self) -> None:
        split = _make_split_with_schnet_graphs()
        spec = _fast_spec()
        trial = _SchNetMockTrial()
        model = spec.fit_selected_model(split, trial, refit_policy="train_plus_val")
        self.assertIsInstance(model, SchNetGateModel)
        preds = spec.predict(model, split.dataset_subsets().test)
        self.assertEqual(preds.shape, (2,))
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_refit_policy_train_only_returns_schnet_gate_model(self) -> None:
        split = _make_split_with_schnet_graphs()
        model = _fast_spec().fit_selected_model(
            split, _SchNetMockTrial(), refit_policy="train_only"
        )
        self.assertIsInstance(model, SchNetGateModel)

    def test_trial_metadata_has_expected_keys(self) -> None:
        split = _make_split_with_schnet_graphs()
        spec = _fast_spec()
        trial = _SchNetMockTrial()
        model = spec.fit_selected_model(split, trial, refit_policy="train_plus_val")
        metadata = spec.trial_metadata(trial, model)
        for key in ("hidden_dim", "n_layers", "n_experts", "n_rbf", "r_max", "bias"):
            self.assertIn(key, metadata)

    def test_trial_metadata_n_experts_matches_dataset(self) -> None:
        split = _make_split_with_schnet_graphs(n_experts=3)
        spec = SchNetGateTuningSpec(
            training_cfg=MoETrainingConfig(epochs=2, seed=0),
            hidden_dims=(8,),
            n_rbf=8,
        )
        model = spec.fit_selected_model(split, _SchNetMockTrial(), refit_policy="train_plus_val")
        self.assertEqual(spec.trial_metadata(_SchNetMockTrial(), model)["n_experts"], 3)


if __name__ == "__main__":
    unittest.main()
