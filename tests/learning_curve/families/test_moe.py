from __future__ import annotations

import types
import unittest
from dataclasses import dataclass

import numpy as np

from oasis.config import MoETrainingConfig
from oasis.learning_curve.families.gating_policy import DenseGatingPolicy, TopKGatingPolicy
from oasis.learning_curve.families.gnn_gate import GnnGateTuningSpec
from oasis.learning_curve.families.moe import MlipBaselineGateTuningSpec, MoEModel
from oasis.learning_curve.learned_specs import (
    _moe_config_runner_kwargs,
    _moe_config_tuning_spec_factory,
    learned_family_registration_specs,
)
from oasis.learning_curve.registry import learned_family_registration
from oasis.learning_curve.runners import ConfiguredSweepModelFamily
from oasis.sweep import SweepDataset, TrainValTestSweepRunnerInput
from oasis.tune import LearnedOptunaModelSelectionSweepRunner, LearnedTrialTuningSpec


@dataclass
class _MockTrial:
    """Minimal trial stub: supports suggest_float during objective and .params afterwards."""

    _logits: list[float]

    def __post_init__(self) -> None:
        self.params = {f"logit_{i}": v for i, v in enumerate(self._logits)}

    def suggest_float(self, name: str, low: float, high: float) -> float:
        del low, high
        return self.params[name]


def _make_split() -> TrainValTestSweepRunnerInput:
    dataset = SweepDataset(
        mlip_features=np.array(
            [
                [1.0, 2.0],
                [2.0, 3.0],
                [3.0, 4.0],
                [4.0, 5.0],
                [5.0, 6.0],
                [6.0, 7.0],
            ]
        ),
        targets=np.array([1.5, 2.5, 3.5, 1.0, 5.5, 6.5]),
    )
    return TrainValTestSweepRunnerInput(
        dataset=dataset,
        sweep_size=4,
        train_idx=np.array([0, 1, 2]),
        val_idx=np.array([3]),
        test_idx=np.array([4, 5]),
    )


class MoEModelTests(unittest.TestCase):
    def test_predict_shape(self) -> None:
        model = MoEModel(weights=np.array([0.3, 0.7]), bias=0.0)
        preds = model.predict(np.array([[1.0, 2.0], [3.0, 4.0]]))
        self.assertEqual(preds.shape, (2,))

    def test_predict_values(self) -> None:
        model = MoEModel(weights=np.array([1.0, 0.0]), bias=0.0)
        preds = model.predict(np.array([[3.0, 999.0], [5.0, 999.0]]))
        np.testing.assert_allclose(preds, [3.0, 5.0])

    def test_bias_added(self) -> None:
        model = MoEModel(weights=np.array([0.0, 1.0]), bias=2.5)
        preds = model.predict(np.array([[999.0, 1.0]]))
        np.testing.assert_allclose(preds, [3.5])


class MlipBaselineGateTuningSpecTests(unittest.TestCase):
    def test_is_learned_trial_tuning_spec(self) -> None:
        self.assertIsInstance(MlipBaselineGateTuningSpec(), LearnedTrialTuningSpec)

    def test_build_trial_objective_returns_finite_rmse(self) -> None:
        spec = MlipBaselineGateTuningSpec()
        objective = spec.build_trial_objective(_make_split())
        rmse = objective(_MockTrial([0.0, 0.0]))
        self.assertTrue(np.isfinite(rmse))
        self.assertGreater(rmse, 0.0)

    def test_round_trip_predict_shape_and_finiteness(self) -> None:
        spec = MlipBaselineGateTuningSpec()
        split = _make_split()
        model = spec.fit_selected_model(
            split, _MockTrial([2.0, -2.0]), refit_policy="train_plus_val"
        )
        self.assertIsInstance(model, MoEModel)
        preds = spec.predict(model, split.dataset_subsets().test)
        self.assertEqual(preds.shape, (2,))
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_refit_policy_train_only_vs_train_plus_val_same_weights(self) -> None:
        spec = MlipBaselineGateTuningSpec()
        split = _make_split()
        trial = _MockTrial([0.0, 0.0])
        m_train = spec.fit_selected_model(split, trial, refit_policy="train_only")
        m_both = spec.fit_selected_model(split, trial, refit_policy="train_plus_val")
        np.testing.assert_allclose(m_train.weights, m_both.weights)

    def test_trial_metadata_weights_sum_to_one(self) -> None:
        spec = MlipBaselineGateTuningSpec()
        split = _make_split()
        model = spec.fit_selected_model(
            split, _MockTrial([1.0, -1.0]), refit_policy="train_plus_val"
        )
        metadata = spec.trial_metadata(_MockTrial([1.0, -1.0]), model)
        self.assertIn("weight_0", metadata)
        self.assertIn("weight_1", metadata)
        self.assertAlmostEqual(metadata["weight_0"] + metadata["weight_1"], 1.0)


class MoERegistrationTests(unittest.TestCase):
    def test_moe_registration_uses_configured_family(self) -> None:
        specs = learned_family_registration_specs()
        moe_spec = next(s for s in specs if s.name == "moe")
        moe_registration = learned_family_registration(moe_spec)
        self.assertIsInstance(moe_registration.family_factory(), ConfiguredSweepModelFamily)

    def test_moe_spec_has_no_placeholder_factory(self) -> None:
        specs = learned_family_registration_specs()
        moe_spec = next(s for s in specs if s.name == "moe")
        self.assertIsNone(moe_spec.family_factory)
        self.assertIsInstance(moe_spec.learned_trial_tuning_spec, MlipBaselineGateTuningSpec)


def _model_cfg(gate_type: str) -> object:
    return types.SimpleNamespace(moe=types.SimpleNamespace(gate_type=gate_type))


def _model_cfg_gnn(hidden_dims: list[int] | None = None) -> object:
    return types.SimpleNamespace(
        moe=types.SimpleNamespace(
            gate_type="gnn",
            training=MoETrainingConfig(epochs=2),
            hidden_dims=hidden_dims if hidden_dims is not None else [32],
        )
    )


class MoEGateDispatchTests(unittest.TestCase):
    def _moe_registration(self) -> object:
        specs = learned_family_registration_specs()
        moe_spec = next(s for s in specs if s.name == "moe")
        return learned_family_registration(moe_spec)

    def test_mlip_baseline_config_produces_configured_family(self) -> None:
        registration = self._moe_registration()
        family = registration.config_factory(_model_cfg("mlip_baseline"))
        self.assertIsInstance(family, ConfiguredSweepModelFamily)

    def test_mlip_baseline_config_uses_mlip_baseline_tuning_spec(self) -> None:
        registration = self._moe_registration()
        family = registration.config_factory(_model_cfg("mlip_baseline"))
        self.assertIsInstance(family.spec.runner.tuning_spec, MlipBaselineGateTuningSpec)

    def test_unknown_gate_type_raises(self) -> None:
        with self.assertRaises(ValueError):
            _moe_config_tuning_spec_factory(_model_cfg("bad_gate"))

    def test_family_factory_fallback_still_works(self) -> None:
        registration = self._moe_registration()
        family = registration.family_factory()
        self.assertIsInstance(family, ConfiguredSweepModelFamily)

    def test_gnn_config_produces_configured_family(self) -> None:
        registration = self._moe_registration()
        family = registration.config_factory(_model_cfg_gnn())
        self.assertIsInstance(family, ConfiguredSweepModelFamily)

    def test_gnn_config_uses_gnn_gate_tuning_spec(self) -> None:
        registration = self._moe_registration()
        family = registration.config_factory(_model_cfg_gnn())
        self.assertIsInstance(family.spec.runner.tuning_spec, GnnGateTuningSpec)

    def test_gnn_config_threads_hidden_dims(self) -> None:
        registration = self._moe_registration()
        family = registration.config_factory(_model_cfg_gnn(hidden_dims=[64, 64]))
        self.assertEqual(family.spec.runner.tuning_spec.hidden_dims, (64, 64))


def _model_cfg_with_optuna(n_trials: int) -> object:
    return types.SimpleNamespace(
        moe=types.SimpleNamespace(
            gate_type="mlip_baseline",
            tuning=types.SimpleNamespace(
                optuna=types.SimpleNamespace(
                    n_trials=n_trials,
                    timeout_s=None,
                    sampler=None,
                    pruner=None,
                    seed=None,
                )
            ),
        )
    )


def _model_cfg_no_optuna() -> object:
    return types.SimpleNamespace(
        moe=types.SimpleNamespace(
            gate_type="mlip_baseline",
            tuning=types.SimpleNamespace(optuna=None),
        )
    )


class MoEOptunaConfigThreadingTests(unittest.TestCase):
    def _moe_registration(self) -> object:
        specs = learned_family_registration_specs()
        moe_spec = next(s for s in specs if s.name == "moe")
        return learned_family_registration(moe_spec)

    def test_runner_kwargs_n_trials_from_config(self) -> None:
        kwargs = _moe_config_runner_kwargs(_model_cfg_with_optuna(3))
        self.assertEqual(kwargs["n_trials"], 3)

    def test_runner_kwargs_fallback_when_optuna_is_none(self) -> None:
        kwargs = _moe_config_runner_kwargs(_model_cfg_no_optuna())
        self.assertEqual(kwargs["n_trials"], 10)
        self.assertNotIn("study_factory", kwargs)

    def test_runner_kwargs_includes_study_factory_when_optuna_set(self) -> None:
        kwargs = _moe_config_runner_kwargs(_model_cfg_with_optuna(5))
        self.assertIn("study_factory", kwargs)
        self.assertTrue(callable(kwargs["study_factory"]))

    def test_integration_config_factory_threads_n_trials(self) -> None:
        registration = self._moe_registration()
        family = registration.config_factory(_model_cfg_with_optuna(3))
        self.assertIsInstance(family.spec.runner, LearnedOptunaModelSelectionSweepRunner)
        self.assertEqual(family.spec.runner.n_trials, 3)


def _model_cfg_with_gating(
    gate_type: str,
    gating_mode: str,
    top_k: int = 2,
) -> object:
    return types.SimpleNamespace(
        moe=types.SimpleNamespace(
            gate_type=gate_type,
            gating_mode=gating_mode,
            top_k=top_k,
        )
    )


def _model_cfg_gnn_with_gating(gating_mode: str, top_k: int = 2) -> object:
    return types.SimpleNamespace(
        moe=types.SimpleNamespace(
            gate_type="gnn",
            gating_mode=gating_mode,
            top_k=top_k,
            training=MoETrainingConfig(epochs=2),
            hidden_dims=[32],
        )
    )


class MoEGatingModeDispatchTests(unittest.TestCase):
    def test_dense_mode_mlip_baseline_produces_dense_policy(self) -> None:
        spec = _moe_config_tuning_spec_factory(
            _model_cfg_with_gating("mlip_baseline", "dense")
        )
        self.assertIsInstance(spec.policy, DenseGatingPolicy)

    def test_top_k_mode_mlip_baseline_produces_top_k_policy(self) -> None:
        spec = _moe_config_tuning_spec_factory(
            _model_cfg_with_gating("mlip_baseline", "top_k", top_k=3)
        )
        self.assertIsInstance(spec.policy, TopKGatingPolicy)
        self.assertEqual(spec.policy.k, 3)

    def test_dense_mode_gnn_produces_dense_policy(self) -> None:
        spec = _moe_config_tuning_spec_factory(_model_cfg_gnn_with_gating("dense"))
        self.assertIsInstance(spec.policy, DenseGatingPolicy)

    def test_top_k_mode_gnn_produces_top_k_policy(self) -> None:
        spec = _moe_config_tuning_spec_factory(
            _model_cfg_gnn_with_gating("top_k", top_k=2)
        )
        self.assertIsInstance(spec.policy, TopKGatingPolicy)
        self.assertEqual(spec.policy.k, 2)

    def test_missing_gating_mode_defaults_to_dense(self) -> None:
        # Old-style config with no gating_mode attribute → should default to dense.
        spec = _moe_config_tuning_spec_factory(_model_cfg("mlip_baseline"))
        self.assertIsInstance(spec.policy, DenseGatingPolicy)

    def test_top_k_value_is_threaded_correctly(self) -> None:
        spec = _moe_config_tuning_spec_factory(
            _model_cfg_with_gating("mlip_baseline", "top_k", top_k=1)
        )
        self.assertEqual(spec.policy.k, 1)


if __name__ == "__main__":
    unittest.main()
