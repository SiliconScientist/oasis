from __future__ import annotations

import types
import unittest
from dataclasses import dataclass

import numpy as np

from oasis.learning_curve.families.moe import MlipBaselineGateTuningSpec, MoEModel
from oasis.learning_curve.learned_specs import (
    _moe_config_tuning_spec_factory,
    learned_family_registration_specs,
)
from oasis.learning_curve.registry import learned_family_registration
from oasis.learning_curve.runners import ConfiguredSweepModelFamily
from oasis.sweep import SweepDataset, TrainValTestSweepRunnerInput
from oasis.tune import LearnedTrialTuningSpec


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


if __name__ == "__main__":
    unittest.main()
