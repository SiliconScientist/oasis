from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from oasis.sweep import SweepDataset, SweepRunnerPayload, TrainValTestSweepRunnerInput

try:
    from oasis.tune import (
        FactoryListHyperparameterSpec,
        GridHyperparameterSpec,
        HyperparameterSelectionSweepRunner,
        SupervisedModelSelectionSweepRunner,
        TrialModelSelectionSweepRunner,
        TrialTuningSpec,
        sweep_model_with_hyperparameter_selection,
        sweep_model_with_trial_tuning,
        sweep_supervised_model_selection,
        sweep_trial_model_selection,
    )

    HAS_SKLEARN = True
except ModuleNotFoundError:
    HAS_SKLEARN = False


class TuneTests(unittest.TestCase):
    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_grid_hyperparameter_spec_expands_candidate_factories(self) -> None:
        from sklearn.linear_model import Ridge

        spec = GridHyperparameterSpec(
            estimator_factory=Ridge,
            grid={"alpha": (0.1, 1.0)},
            fixed_params={"fit_intercept": False},
        )

        candidates = [factory() for factory in spec.candidate_factories()]

        self.assertEqual(len(candidates), 2)
        self.assertEqual([candidate.alpha for candidate in candidates], [0.1, 1.0])
        self.assertTrue(all(candidate.fit_intercept is False for candidate in candidates))

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_factory_list_hyperparameter_spec_returns_declared_candidates(self) -> None:
        from sklearn.linear_model import Ridge

        spec = FactoryListHyperparameterSpec(
            factories=(
                lambda: Ridge(alpha=0.1),
                lambda: Ridge(alpha=1.0),
            )
        )

        candidates = [factory() for factory in spec.candidate_factories()]

        self.assertEqual([candidate.alpha for candidate in candidates], [0.1, 1.0])

    def test_hyperparameter_selection_runner_picks_best_validation_candidate(self) -> None:
        dataset = SweepDataset(
            mlip_features=np.array(
                [
                    [1.0],
                    [2.0],
                    [3.0],
                    [4.0],
                    [5.0],
                    [6.0],
                ]
            ),
            targets=np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0]),
        )
        payload = SweepRunnerPayload(
            splits=(
                TrainValTestSweepRunnerInput(
                    dataset=dataset,
                    sweep_size=4,
                    train_idx=np.array([0, 1, 2]),
                    val_idx=np.array([3]),
                    test_idx=np.array([4, 5]),
                ),
            )
        )

        class ConstantPredictor:
            def __init__(self, constant: float) -> None:
                self.constant = constant
                self.coef_ = np.array([self.constant], dtype=float)

            def fit(self, X, y) -> ConstantPredictor:
                del X, y
                return self

            def predict(self, X):
                return np.full(len(X), self.constant, dtype=float)

        spec = FactoryListHyperparameterSpec(
            factories=(
                lambda: ConstantPredictor(0.0),
                lambda: ConstantPredictor(8.0),
            )
        )

        result = sweep_model_with_hyperparameter_selection(payload, spec)
        runner_result = HyperparameterSelectionSweepRunner(spec).run_with_validation(
            payload
        )

        np.testing.assert_allclose(
            result["rmse_mean"].to_numpy(),
            [np.sqrt(10.0)],
            atol=1e-12,
        )
        pd.testing.assert_frame_equal(result, runner_result.metrics)

    def test_supervised_model_selection_uses_val_idx_not_test_idx(self) -> None:
        dataset = SweepDataset(
            mlip_features=np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]),
            targets=np.array([0.0, 0.0, 0.0, 100.0, 100.0, 100.0]),
        )
        payload = SweepRunnerPayload(
            splits=(
                TrainValTestSweepRunnerInput(
                    dataset=dataset,
                    sweep_size=4,
                    train_idx=np.array([0, 1]),
                    val_idx=np.array([2, 3]),
                    test_idx=np.array([4, 5]),
                ),
            )
        )

        class ConstantPredictor:
            def __init__(self, constant: float) -> None:
                self.constant = constant
                self.coef_ = np.array([self.constant], dtype=float)

            def fit(self, X, y):
                del X, y
                return self

            def predict(self, X):
                return np.full(len(X), self.constant, dtype=float)

        spec = FactoryListHyperparameterSpec(
            factories=(
                lambda: ConstantPredictor(0.0),
                lambda: ConstantPredictor(100.0),
            )
        )

        result = sweep_supervised_model_selection(
            payload,
            spec,
            refit_policy="train_only",
        )

        np.testing.assert_allclose(
            result.metrics["rmse_mean"].to_numpy(),
            [100.0],
            atol=1e-12,
        )

    def test_supervised_model_selection_leaves_outer_test_untouched_during_selection(
        self,
    ) -> None:
        dataset = SweepDataset(
            mlip_features=np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]),
            targets=np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        )
        payload = SweepRunnerPayload(
            splits=(
                TrainValTestSweepRunnerInput(
                    dataset=dataset,
                    sweep_size=4,
                    train_idx=np.array([0, 1]),
                    val_idx=np.array([2, 3]),
                    test_idx=np.array([4, 5]),
                ),
            )
        )
        val_X = dataset.mlip_features[np.array([2, 3])]
        test_X = dataset.mlip_features[np.array([4, 5])]
        events: list[tuple[str, str, tuple[float, ...]]] = []

        class SpyPredictor:
            def __init__(self, name: str) -> None:
                self.name = name
                self.coef_ = np.array([0.0], dtype=float)

            def fit(self, X, y):
                del X, y
                events.append(("fit", self.name, ()))
                return self

            def predict(self, X):
                signature = tuple(float(v) for v in np.ravel(X))
                events.append(("predict", self.name, signature))
                return np.zeros(len(X), dtype=float)

        spec = FactoryListHyperparameterSpec(
            factories=(
                lambda: SpyPredictor("first"),
                lambda: SpyPredictor("second"),
            )
        )

        sweep_supervised_model_selection(payload, spec, refit_policy="train_only")

        val_signature = tuple(float(v) for v in np.ravel(val_X))
        test_signature = tuple(float(v) for v in np.ravel(test_X))
        predict_events = [event for event in events if event[0] == "predict"]

        self.assertEqual(
            sum(1 for _, _, signature in predict_events if signature == val_signature),
            2,
        )
        self.assertEqual(
            sum(1 for _, _, signature in predict_events if signature == test_signature),
            1,
        )

    def test_supervised_model_selection_breaks_validation_ties_deterministically(self) -> None:
        dataset = SweepDataset(
            mlip_features=np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]),
            targets=np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0]),
        )
        payload = SweepRunnerPayload(
            splits=(
                TrainValTestSweepRunnerInput(
                    dataset=dataset,
                    sweep_size=4,
                    train_idx=np.array([0, 1]),
                    val_idx=np.array([2, 3]),
                    test_idx=np.array([4, 5]),
                ),
            )
        )
        val_signature = tuple(
            float(v) for v in np.ravel(dataset.mlip_features[np.array([2, 3])])
        )

        class TiePredictor:
            def __init__(self, *, val_constant: float, test_constant: float) -> None:
                self.val_constant = val_constant
                self.test_constant = test_constant
                self.coef_ = np.array([self.test_constant], dtype=float)

            def fit(self, X, y):
                del X, y
                return self

            def predict(self, X):
                signature = tuple(float(v) for v in np.ravel(X))
                if signature == val_signature:
                    return np.full(len(X), self.val_constant, dtype=float)
                return np.full(len(X), self.test_constant, dtype=float)

        spec = FactoryListHyperparameterSpec(
            factories=(
                lambda: TiePredictor(val_constant=0.0, test_constant=0.0),
                lambda: TiePredictor(val_constant=0.0, test_constant=10.0),
            )
        )

        first = sweep_supervised_model_selection(
            payload,
            spec,
            refit_policy="train_only",
        )
        second = sweep_supervised_model_selection(
            payload,
            spec,
            refit_policy="train_only",
        )

        np.testing.assert_allclose(first.metrics["rmse_mean"].to_numpy(), [0.0], atol=1e-12)
        pd.testing.assert_frame_equal(first.metrics, second.metrics)

    def test_supervised_model_selection_runner_supports_refit_policy(self) -> None:
        dataset = SweepDataset(
            mlip_features=np.array([[0.0], [1.0], [2.0], [10.0], [4.0], [5.0]]),
            targets=np.array([0.0, 1.0, 2.0, 10.0, 4.0, 5.0]),
        )
        payload = SweepRunnerPayload(
            splits=(
                TrainValTestSweepRunnerInput(
                    dataset=dataset,
                    sweep_size=4,
                    train_idx=np.array([0, 1]),
                    val_idx=np.array([2, 3]),
                    test_idx=np.array([4, 5]),
                ),
            )
        )

        class MeanPredictor:
            def fit(self, X, y):
                del X
                self.mean_ = float(np.mean(y))
                self.coef_ = np.array([self.mean_], dtype=float)
                return self

            def predict(self, X):
                return np.full(len(X), self.mean_, dtype=float)

        spec = FactoryListHyperparameterSpec(factories=(MeanPredictor,))

        train_only = sweep_supervised_model_selection(
            payload,
            spec,
            refit_policy="train_only",
        )
        refit_train_plus_val = sweep_supervised_model_selection(
            payload,
            spec,
            refit_policy="train_plus_val",
        )
        runner_result = SupervisedModelSelectionSweepRunner(
            spec,
            refit_policy="train_plus_val",
        ).run_with_validation(payload)

        np.testing.assert_allclose(
            train_only.metrics["rmse_mean"].to_numpy(),
            [np.sqrt(16.25)],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            refit_train_plus_val.metrics["rmse_mean"].to_numpy(),
            [np.sqrt(1.8125)],
            atol=1e-12,
        )
        pd.testing.assert_frame_equal(refit_train_plus_val.metrics, runner_result.metrics)

    def test_supervised_model_selection_runner_rejects_unknown_refit_policy(self) -> None:
        dataset = SweepDataset(
            mlip_features=np.array([[0.0], [1.0], [2.0], [3.0]]),
            targets=np.array([0.0, 1.0, 2.0, 3.0]),
        )
        payload = SweepRunnerPayload(
            splits=(
                TrainValTestSweepRunnerInput(
                    dataset=dataset,
                    sweep_size=3,
                    train_idx=np.array([0]),
                    val_idx=np.array([1, 2]),
                    test_idx=np.array([3]),
                ),
            )
        )

        class ConstantPredictor:
            def fit(self, X, y):
                del X, y
                self.coef_ = np.array([0.0], dtype=float)
                return self

            def predict(self, X):
                return np.zeros(len(X), dtype=float)

        spec = FactoryListHyperparameterSpec(factories=(ConstantPredictor,))

        with self.assertRaisesRegex(ValueError, "unsupported selection refit policy"):
            sweep_supervised_model_selection(
                payload,
                spec,
                refit_policy="bad_policy",  # type: ignore[arg-type]
            )

    def test_trial_tuning_spec_runtime_protocol_and_runner_pick_best_trial(self) -> None:
        dataset = SweepDataset(
            mlip_features=np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]),
            targets=np.array([0.0, 0.0, 1.0, 3.0, 4.0, 5.0]),
        )
        payload = SweepRunnerPayload(
            splits=(
                TrainValTestSweepRunnerInput(
                    dataset=dataset,
                    sweep_size=4,
                    train_idx=np.array([0, 1]),
                    val_idx=np.array([2, 3]),
                    test_idx=np.array([4, 5]),
                ),
            )
        )
        seen_values: list[float] = []

        class ConstantPredictor:
            def __init__(self, constant: float) -> None:
                self.constant = constant

            def predict(self, X):
                return np.full(len(X), self.constant, dtype=float)

        class FakeTrialTuningSpec:
            def build_trial_objective(self, split):
                y_val = split.dataset.targets[split.val_idx]

                def objective(trial):
                    seen_values.append(float(trial["constant"]))
                    preds = np.full(len(y_val), trial["constant"], dtype=float)
                    return float(np.sqrt(np.mean((y_val - preds) ** 2)))

                return objective

            def fit_selected_model(self, split, best_trial, *, refit_policy):
                del split, refit_policy
                return ConstantPredictor(float(best_trial["constant"]))

            def trial_metadata(self, best_trial, model):
                return {
                    "constant": float(best_trial["constant"]),
                    "selected_constant": float(model.constant),
                }

        spec = FakeTrialTuningSpec()
        self.assertIsInstance(spec, TrialTuningSpec)
        trial_factory = lambda split: (  # noqa: E731
            {"constant": 0.0},
            {"constant": 3.0},
        )

        artifacts = sweep_trial_model_selection(payload, spec, trial_factory)
        runner_artifacts = TrialModelSelectionSweepRunner(
            spec,
            trial_factory,
        ).run_artifacts_with_validation(payload)
        result = sweep_model_with_trial_tuning(payload, spec, trial_factory)

        self.assertEqual(seen_values, [0.0, 3.0, 0.0, 3.0, 0.0, 3.0])
        np.testing.assert_allclose(
            artifacts.metrics["rmse_mean"].to_numpy(),
            [np.sqrt(2.5)],
            atol=1e-12,
        )
        self.assertEqual(
            artifacts.selection_metadata.columns.tolist(),
            ["n_train", "constant", "selected_constant"],
        )
        self.assertEqual(artifacts.selection_metadata["constant"].tolist(), [3.0])
        pd.testing.assert_frame_equal(artifacts.metrics, runner_artifacts.metrics)
        pd.testing.assert_frame_equal(artifacts.selection_metadata, runner_artifacts.selection_metadata)
        pd.testing.assert_frame_equal(artifacts.metrics, result)

    def test_trial_tuning_spec_refit_policy_is_passed_into_refit_step(self) -> None:
        dataset = SweepDataset(
            mlip_features=np.array([[0.0], [1.0], [2.0], [10.0], [4.0], [5.0]]),
            targets=np.array([0.0, 1.0, 2.0, 10.0, 4.0, 5.0]),
        )
        payload = SweepRunnerPayload(
            splits=(
                TrainValTestSweepRunnerInput(
                    dataset=dataset,
                    sweep_size=4,
                    train_idx=np.array([0, 1]),
                    val_idx=np.array([2, 3]),
                    test_idx=np.array([4, 5]),
                ),
            )
        )

        class MeanPredictor:
            def __init__(self, mean_: float) -> None:
                self.mean_ = mean_

            def predict(self, X):
                return np.full(len(X), self.mean_, dtype=float)

        class FakeTrialTuningSpec:
            def build_trial_objective(self, split):
                def objective(trial):
                    return float(trial["score"])

                return objective

            def fit_selected_model(self, split, best_trial, *, refit_policy):
                del best_trial
                if refit_policy == "train_only":
                    fit_idx = split.train_idx
                else:
                    fit_idx = np.concatenate([split.train_idx, split.val_idx])
                mean_ = float(np.mean(split.dataset.targets[fit_idx]))
                return MeanPredictor(mean_)

            def trial_metadata(self, best_trial, model):
                return {"score": float(best_trial["score"]), "mean": float(model.mean_)}

        spec = FakeTrialTuningSpec()
        trial_factory = lambda split: ({"score": 1.0},)  # noqa: E731

        train_only = sweep_trial_model_selection(
            payload,
            spec,
            trial_factory,
            refit_policy="train_only",
        )
        train_plus_val = sweep_trial_model_selection(
            payload,
            spec,
            trial_factory,
            refit_policy="train_plus_val",
        )

        np.testing.assert_allclose(
            train_only.metrics["rmse_mean"].to_numpy(),
            [np.sqrt(16.25)],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            train_plus_val.metrics["rmse_mean"].to_numpy(),
            [np.sqrt(1.8125)],
            atol=1e-12,
        )
        self.assertEqual(train_only.selection_metadata["mean"].tolist(), [0.5])
        self.assertEqual(train_plus_val.selection_metadata["mean"].tolist(), [3.25])
