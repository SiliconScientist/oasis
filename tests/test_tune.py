from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from oasis.sweep import (
    build_sweep_batches,
    DatasetLoaderAdapter,
    GraphDatasetView,
    GraphRecord,
    LoaderAdapterInput,
    LoaderBatching,
    SweepDataset,
    SweepDatasetBatchLoaderAdapter,
    SweepRunnerPayload,
    TrainEvalLoaderPolicy,
    TrainValTestSweepRunnerInput,
)

try:
    from oasis.tune import (
        FactoryListHyperparameterSpec,
        GridHyperparameterSpec,
        HyperparameterSelectionSweepRunner,
        LearnedOptunaModelSelectionSweepRunner,
        LearnedTrialModelSelectionSweepRunner,
        LearnedTrialTuningSpec,
        OptunaModelSelectionSweepRunner,
        OptunaTuningConfig,
        SupervisedModelSelectionSweepRunner,
        TrialModelSelectionSweepRunner,
        TrialTuningSpec,
        study_factory_from_optuna_cfg,
        sweep_learned_model_with_optuna_selection,
        sweep_learned_model_with_trial_tuning,
        sweep_learned_optuna_model_selection,
        sweep_learned_trial_model_selection,
        sweep_model_with_hyperparameter_selection,
        sweep_model_with_optuna_selection,
        sweep_model_with_trial_tuning,
        sweep_optuna_model_selection,
        sweep_supervised_model_selection,
        sweep_trial_model_selection,
    )

    HAS_TUNE = True
except ModuleNotFoundError:
    HAS_TUNE = False

try:
    import sklearn  # noqa: F401

    HAS_SKLEARN = True
except ModuleNotFoundError:
    HAS_SKLEARN = False


@unittest.skipUnless(HAS_TUNE, "requires oasis.tune dependencies")
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

    def test_learned_trial_tuning_spec_uses_split_aware_datasets_and_runner(self) -> None:
        dataset = SweepDataset(
            mlip_features=np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]),
            targets=np.array([0.0, 0.0, 1.0, 3.0, 4.0, 5.0]),
            sample_ids=np.array(["s0", "s1", "s2", "s3", "s4", "s5"]),
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
        seen_loader_splits: list[tuple[str, tuple[str, ...], int | None, bool, str | None]] = []

        class ConstantPredictor:
            def __init__(self, constant: float) -> None:
                self.constant = constant

        class FakeLearnedTrialTuningSpec:
            @staticmethod
            def _train_collate(samples: object) -> object:
                return samples

            @staticmethod
            def _eval_collate(samples: object) -> object:
                return samples

            class _LoaderAdapter:
                def batching_for_split(self, *, split_name: str) -> LoaderBatching:
                    if split_name == "train":
                        return LoaderBatching(
                            batch_size=2,
                            shuffle=True,
                            collate_fn=FakeLearnedTrialTuningSpec._train_collate,
                        )
                    return LoaderBatching(
                        batch_size=4,
                        shuffle=False,
                        collate_fn=FakeLearnedTrialTuningSpec._eval_collate,
                    )

                def build_loader(self, loader_input: LoaderAdapterInput) -> dict[str, object]:
                    collate_fn = loader_input.batching.collate_fn
                    return {
                        "split_name": loader_input.split_name,
                        "sample_ids": tuple(loader_input.dataset.sample_ids.tolist()),
                        "targets": loader_input.dataset.targets,
                        "batch_size": loader_input.batching.batch_size,
                        "shuffle": loader_input.batching.shuffle,
                        "collate_name": None if collate_fn is None else collate_fn.__name__,
                    }

            _loader_adapter = _LoaderAdapter()

            def build_trial_objective(self, split):
                self_ref = self
                self_ref.assertIsInstance(self._loader_adapter, DatasetLoaderAdapter)
                loaders = split.loaders(self._loader_adapter)
                seen_loader_splits.append(
                    (
                        "train",
                        loaders.train["sample_ids"],
                        loaders.train["batch_size"],
                        loaders.train["shuffle"],
                        loaders.train["collate_name"],
                    )
                )
                seen_loader_splits.append(
                    (
                        "val",
                        loaders.val["sample_ids"],
                        loaders.val["batch_size"],
                        loaders.val["shuffle"],
                        loaders.val["collate_name"],
                    )
                )
                y_val = np.asarray(loaders.val["targets"], dtype=float)

                def objective(trial):
                    seen_values.append(float(trial["constant"]))
                    preds = np.full(len(y_val), trial["constant"], dtype=float)
                    return float(np.sqrt(np.mean((y_val - preds) ** 2)))

                return objective

            def fit_selected_model(self, split, best_trial, *, refit_policy):
                del split, refit_policy
                return ConstantPredictor(float(best_trial["constant"]))

            def predict(self, model, dataset):
                return np.full(len(dataset), model.constant, dtype=float)

            def trial_metadata(self, best_trial, model):
                return {
                    "constant": float(best_trial["constant"]),
                    "selected_constant": float(model.constant),
                }

        spec = FakeLearnedTrialTuningSpec()
        spec.assertIsInstance = self.assertIsInstance
        self.assertIsInstance(spec, LearnedTrialTuningSpec)
        trial_factory = lambda split: (  # noqa: E731
            {"constant": 0.0},
            {"constant": 3.0},
        )

        artifacts = sweep_learned_trial_model_selection(payload, spec, trial_factory)
        runner_artifacts = LearnedTrialModelSelectionSweepRunner(
            spec,
            trial_factory,
        ).run_artifacts_with_validation(payload)
        result = sweep_learned_model_with_trial_tuning(payload, spec, trial_factory)

        self.assertEqual(seen_values, [0.0, 3.0, 0.0, 3.0, 0.0, 3.0])
        self.assertEqual(
            seen_loader_splits,
            [
                ("train", ("s0", "s1"), 2, True, "_train_collate"),
                ("val", ("s2", "s3"), 4, False, "_eval_collate"),
                ("train", ("s0", "s1"), 2, True, "_train_collate"),
                ("val", ("s2", "s3"), 4, False, "_eval_collate"),
                ("train", ("s0", "s1"), 2, True, "_train_collate"),
                ("val", ("s2", "s3"), 4, False, "_eval_collate"),
            ],
        )
        np.testing.assert_allclose(
            artifacts.metrics["rmse_mean"].to_numpy(),
            [np.sqrt(2.5)],
            atol=1e-12,
        )
        self.assertEqual(
            artifacts.selection_metadata.columns.tolist(),
            ["n_train", "constant", "selected_constant"],
        )
        pd.testing.assert_frame_equal(artifacts.metrics, runner_artifacts.metrics)
        pd.testing.assert_frame_equal(
            artifacts.selection_metadata,
            runner_artifacts.selection_metadata,
        )
        pd.testing.assert_frame_equal(artifacts.metrics, result)

    def test_learned_trial_tuning_refit_policy_uses_split_subsets(self) -> None:
        dataset = SweepDataset(
            mlip_features=np.array([[0.0], [1.0], [2.0], [10.0], [4.0], [5.0]]),
            targets=np.array([0.0, 1.0, 2.0, 10.0, 4.0, 5.0]),
            sample_ids=np.array(["s0", "s1", "s2", "s3", "s4", "s5"]),
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

        class FakeLearnedTrialTuningSpec:
            def build_trial_objective(self, split):
                def objective(trial):
                    return float(trial["score"])

                return objective

            def fit_selected_model(self, split, best_trial, *, refit_policy):
                del best_trial
                subsets = split.dataset_subsets()
                if refit_policy == "train_only":
                    fit_targets = subsets.train.targets
                else:
                    fit_targets = np.concatenate([subsets.train.targets, subsets.val.targets])
                return MeanPredictor(float(np.mean(fit_targets)))

            def predict(self, model, dataset):
                return np.full(len(dataset), model.mean_, dtype=float)

            def trial_metadata(self, best_trial, model):
                return {"score": float(best_trial["score"]), "mean": float(model.mean_)}

        spec = FakeLearnedTrialTuningSpec()
        trial_factory = lambda split: ({"score": 1.0},)  # noqa: E731

        train_only = sweep_learned_trial_model_selection(
            payload,
            spec,
            trial_factory,
            refit_policy="train_only",
        )
        train_plus_val = sweep_learned_trial_model_selection(
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

    def test_learned_trial_tuning_spec_can_use_batched_mlip_only_loaders(self) -> None:
        dataset = SweepDataset(
            mlip_features=np.array([[0.0], [1.0], [2.0], [3.0], [40.0], [50.0]]),
            targets=np.array([0.0, 0.0, 2.0, 3.0, 4.0, 5.0]),
            sample_ids=np.array(["s0", "s1", "s2", "s3", "s4", "s5"]),
        )
        payload = SweepRunnerPayload(
            splits=(
                TrainValTestSweepRunnerInput(
                    dataset=dataset,
                    sweep_size=4,
                    train_idx=np.array([1, 0]),
                    val_idx=np.array([3, 2]),
                    test_idx=np.array([5, 4]),
                ),
            )
        )
        objective_batch_signatures: list[tuple[str, tuple[str, ...], tuple[float, ...]]] = []
        test_predict_signatures: list[tuple[str, ...]] = []

        class ConstantPredictor:
            def __init__(self, constant: float) -> None:
                self.constant = constant

        class FakeLearnedTrialTuningSpec:
            _loader_adapter = SweepDatasetBatchLoaderAdapter(
                policy=TrainEvalLoaderPolicy(
                    batch_size=2,
                    eval_batch_size=1,
                    train_shuffle=False,
                    eval_shuffle=False,
                )
            )

            def build_trial_objective(self, split):
                loaders = split.loaders(self._loader_adapter)
                for batch in loaders.train:
                    objective_batch_signatures.append(
                        (
                            batch.split_name,
                            batch.sample_ids,
                            tuple(float(v) for v in batch.targets),
                        )
                    )
                for batch in loaders.val:
                    objective_batch_signatures.append(
                        (
                            batch.split_name,
                            batch.sample_ids,
                            tuple(float(v) for v in batch.targets),
                        )
                    )
                y_val = np.concatenate([batch.targets for batch in loaders.val])

                def objective(trial):
                    preds = np.full(len(y_val), trial["constant"], dtype=float)
                    return float(np.sqrt(np.mean((y_val - preds) ** 2)))

                return objective

            def fit_selected_model(self, split, best_trial, *, refit_policy):
                del split, refit_policy
                return ConstantPredictor(float(best_trial["constant"]))

            def predict(self, model, dataset):
                test_predict_signatures.append(tuple(dataset.sample_ids.tolist()))
                return np.full(len(dataset), model.constant, dtype=float)

            def trial_metadata(self, best_trial, model):
                return {
                    "constant": float(best_trial["constant"]),
                    "selected_constant": float(model.constant),
                }

        spec = FakeLearnedTrialTuningSpec()
        self.assertIsInstance(spec, LearnedTrialTuningSpec)
        trial_factory = lambda split: (  # noqa: E731
            {"constant": 0.0},
            {"constant": 3.0},
        )

        artifacts = sweep_learned_trial_model_selection(payload, spec, trial_factory)

        self.assertEqual(
            objective_batch_signatures,
            [
                ("train", ("s1", "s0"), (0.0, 0.0)),
                ("val", ("s3",), (3.0,)),
                ("val", ("s2",), (2.0,)),
            ],
        )
        self.assertEqual(test_predict_signatures, [("s5", "s4")])
        np.testing.assert_allclose(
            artifacts.metrics["rmse_mean"].to_numpy(),
            [np.sqrt(2.5)],
            atol=1e-12,
        )
        self.assertEqual(
            artifacts.selection_metadata["selected_constant"].tolist(),
            [3.0],
        )

    def test_learned_trial_tuning_spec_can_use_batched_graph_backed_loaders(self) -> None:
        dataset = SweepDataset(
            mlip_features=np.empty((5, 0), dtype=float),
            targets=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            sample_ids=np.array(["s0", "s1", "s2", "s3", "s4"]),
            graph_view=GraphDatasetView.from_records(
                (
                    GraphRecord(
                        sample_id="s0",
                        node_features=np.array([[1.0], [1.1]], dtype=float),
                        edge_index=np.array([[0], [1]], dtype=np.int64),
                    ),
                    GraphRecord(
                        sample_id="s1",
                        node_features=np.array([[2.0], [2.1]], dtype=float),
                        edge_index=np.array([[0], [1]], dtype=np.int64),
                    ),
                    GraphRecord(
                        sample_id="s2",
                        node_features=np.array([[3.0], [3.1]], dtype=float),
                        edge_index=np.array([[0], [1]], dtype=np.int64),
                    ),
                    GraphRecord(
                        sample_id="s3",
                        node_features=np.array([[4.0], [4.1]], dtype=float),
                        edge_index=np.array([[0], [1]], dtype=np.int64),
                    ),
                    GraphRecord(
                        sample_id="s4",
                        node_features=np.array([[5.0], [5.1]], dtype=float),
                        edge_index=np.array([[0], [1]], dtype=np.int64),
                    ),
                )
            ),
        )
        payload = SweepRunnerPayload(
            splits=(
                TrainValTestSweepRunnerInput(
                    dataset=dataset,
                    sweep_size=3,
                    train_idx=np.array([4, 1, 0]),
                    val_idx=np.array([3]),
                    test_idx=np.array([2]),
                ),
            )
        )
        objective_graph_signatures: list[tuple[str, tuple[str, ...], tuple[tuple[float, ...], ...]]] = []
        test_predict_signatures: list[tuple[str, ...]] = []

        class ConstantPredictor:
            def __init__(self, constant: float) -> None:
                self.constant = constant

        class FakeLearnedTrialTuningSpec:
            _loader_adapter = SweepDatasetBatchLoaderAdapter(
                policy=TrainEvalLoaderPolicy(
                    batch_size=2,
                    eval_batch_size=1,
                    train_shuffle=False,
                    eval_shuffle=False,
                )
            )

            def build_trial_objective(self, split):
                loaders = split.loaders(self._loader_adapter)
                for batch in loaders.train:
                    objective_graph_signatures.append(
                        (
                            batch.split_name,
                            batch.sample_ids,
                            tuple(
                                tuple(float(v) for v in np.ravel(graph.node_features))
                                for graph in batch.graphs
                                if graph is not None
                            ),
                        )
                    )
                    self_ref = self
                    self_ref.assertEqual(
                        batch.sample_ids,
                        tuple(
                            graph.sample_id for graph in batch.graphs if graph is not None
                        ),
                    )
                for batch in loaders.val:
                    objective_graph_signatures.append(
                        (
                            batch.split_name,
                            batch.sample_ids,
                            tuple(
                                tuple(float(v) for v in np.ravel(graph.node_features))
                                for graph in batch.graphs
                                if graph is not None
                            ),
                        )
                    )
                    self_ref = self
                    self_ref.assertEqual(
                        batch.sample_ids,
                        tuple(
                            graph.sample_id for graph in batch.graphs if graph is not None
                        ),
                    )
                y_val = np.concatenate([batch.targets for batch in loaders.val])

                def objective(trial):
                    preds = np.full(len(y_val), trial["constant"], dtype=float)
                    return float(np.sqrt(np.mean((y_val - preds) ** 2)))

                return objective

            def fit_selected_model(self, split, best_trial, *, refit_policy):
                del split, refit_policy
                return ConstantPredictor(float(best_trial["constant"]))

            def predict(self, model, dataset):
                test_predict_signatures.append(tuple(dataset.sample_ids.tolist()))
                return np.full(len(dataset), model.constant, dtype=float)

            def trial_metadata(self, best_trial, model):
                return {
                    "constant": float(best_trial["constant"]),
                    "selected_constant": float(model.constant),
                }

        spec = FakeLearnedTrialTuningSpec()
        spec.assertEqual = self.assertEqual
        self.assertIsInstance(spec, LearnedTrialTuningSpec)
        trial_factory = lambda split: (  # noqa: E731
            {"constant": 0.0},
            {"constant": 4.0},
        )

        artifacts = sweep_learned_trial_model_selection(payload, spec, trial_factory)

        self.assertEqual(
            objective_graph_signatures,
            [
                ("train", ("s4", "s1"), ((5.0, 5.1), (2.0, 2.1))),
                ("train", ("s0",), ((1.0, 1.1),)),
                ("val", ("s3",), ((4.0, 4.1),)),
            ],
        )
        self.assertEqual(test_predict_signatures, [("s2",)])
        np.testing.assert_allclose(
            artifacts.metrics["rmse_mean"].to_numpy(),
            [1.0],
            atol=1e-12,
        )
        self.assertEqual(
            artifacts.selection_metadata["selected_constant"].tolist(),
            [4.0],
        )

    def test_learned_trial_tuning_train_loader_shuffle_changes_order_without_leakage(
        self,
    ) -> None:
        dataset = SweepDataset(
            mlip_features=np.arange(24, dtype=float).reshape(8, 3),
            targets=np.arange(8, dtype=float),
            sample_ids=np.array(["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"]),
        )
        payload = SweepRunnerPayload(
            splits=(
                TrainValTestSweepRunnerInput(
                    dataset=dataset,
                    sweep_size=5,
                    train_idx=np.array([0, 1, 2, 3]),
                    val_idx=np.array([4, 5]),
                    test_idx=np.array([6, 7]),
                ),
            )
        )
        seen_train_orders: list[tuple[tuple[str, ...], ...]] = []
        seen_val_orders: list[tuple[tuple[str, ...], ...]] = []
        seen_test_ids: list[tuple[str, ...]] = []

        class ConstantPredictor:
            def __init__(self, constant: float) -> None:
                self.constant = constant

        class FakeLearnedTrialTuningSpec:
            _loader_adapter = SweepDatasetBatchLoaderAdapter(
                policy=TrainEvalLoaderPolicy(
                    batch_size=2,
                    eval_batch_size=1,
                    train_shuffle=True,
                    eval_shuffle=False,
                )
            )

            def build_trial_objective(self, split):
                loaders = split.loaders(self._loader_adapter)
                seen_train_orders.append(tuple(batch.sample_ids for batch in loaders.train))
                seen_val_orders.append(tuple(batch.sample_ids for batch in loaders.val))

                train_ids = set().union(*(set(batch.sample_ids) for batch in loaders.train))
                val_ids = set().union(*(set(batch.sample_ids) for batch in loaders.val))

                self_ref = self
                self_ref.assertEqual(train_ids, {"s0", "s1", "s2", "s3"})
                self_ref.assertEqual(val_ids, {"s4", "s5"})
                self_ref.assertTrue(train_ids.isdisjoint(val_ids))
                self_ref.assertTrue(train_ids.isdisjoint({"s6", "s7"}))
                self_ref.assertEqual(tuple(batch.sample_ids for batch in loaders.val), (("s4",), ("s5",)))

                y_val = np.concatenate([batch.targets for batch in loaders.val])

                def objective(trial):
                    preds = np.full(len(y_val), trial["constant"], dtype=float)
                    return float(np.sqrt(np.mean((y_val - preds) ** 2)))

                return objective

            def fit_selected_model(self, split, best_trial, *, refit_policy):
                del split, refit_policy
                return ConstantPredictor(float(best_trial["constant"]))

            def predict(self, model, dataset):
                seen_test_ids.append(tuple(dataset.sample_ids.tolist()))
                return np.full(len(dataset), model.constant, dtype=float)

            def trial_metadata(self, best_trial, model):
                return {
                    "constant": float(best_trial["constant"]),
                    "selected_constant": float(model.constant),
                }

        spec = FakeLearnedTrialTuningSpec()
        spec.assertEqual = self.assertEqual
        spec.assertTrue = self.assertTrue
        self.assertIsInstance(spec, LearnedTrialTuningSpec)
        trial_factory = lambda split: (  # noqa: E731
            {"constant": 0.0},
            {"constant": 5.0},
        )

        artifacts = sweep_learned_trial_model_selection(payload, spec, trial_factory)
        runner_artifacts = LearnedTrialModelSelectionSweepRunner(
            spec,
            trial_factory,
        ).run_artifacts_with_validation(payload)
        result = sweep_learned_model_with_trial_tuning(payload, spec, trial_factory)

        self.assertEqual(
            seen_train_orders,
            [
                (("s2", "s0"), ("s1", "s3")),
                (("s2", "s0"), ("s1", "s3")),
                (("s2", "s0"), ("s1", "s3")),
            ],
        )
        self.assertEqual(
            seen_val_orders,
            [
                (("s4",), ("s5",)),
                (("s4",), ("s5",)),
                (("s4",), ("s5",)),
            ],
        )
        self.assertEqual(seen_test_ids, [("s6", "s7"), ("s6", "s7"), ("s6", "s7")])
        self.assertNotEqual(
            seen_train_orders[0],
            (("s0", "s1"), ("s2", "s3")),
        )
        pd.testing.assert_frame_equal(artifacts.metrics, runner_artifacts.metrics)
        pd.testing.assert_frame_equal(artifacts.metrics, result)

    def test_learned_trial_tuning_end_to_end_batched_loaders_touch_outer_test_once(
        self,
    ) -> None:
        dataset = SweepDataset(
            mlip_features=np.arange(24, dtype=float).reshape(8, 3),
            targets=np.array([0.0, 0.0, 1.0, 1.0, 4.0, 4.0, 9.0, 9.0]),
            sample_ids=np.array(["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"]),
        )
        payload = SweepRunnerPayload(
            splits=(
                TrainValTestSweepRunnerInput(
                    dataset=dataset,
                    sweep_size=5,
                    train_idx=np.array([0, 1, 2, 3]),
                    val_idx=np.array([4, 5]),
                    test_idx=np.array([6, 7]),
                ),
            )
        )
        objective_loader_signatures: list[tuple[tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...]]] = []
        fit_loader_signatures: list[tuple[tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...]]] = []
        test_predict_signatures: list[tuple[str, ...]] = []

        class ConstantPredictor:
            def __init__(self, constant: float) -> None:
                self.constant = constant

        class FakeLearnedTrialTuningSpec:
            _loader_adapter = SweepDatasetBatchLoaderAdapter(
                policy=TrainEvalLoaderPolicy(
                    batch_size=2,
                    eval_batch_size=1,
                    train_shuffle=True,
                    eval_shuffle=False,
                )
            )

            def build_trial_objective(self, split):
                loaders = split.loaders(self._loader_adapter)
                train_batches = tuple(batch.sample_ids for batch in loaders.train)
                val_batches = tuple(batch.sample_ids for batch in loaders.val)
                objective_loader_signatures.append((train_batches, val_batches))
                self_ref = self
                self_ref.assertEqual(
                    set().union(*(set(batch) for batch in train_batches)),
                    {"s0", "s1", "s2", "s3"},
                )
                self_ref.assertEqual(val_batches, (("s4",), ("s5",)))
                y_val = np.concatenate([batch.targets for batch in loaders.val])

                def objective(trial):
                    preds = np.full(len(y_val), trial["constant"], dtype=float)
                    return float(np.sqrt(np.mean((y_val - preds) ** 2)))

                return objective

            def fit_selected_model(self, split, best_trial, *, refit_policy):
                loaders = split.loaders(self._loader_adapter)
                fit_loader_signatures.append(
                    (
                        tuple(batch.sample_ids for batch in loaders.train),
                        tuple(batch.sample_ids for batch in loaders.val),
                    )
                )
                del split, refit_policy
                return ConstantPredictor(float(best_trial["constant"]))

            def predict(self, model, dataset):
                test_predict_signatures.append(tuple(dataset.sample_ids.tolist()))
                return np.full(len(dataset), model.constant, dtype=float)

            def trial_metadata(self, best_trial, model):
                return {
                    "constant": float(best_trial["constant"]),
                    "selected_constant": float(model.constant),
                }

        spec = FakeLearnedTrialTuningSpec()
        spec.assertEqual = self.assertEqual
        self.assertIsInstance(spec, LearnedTrialTuningSpec)
        trial_factory = lambda split: (  # noqa: E731
            {"constant": 0.0},
            {"constant": 4.0},
        )

        artifacts = sweep_learned_trial_model_selection(payload, spec, trial_factory)
        runner_artifacts = LearnedTrialModelSelectionSweepRunner(
            spec,
            trial_factory,
        ).run_artifacts_with_validation(payload)
        result = sweep_learned_model_with_trial_tuning(payload, spec, trial_factory)

        self.assertEqual(
            objective_loader_signatures,
            [
                ((("s2", "s0"), ("s1", "s3")), (("s4",), ("s5",))),
                ((("s2", "s0"), ("s1", "s3")), (("s4",), ("s5",))),
                ((("s2", "s0"), ("s1", "s3")), (("s4",), ("s5",))),
            ],
        )
        self.assertEqual(
            fit_loader_signatures,
            [
                ((("s2", "s0"), ("s1", "s3")), (("s4",), ("s5",))),
                ((("s2", "s0"), ("s1", "s3")), (("s4",), ("s5",))),
                ((("s2", "s0"), ("s1", "s3")), (("s4",), ("s5",))),
            ],
        )
        self.assertEqual(
            test_predict_signatures,
            [("s6", "s7"), ("s6", "s7"), ("s6", "s7")],
        )
        pd.testing.assert_frame_equal(artifacts.metrics, runner_artifacts.metrics)
        pd.testing.assert_frame_equal(artifacts.metrics, result)

    def test_optuna_runner_objective_sees_only_train_and_val(self) -> None:
        dataset = SweepDataset(
            mlip_features=np.array([[0.0], [1.0], [2.0], [3.0], [40.0], [50.0]]),
            targets=np.array([0.0, 0.0, 2.0, 3.0, 4.0, 5.0]),
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
        objective_signatures: list[tuple[tuple[float, ...], tuple[float, ...]]] = []
        test_predict_signatures: list[tuple[float, ...]] = []

        class SpyModel:
            def __init__(self, constant: float) -> None:
                self.constant = constant

            def predict(self, X):
                signature = tuple(float(v) for v in np.ravel(X))
                test_predict_signatures.append(signature)
                return np.full(len(X), self.constant, dtype=float)

        class FakeOptunaSpec:
            def build_trial_objective(self, split):
                train_signature = tuple(
                    float(v)
                    for v in np.ravel(split.dataset.mlip_features[split.train_idx])
                )
                val_signature = tuple(
                    float(v)
                    for v in np.ravel(split.dataset.mlip_features[split.val_idx])
                )
                y_val = split.dataset.targets[split.val_idx]

                def objective(trial):
                    objective_signatures.append((train_signature, val_signature))
                    preds = np.full(len(y_val), trial.params["constant"], dtype=float)
                    return float(np.sqrt(np.mean((y_val - preds) ** 2)))

                return objective

            def fit_selected_model(self, split, best_trial, *, refit_policy):
                del split, refit_policy
                return SpyModel(float(best_trial.params["constant"]))

            def trial_metadata(self, best_trial, model):
                return {
                    "value": float(best_trial.params["constant"]),
                    "selected": float(model.constant),
                }

        class FakeTrial:
            def __init__(self, constant: float) -> None:
                self.params = {"constant": constant}
                self.value: float | None = None

        class FakeSampler:
            pass

        class FakePruner:
            pass

        class FakeStudy:
            def __init__(self, trials):
                self.trials = tuple(trials)
                self.best_trial = None
                self.sampler = FakeSampler()
                self.pruner = FakePruner()

            def optimize(self, objective, *, n_trials, timeout):
                del timeout
                best_value = np.inf
                for trial in self.trials[:n_trials]:
                    objective_value = objective(trial)
                    trial.value = objective_value
                    if objective_value < best_value:
                        best_value = objective_value
                        self.best_trial = trial

        study_factory = lambda split: FakeStudy((FakeTrial(0.0), FakeTrial(3.0)))  # noqa: E731
        spec = FakeOptunaSpec()

        artifacts = sweep_optuna_model_selection(
            payload,
            spec,
            n_trials=2,
            study_factory=study_factory,
        )

        self.assertEqual(
            objective_signatures,
            [((0.0, 1.0), (2.0, 3.0)), ((0.0, 1.0), (2.0, 3.0))],
        )
        self.assertEqual(test_predict_signatures, [(40.0, 50.0)])
        self.assertEqual(
            artifacts.metrics.columns.tolist(),
            ["n_train", "rmse_mean", "rmse_std"],
        )
        self.assertEqual(
            artifacts.selection_metadata.columns.tolist(),
            [
                "n_train",
                "best_validation_score",
                "constant",
                "pruner",
                "sampler",
                "selected",
                "trial_count",
                "value",
            ],
        )
        self.assertEqual(artifacts.selection_metadata["n_train"].tolist(), [4])
        self.assertEqual(artifacts.selection_metadata["constant"].tolist(), [3.0])
        self.assertEqual(
            artifacts.selection_metadata["best_validation_score"].tolist(),
            [np.sqrt(0.5)],
        )
        self.assertEqual(artifacts.selection_metadata["trial_count"].tolist(), [2])
        self.assertEqual(artifacts.selection_metadata["sampler"].tolist(), ["FakeSampler"])
        self.assertEqual(artifacts.selection_metadata["pruner"].tolist(), ["FakePruner"])

    def test_learned_optuna_runner_objective_sees_only_train_and_val(self) -> None:
        dataset = SweepDataset(
            mlip_features=np.array([[0.0], [1.0], [2.0], [3.0], [40.0], [50.0]]),
            targets=np.array([0.0, 0.0, 2.0, 3.0, 4.0, 5.0]),
            sample_ids=np.array(["s0", "s1", "s2", "s3", "s4", "s5"]),
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
        objective_signatures: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
        test_predict_signatures: list[tuple[str, ...]] = []

        class SpyModel:
            def __init__(self, constant: float) -> None:
                self.constant = constant

        class FakeLearnedOptunaSpec:
            def build_trial_objective(self, split):
                subsets = split.dataset_subsets()
                train_signature = tuple(subsets.train.sample_ids.tolist())
                val_signature = tuple(subsets.val.sample_ids.tolist())
                y_val = subsets.val.targets

                def objective(trial):
                    objective_signatures.append((train_signature, val_signature))
                    preds = np.full(len(y_val), trial.params["constant"], dtype=float)
                    return float(np.sqrt(np.mean((y_val - preds) ** 2)))

                return objective

            def fit_selected_model(self, split, best_trial, *, refit_policy):
                del split, refit_policy
                return SpyModel(float(best_trial.params["constant"]))

            def predict(self, model, dataset):
                test_predict_signatures.append(tuple(dataset.sample_ids.tolist()))
                return np.full(len(dataset), model.constant, dtype=float)

            def trial_metadata(self, best_trial, model):
                return {
                    "value": float(best_trial.params["constant"]),
                    "selected": float(model.constant),
                }

        class FakeTrial:
            def __init__(self, constant: float) -> None:
                self.params = {"constant": constant}
                self.value: float | None = None

        class FakeSampler:
            pass

        class FakePruner:
            pass

        class FakeStudy:
            def __init__(self, trials):
                self.trials = tuple(trials)
                self.best_trial = None
                self.sampler = FakeSampler()
                self.pruner = FakePruner()

            def optimize(self, objective, *, n_trials, timeout):
                del timeout
                best_value = np.inf
                for trial in self.trials[:n_trials]:
                    objective_value = objective(trial)
                    trial.value = objective_value
                    if objective_value < best_value:
                        best_value = objective_value
                        self.best_trial = trial

        study_factory = lambda split: FakeStudy((FakeTrial(0.0), FakeTrial(3.0)))  # noqa: E731
        spec = FakeLearnedOptunaSpec()

        artifacts = sweep_learned_optuna_model_selection(
            payload,
            spec,
            n_trials=2,
            study_factory=study_factory,
        )

        self.assertEqual(
            objective_signatures,
            [(("s0", "s1"), ("s2", "s3")), (("s0", "s1"), ("s2", "s3"))],
        )
        self.assertEqual(test_predict_signatures, [("s4", "s5")])
        self.assertEqual(
            artifacts.selection_metadata.columns.tolist(),
            [
                "n_train",
                "best_validation_score",
                "constant",
                "pruner",
                "sampler",
                "selected",
                "trial_count",
                "value",
            ],
        )
        self.assertEqual(artifacts.selection_metadata["constant"].tolist(), [3.0])

    def test_learned_optuna_runner_uses_outer_test_once_after_selection(self) -> None:
        dataset = SweepDataset(
            mlip_features=np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]),
            targets=np.array([0.0, 0.0, 1.0, 3.0, 4.0, 5.0]),
            sample_ids=np.array(["s0", "s1", "s2", "s3", "s4", "s5"]),
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
        events: list[tuple[str, float | tuple[str, ...]]] = []

        class FakeTrial:
            def __init__(self, value: float) -> None:
                self.params = {"value": value}
                self.value: float | None = None

        class FakeStudy:
            def __init__(self, trials):
                self.trials = tuple(trials)
                self.best_trial = None

            def optimize(self, objective, *, n_trials, timeout):
                del timeout
                best_value = np.inf
                for trial in self.trials[:n_trials]:
                    events.append(("objective", trial.params["value"]))
                    objective_value = objective(trial)
                    trial.value = objective_value
                    if objective_value < best_value:
                        best_value = objective_value
                        self.best_trial = trial

        class PredictSpyModel:
            def __init__(self, value: float) -> None:
                self.value = value

        class FakeLearnedOptunaSpec:
            def build_trial_objective(self, split):
                y_val = split.dataset_subsets().val.targets

                def objective(trial):
                    preds = np.full(len(y_val), trial.params["value"], dtype=float)
                    return float(np.sqrt(np.mean((y_val - preds) ** 2)))

                return objective

            def fit_selected_model(self, split, best_trial, *, refit_policy):
                del split, refit_policy
                events.append(("refit", float(best_trial.params["value"])))
                return PredictSpyModel(float(best_trial.params["value"]))

            def predict(self, model, dataset):
                events.append(("predict", tuple(dataset.sample_ids.tolist())))
                return np.full(len(dataset), model.value, dtype=float)

            def trial_metadata(self, best_trial, model):
                return {
                    "value": float(best_trial.params["value"]),
                    "selected": float(model.value),
                }

        study_factory = lambda split: FakeStudy((FakeTrial(0.0), FakeTrial(3.0)))  # noqa: E731

        result = sweep_learned_model_with_optuna_selection(
            payload,
            FakeLearnedOptunaSpec(),
            n_trials=2,
            study_factory=study_factory,
        )
        runner_result = LearnedOptunaModelSelectionSweepRunner(
            FakeLearnedOptunaSpec(),
            n_trials=2,
            study_factory=study_factory,
        ).run_artifacts_with_validation(payload)

        self.assertEqual(
            events,
            [
                ("objective", 0.0),
                ("objective", 3.0),
                ("refit", 3.0),
                ("predict", ("s4", "s5")),
                ("objective", 0.0),
                ("objective", 3.0),
                ("refit", 3.0),
                ("predict", ("s4", "s5")),
            ],
        )
        np.testing.assert_allclose(result["rmse_mean"].to_numpy(), [np.sqrt(2.5)], atol=1e-12)
        self.assertEqual(result.columns.tolist(), ["n_train", "rmse_mean", "rmse_std"])
        self.assertEqual(runner_result.metrics["n_train"].tolist(), [4])

    def test_optuna_runner_uses_outer_test_once_after_selection(self) -> None:
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
        events: list[tuple[str, float | tuple[float, ...]]] = []

        class FakeTrial:
            def __init__(self, value: float) -> None:
                self.params = {"value": value}
                self.value: float | None = None

        class FakeStudy:
            def __init__(self, trials):
                self.trials = tuple(trials)
                self.best_trial = None

            def optimize(self, objective, *, n_trials, timeout):
                del timeout
                best_value = np.inf
                for trial in self.trials[:n_trials]:
                    events.append(("objective", trial.params["value"]))
                    objective_value = objective(trial)
                    trial.value = objective_value
                    if objective_value < best_value:
                        best_value = objective_value
                        self.best_trial = trial

        class PredictSpyModel:
            def __init__(self, value: float) -> None:
                self.value = value

            def predict(self, X):
                events.append(("predict", tuple(float(v) for v in np.ravel(X))))
                return np.full(len(X), self.value, dtype=float)

        class FakeOptunaSpec:
            def build_trial_objective(self, split):
                y_val = split.dataset.targets[split.val_idx]

                def objective(trial):
                    preds = np.full(len(y_val), trial.params["value"], dtype=float)
                    return float(np.sqrt(np.mean((y_val - preds) ** 2)))

                return objective

            def fit_selected_model(self, split, best_trial, *, refit_policy):
                del split, refit_policy
                events.append(("refit", float(best_trial.params["value"])))
                return PredictSpyModel(float(best_trial.params["value"]))

            def trial_metadata(self, best_trial, model):
                return {
                    "value": float(best_trial.params["value"]),
                    "selected": float(model.value),
                }

        study_factory = lambda split: FakeStudy((FakeTrial(0.0), FakeTrial(3.0)))  # noqa: E731

        result = sweep_model_with_optuna_selection(
            payload,
            FakeOptunaSpec(),
            n_trials=2,
            study_factory=study_factory,
        )

        self.assertEqual(
            events,
            [
                ("objective", 0.0),
                ("objective", 3.0),
                ("refit", 3.0),
                ("predict", (4.0, 5.0)),
            ],
        )
        np.testing.assert_allclose(result["rmse_mean"].to_numpy(), [np.sqrt(2.5)], atol=1e-12)
        self.assertEqual(result.columns.tolist(), ["n_train", "rmse_mean", "rmse_std"])

    def test_optuna_runner_is_deterministic_for_seeded_study_factory(self) -> None:
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

        class FakeTrial:
            def __init__(self, name: str, val_constant: float, test_constant: float) -> None:
                self.name = name
                self.val_constant = val_constant
                self.test_constant = test_constant
                self.value: float | None = None
                self.params = {
                    "name": name,
                    "val_constant": val_constant,
                    "test_constant": test_constant,
                }

        class FakeStudy:
            def __init__(self, trials):
                self.trials = tuple(trials)
                self.best_trial = None

            def optimize(self, objective, *, n_trials, timeout):
                del timeout
                best_value = np.inf
                for trial in self.trials[:n_trials]:
                    objective_value = objective(trial)
                    trial.value = objective_value
                    if objective_value < best_value:
                        best_value = objective_value
                        self.best_trial = trial

        class PredictConstantModel:
            def __init__(self, value: float) -> None:
                self.value = value

            def predict(self, X):
                return np.full(len(X), self.value, dtype=float)

        class FakeOptunaSpec:
            def build_trial_objective(self, split):
                y_val = split.dataset.targets[split.val_idx]

                def objective(trial):
                    preds = np.full(len(y_val), trial.val_constant, dtype=float)
                    return float(np.sqrt(np.mean((y_val - preds) ** 2)))

                return objective

            def fit_selected_model(self, split, best_trial, *, refit_policy):
                del split, refit_policy
                return PredictConstantModel(float(best_trial.test_constant))

            def trial_metadata(self, best_trial, model):
                return {
                    "name": best_trial.name,
                    "test_constant": float(model.value),
                }

        def make_study_factory(seed: int):
            rng = np.random.default_rng(seed)
            ordered_trials = [
                FakeTrial("first", val_constant=0.0, test_constant=0.0),
                FakeTrial("second", val_constant=0.0, test_constant=10.0),
            ]
            if rng.integers(0, 2) == 1:
                ordered_trials = list(reversed(ordered_trials))

            def factory(split):
                del split
                return FakeStudy(ordered_trials)

            return factory

        spec = FakeOptunaSpec()
        first = OptunaModelSelectionSweepRunner(
            spec,
            n_trials=2,
            study_factory=make_study_factory(7),
        ).run_artifacts_with_validation(payload)
        second = OptunaModelSelectionSweepRunner(
            spec,
            n_trials=2,
            study_factory=make_study_factory(7),
        ).run_artifacts_with_validation(payload)

        pd.testing.assert_frame_equal(first.metrics, second.metrics)
        pd.testing.assert_frame_equal(first.selection_metadata, second.selection_metadata)
        self.assertEqual(first.selection_metadata["n_train"].tolist(), [4])


@unittest.skipUnless(HAS_TUNE, "requires tune dependencies")
class StudyFactoryFromOptunaCfgTests(unittest.TestCase):
    def _make_split(self) -> TrainValTestSweepRunnerInput:
        dataset = SweepDataset(
            mlip_features=np.zeros((4, 1)),
            targets=np.zeros(4),
        )
        return TrainValTestSweepRunnerInput(
            dataset=dataset,
            sweep_size=3,
            train_idx=np.array([0, 1]),
            val_idx=np.array([2]),
            test_idx=np.array([3]),
        )

    def test_tpe_sampler_by_name(self) -> None:
        import optuna

        cfg = OptunaTuningConfig(n_trials=5, sampler="tpe", seed=7)
        study = study_factory_from_optuna_cfg(cfg)(self._make_split())
        self.assertIsInstance(study.sampler, optuna.samplers.TPESampler)

    def test_default_sampler_is_tpe(self) -> None:
        import optuna

        cfg = OptunaTuningConfig(n_trials=5, sampler=None, seed=3)
        study = study_factory_from_optuna_cfg(cfg)(self._make_split())
        self.assertIsInstance(study.sampler, optuna.samplers.TPESampler)

    def test_random_sampler_by_name(self) -> None:
        import optuna

        cfg = OptunaTuningConfig(n_trials=5, sampler="random", seed=1)
        study = study_factory_from_optuna_cfg(cfg)(self._make_split())
        self.assertIsInstance(study.sampler, optuna.samplers.RandomSampler)

    def test_median_pruner_by_name(self) -> None:
        import optuna

        cfg = OptunaTuningConfig(n_trials=5, pruner="median")
        study = study_factory_from_optuna_cfg(cfg)(self._make_split())
        self.assertIsInstance(study.pruner, optuna.pruners.MedianPruner)

    def test_unknown_sampler_raises(self) -> None:
        cfg = OptunaTuningConfig(n_trials=5, sampler="bad_sampler")
        with self.assertRaises(ValueError):
            study_factory_from_optuna_cfg(cfg)(self._make_split())

    def test_unknown_pruner_raises(self) -> None:
        cfg = OptunaTuningConfig(n_trials=5, pruner="bad_pruner")
        with self.assertRaises(ValueError):
            study_factory_from_optuna_cfg(cfg)(self._make_split())
