from __future__ import annotations

import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pandas as pd

from oasis.exp import (
    build_sweep_split_collection,
    generate_inner_validation_sweep_splits,
    generate_sweep_splits,
    run_learning_curve_experiments,
)
from oasis.sweep import (
    LearningCurveResults,
    SweepDataset,
    SweepFamilyRequirements,
    SweepModelCapabilities,
    SweepRunnerPayload,
    SweepSplitCollection,
    SweepRunPayload,
    TrainTestSweepRunnerInput,
    TrainValTestSweepRunnerInput,
)
from tests.support import regression_dataset, regression_train_test_payload, weighted_fixed_payload, weighted_toy_dataset

try:
    import oasis.method as method_module

    from oasis.method import (
        ConfiguredSweepModelFamily,
        FactoryListHyperparameterSpec,
        GridHyperparameterSpec,
        HyperparameterSelectionSweepRunner,
        SupervisedModelSelectionSweepRunner,
        SupervisedModelSweepRunner,
        SweepFamilySpec,
        ValidationAwareSupervisedModelSweepRunner,
        enabled_learning_curve_model_names_from_config,
        learning_curve_model_registry,
        residual_sweep,
        residual_sweep_trimmed,
        sklearn_model_families,
        sklearn_sweep_model_specs,
        sweep_model,
        sweep_model_trimmed,
        sweep_model_with_hyperparameter_selection,
        sweep_model_with_hyperparameter_selection_trimmed,
        sweep_model_with_validation,
        sweep_supervised_model_selection,
        weighted_linear_sweep,
        weighted_simplex_sweep,
    )

    HAS_SKLEARN = True
except ModuleNotFoundError:
    HAS_SKLEARN = False


class SweepOutputRegressionTests(unittest.TestCase):
    _regression_dataset = staticmethod(regression_dataset)
    _train_test_payload = staticmethod(regression_train_test_payload)

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_sklearn_methods_are_registered_from_specs(self) -> None:
        specs = tuple(
            spec
            for spec in sklearn_sweep_model_specs()
            if spec[0] in {"ridge", "lasso", "elastic"}
        )

        families = sklearn_model_families(specs)

        self.assertTrue(
            all(isinstance(family, ConfiguredSweepModelFamily) for family in families)
        )
        self.assertEqual(
            [family.spec.result_field for family in families],
            ["ridge_df", "lasso_df", "elastic_df"],
        )
        self.assertEqual(
            [family.spec.trimmed_result_field for family in families],
            ["ridge_trimmed_df", "lasso_trimmed_df", "elastic_trimmed_df"],
        )

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_non_sklearn_methods_are_configured_as_first_class_families(self) -> None:
        from oasis.method import default_sweep_model_families

        families = default_sweep_model_families(
            ["residual", "weighted_linear", "weighted_simplex"]
        )

        self.assertTrue(
            all(isinstance(family, ConfiguredSweepModelFamily) for family in families)
        )
        self.assertEqual(
            [family.spec.result_field for family in families],
            [
                "resid_df",
                "weighted_linear_df",
                "weighted_simplex_df",
            ],
        )
        self.assertEqual(
            [family.spec.trimmed_result_field for family in families],
            ["resid_trimmed_df", None, None],
        )

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_registry_drives_enabled_model_assembly_from_config(self) -> None:
        registry = learning_curve_model_registry()
        self.assertEqual(
            [registration.name for registration in registry],
            [
                "ridge",
                "kernel_ridge",
                "lasso",
                "elastic",
                "residual",
                "weighted_linear",
                "weighted_simplex",
            ],
        )

        model_cfg = SimpleNamespace(
            use_ridge=True,
            use_kernel_ridge=False,
            use_lasso=True,
            use_elastic_net=False,
            use_residual=True,
            use_weighted_linear=True,
            use_weighted_simplex=True,
        )

        self.assertEqual(
            enabled_learning_curve_model_names_from_config(model_cfg),
            (
                "ridge",
                "lasso",
                "residual",
                "weighted_linear",
                "weighted_simplex",
            ),
        )

        built_in_requirements = [registration.family_factory().requirements() for registration in registry]
        self.assertEqual(
            built_in_requirements,
            [
                SweepFamilyRequirements(requires_inner_validation=True),
                SweepFamilyRequirements(requires_inner_validation=True),
                SweepFamilyRequirements(requires_inner_validation=True),
                SweepFamilyRequirements(requires_inner_validation=True),
                SweepFamilyRequirements(),
                SweepFamilyRequirements(),
                SweepFamilyRequirements(),
            ],
        )
        built_in_capabilities = [registration.family_factory().capabilities() for registration in registry]
        self.assertEqual(
            built_in_capabilities,
            [
                SweepModelCapabilities(requires_validation=True),
                SweepModelCapabilities(requires_validation=True),
                SweepModelCapabilities(requires_validation=True),
                SweepModelCapabilities(requires_validation=True),
                SweepModelCapabilities(),
                SweepModelCapabilities(),
                SweepModelCapabilities(),
            ],
        )

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_all_methods_consume_same_split_counts_and_keep_result_shape(self) -> None:
        X, y = self._regression_dataset()
        payload = SweepRunPayload(
            dataset=SweepDataset(mlip_features=X, targets=y),
            split_collection=SweepSplitCollection(
                splits=tuple(
                    generate_sweep_splits(
                        n_samples=len(X),
                        min_train=2,
                        max_train=4,
                        n_repeats=2,
                        rng=np.random.default_rng(7),
                    )
                )
            ),
            use_trim=True,
        )
        expected_counts = [2, 3, 4]
        expected_columns = ["n_train", "rmse_mean", "rmse_std"]

        class DummyModel:
            def fit(self, X, y):
                X = np.asarray(X)
                self.coef_ = np.ones(X.shape[1], dtype=float)
                return self

            def predict(self, X):
                return np.asarray(X).mean(axis=1)

        results = [
            sweep_model(payload, lambda: DummyModel()),
            sweep_model_trimmed(payload, lambda: DummyModel()),
            residual_sweep(payload),
            residual_sweep_trimmed(payload),
            weighted_linear_sweep(payload),
            weighted_simplex_sweep(payload),
        ]

        for df in results:
            self.assertEqual(df["n_train"].tolist(), expected_counts)
            self.assertEqual(df.columns.tolist(), expected_columns)

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_registry_pipeline_matches_direct_method_outputs(self) -> None:
        X, y = self._regression_dataset()
        dataset = SweepDataset(mlip_features=X, targets=y)
        payload = self._train_test_payload(seed=13)
        sklearn_specs = {
            name: spec for name, _, spec in sklearn_sweep_model_specs()
        }
        validation_payload = SweepRunPayload(
            dataset=dataset,
            split_collection=build_sweep_split_collection(
                dataset.n_samples,
                min_train=2,
                max_train=4,
                n_repeats=2,
                seed=13,
                requirements=SweepFamilyRequirements(
                    requires_inner_validation=True,
                ),
            ),
            use_trim=True,
        )

        expected = {
            "ridge_df": sweep_model_with_hyperparameter_selection(
                validation_payload,
                sklearn_specs["ridge"].hyperparameter_spec,
            ),
            "kernel_ridge_df": sweep_model_with_hyperparameter_selection(
                validation_payload,
                sklearn_specs["kernel_ridge"].hyperparameter_spec,
            ),
            "ridge_trimmed_df": sweep_model_with_hyperparameter_selection_trimmed(
                validation_payload,
                sklearn_specs["ridge"].hyperparameter_spec,
                z_thresh=sklearn_specs["ridge"].trim_z_thresh,
            ),
            "lasso_df": sweep_model_with_hyperparameter_selection(
                validation_payload,
                sklearn_specs["lasso"].hyperparameter_spec,
            ),
            "lasso_trimmed_df": sweep_model_with_hyperparameter_selection_trimmed(
                validation_payload,
                sklearn_specs["lasso"].hyperparameter_spec,
                z_thresh=sklearn_specs["lasso"].trim_z_thresh,
            ),
            "elastic_df": sweep_model_with_hyperparameter_selection(
                validation_payload,
                sklearn_specs["elastic"].hyperparameter_spec,
            ),
            "elastic_trimmed_df": sweep_model_with_hyperparameter_selection_trimmed(
                validation_payload,
                sklearn_specs["elastic"].hyperparameter_spec,
                z_thresh=sklearn_specs["elastic"].trim_z_thresh,
            ),
            "resid_df": residual_sweep(payload),
            "resid_trimmed_df": residual_sweep_trimmed(payload),
            "weighted_linear_df": weighted_linear_sweep(payload),
            "weighted_simplex_df": weighted_simplex_sweep(payload),
        }

        actual = run_learning_curve_experiments(
            dataset,
            min_train=2,
            max_train=4,
            n_repeats=2,
            seed=13,
            use_trim=True,
            enabled_model_names=[
                "ridge",
                "kernel_ridge",
                "lasso",
                "elastic",
                "residual",
                "weighted_linear",
                "weighted_simplex",
            ],
        )

        for field_name, expected_df in expected.items():
            actual_df = getattr(actual, field_name)
            self.assertIsNotNone(actual_df, field_name)
            pd.testing.assert_frame_equal(actual_df, expected_df)

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_ridge_is_registered_as_validation_aware_selection_method(self) -> None:
        ridge_registration = next(
            registration
            for registration in learning_curve_model_registry()
            if registration.name == "ridge"
        )
        ridge_family = ridge_registration.family_factory()

        self.assertEqual(
            ridge_family.capabilities(),
            SweepModelCapabilities(requires_validation=True),
        )
        self.assertEqual(
            ridge_family.requirements(),
            SweepFamilyRequirements(requires_inner_validation=True),
        )
        self.assertIsInstance(
            ridge_family.spec.runner,
            SupervisedModelSelectionSweepRunner,
        )

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_ridge_mixed_sweep_integration_keeps_expected_output_shape(self) -> None:
        X, y = self._regression_dataset()

        results = run_learning_curve_experiments(
            SweepDataset(mlip_features=X, targets=y),
            min_train=2,
            max_train=4,
            n_repeats=1,
            seed=17,
            use_trim=False,
            enabled_model_names=["ridge", "weighted_linear"],
        )

        self.assertIsNotNone(results.ridge_df)
        self.assertIsNotNone(results.weighted_linear_df)
        self.assertEqual(
            results.ridge_df.columns.tolist(),
            ["n_train", "rmse_mean", "rmse_std"],
        )
        self.assertEqual(results.ridge_df["n_train"].tolist(), [2, 3, 4])
        self.assertEqual(
            results.weighted_linear_df.columns.tolist(),
            ["n_train", "rmse_mean", "rmse_std"],
        )

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_ridge_skips_undersized_budgets_predictably(self) -> None:
        X, y = self._regression_dataset()

        results = run_learning_curve_experiments(
            SweepDataset(mlip_features=X, targets=y),
            min_train=1,
            max_train=3,
            n_repeats=1,
            seed=17,
            use_trim=False,
            enabled_model_names=["ridge"],
        )

        self.assertIsNotNone(results.ridge_df)
        self.assertEqual(results.ridge_df["n_train"].tolist(), [2, 3])

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_ridge_selection_is_deterministic_for_fixed_seed(self) -> None:
        X, y = self._regression_dataset()
        dataset = SweepDataset(mlip_features=X, targets=y)

        first = run_learning_curve_experiments(
            dataset,
            min_train=2,
            max_train=4,
            n_repeats=2,
            seed=19,
            use_trim=False,
            enabled_model_names=["ridge"],
        )
        second = run_learning_curve_experiments(
            dataset,
            min_train=2,
            max_train=4,
            n_repeats=2,
            seed=19,
            use_trim=False,
            enabled_model_names=["ridge"],
        )

        pd.testing.assert_frame_equal(first.ridge_df, second.ridge_df)

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_selection_metadata_surfaces_without_changing_rmse_frames(self) -> None:
        X, y = self._regression_dataset()

        results = run_learning_curve_experiments(
            SweepDataset(mlip_features=X, targets=y),
            min_train=2,
            max_train=4,
            n_repeats=1,
            seed=37,
            use_trim=False,
            enabled_model_names=["ridge", "kernel_ridge"],
        )

        self.assertEqual(
            results.ridge_df.columns.tolist(),
            ["n_train", "rmse_mean", "rmse_std"],
        )
        self.assertIsNotNone(results.ridge_selection_df)
        self.assertEqual(
            results.ridge_selection_df.columns.tolist(),
            ["n_train", "alpha"],
        )
        self.assertEqual(
            results.ridge_selection_df["n_train"].tolist(),
            results.ridge_df["n_train"].tolist(),
        )
        self.assertTrue(
            set(results.ridge_selection_df["alpha"]).issubset({0.01, 0.1, 1.0, 10.0})
        )

        self.assertIsNotNone(results.kernel_ridge_selection_df)
        self.assertEqual(
            results.kernel_ridge_selection_df.columns.tolist(),
            ["n_train", "alpha", "gamma", "kernel"],
        )
        self.assertEqual(
            results.kernel_ridge_selection_df["n_train"].tolist(),
            results.kernel_ridge_df["n_train"].tolist(),
        )

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_ridge_train_test_runner_payload_preserves_legacy_behavior(self) -> None:
        payload = self._train_test_payload(seed=19)
        runner_payload = payload.to_runner_payload()
        ridge_spec = {
            name: spec for name, _, spec in sklearn_sweep_model_specs()
        }["ridge"]

        self.assertTrue(
            all(
                isinstance(split, TrainTestSweepRunnerInput)
                for split in runner_payload.splits
            )
        )

        legacy = sweep_model(payload, ridge_spec.model_factory)
        split_aware = sweep_model(runner_payload, ridge_spec.model_factory)

        pd.testing.assert_frame_equal(split_aware, legacy)

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_lasso_train_test_runner_payload_preserves_legacy_behavior(self) -> None:
        payload = self._train_test_payload(seed=19)
        runner_payload = payload.to_runner_payload()
        lasso_spec = {
            name: spec for name, _, spec in sklearn_sweep_model_specs()
        }["lasso"]

        self.assertTrue(
            all(
                isinstance(split, TrainTestSweepRunnerInput)
                for split in runner_payload.splits
            )
        )

        legacy = sweep_model(payload, lasso_spec.model_factory)
        split_aware = sweep_model(runner_payload, lasso_spec.model_factory)

        pd.testing.assert_frame_equal(split_aware, legacy)

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_residual_train_test_runner_payload_preserves_legacy_behavior(self) -> None:
        payload = self._train_test_payload(seed=19)
        runner_payload = payload.to_runner_payload()

        self.assertTrue(
            all(
                isinstance(split, TrainTestSweepRunnerInput)
                for split in runner_payload.splits
            )
        )

        legacy = residual_sweep(payload)
        split_aware = residual_sweep(runner_payload)

        pd.testing.assert_frame_equal(split_aware, legacy)


class WeightedBaselineRegressionTests(unittest.TestCase):
    _toy_dataset = staticmethod(weighted_toy_dataset)
    _fixed_payload = staticmethod(weighted_fixed_payload)

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_weighted_linear_keeps_standard_result_shape(self) -> None:
        result = weighted_linear_sweep(self._fixed_payload())

        self.assertEqual(
            result.columns.tolist(),
            ["n_train", "rmse_mean", "rmse_std"],
        )
        self.assertEqual(result["n_train"].tolist(), [4, 5])
        self.assertEqual(len(result), 2)

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_weighted_linear_is_deterministic_for_fixed_splits(self) -> None:
        payload = self._fixed_payload()

        first = weighted_linear_sweep(payload)
        second = weighted_linear_sweep(payload)

        pd.testing.assert_frame_equal(first, second)

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_weighted_linear_has_near_zero_error_on_noiseless_toy_data(self) -> None:
        result = weighted_linear_sweep(self._fixed_payload())

        np.testing.assert_allclose(result["rmse_mean"].to_numpy(), 0.0, atol=1e-12)
        np.testing.assert_allclose(result["rmse_std"].to_numpy(), 0.0, atol=1e-12)

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_weighted_simplex_keeps_standard_result_shape(self) -> None:
        result = weighted_simplex_sweep(self._fixed_payload())

        self.assertEqual(
            result.columns.tolist(),
            ["n_train", "rmse_mean", "rmse_std"],
        )
        self.assertEqual(result["n_train"].tolist(), [4, 5])
        self.assertEqual(len(result), 2)

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_weighted_simplex_is_deterministic_for_fixed_splits(self) -> None:
        payload = self._fixed_payload()

        first = weighted_simplex_sweep(payload)
        second = weighted_simplex_sweep(payload)

        pd.testing.assert_frame_equal(first, second)

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_weighted_simplex_exactly_recovers_convex_toy_target(
        self,
    ) -> None:
        X, _ = self._toy_dataset()
        y = 0.75 * X[:, 0] + 0.25 * X[:, 1]
        payload = SweepRunPayload(
            dataset=SweepDataset(mlip_features=X, targets=y),
            split_collection=self._fixed_payload().split_collection,
            use_trim=False,
        )

        result = weighted_simplex_sweep(payload)

        np.testing.assert_allclose(result["rmse_mean"].to_numpy(), 0.0, atol=1e-12)
        np.testing.assert_allclose(result["rmse_std"].to_numpy(), 0.0, atol=1e-12)

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_weighted_linear_ignores_unrelated_auxiliary_views(self) -> None:
        X, y = self._toy_dataset()
        split_collection = self._fixed_payload().split_collection
        base_payload = SweepRunPayload(
            dataset=SweepDataset(mlip_features=X, targets=y),
            split_collection=split_collection,
            use_trim=False,
        )
        payload_with_auxiliary_views = SweepRunPayload(
            dataset=SweepDataset(
                mlip_features=X,
                targets=y,
                auxiliary_views={
                    "graphs": np.array(
                        [{"edges": 1}, {"edges": 2}, {"edges": 3}, {"edges": 4}, {"edges": 5}, {"edges": 6}],
                        dtype=object,
                    )
                },
            ),
            split_collection=split_collection,
            use_trim=False,
        )

        baseline_result = weighted_linear_sweep(base_payload)
        result_with_auxiliary_views = weighted_linear_sweep(
            payload_with_auxiliary_views
        )

        pd.testing.assert_frame_equal(
            baseline_result,
            result_with_auxiliary_views,
        )


class BoundaryTests(unittest.TestCase):
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

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_sklearn_specs_declare_validation_search_spaces(self) -> None:
        specs = {name: spec for name, _, spec in sklearn_sweep_model_specs()}

        self.assertIsNotNone(specs["ridge"].hyperparameter_spec)
        self.assertIsNotNone(specs["kernel_ridge"].hyperparameter_spec)
        self.assertIsNotNone(specs["lasso"].hyperparameter_spec)
        self.assertIsNotNone(specs["elastic"].hyperparameter_spec)

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_lasso_selection_can_choose_different_hyperparameters_by_split(self) -> None:
        class SpyLasso:
            def __init__(self, alpha: float, max_iter: int = 10000) -> None:
                del max_iter
                self.alpha = alpha
                self.coef_ = np.array([alpha], dtype=float)

            def fit(self, X, y):
                del X, y
                return self

            def predict(self, X):
                signature = tuple(float(v) for v in np.ravel(X))
                if signature == (2.0,):
                    return np.full(len(X), self.alpha, dtype=float)
                return np.full(len(X), 1.0 - self.alpha, dtype=float)

        with patch.object(method_module, "Lasso", SpyLasso):
            lasso_spec = {
                name: spec for name, _, spec in method_module.sklearn_sweep_model_specs()
            }["lasso"]
            dataset = SweepDataset(
                mlip_features=np.array([[0.0], [1.0], [2.0], [3.0], [4.0]]),
                targets=np.array([0.0, 0.0, 0.001, 0.99, 0.0]),
            )
            first_split = TrainValTestSweepRunnerInput(
                dataset=dataset,
                sweep_size=3,
                train_idx=np.array([0, 1]),
                val_idx=np.array([2]),
                test_idx=np.array([4]),
            )
            second_split = TrainValTestSweepRunnerInput(
                dataset=dataset,
                sweep_size=3,
                train_idx=np.array([0, 1]),
                val_idx=np.array([3]),
                test_idx=np.array([4]),
            )

            first_model = method_module._select_candidate_factory_by_validation(
                first_split,
                lasso_spec.hyperparameter_spec,
            )()
            second_model = method_module._select_candidate_factory_by_validation(
                second_split,
                lasso_spec.hyperparameter_spec,
            )()

        self.assertNotEqual(first_model.alpha, second_model.alpha)

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_elastic_selection_can_choose_different_hyperparameters_by_split(self) -> None:
        class SpyElasticNet:
            def __init__(
                self,
                alpha: float,
                l1_ratio: float,
                max_iter: int = 20000,
            ) -> None:
                del max_iter
                self.alpha = alpha
                self.l1_ratio = l1_ratio
                self.coef_ = np.array([alpha + l1_ratio], dtype=float)

            def fit(self, X, y):
                del X, y
                return self

            def predict(self, X):
                signature = tuple(float(v) for v in np.ravel(X))
                if signature == (2.0,):
                    return np.full(len(X), self.alpha + self.l1_ratio, dtype=float)
                return np.full(
                    len(X),
                    abs((self.alpha + self.l1_ratio) - 1.1),
                    dtype=float,
                )

        with patch.object(method_module, "ElasticNet", SpyElasticNet):
            elastic_spec = {
                name: spec for name, _, spec in method_module.sklearn_sweep_model_specs()
            }["elastic"]
            dataset = SweepDataset(
                mlip_features=np.array([[0.0], [1.0], [2.0], [3.0], [4.0]]),
                targets=np.array([0.0, 0.0, 0.201, 0.299, 0.0]),
            )
            first_split = TrainValTestSweepRunnerInput(
                dataset=dataset,
                sweep_size=3,
                train_idx=np.array([0, 1]),
                val_idx=np.array([2]),
                test_idx=np.array([4]),
            )
            second_split = TrainValTestSweepRunnerInput(
                dataset=dataset,
                sweep_size=3,
                train_idx=np.array([0, 1]),
                val_idx=np.array([3]),
                test_idx=np.array([4]),
            )

            first_model = method_module._select_candidate_factory_by_validation(
                first_split,
                elastic_spec.hyperparameter_spec,
            )()
            second_model = method_module._select_candidate_factory_by_validation(
                second_split,
                elastic_spec.hyperparameter_spec,
            )()

        self.assertNotEqual(
            (first_model.alpha, first_model.l1_ratio),
            (second_model.alpha, second_model.l1_ratio),
        )

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_lasso_and_elastic_skip_undersized_budgets_predictably(self) -> None:
        X, y = SweepOutputRegressionTests._regression_dataset()

        results = run_learning_curve_experiments(
            SweepDataset(mlip_features=X, targets=y),
            min_train=1,
            max_train=3,
            n_repeats=1,
            seed=23,
            use_trim=False,
            enabled_model_names=["lasso", "elastic"],
        )

        self.assertEqual(results.lasso_df["n_train"].tolist(), [2, 3])
        self.assertEqual(results.elastic_df["n_train"].tolist(), [2, 3])

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_train_test_baselines_remain_unaffected_by_lasso_and_elastic(self) -> None:
        X, y = SweepOutputRegressionTests._regression_dataset()
        dataset = SweepDataset(mlip_features=X, targets=y)

        baseline_only = run_learning_curve_experiments(
            dataset,
            min_train=2,
            max_train=4,
            n_repeats=1,
            seed=29,
            use_trim=False,
            enabled_model_names=["weighted_linear"],
        )
        with_selection_families = run_learning_curve_experiments(
            dataset,
            min_train=2,
            max_train=4,
            n_repeats=1,
            seed=29,
            use_trim=False,
            enabled_model_names=["weighted_linear", "lasso", "elastic"],
        )

        pd.testing.assert_frame_equal(
            baseline_only.weighted_linear_df,
            with_selection_families.weighted_linear_df,
        )

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_kernel_ridge_candidate_order_is_deterministic_and_bounded(self) -> None:
        kernel_ridge_spec = {
            name: spec for name, _, spec in sklearn_sweep_model_specs()
        }["kernel_ridge"]

        candidates = [
            factory() for factory in kernel_ridge_spec.hyperparameter_spec.candidate_factories()
        ]

        self.assertEqual(len(candidates), 6)
        self.assertEqual(
            [(candidate.alpha, candidate.gamma) for candidate in candidates],
            [
                (0.1, 0.1),
                (0.1, 1.0),
                (1.0, 0.1),
                (1.0, 1.0),
                (10.0, 0.1),
                (10.0, 1.0),
            ],
        )

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_kernel_ridge_skips_expected_small_budget_ranges(self) -> None:
        X, y = SweepOutputRegressionTests._regression_dataset()

        results = run_learning_curve_experiments(
            SweepDataset(mlip_features=X, targets=y),
            min_train=1,
            max_train=3,
            n_repeats=1,
            seed=31,
            use_trim=False,
            enabled_model_names=["kernel_ridge"],
        )

        self.assertIsNotNone(results.kernel_ridge_df)
        self.assertEqual(results.kernel_ridge_df["n_train"].tolist(), [2, 3])

    def test_validation_aware_supervised_runner_uses_train_val_test_path(self) -> None:
        class ValidationAwareLinearModel:
            def fit(self, X_train, y_train, X_val, y_val) -> None:
                del X_val, y_val
                self.coef_, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)

            def predict(self, X):
                return X @ self.coef_

        dataset = SweepDataset(
            mlip_features=np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [2.0, 1.0],
                    [1.0, 2.0],
                    [2.0, 2.0],
                ]
            ),
            targets=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
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

        result = sweep_model_with_validation(
            payload,
            ValidationAwareLinearModel,
        )
        runner_result = ValidationAwareSupervisedModelSweepRunner(
            ValidationAwareLinearModel
        ).run_with_validation(payload)

        self.assertEqual(
            result.columns.tolist(),
            ["n_train", "rmse_mean", "rmse_std"],
        )
        self.assertEqual(result["n_train"].tolist(), [4])
        pd.testing.assert_frame_equal(result, runner_result)

    def test_configured_family_capabilities_round_trip_to_requirements(self) -> None:
        if not HAS_SKLEARN:
            self.skipTest("requires scikit-learn")

        family = ConfiguredSweepModelFamily(
            spec=SweepFamilySpec(
                result_field="ridge_df",
                trimmed_result_field=None,
                runner=SupervisedModelSweepRunner(lambda: None),
                capabilities=SweepModelCapabilities(
                    min_train_size=6,
                    requires_validation=True,
                ),
            )
        )

        self.assertEqual(
            family.capabilities(),
            SweepModelCapabilities(
                min_train_size=6,
                requires_validation=True,
            ),
        )
        self.assertEqual(
            family.requirements(),
            SweepFamilyRequirements(
                min_train_size=6,
                requires_inner_validation=True,
            ),
        )
