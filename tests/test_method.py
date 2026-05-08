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
from oasis.tune import (
    OptunaModelSelectionSweepRunner,
    SupervisedModelSelectionSweepRunner,
    sweep_model_with_hyperparameter_selection,
    sweep_model_with_hyperparameter_selection_trimmed,
    sweep_model_with_optuna_selection,
    sweep_model_with_optuna_selection_trimmed,
)
from tests.support import regression_dataset, regression_train_test_payload, weighted_fixed_payload, weighted_toy_dataset

try:
    import oasis.method as method_module

    from oasis.method import (
        ConfiguredSweepModelFamily,
        LearnedFamilyRegistrationSpec,
        PlaceholderLearnedSweepModelFamily,
        SupervisedModelSweepRunner,
        SweepFamilySpec,
        ValidationAwareSupervisedModelSweepRunner,
        default_sweep_model_families,
        enabled_learning_curve_model_names_from_config,
        learned_family_registration,
        learning_curve_model_registry,
        residual_sweep,
        residual_sweep_trimmed,
        sklearn_model_families,
        sklearn_sweep_model_specs,
        sweep_model,
        sweep_model_trimmed,
        sweep_model_with_validation,
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
                "moe",
            ],
        )

        model_cfg = SimpleNamespace(
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(
                    models=SimpleNamespace(
                        use_ridge=True,
                        use_kernel_ridge=False,
                        use_lasso=True,
                        use_elastic_net=False,
                        use_residual=True,
                        use_weighted_linear=True,
                        use_weighted_simplex=True,
                    )
                )
            )
        )

        self.assertEqual(
            enabled_learning_curve_model_names_from_config(
                model_cfg.experiment.learning_curve.models
            ),
            (
                "ridge",
                "lasso",
                "residual",
                "weighted_linear",
                "weighted_simplex",
            ),
        )

        self.assertEqual(
            enabled_learning_curve_model_names_from_config(None),
            (
                "ridge",
                "kernel_ridge",
                "lasso",
                "elastic",
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
                SweepFamilyRequirements(requires_inner_validation=True),
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
                SweepModelCapabilities(requires_validation=True),
            ],
        )

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_learned_family_registration_supports_stub_non_sklearn_family(self) -> None:
        class StubLearnedFamily:
            def capabilities(self) -> SweepModelCapabilities:
                return SweepModelCapabilities(requires_validation=True)

            def requirements(self) -> SweepFamilyRequirements:
                return SweepFamilyRequirements(requires_inner_validation=True)

            def run(self, payload) -> LearningCurveResults:
                del payload
                return LearningCurveResults.empty()

        registration = learned_family_registration(
            LearnedFamilyRegistrationSpec(
                name="stub_learned",
                config_key="use_stub_learned",
                capabilities=SweepModelCapabilities(requires_validation=True),
                family_factory=StubLearnedFamily,
            )
        )

        self.assertEqual(registration.name, "stub_learned")
        self.assertTrue(registration.is_enabled(SimpleNamespace(use_stub_learned=True)))
        self.assertFalse(registration.is_enabled(SimpleNamespace(use_stub_learned=False)))

        family = registration.family_factory()
        self.assertEqual(
            family.capabilities(),
            SweepModelCapabilities(requires_validation=True),
        )
        self.assertEqual(
            family.requirements(),
            SweepFamilyRequirements(requires_inner_validation=True),
        )

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_placeholder_moe_registration_instantiates_cleanly(self) -> None:
        registry = {
            registration.name: registration
            for registration in learning_curve_model_registry()
        }

        moe_family = registry["moe"].family_factory()

        self.assertIsInstance(moe_family, PlaceholderLearnedSweepModelFamily)
        self.assertEqual(
            moe_family.capabilities(),
            SweepModelCapabilities(requires_validation=True),
        )
        self.assertEqual(
            moe_family.requirements(),
            SweepFamilyRequirements(requires_inner_validation=True),
        )

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_default_built_in_families_still_instantiate_and_run(self) -> None:
        X, y = self._regression_dataset()

        results = run_learning_curve_experiments(
            SweepDataset(mlip_features=X, targets=y),
            min_train=2,
            max_train=4,
            n_repeats=1,
            seed=11,
            use_trim=False,
            model_families=default_sweep_model_families(),
        )

        self.assertIsNotNone(results.ridge_df)
        self.assertIsNotNone(results.kernel_ridge_df)
        self.assertIsNotNone(results.lasso_df)
        self.assertIsNotNone(results.elastic_df)
        self.assertIsNotNone(results.resid_df)
        self.assertIsNotNone(results.weighted_linear_df)
        self.assertIsNotNone(results.weighted_simplex_df)

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
            "ridge_df": sweep_model_with_optuna_selection(
                validation_payload,
                sklearn_specs["ridge"].trial_tuning_spec,
                n_trials=sklearn_specs["ridge"].optuna_n_trials,
                study_factory=sklearn_specs["ridge"].optuna_study_factory,
            ),
            "kernel_ridge_df": sweep_model_with_hyperparameter_selection(
                validation_payload,
                sklearn_specs["kernel_ridge"].hyperparameter_spec,
            ),
            "ridge_trimmed_df": sweep_model_with_optuna_selection_trimmed(
                validation_payload,
                sklearn_specs["ridge"].trial_tuning_spec,
                n_trials=sklearn_specs["ridge"].optuna_n_trials,
                study_factory=sklearn_specs["ridge"].optuna_study_factory,
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
            OptunaModelSelectionSweepRunner,
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
            [
                "n_train",
                "alpha",
                "best_validation_score",
                "pruner",
                "sampler",
                "trial_count",
            ],
        )
        self.assertEqual(
            results.ridge_selection_df["n_train"].tolist(),
            results.ridge_df["n_train"].tolist(),
        )
        self.assertTrue(
            set(results.ridge_selection_df["alpha"]).issubset({0.01, 0.1, 1.0, 10.0})
        )
        self.assertEqual(set(results.ridge_selection_df["sampler"]), {"GridSampler"})
        self.assertEqual(len(set(results.ridge_selection_df["pruner"])), 1)

        self.assertTrue(
            all(
                count == 4
                for count in results.ridge_selection_df["trial_count"].tolist()
            )
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
    def test_sklearn_specs_declare_validation_search_spaces(self) -> None:
        specs = {name: spec for name, _, spec in sklearn_sweep_model_specs()}

        self.assertIsNotNone(specs["ridge"].trial_tuning_spec)
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
