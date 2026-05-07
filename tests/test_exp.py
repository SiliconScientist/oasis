from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oasis.exp import (
    LearningCurveResults,
    SweepDataset,
    SweepFamilyRequirements,
    SweepSplit,
    SweepSplitCollection,
    SweepRunPayload,
    generate_sweep_splits,
    prepare_parity_plot_data,
    run_learning_curve_experiments,
    run_learning_curve_experiments_from_config,
)
from oasis.plot import learning_curve_plot

try:
    from oasis.method import (
        ConfiguredSweepModelFamily,
        enabled_learning_curve_model_names_from_config,
        linearization_sweep,
        linearization_sweep_trimmed,
        learning_curve_model_registry,
        residual_sweep,
        residual_sweep_trimmed,
        sklearn_model_families,
        sklearn_sweep_model_specs,
        sweep_model,
        sweep_model_trimmed,
        weighted_combiner_sweep,
    )

    HAS_SKLEARN = True
except ModuleNotFoundError:
    HAS_SKLEARN = False


class GenerateSweepSplitsTests(unittest.TestCase):
    def test_generate_sweep_splits_yields_disjoint_full_partitions(self) -> None:
        rng = np.random.default_rng(123)

        splits = list(
            generate_sweep_splits(
                n_samples=6,
                min_train=2,
                max_train=3,
                n_repeats=2,
                rng=rng,
            )
        )

        self.assertEqual(len(splits), 4)
        self.assertTrue(all(isinstance(split, SweepSplit) for split in splits))

        expected_sizes = [2, 2, 3, 3]
        observed_sizes = [split.sweep_size for split in splits]
        self.assertEqual(observed_sizes, expected_sizes)

        full_idx = np.arange(6)
        for split in splits:
            self.assertEqual(len(split.train_idx), split.sweep_size)
            self.assertEqual(len(split.test_idx), 6 - split.sweep_size)
            self.assertEqual(
                len(np.intersect1d(split.train_idx, split.test_idx)),
                0,
            )
            np.testing.assert_array_equal(
                np.sort(np.concatenate([split.train_idx, split.test_idx])),
                full_idx,
            )

    def test_same_seed_gives_same_splits(self) -> None:
        splits_a = list(
            generate_sweep_splits(
                n_samples=8,
                min_train=2,
                max_train=4,
                n_repeats=3,
                rng=np.random.default_rng(42),
            )
        )
        splits_b = list(
            generate_sweep_splits(
                n_samples=8,
                min_train=2,
                max_train=4,
                n_repeats=3,
                rng=np.random.default_rng(42),
            )
        )

        self.assertEqual(len(splits_a), len(splits_b))
        for split_a, split_b in zip(splits_a, splits_b, strict=True):
            self.assertEqual(split_a.sweep_size, split_b.sweep_size)
            np.testing.assert_array_equal(split_a.train_idx, split_b.train_idx)
            np.testing.assert_array_equal(split_a.test_idx, split_b.test_idx)


class SweepOutputRegressionTests(unittest.TestCase):
    @staticmethod
    def _regression_dataset() -> tuple[np.ndarray, np.ndarray]:
        X = np.array(
            [
                [1.0, 1.2, 0.8],
                [1.8, 2.1, 1.9],
                [2.7, 3.0, 2.9],
                [3.9, 4.2, 3.8],
                [5.1, 5.0, 4.9],
                [6.2, 6.0, 5.8],
            ]
        )
        y = np.array([1.1, 2.0, 2.9, 4.0, 5.0, 6.1])
        return X, y

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

        families = default_sweep_model_families(["residual", "linearization"])

        self.assertTrue(
            all(isinstance(family, ConfiguredSweepModelFamily) for family in families)
        )
        self.assertEqual(
            [family.spec.result_field for family in families],
            ["resid_df", "linear_df"],
        )
        self.assertEqual(
            [family.spec.trimmed_result_field for family in families],
            ["resid_trimmed_df", "linear_trimmed_df"],
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
                "linearization",
                "weighted_combiner",
            ],
        )

        plot_cfg = SimpleNamespace(
            use_ridge=True,
            use_kernel_ridge=False,
            use_lasso=True,
            use_elastic_net=False,
            use_residual=True,
            use_linearization=False,
            use_weighted_combiner=True,
        )

        self.assertEqual(
            enabled_learning_curve_model_names_from_config(plot_cfg),
            ("ridge", "lasso", "residual", "weighted_combiner"),
        )

        built_in_requirements = [registration.family_factory().requirements() for registration in registry]
        self.assertTrue(
            all(requirement == SweepFamilyRequirements() for requirement in built_in_requirements)
        )

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_all_methods_consume_same_split_counts_and_keep_result_shape(self) -> None:
        X, y = self._regression_dataset()
        payload = SweepRunPayload(
            dataset=SweepDataset(X=X, y=y),
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
            linearization_sweep(payload),
            linearization_sweep_trimmed(payload),
            weighted_combiner_sweep(payload),
        ]

        for df in results:
            self.assertEqual(df["n_train"].tolist(), expected_counts)
            self.assertEqual(df.columns.tolist(), expected_columns)

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_registry_pipeline_matches_direct_method_outputs(self) -> None:
        X, y = self._regression_dataset()
        dataset = SweepDataset(X=X, y=y)
        payload = SweepRunPayload(
            dataset=dataset,
            split_collection=SweepSplitCollection(
                splits=tuple(
                    generate_sweep_splits(
                        n_samples=len(X),
                        min_train=2,
                        max_train=4,
                        n_repeats=2,
                        rng=np.random.default_rng(13),
                    )
                )
            ),
            use_trim=True,
        )
        sklearn_specs = {
            name: spec for name, _, spec in sklearn_sweep_model_specs()
        }

        expected = {
            "ridge_df": sweep_model(payload, sklearn_specs["ridge"].model_factory),
            "kernel_ridge_df": sweep_model(
                payload,
                sklearn_specs["kernel_ridge"].model_factory,
            ),
            "ridge_trimmed_df": sweep_model_trimmed(
                payload,
                sklearn_specs["ridge"].model_factory,
            ),
            "lasso_df": sweep_model(payload, sklearn_specs["lasso"].model_factory),
            "lasso_trimmed_df": sweep_model_trimmed(
                payload,
                sklearn_specs["lasso"].model_factory,
            ),
            "elastic_df": sweep_model(payload, sklearn_specs["elastic"].model_factory),
            "elastic_trimmed_df": sweep_model_trimmed(
                payload,
                sklearn_specs["elastic"].model_factory,
            ),
            "resid_df": residual_sweep(payload),
            "resid_trimmed_df": residual_sweep_trimmed(payload),
            "linear_df": linearization_sweep(payload),
            "linear_trimmed_df": linearization_sweep_trimmed(payload),
            "weighted_combiner_df": weighted_combiner_sweep(payload),
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
                "linearization",
                "weighted_combiner",
            ],
        )

        for field_name, expected_df in expected.items():
            actual_df = getattr(actual, field_name)
            self.assertIsNotNone(actual_df, field_name)
            pd.testing.assert_frame_equal(actual_df, expected_df)


class BoundaryTests(unittest.TestCase):
    def test_run_learning_curve_experiments_accepts_injected_model_families(
        self,
    ) -> None:
        X = np.array(
            [
                [1.0, 1.1],
                [2.0, 2.1],
                [3.0, 3.1],
                [4.0, 4.1],
                [5.0, 5.1],
                [6.0, 6.1],
            ]
        )
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result_df = pd.DataFrame(
            {
                "n_train": [2, 3, 4],
                "rmse_mean": [0.4, 0.3, 0.2],
                "rmse_std": [0.05, 0.04, 0.03],
            }
        )

        class StubFamily:
            def __init__(self, field_name: str) -> None:
                self.field_name = field_name
                self.calls = 0

            def requirements(self) -> SweepFamilyRequirements:
                return SweepFamilyRequirements()

            def run(self, payload):
                self.calls += 1
                self.last_payload = payload
                return LearningCurveResults.from_mapping(
                    {self.field_name: result_df}
                )

        ridge_family = StubFamily("ridge_df")
        linear_family = StubFamily("linear_df")

        results = run_learning_curve_experiments(
            SweepDataset(X=X, y=y),
            min_train=2,
            max_train=4,
            n_repeats=1,
            seed=9,
            use_trim=False,
            model_families=[ridge_family, linear_family],
        )

        self.assertIs(results.ridge_df, result_df)
        self.assertIs(results.linear_df, result_df)
        self.assertIsNone(results.kernel_ridge_df)
        self.assertIsNone(results.ridge_trimmed_df)
        self.assertIsNone(results.weighted_combiner_df)
        self.assertEqual(ridge_family.calls, 1)
        self.assertEqual(linear_family.calls, 1)
        self.assertIsInstance(ridge_family.last_payload, SweepRunPayload)
        self.assertFalse(ridge_family.last_payload.use_trim)
        self.assertEqual(
            [split.sweep_size for split in ridge_family.last_payload.split_collection.splits],
            [2, 3, 4],
        )

    def test_run_learning_curve_experiments_honors_family_split_requirements(
        self,
    ) -> None:
        X = np.array(
            [
                [1.0, 1.1],
                [2.0, 2.1],
                [3.0, 3.1],
                [4.0, 4.1],
                [5.0, 5.1],
                [6.0, 6.1],
                [7.0, 7.1],
            ]
        )
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        result_df = pd.DataFrame(
            {
                "n_train": [4, 5],
                "rmse_mean": [0.4, 0.3],
                "rmse_std": [0.05, 0.04],
            }
        )

        class ValidationAwareStubFamily:
            def requirements(self) -> SweepFamilyRequirements:
                return SweepFamilyRequirements(
                    min_train_size=4,
                    requires_inner_validation=True,
                )

            def run(self, payload):
                self.last_payload = payload
                return LearningCurveResults.from_mapping({"ridge_df": result_df})

        family = ValidationAwareStubFamily()

        results = run_learning_curve_experiments(
            SweepDataset(X=X, y=y),
            min_train=2,
            max_train=5,
            n_repeats=1,
            seed=3,
            use_trim=False,
            model_families=[family],
        )

        self.assertIs(results.ridge_df, result_df)
        self.assertEqual(
            [split.sweep_size for split in family.last_payload.split_collection.splits],
            [4, 5],
        )
        self.assertEqual(
            family.last_payload.split_collection.planning_requirements,
            SweepFamilyRequirements(
                min_train_size=4,
                requires_inner_validation=True,
            ),
        )

    def test_prepare_parity_plot_data_extracts_render_inputs(self) -> None:
        df = pd.DataFrame(
            {
                "reference_ads_eng": [1.0, 2.0],
                "ridge_mlip_ads_eng_median": [1.1, 2.1],
                "lasso_mlip_ads_eng_median": [0.9, 1.9],
            }
        )

        plot_data = prepare_parity_plot_data(df)

        np.testing.assert_array_equal(plot_data.reference, np.array([1.0, 2.0]))
        self.assertEqual(set(plot_data.predictions), {"ridge", "lasso"})
        np.testing.assert_array_equal(
            plot_data.predictions["ridge"],
            np.array([1.1, 2.1]),
        )

    def test_run_learning_curve_experiments_from_config_uses_defaults_without_cfg(
        self,
    ) -> None:
        df = pd.DataFrame(
            {
                "reference_ads_eng": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "ridge_mlip_ads_eng_median": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
                "lasso_mlip_ads_eng_median": [0.9, 1.9, 2.9, 3.9, 4.9, 5.9],
            }
        )

        if not HAS_SKLEARN:
            self.skipTest("requires scikit-learn")

        results = run_learning_curve_experiments_from_config(df, cfg=None)

        self.assertIsInstance(results, LearningCurveResults)
        self.assertIsNotNone(results.ridge_df)
        self.assertEqual(results.ridge_df["n_train"].tolist(), [5])

    def test_learning_curve_plot_renders_from_results_only(self) -> None:
        result_df = pd.DataFrame(
            {
                "n_train": [2, 3, 4],
                "rmse_mean": [0.4, 0.3, 0.2],
                "rmse_std": [0.05, 0.04, 0.03],
            }
        )
        results = LearningCurveResults(
            ridge_df=result_df,
            kernel_ridge_df=None,
            ridge_trimmed_df=None,
            lasso_df=None,
            lasso_trimmed_df=None,
            elastic_df=None,
            elastic_trimmed_df=None,
            resid_df=None,
            resid_trimmed_df=None,
            linear_df=None,
            linear_trimmed_df=None,
            weighted_combiner_df=result_df,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "learning_curve.png"
            saved_path = learning_curve_plot(results, output_path=output_path)

            self.assertEqual(saved_path, output_path)
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
