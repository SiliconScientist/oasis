from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oasis.exp import (
    LearningCurveResults,
    SweepDataset,
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
        linearization_sweep,
        linearization_sweep_trimmed,
        residual_sweep,
        residual_sweep_trimmed,
        sweep_model,
        sweep_model_trimmed,
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
    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_all_methods_consume_same_split_counts_and_keep_result_shape(self) -> None:
        X = np.array(
            [
                [1.0, 1.1, 0.9],
                [2.0, 2.1, 1.9],
                [3.0, 3.1, 2.9],
                [4.0, 4.1, 3.9],
                [5.0, 5.1, 4.9],
                [6.0, 6.1, 5.9],
            ]
        )
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
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
        ]

        for df in results:
            self.assertEqual(df["n_train"].tolist(), expected_counts)
            self.assertEqual(df.columns.tolist(), expected_columns)


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
        self.assertEqual(ridge_family.calls, 1)
        self.assertEqual(linear_family.calls, 1)
        self.assertIsInstance(ridge_family.last_payload, SweepRunPayload)
        self.assertFalse(ridge_family.last_payload.use_trim)
        self.assertEqual(
            [split.sweep_size for split in ridge_family.last_payload.split_collection.splits],
            [2, 3, 4],
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
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "learning_curve.png"
            saved_path = learning_curve_plot(results, output_path=output_path)

            self.assertEqual(saved_path, output_path)
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
