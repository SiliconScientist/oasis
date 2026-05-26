from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from oasis.learning_curve.results_io import (
    LearningCurveSweepMetadata,
    load_learning_curve_results_from_method_artifacts,
    save_learning_curve_method_artifacts,
)
from oasis.plot import learning_curve_plot
from oasis.sweep import LearningCurveResults


class PlotTests(unittest.TestCase):
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
            lasso_df=None,
            elastic_df=None,
            resid_df=None,
            weighted_linear_df=result_df,
            weighted_simplex_df=None,
            graph_mean_df=result_df,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "learning_curve.png"
            saved_path = learning_curve_plot(results, output_path=output_path)

            self.assertEqual(saved_path, output_path)
            self.assertTrue(output_path.exists())

    def test_learning_curve_plot_matches_reloaded_artifacts(self) -> None:
        result_df = pd.DataFrame(
            {
                "n_train": [2, 3, 4],
                "rmse_mean": [0.4, 0.3, 0.2],
                "rmse_std": [0.05, 0.04, 0.03],
            }
        )
        selection_df = pd.DataFrame(
            {
                "n_train": [2, 3, 4],
                "alpha": [0.1, 1.0, 10.0],
            }
        )
        results = LearningCurveResults(
            ridge_df=result_df,
            ridge_selection_df=selection_df,
            weighted_linear_df=result_df,
        )
        metadata = LearningCurveSweepMetadata(
            seed=23,
            min_train=2,
            max_train=4,
            step=1,
            n_repeats=2,
            enabled_models=("ridge", "weighted_linear"),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "artifacts"
            save_learning_curve_method_artifacts(results, metadata, artifact_dir)
            reloaded = load_learning_curve_results_from_method_artifacts(
                artifact_dir,
                expected_metadata=metadata,
            )
            pd.testing.assert_frame_equal(reloaded.ridge_df, results.ridge_df)
            pd.testing.assert_frame_equal(
                reloaded.ridge_selection_df,
                results.ridge_selection_df,
            )
            pd.testing.assert_frame_equal(
                reloaded.weighted_linear_df,
                results.weighted_linear_df,
            )
            original_path = Path(tmpdir) / "original.png"
            reloaded_path = Path(tmpdir) / "reloaded.png"

            learning_curve_plot(results, output_path=original_path)
            learning_curve_plot(reloaded, output_path=reloaded_path)

            self.assertTrue(original_path.exists())
            self.assertTrue(reloaded_path.exists())
