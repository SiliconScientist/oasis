from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from oasis.learning_curve.results_io import (
    LearningCurveSweepMetadata,
    load_learning_curve_results_artifact,
    load_learning_curve_results_from_method_artifacts,
    save_learning_curve_results_artifact,
    save_learning_curve_method_artifacts,
)
from oasis.plot import learning_curve_plot, screening_budget_plot
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

    def test_learning_curve_plot_renders_sparse_incremental_bundle(self) -> None:
        results = LearningCurveResults(
            ridge_df=pd.DataFrame(
                {
                    "n_train": [4, 1, 3],
                    "rmse_mean": [0.2, 0.5, 0.3],
                    "rmse_std": [0.03, 0.06, 0.04],
                }
            ),
            weighted_simplex_df=pd.DataFrame(
                {
                    "n_train": [4, 2],
                    "rmse_mean": [0.18, 0.28],
                    "rmse_std": [0.025, 0.035],
                }
            ),
        )
        metadata = LearningCurveSweepMetadata(
            seed=23,
            min_train=1,
            max_train=4,
            step=1,
            n_repeats=2,
            enabled_models=("ridge", "weighted_simplex"),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "learning_curve_results.json"
            output_path = Path(tmpdir) / "learning_curve.png"
            save_learning_curve_results_artifact(results, metadata, bundle_path)

            reloaded = load_learning_curve_results_artifact(
                bundle_path,
                expected_metadata=metadata,
            ).results
            saved_path = learning_curve_plot(reloaded, output_path=output_path)

            self.assertEqual(saved_path, output_path)
            self.assertTrue(output_path.exists())

    def test_screening_budget_plot_renders_from_results_only(self) -> None:
        result_df = pd.DataFrame(
            {
                "n_budget": [4, 6, 8],
                "n_train": [3, 4, 6],
                "n_screen": [1, 2, 2],
                "screen_fraction": [0.25, 1 / 3, 0.25],
                "n_cv_folds": [4, 3, 4],
                "cv_rmse_mean": [0.4, 0.3, 0.2],
                "cv_rmse_std": [0.05, 0.04, 0.03],
            }
        )
        results = LearningCurveResults(
            ridge_df=result_df,
            weighted_linear_df=result_df,
            graph_mean_df=result_df,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "screening_curve.png"
            saved_path = screening_budget_plot(results, output_path=output_path)

            self.assertEqual(saved_path, output_path)
            self.assertTrue(output_path.exists())

    def test_screening_budget_plot_matches_reloaded_artifacts(self) -> None:
        result_df = pd.DataFrame(
            {
                "n_budget": [4, 6, 8],
                "n_train": [3, 4, 6],
                "n_screen": [1, 2, 2],
                "screen_fraction": [0.25, 1 / 3, 0.25],
                "n_cv_folds": [4, 3, 4],
                "cv_rmse_mean": [0.4, 0.3, 0.2],
                "cv_rmse_std": [0.05, 0.04, 0.03],
            }
        )
        selection_df = pd.DataFrame(
            {
                "n_budget": [4, 6, 8],
                "n_train": [3, 4, 6],
                "n_screen": [1, 2, 2],
                "screen_fraction": [0.25, 1 / 3, 0.25],
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
            min_train=4,
            max_train=8,
            step=2,
            n_repeats=2,
            enabled_models=("ridge", "weighted_linear"),
            budget_mode="screening_fraction",
            screen_fraction=0.25,
            min_screen_size=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "artifacts"
            save_learning_curve_method_artifacts(results, metadata, artifact_dir)
            reloaded = load_learning_curve_results_from_method_artifacts(
                artifact_dir,
                expected_metadata=metadata,
            )
            original_path = Path(tmpdir) / "original_screening.png"
            reloaded_path = Path(tmpdir) / "reloaded_screening.png"

            screening_budget_plot(results, output_path=original_path)
            screening_budget_plot(reloaded, output_path=reloaded_path)

            self.assertTrue(original_path.exists())
            self.assertTrue(reloaded_path.exists())

    def test_screening_budget_plot_renders_sparse_incremental_bundle(self) -> None:
        results = LearningCurveResults(
            ridge_df=pd.DataFrame(
                {
                    "n_budget": [8, 4, 6],
                    "n_train": [6, 3, 4],
                    "n_screen": [2, 1, 2],
                    "screen_fraction": [0.25, 0.25, 1 / 3],
                    "n_cv_folds": [4, 4, 3],
                    "cv_rmse_mean": [0.2, 0.5, 0.3],
                    "cv_rmse_std": [0.03, 0.06, 0.04],
                }
            ),
            weighted_simplex_df=pd.DataFrame(
                {
                    "n_budget": [8, 6],
                    "n_train": [6, 4],
                    "n_screen": [2, 2],
                    "screen_fraction": [0.25, 1 / 3],
                    "n_cv_folds": [4, 3],
                    "cv_rmse_mean": [0.18, 0.28],
                    "cv_rmse_std": [0.025, 0.035],
                }
            ),
        )
        metadata = LearningCurveSweepMetadata(
            seed=23,
            min_train=4,
            max_train=8,
            step=2,
            n_repeats=2,
            enabled_models=("ridge", "weighted_simplex"),
            budget_mode="screening_fraction",
            screen_fraction=0.25,
            min_screen_size=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "screening_results.json"
            output_path = Path(tmpdir) / "screening_curve.png"
            save_learning_curve_results_artifact(results, metadata, bundle_path)

            reloaded = load_learning_curve_results_artifact(
                bundle_path,
                expected_metadata=metadata,
            ).results
            saved_path = screening_budget_plot(reloaded, output_path=output_path)

            self.assertEqual(saved_path, output_path)
            self.assertTrue(output_path.exists())
