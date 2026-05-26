from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from oasis.learning_curve.results_io import (
    dump_learning_curve_results,
    load_learning_curve_results,
    load_learning_curve_results_mapping,
    save_learning_curve_results,
)
from oasis.sweep import LearningCurveResults


class LearningCurveResultsIoTests(unittest.TestCase):
    def test_round_trip_serialization_preserves_mixed_result_frames(self) -> None:
        results = LearningCurveResults(
            ridge_df=pd.DataFrame(
                {
                    "n_train": [4, 8],
                    "rmse_mean": [0.41, 0.32],
                    "rmse_std": [0.06, 0.03],
                }
            ),
            moe_df=pd.DataFrame(
                {
                    "n_train": [8],
                    "rmse_mean": [0.28],
                    "rmse_std": [0.02],
                }
            ),
            ridge_selection_df=pd.DataFrame(
                {
                    "n_train": [4, 8],
                    "alpha": [0.1, 1.0],
                    "best_validation_score": [0.52, 0.44],
                }
            ),
            probe_gnn_selection_df=pd.DataFrame(
                {
                    "n_train": [8],
                    "hidden_dim": [32],
                    "aggregator": ["mean"],
                }
            ),
        )

        restored = load_learning_curve_results_mapping(
            dump_learning_curve_results(results)
        )

        self.assertIsNone(restored.lasso_df)
        pd.testing.assert_frame_equal(restored.ridge_df, results.ridge_df)
        pd.testing.assert_frame_equal(restored.moe_df, results.moe_df)
        pd.testing.assert_frame_equal(
            restored.ridge_selection_df,
            results.ridge_selection_df,
        )
        pd.testing.assert_frame_equal(
            restored.probe_gnn_selection_df,
            results.probe_gnn_selection_df,
        )

    def test_save_and_load_learning_curve_results_bundle(self) -> None:
        results = LearningCurveResults(
            weighted_simplex_df=pd.DataFrame(
                {
                    "n_train": [3, 5],
                    "rmse_mean": [0.7, 0.55],
                    "rmse_std": [0.1, 0.08],
                }
            )
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "learning_curve_results.json"
            saved_path = save_learning_curve_results(results, path)
            restored = load_learning_curve_results(saved_path)

        self.assertEqual(saved_path, path)
        pd.testing.assert_frame_equal(
            restored.weighted_simplex_df,
            results.weighted_simplex_df,
        )
