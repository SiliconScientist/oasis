from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

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
            ridge_trimmed_df=None,
            lasso_df=None,
            lasso_trimmed_df=None,
            elastic_df=None,
            elastic_trimmed_df=None,
            resid_df=None,
            resid_trimmed_df=None,
            weighted_linear_df=result_df,
            weighted_simplex_df=None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "learning_curve.png"
            saved_path = learning_curve_plot(results, output_path=output_path)

            self.assertEqual(saved_path, output_path)
            self.assertTrue(output_path.exists())
