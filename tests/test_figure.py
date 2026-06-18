from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from oasis.figure import learning_screening_figure, vertical_panel_figure
from oasis.plot import learning_curve_plot, screening_budget_plot
from oasis.sweep import LearningCurveResults


class FigureTests(unittest.TestCase):
    def test_vertical_panel_figure_requires_matching_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            panel_path = tmp_path / "panel.png"
            panel_path.write_bytes(b"not an image")

            with self.assertRaisesRegex(ValueError, "panel_labels must match"):
                vertical_panel_figure(
                    [panel_path],
                    output_path=tmp_path / "figure.png",
                    panel_labels=("a)", "b)"),
                )

    def test_vertical_panel_figure_requires_matching_label_positions(self) -> None:
        learning_results = LearningCurveResults(
            ridge_df=pd.DataFrame(
                {
                    "n_train": [2, 3, 4],
                    "rmse_mean": [0.4, 0.3, 0.2],
                    "rmse_std": [0.05, 0.04, 0.03],
                }
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            panel_path = learning_curve_plot(
                learning_results,
                output_path=tmp_path / "panel.png",
            )

            with self.assertRaisesRegex(ValueError, "panel_label_positions must match"):
                vertical_panel_figure(
                    [panel_path],
                    output_path=tmp_path / "figure.png",
                    panel_label_positions=((0.02, 0.98), (0.02, 1.01)),
                )

    def test_learning_screening_figure_stitches_two_plot_panels(self) -> None:
        learning_results = LearningCurveResults(
            ridge_df=pd.DataFrame(
                {
                    "n_train": [2, 3, 4],
                    "rmse_mean": [0.4, 0.3, 0.2],
                    "rmse_std": [0.05, 0.04, 0.03],
                }
            )
        )
        screening_results = LearningCurveResults(
            ridge_df=pd.DataFrame(
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
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            learning_path = learning_curve_plot(
                learning_results,
                output_path=tmp_path / "learning_curve.png",
            )
            screening_path = screening_budget_plot(
                screening_results,
                output_path=tmp_path / "screening_curve.png",
            )

            output_path = learning_screening_figure(
                learning_curve_path=learning_path,
                screening_curve_path=screening_path,
                output_path=tmp_path / "learning_screening_figure.png",
            )

            self.assertEqual(output_path, tmp_path / "learning_screening_figure.png")
            self.assertTrue(output_path.exists())
