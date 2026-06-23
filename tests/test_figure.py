from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from unittest.mock import patch

from oasis.figure import learning_screening_figure, uq_summary_figure, vertical_panel_figure
from oasis.plot import (
    dispersion_plot,
    learning_curve_plot,
    miscalibration_area_plot,
    screening_budget_plot,
    sharpness_plot,
)
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

    def test_uq_summary_figure_stitches_three_uq_metric_panels(self) -> None:
        uq_results = LearningCurveResults(
            ridge_uq_df=pd.DataFrame(
                {
                    "n_train": [2, 3, 4],
                    "miscalibration_area": [0.2, 0.15, 0.1],
                    "sharpness": [0.3, 0.25, 0.2],
                    "dispersion": [0.4, 0.35, 0.3],
                    "uncertainty_kind": ["spread_only", "spread_only", "spread_only"],
                }
            ),
            weighted_simplex_uq_df=pd.DataFrame(
                {
                    "n_train": [2, 3, 4],
                    "miscalibration_area": [0.18, 0.14, 0.09],
                    "sharpness": [0.28, 0.24, 0.19],
                    "dispersion": [0.38, 0.34, 0.29],
                    "uncertainty_kind": ["spread_only", "spread_only", "spread_only"],
                }
            ),
            moe_uq_df=pd.DataFrame(
                {
                    "n_train": [2, 3, 4],
                    "miscalibration_area": [0.17, 0.12, 0.08],
                    "sharpness": [0.27, 0.22, 0.18],
                    "dispersion": [0.37, 0.32, 0.28],
                    "uncertainty_kind": ["spread_only", "spread_only", "spread_only"],
                }
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            miscalibration_path = miscalibration_area_plot(
                uq_results,
                output_path=tmp_path / "miscalibration.png",
            )
            sharpness_path = sharpness_plot(
                uq_results,
                output_path=tmp_path / "sharpness.png",
            )
            dispersion_path = dispersion_plot(
                uq_results,
                output_path=tmp_path / "dispersion.png",
            )

            output_path = uq_summary_figure(
                miscalibration_area_path=miscalibration_path,
                sharpness_path=sharpness_path,
                dispersion_path=dispersion_path,
                output_path=tmp_path / "uq_summary_figure.png",
            )

            self.assertEqual(output_path, tmp_path / "uq_summary_figure.png")
            self.assertTrue(output_path.exists())

    def test_miscalibration_area_plot_uses_budget_axis_for_screening_uq_results(self) -> None:
        screening_uq_results = LearningCurveResults(
            ridge_uq_df=pd.DataFrame(
                {
                    "n_budget": [4, 8],
                    "n_train": [3, 6],
                    "n_screen": [1, 2],
                    "screen_fraction": [0.25, 0.25],
                    "n_cv_folds": [4, 4],
                    "miscalibration_area": [0.2, 0.1],
                    "miscalibration_area_std": [0.01, 0.02],
                    "uncertainty_kind": ["spread_only", "spread_only"],
                }
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            with patch("matplotlib.pyplot.close") as mock_close:
                output_path = miscalibration_area_plot(
                    screening_uq_results,
                    output_path=tmp_path / "screening_miscalibration.png",
                    include_x=[4, 8],
                )

            self.assertTrue(mock_close.called)
            self.assertEqual(output_path, tmp_path / "screening_miscalibration.png")
            ax = plt.gcf().axes[0]
            self.assertEqual(ax.get_xlabel(), "Sample budget")
            self.assertEqual(ax.get_title(), "Miscalibration area vs budget")
            self.assertEqual(list(ax.lines[0].get_xdata()), [4, 8])
            plt.close("all")
