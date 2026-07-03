from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from oasis.learning_curve.results_io import (
    LearningCurveSweepMetadata,
    load_learning_curve_results_artifact,
    load_learning_curve_results_from_method_artifacts,
    save_learning_curve_results_artifact,
    save_learning_curve_method_artifacts,
)
from oasis.mlip.timing import MlipGenerationTimingSummary
from oasis.plot import (
    dispersion_plot,
    fixed_split_total_time_accuracy_plot,
    fixed_split_training_time_accuracy_plot,
    generation_time_accuracy_plot,
    learning_curve_plot,
    miscalibration_area_plot,
    oracle_learning_curve_frame,
    oracle_learning_curve_plot,
    parity_plot,
    policy_regret_plot,
    policy_selected_vs_oracle_plot,
    screening_budget_plot,
    sharpness_plot,
    total_time_accuracy_plot,
    training_time_accuracy_plot,
    zero_shot_rmse_stage_plot,
)
from oasis.sweep import LearningCurveResults


class PlotTests(unittest.TestCase):
    @staticmethod
    def _generation_summaries() -> dict[str, MlipGenerationTimingSummary]:
        return {
            "mace": MlipGenerationTimingSummary(
                model_name="mace",
                reaction_count=2,
                generation_time_total_s=10.0,
                generation_time_slab_s=2.0,
                generation_time_adslab_s=8.0,
                generation_steps_total=20,
                generation_steps_slab=4,
                generation_steps_adslab=16,
                time_per_step_s=0.5,
            ),
            "orb": MlipGenerationTimingSummary(
                model_name="orb",
                reaction_count=2,
                generation_time_total_s=5.0,
                generation_time_slab_s=1.0,
                generation_time_adslab_s=4.0,
                generation_steps_total=10,
                generation_steps_slab=2,
                generation_steps_adslab=8,
                time_per_step_s=0.5,
            ),
        }

    def test_policy_selected_vs_oracle_plot_renders_from_summary_frame(self) -> None:
        summary_df = pd.DataFrame(
            {
                "policy_name": ["min_screening_rmse", "min_screening_rmse"],
                "budget": [4, 8],
                "mean_regret": [0.05, 0.01],
                "std_regret": [0.02, 0.01],
                "se_regret": [0.014, 0.007],
                "ci95_low": [0.022, -0.004],
                "ci95_high": [0.078, 0.024],
                "agreement_rate": [0.5, 1.0],
                "oracle_outer_rmse_mean": [0.2, 0.18],
                "screening_selected_outer_rmse_mean": [0.25, 0.19],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "policy_selected_vs_oracle.png"
            saved_path = policy_selected_vs_oracle_plot(
                summary_df,
                output_path=output_path,
            )
            exists = saved_path.exists()

        self.assertEqual(saved_path, output_path)
        self.assertTrue(exists)

    def test_policy_regret_plot_renders_and_filters_budget_window(self) -> None:
        summary_df = pd.DataFrame(
            {
                "policy_name": [
                    "min_screening_rmse",
                    "min_screening_rmse",
                    "min_screening_rmse",
                ],
                "budget": [4, 8, 12],
                "mean_regret": [0.05, 0.01, 0.0],
                "std_regret": [0.02, 0.01, 0.0],
                "se_regret": [0.014, 0.007, 0.0],
                "ci95_low": [0.022, -0.004, 0.0],
                "ci95_high": [0.078, 0.024, 0.0],
                "agreement_rate": [0.5, 1.0, 1.0],
                "oracle_outer_rmse_mean": [0.2, 0.18, 0.16],
                "screening_selected_outer_rmse_mean": [0.25, 0.19, 0.16],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "policy_regret.png"
            saved_path = policy_regret_plot(
                summary_df,
                output_path=output_path,
                min_x=8,
                max_x=12,
                include_x=[8, 12],
            )
            exists = saved_path.exists()

        self.assertEqual(saved_path, output_path)
        self.assertTrue(exists)

    def test_policy_plots_render_multiple_policy_lines(self) -> None:
        summary_df = pd.DataFrame(
            {
                "policy_name": [
                    "min_screening_rmse",
                    "min_screening_rmse",
                    "combined_screening_rmse_miscalibration",
                    "combined_screening_rmse_miscalibration",
                ],
                "budget": [4, 8, 4, 8],
                "mean_regret": [0.05, 0.01, 0.02, 0.0],
                "std_regret": [0.02, 0.01, 0.01, 0.0],
                "se_regret": [0.014, 0.007, 0.007, 0.0],
                "ci95_low": [0.022, -0.004, 0.006, 0.0],
                "ci95_high": [0.078, 0.024, 0.034, 0.0],
                "agreement_rate": [0.5, 1.0, 0.5, 1.0],
                "oracle_outer_rmse_mean": [0.2, 0.18, 0.2, 0.18],
                "screening_selected_outer_rmse_mean": [0.25, 0.19, 0.22, 0.18],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            selected_path = policy_selected_vs_oracle_plot(
                summary_df,
                output_path=tmp_path / "policy_selected_vs_oracle_multi.png",
            )
            regret_path = policy_regret_plot(
                summary_df,
                output_path=tmp_path / "policy_regret_multi.png",
            )

            self.assertTrue(selected_path.exists())
            self.assertTrue(regret_path.exists())

    def test_uq_metric_plots_render_from_results_only(self) -> None:
        uq_df = pd.DataFrame(
            {
                "n_train": [2, 3, 4],
                "miscalibration_area": [0.2, 0.15, 0.1],
                "miscalibration_area_std": [0.03, 0.02, 0.01],
                "sharpness": [0.3, 0.25, 0.2],
                "sharpness_std": [0.04, 0.03, 0.02],
                "dispersion": [0.4, 0.35, 0.3],
                "dispersion_std": [0.05, 0.04, 0.03],
                "uncertainty_kind": ["spread_only", "spread_only", "spread_only"],
            }
        )
        results = LearningCurveResults(
            ridge_uq_df=uq_df,
            weighted_simplex_uq_df=uq_df,
            moe_uq_df=uq_df,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            miscalibration_path = miscalibration_area_plot(
                results,
                output_path=tmp_path / "miscalibration.png",
            )
            sharpness_path = sharpness_plot(
                results,
                output_path=tmp_path / "sharpness.png",
            )
            dispersion_path = dispersion_plot(
                results,
                output_path=tmp_path / "dispersion.png",
            )

            self.assertEqual(miscalibration_path, tmp_path / "miscalibration.png")
            self.assertEqual(sharpness_path, tmp_path / "sharpness.png")
            self.assertEqual(dispersion_path, tmp_path / "dispersion.png")
            self.assertTrue(miscalibration_path.exists())
            self.assertTrue(sharpness_path.exists())
            self.assertTrue(dispersion_path.exists())

    def test_uq_metric_plots_filter_to_requested_x_window(self) -> None:
        uq_df = pd.DataFrame(
            {
                "n_train": [2, 3, 4, 5],
                "miscalibration_area": [0.2, 0.15, 0.1, 0.08],
                "miscalibration_area_std": [0.03, 0.02, 0.01, 0.01],
                "sharpness": [0.3, 0.25, 0.2, 0.18],
                "sharpness_std": [0.04, 0.03, 0.02, 0.02],
                "dispersion": [0.4, 0.35, 0.3, 0.28],
                "dispersion_std": [0.05, 0.04, 0.03, 0.02],
                "uncertainty_kind": ["spread_only"] * 4,
            }
        )
        results = LearningCurveResults(ridge_uq_df=uq_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "miscalibration_windowed.png"
            saved_path = miscalibration_area_plot(
                results,
                output_path=output_path,
                min_x=3,
                max_x=4,
            )

            self.assertEqual(saved_path, output_path)
            self.assertTrue(output_path.exists())

    def test_uq_metric_plots_use_integer_x_ticks(self) -> None:
        uq_df = pd.DataFrame(
            {
                "n_train": [2, 3, 4, 5],
                "miscalibration_area": [0.2, 0.15, 0.1, 0.08],
                "miscalibration_area_std": [0.03, 0.02, 0.01, 0.01],
                "sharpness": [0.3, 0.25, 0.2, 0.18],
                "sharpness_std": [0.04, 0.03, 0.02, 0.02],
                "dispersion": [0.4, 0.35, 0.3, 0.28],
                "dispersion_std": [0.05, 0.04, 0.03, 0.02],
                "uncertainty_kind": ["spread_only"] * 4,
            }
        )
        results = LearningCurveResults(ridge_uq_df=uq_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sharpness_ticks.png"
            with patch("oasis.plot.plt.close"):
                sharpness_plot(results, output_path=output_path)
                fig = sharpness_plot.__globals__["plt"].gcf()
                locator = fig.axes[0].xaxis.get_major_locator()

            self.assertTrue(getattr(locator, "_integer", False))

    def test_uq_metric_plot_can_render_zero_shot_baseline_line(self) -> None:
        uq_df = pd.DataFrame(
            {
                "n_train": [2, 3, 4],
                "miscalibration_area": [0.2, 0.15, 0.1],
                "miscalibration_area_std": [0.03, 0.02, 0.01],
                "sharpness": [0.3, 0.25, 0.2],
                "sharpness_std": [0.04, 0.03, 0.02],
                "dispersion": [0.4, 0.35, 0.3],
                "dispersion_std": [0.05, 0.04, 0.03],
                "uncertainty_kind": ["spread_only"] * 3,
            }
        )
        results = LearningCurveResults(ridge_uq_df=uq_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "miscalibration_zero_shot.png"
            with patch("oasis.plot.plt.close"):
                miscalibration_area_plot(
                    results,
                    output_path=output_path,
                    zero_shot_value=0.12,
                )
                fig = miscalibration_area_plot.__globals__["plt"].gcf()
                labels = [collection.get_label() for collection in fig.axes[0].collections]

            self.assertIn("Zero-shot mean-MLIP baseline", labels)

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

    def test_oracle_learning_curve_frame_selects_best_enabled_method_per_train_size(
        self,
    ) -> None:
        ridge_df = pd.DataFrame(
            {
                "n_train": [2, 4, 8],
                "rmse_mean": [0.35, 0.30, 0.26],
                "rmse_std": [0.02, 0.02, 0.01],
            }
        )
        probe_gnn_df = pd.DataFrame(
            {
                "n_train": [2, 4, 8],
                "rmse_mean": [0.33, 0.31, 0.22],
                "rmse_std": [0.03, 0.02, 0.02],
            }
        )
        results = LearningCurveResults(
            ridge_df=ridge_df,
            probe_gnn_df=probe_gnn_df,
        )

        oracle_df = oracle_learning_curve_frame(
            results,
            enabled_method_names=["ridge", "probe_gnn"],
            dataset="bio_mass",
            dataset_label="Bio-Mass",
        )

        expected = pd.DataFrame(
            {
                "dataset": ["bio_mass", "bio_mass", "bio_mass"],
                "dataset_label": ["Bio-Mass", "Bio-Mass", "Bio-Mass"],
                "n_train": [2, 4, 8],
                "oracle_rmse": [0.33, 0.30, 0.22],
                "oracle_method": ["probe_gnn", "ridge", "probe_gnn"],
            }
        )
        pd.testing.assert_frame_equal(oracle_df, expected)

    def test_oracle_learning_curve_frame_ignores_disabled_methods(self) -> None:
        ridge_df = pd.DataFrame(
            {
                "n_train": [2, 4],
                "rmse_mean": [0.35, 0.30],
                "rmse_std": [0.02, 0.02],
            }
        )
        latent_df = pd.DataFrame(
            {
                "n_train": [2, 4],
                "rmse_mean": [0.10, 0.09],
                "rmse_std": [0.01, 0.01],
            }
        )
        results = LearningCurveResults(
            ridge_df=ridge_df,
            latent_df=latent_df,
        )

        oracle_df = oracle_learning_curve_frame(
            results,
            enabled_method_names=["ridge"],
            dataset="bio_mass",
        )

        expected = pd.DataFrame(
            {
                "dataset": ["bio_mass", "bio_mass"],
                "dataset_label": ["bio_mass", "bio_mass"],
                "n_train": [2, 4],
                "oracle_rmse": [0.35, 0.30],
                "oracle_method": ["ridge", "ridge"],
            }
        )
        pd.testing.assert_frame_equal(oracle_df, expected)

    def test_oracle_learning_curve_plot_renders_single_dataset_curve(self) -> None:
        oracle_df = pd.DataFrame(
            {
                "dataset": ["bio_mass", "bio_mass", "bio_mass"],
                "dataset_label": ["Bio-Mass", "Bio-Mass", "Bio-Mass"],
                "n_train": [2, 4, 8],
                "oracle_rmse": [0.33, 0.30, 0.22],
                "oracle_method": ["probe_gnn", "ridge", "probe_gnn"],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "oracle_learning_curve.png"
            with patch("oasis.plot.plt.close"):
                saved_path = oracle_learning_curve_plot(
                    oracle_df,
                    output_path=output_path,
                )
                fig = oracle_learning_curve_plot.__globals__["plt"].gcf()
                ax = fig.axes[0]

            self.assertEqual(saved_path, output_path)
            self.assertTrue(output_path.exists())
            self.assertEqual(ax.get_ylabel(), "Oracle RMSE (eV)")
            self.assertEqual(ax.get_xlabel(), "Train size")
            self.assertEqual(ax.get_title(), "Oracle learning curve by dataset")
            self.assertEqual(len(ax.lines), 1)
            self.assertEqual(ax.get_legend().get_texts()[0].get_text(), "Bio-Mass")

    def test_oracle_learning_curve_plot_renders_multiple_dataset_curves(self) -> None:
        oracle_df = pd.DataFrame(
            {
                "dataset": [
                    "bio_mass",
                    "bio_mass",
                    "khlohc",
                    "khlohc",
                ],
                "dataset_label": [
                    "Bio-Mass",
                    "Bio-Mass",
                    "KHLOHC-TOL",
                    "KHLOHC-TOL",
                ],
                "n_train": [2, 4, 2, 4],
                "oracle_rmse": [0.33, 0.30, 0.41, 0.29],
                "oracle_method": ["probe_gnn", "ridge", "ridge", "probe_gnn"],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "oracle_learning_curve_multi.png"
            with patch("oasis.plot.plt.close"):
                saved_path = oracle_learning_curve_plot(
                    oracle_df,
                    output_path=output_path,
                )
                fig = oracle_learning_curve_plot.__globals__["plt"].gcf()
                ax = fig.axes[0]
                locator = ax.xaxis.get_major_locator()

            self.assertEqual(saved_path, output_path)
            self.assertTrue(output_path.exists())
            self.assertEqual(len(ax.lines), 2)
            self.assertEqual(
                [text.get_text() for text in ax.get_legend().get_texts()],
                ["Bio-Mass", "KHLOHC-TOL"],
            )
            self.assertTrue(getattr(locator, "_integer", False))

    def test_parity_plot_can_filter_invalid_points_per_mlip(self) -> None:
        df = pd.DataFrame(
            {
                "reference_ads_eng": [1.0, 2.0, 3.0],
                "ridge_mlip_ads_eng_median": [1.1, 2.1, 3.1],
                "lasso_mlip_ads_eng_median": [0.9, 1.9, 2.9],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "parity_filtered.png"
            with patch("oasis.plot.plt.close"):
                saved_path = parity_plot(
                    df,
                    output_path=output_path,
                    title="Parity plot (filtered)",
                    validity_mask_by_prediction={
                        "ridge": pd.Series([True, False, True]).to_numpy(),
                        "lasso": pd.Series([False, True, True]).to_numpy(),
                    },
                )
                fig = parity_plot.__globals__["plt"].gcf()
                ax = fig.axes[0]

            self.assertEqual(saved_path, output_path)
            self.assertTrue(output_path.exists())
            self.assertEqual(ax.get_title(), "Parity plot (filtered)")
            point_counts = [len(collection.get_offsets()) for collection in ax.collections]
            self.assertEqual(point_counts, [2, 2])

    def test_zero_shot_rmse_stage_plot_renders_three_stage_bars(self) -> None:
        stage_df = pd.DataFrame(
            {
                "dataset": ["mamun_oh", "mamun_oh", "mamun_oh"],
                "dataset_label": ["OH-BMA", "OH-BMA", "OH-BMA"],
                "stage": [
                    "Full / all MLIPs",
                    "Matched subset / all MLIPs",
                    "Matched subset / anomaly-aware selection",
                ],
                "rmse": [0.55, 0.51, 0.47],
                "n_samples": [1235, 1094, 1094],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "zero_shot_stage.png"
            with patch("oasis.plot.plt.close"):
                saved_path = zero_shot_rmse_stage_plot(
                    stage_df,
                    output_path=output_path,
                )
                fig = zero_shot_rmse_stage_plot.__globals__["plt"].gcf()
                ax = fig.axes[0]

            self.assertEqual(saved_path, output_path)
            self.assertTrue(output_path.exists())
            self.assertEqual(ax.get_ylabel(), "Zero-shot RMSE (eV)")
            self.assertEqual(len(ax.patches), 3)
            self.assertEqual([tick.get_text() for tick in ax.get_xticklabels()], ["OH-BMA"])

    def test_zero_shot_rmse_stage_plot_renders_multiple_datasets(self) -> None:
        stage_df = pd.DataFrame(
            {
                "dataset": [
                    "mamun_oh",
                    "mamun_oh",
                    "mamun_oh",
                    "khlohc",
                    "khlohc",
                    "khlohc",
                ],
                "dataset_label": [
                    "OH-BMA",
                    "OH-BMA",
                    "OH-BMA",
                    "KHLOHC-TOL",
                    "KHLOHC-TOL",
                    "KHLOHC-TOL",
                ],
                "stage": [
                    "Full / all MLIPs",
                    "Matched subset / all MLIPs",
                    "Matched subset / anomaly-aware selection",
                    "Full / all MLIPs",
                    "Matched subset / all MLIPs",
                    "Matched subset / anomaly-aware selection",
                ],
                "rmse": [0.55, 0.51, 0.47, 0.62, 0.58, 0.5],
                "n_samples": [1235, 1094, 1094, 800, 740, 740],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "zero_shot_stage_multi.png"
            with patch("oasis.plot.plt.close"):
                zero_shot_rmse_stage_plot(stage_df, output_path=output_path)
                fig = zero_shot_rmse_stage_plot.__globals__["plt"].gcf()
                ax = fig.axes[0]

            self.assertTrue(output_path.exists())
            self.assertEqual(len(ax.patches), 6)
            self.assertEqual(
                [tick.get_text() for tick in ax.get_xticklabels()],
                ["OH-BMA", "KHLOHC-TOL"],
            )

    def test_learning_curve_plot_filters_to_requested_x_window(self) -> None:
        result_df = pd.DataFrame(
            {
                "n_train": [2, 3, 4, 5],
                "rmse_mean": [0.4, 0.3, 0.2, 0.1],
                "rmse_std": [0.05, 0.04, 0.03, 0.02],
            }
        )
        results = LearningCurveResults(ridge_df=result_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "learning_curve_windowed.png"
            saved_path = learning_curve_plot(
                results,
                output_path=output_path,
                min_x=3,
                max_x=4,
            )

            self.assertEqual(saved_path, output_path)
            self.assertTrue(output_path.exists())

    def test_learning_curve_plot_uses_integer_x_ticks(self) -> None:
        result_df = pd.DataFrame(
            {
                "n_train": [2, 3, 4, 5],
                "rmse_mean": [0.4, 0.3, 0.2, 0.1],
                "rmse_std": [0.05, 0.04, 0.03, 0.02],
            }
        )
        results = LearningCurveResults(ridge_df=result_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "learning_curve_ticks.png"
            with patch("oasis.plot.plt.close"):
                learning_curve_plot(results, output_path=output_path)
                fig = learning_curve_plot.__globals__["plt"].gcf()
                locator = fig.axes[0].xaxis.get_major_locator()

            self.assertTrue(getattr(locator, "_integer", False))

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

    def test_generation_time_accuracy_plot_renders_one_point_per_method(self) -> None:
        results = LearningCurveResults(
            ridge_df=pd.DataFrame(
                {
                    "n_train": [2, 3, 4],
                    "rmse_mean": [0.4, 0.3, 0.2],
                    "rmse_std": [0.05, 0.04, 0.03],
                    "fit_time_mean_s": [0.12, 0.18, 0.24],
                    "fit_time_std_s": [0.01, 0.01, 0.02],
                }
            ),
            weighted_linear_df=pd.DataFrame(
                {
                    "n_train": [2, 3, 4],
                    "rmse_mean": [0.35, 0.28, 0.22],
                    "rmse_std": [0.04, 0.03, 0.02],
                    "fit_time_mean_s": [0.08, 0.12, 0.16],
                    "fit_time_std_s": [0.01, 0.01, 0.01],
                }
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "generation_time_accuracy.png"
            with patch("oasis.plot.plt.close"):
                saved_path = generation_time_accuracy_plot(
                    results,
                    self._generation_summaries(),
                    output_path=output_path,
                    mlip_feature_names=("mace", "orb"),
                )
                fig = generation_time_accuracy_plot.__globals__["plt"].gcf()
                labels = [
                    collection.get_label()
                    for collection in fig.axes[0].collections
                ]

            self.assertEqual(saved_path, output_path)
            self.assertTrue(output_path.exists())
            self.assertEqual(labels, ["Ridge", "Weighted linear"])

    def test_training_time_accuracy_plot_uses_expected_axis_labels(self) -> None:
        results = LearningCurveResults(
            ridge_df=pd.DataFrame(
                {
                    "n_train": [2, 3],
                    "rmse_mean": [0.4, 0.3],
                    "rmse_std": [0.05, 0.04],
                    "fit_time_mean_s": [0.12, 0.18],
                    "fit_time_std_s": [0.01, 0.02],
                }
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "training_time_accuracy.png"
            with patch("oasis.plot.plt.close"):
                training_time_accuracy_plot(
                    results,
                    self._generation_summaries(),
                    output_path=output_path,
                    mlip_feature_names=("mace", "orb"),
                )
                fig = training_time_accuracy_plot.__globals__["plt"].gcf()
                ax = fig.axes[0]

            self.assertEqual(ax.get_xlabel(), "Training time (s)")
            self.assertEqual(ax.get_ylabel(), "RMSE (eV)")

    def test_total_time_accuracy_plot_skips_excluded_methods(self) -> None:
        results = LearningCurveResults(
            probe_gnn_df=pd.DataFrame(
                {
                    "n_train": [2],
                    "rmse_mean": [0.2],
                    "rmse_std": [0.02],
                    "fit_time_mean_s": [9.0],
                    "fit_time_std_s": [0.1],
                }
            ),
            latent_df=pd.DataFrame(
                {
                    "n_train": [2],
                    "rmse_mean": [0.25],
                    "rmse_std": [0.03],
                    "fit_time_mean_s": [8.0],
                    "fit_time_std_s": [0.2],
                }
            ),
            ridge_df=pd.DataFrame(
                {
                    "n_train": [2],
                    "rmse_mean": [0.4],
                    "rmse_std": [0.05],
                    "fit_time_mean_s": [0.12],
                    "fit_time_std_s": [0.01],
                }
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "total_time_accuracy.png"
            with patch("oasis.plot.plt.close"):
                total_time_accuracy_plot(
                    results,
                    self._generation_summaries(),
                    output_path=output_path,
                    mlip_feature_names=("mace", "orb"),
                )
                fig = total_time_accuracy_plot.__globals__["plt"].gcf()
                labels = [
                    collection.get_label()
                    for collection in fig.axes[0].collections
                ]

            self.assertEqual(labels, ["Ridge"])

    def test_fixed_split_training_time_accuracy_plot_uses_expected_axis_labels(self) -> None:
        results = LearningCurveResults(
            ridge_df=pd.DataFrame(
                {
                    "n_train": [8],
                    "rmse_mean": [0.3],
                    "rmse_std": [0.04],
                    "fit_time_mean_s": [0.18],
                    "fit_time_std_s": [0.02],
                }
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "fixed_split_training_time_accuracy.png"
            with patch("oasis.plot.plt.close"):
                fixed_split_training_time_accuracy_plot(
                    results,
                    self._generation_summaries(),
                    dataset_size=10,
                    output_path=output_path,
                    train_fraction=0.5,
                    mlip_feature_names=("mace", "orb"),
                )
                fig = fixed_split_training_time_accuracy_plot.__globals__["plt"].gcf()
                ax = fig.axes[0]

            self.assertEqual(ax.get_xlabel(), "Training time (s)")
            self.assertEqual(ax.get_ylabel(), "RMSE (eV)")
            self.assertEqual(
                ax.get_title(),
                "Fixed-split training time vs RMSE (train=50%)",
            )
            self.assertEqual(ax.xaxis.get_offset_text().get_text(), "")

    def test_fixed_split_total_time_accuracy_plot_renders_one_point_per_method(self) -> None:
        results = LearningCurveResults(
            ridge_df=pd.DataFrame(
                {
                    "n_train": [8],
                    "rmse_mean": [0.3],
                    "rmse_std": [0.04],
                    "fit_time_mean_s": [0.18],
                    "fit_time_std_s": [0.02],
                }
            ),
            resid_df=pd.DataFrame(
                {
                    "n_train": [8],
                    "rmse_mean": [0.25],
                    "rmse_std": [0.03],
                    "fit_time_mean_s": [0.08],
                    "fit_time_std_s": [0.01],
                }
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "fixed_split_total_time_accuracy.png"
            with patch("oasis.plot.plt.close"):
                saved_path = fixed_split_total_time_accuracy_plot(
                    results,
                    self._generation_summaries(),
                    dataset_size=10,
                    output_path=output_path,
                    train_fraction=0.8,
                    mlip_feature_names=("mace", "orb"),
                )
                fig = fixed_split_total_time_accuracy_plot.__globals__["plt"].gcf()
                ax = fig.axes[0]

            self.assertEqual(saved_path, output_path)
            self.assertTrue(output_path.exists())
            self.assertEqual(
                ax.get_title(),
                "Fixed-split total time vs RMSE (train=80%)",
            )

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

    def test_screening_budget_plot_filters_to_requested_x_window(self) -> None:
        result_df = pd.DataFrame(
            {
                "n_budget": [4, 6, 8, 10],
                "n_train": [3, 4, 6, 8],
                "n_screen": [1, 2, 2, 2],
                "screen_fraction": [0.25, 1 / 3, 0.25, 0.2],
                "n_cv_folds": [4, 3, 4, 5],
                "cv_rmse_mean": [0.4, 0.3, 0.2, 0.1],
                "cv_rmse_std": [0.05, 0.04, 0.03, 0.02],
            }
        )
        results = LearningCurveResults(ridge_df=result_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "screening_curve_windowed.png"
            saved_path = screening_budget_plot(
                results,
                output_path=output_path,
                include_x=[6, 10],
            )

            self.assertEqual(saved_path, output_path)
            self.assertTrue(output_path.exists())

    def test_screening_budget_plot_uses_integer_x_ticks(self) -> None:
        result_df = pd.DataFrame(
            {
                "n_budget": [4, 6, 8, 10],
                "n_train": [3, 4, 6, 8],
                "n_screen": [1, 2, 2, 2],
                "screen_fraction": [0.25, 1 / 3, 0.25, 0.2],
                "n_cv_folds": [4, 3, 4, 5],
                "cv_rmse_mean": [0.4, 0.3, 0.2, 0.1],
                "cv_rmse_std": [0.05, 0.04, 0.03, 0.02],
            }
        )
        results = LearningCurveResults(ridge_df=result_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "screening_curve_ticks.png"
            with patch("oasis.plot.plt.close"):
                screening_budget_plot(results, output_path=output_path)
                fig = screening_budget_plot.__globals__["plt"].gcf()
                locator = fig.axes[0].xaxis.get_major_locator()

            self.assertTrue(getattr(locator, "_integer", False))

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
