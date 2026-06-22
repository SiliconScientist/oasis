from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from oasis.experiment_runner import (
    load_filtered_wide_predictions,
    run_experiment,
    run_experiment_from_config,
)
from oasis.mlip.timing import MlipGenerationTimingSummary
from oasis.sweep import LearningCurveResults


class _FakeColumn:
    def __init__(self, values):
        self._values = values

    def to_list(self):
        return list(self._values)

    def to_numpy(self):
        return np.asarray(self._values)


class _FakeWideFrame:
    def __init__(self, reactions=None) -> None:
        reactions = ["r0", "r1"] if reactions is None else list(reactions)
        self._columns = {
            "reaction": _FakeColumn(reactions),
            "reference_ads_eng": _FakeColumn([float(i + 1) for i in range(len(reactions))]),
            "model_a_mlip_ads_eng_median": _FakeColumn(
                [float(i + 1) + 0.1 for i in range(len(reactions))]
            ),
            "model_b_mlip_ads_eng_median": _FakeColumn(
                [float(i + 1) - 0.1 for i in range(len(reactions))]
            ),
        }

    def get_column(self, name: str):
        return self._columns[name]

    @property
    def columns(self) -> list[str]:
        return list(self._columns)
    def __getitem__(self, name: str):
        return self._columns[name]

    def __len__(self) -> int:
        return len(self._columns["reaction"].to_list())


class ExperimentRunnerTests(unittest.TestCase):
    @staticmethod
    def _uq_results() -> LearningCurveResults:
        uq_frame = pd.DataFrame(
            {
                "n_train": [5, 10],
                "miscalibration_area": [0.1, 0.05],
                "sharpness": [0.2, 0.15],
                "dispersion": [0.3, 0.25],
                "uncertainty_kind": ["calibrated", "calibrated"],
            }
        )
        spread_only_frame = pd.DataFrame(
            {
                "n_train": [5, 10],
                "miscalibration_area": [0.4, 0.35],
                "sharpness": [0.5, 0.45],
                "dispersion": [0.6, 0.55],
                "uncertainty_kind": ["spread_only", "spread_only"],
            }
        )
        return LearningCurveResults(
            resid_uq_df=uq_frame,
            weighted_simplex_uq_df=uq_frame,
            ridge_uq_df=spread_only_frame,
            moe_uq_df=spread_only_frame,
        )

    @staticmethod
    def _timed_learning_curve_results() -> LearningCurveResults:
        ridge_frame = pd.DataFrame(
            {
                "n_train": [5, 10],
                "rmse_mean": [0.15, 0.12],
                "rmse_std": [0.01, 0.02],
                "fit_time_mean_s": [0.5, 0.8],
                "fit_time_std_s": [0.05, 0.08],
            }
        )
        return LearningCurveResults(ridge_df=ridge_frame)

    @staticmethod
    def _fixed_split_timed_learning_curve_results() -> LearningCurveResults:
        ridge_frame = pd.DataFrame(
            {
                "n_train": [8],
                "rmse_mean": [0.15],
                "rmse_std": [0.01],
                "fit_time_mean_s": [0.5],
                "fit_time_std_s": [0.05],
            }
        )
        return LearningCurveResults(ridge_df=ridge_frame)

    def test_run_experiment_from_config_loads_config_then_runs(self) -> None:
        cfg = SimpleNamespace()

        with patch("oasis.experiment_runner.get_config", return_value=cfg) as mock_get:
            with patch(
                "oasis.experiment_runner.run_experiment",
                return_value="results",
            ) as mock_run:
                result = run_experiment_from_config(["mlip.toml", "experiment.toml"])

        mock_get.assert_called_once_with(["mlip.toml", "experiment.toml"])
        mock_run.assert_called_once_with(cfg)
        self.assertEqual(result, "results")

    def test_run_experiment_skips_probe_dataset_build_when_probe_gnn_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cfg = SimpleNamespace(
                mlip=SimpleNamespace(dataset=str(tmp_path / "mamun_oh.json")),
                probe_features=SimpleNamespace(
                    dataset_path=tmp_path / "probe_dataset.json",
                    mlip_results_dir=tmp_path / "probe_features",
                ),
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        graph_dataset=None,
                        models=SimpleNamespace(
                            use_latent=False,
                            probe_gnn=SimpleNamespace(enabled=False),
                        ),
                    )
                ),
                analysis=SimpleNamespace(base_dir=tmp_path / "mlips"),
                plot=SimpleNamespace(
                    output_dir=tmp_path / "plots",
                ),
            )
            fake_wide_df = _FakeWideFrame()

            with patch(
                "oasis.experiment_runner.build_probe_dataset"
            ) as mock_build_probe:
                with patch(
                    "oasis.experiment_runner.add_mlip_feature_matrices_to_dataset"
                ) as mock_add_probe_features:
                    with patch(
                        "oasis.experiment_runner.find_result_files",
                        return_value=[],
                    ):
                        with patch(
                            "oasis.experiment_runner.load_wide_predictions",
                            return_value=fake_wide_df,
                        ):
                            with patch(
                                "oasis.experiment_runner.parity_plot",
                                return_value=tmp_path / "plots" / "parity.png",
                            ):
                                with patch(
                                    "oasis.experiment_runner.load_sample_atoms_for_wide_df",
                                    return_value=[],
                                ):
                                    with patch(
                                        "oasis.experiment_runner.atoms_to_graph_dataset_view",
                                        return_value=[],
                                    ):
                                        with patch(
                                            "oasis.experiment_runner.load_or_run_learning_curve_results_from_config",
                                            return_value=LearningCurveResults.empty(),
                                        ):
                                            with patch(
                                                "oasis.experiment_runner.learning_curve_plot",
                                                return_value=tmp_path
                                                / "plots"
                                                / "learning_curve.png",
                                            ):
                                                run_experiment(cfg)

            self.assertFalse(mock_build_probe.called)
            self.assertFalse(mock_add_probe_features.called)

    def test_load_filtered_wide_predictions_applies_anomaly_aware_mlip_selection(
        self,
    ) -> None:
        cfg = SimpleNamespace(
            analysis=SimpleNamespace(base_dir=Path("data/mlips/OH-BMA")),
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(
                    mlip_selection=SimpleNamespace(
                        exclude_anomalous=True,
                        label_allowlist=["normal", "energy_anomaly"],
                        strict_inference_anomaly=True,
                    )
                )
            ),
        )
        fake_wide_df = _FakeWideFrame()

        with patch(
            "oasis.experiment_runner.find_result_files",
            return_value=[],
        ):
            with patch(
                "oasis.experiment_runner.load_wide_predictions",
                return_value=fake_wide_df,
            ):
                with patch(
                    "oasis.experiment_runner.filter_structures_with_insufficient_valid_mlips",
                    return_value=fake_wide_df,
                ) as mock_filter_structures:
                    with patch(
                        "oasis.experiment_runner.filter_anomalous_mlip_columns",
                        return_value=fake_wide_df,
                    ) as mock_filter_mlips:
                        wide_df, *_ = load_filtered_wide_predictions(cfg)

        self.assertIs(wide_df, fake_wide_df)
        mock_filter_structures.assert_called_once_with(
            fake_wide_df,
            enabled=True,
            label_allowlist=["normal", "energy_anomaly"],
            strict_inference_anomaly=True,
        )
        mock_filter_mlips.assert_called_once_with(
            fake_wide_df,
            enabled=True,
            label_allowlist=["normal", "energy_anomaly"],
            strict_inference_anomaly=True,
        )

    def test_run_experiment_uses_filtered_wide_df_from_auxiliary_view_builder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            original_wide_df = _FakeWideFrame([f"r{i}" for i in range(10)])
            filtered_wide_df = _FakeWideFrame([f"r{i}" for i in range(7)])
            cfg = SimpleNamespace(
                mlip=SimpleNamespace(dataset=str(tmp_path / "mamun_oh.json")),
                probe_features=None,
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        graph_dataset=None,
                        models=SimpleNamespace(
                            use_latent=True,
                            probe_gnn=SimpleNamespace(enabled=False),
                        ),
                    )
                ),
                analysis=SimpleNamespace(base_dir=tmp_path / "mlips"),
                plot=SimpleNamespace(
                    output_dir=tmp_path / "plots",
                ),
            )

            with patch(
                "oasis.experiment_runner.find_result_files",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.load_wide_predictions",
                return_value=original_wide_df,
            ), patch(
                "oasis.experiment_runner.build_auxiliary_views",
                return_value=(filtered_wide_df, {}),
            ), patch(
                "oasis.experiment_runner.ensure_probe_artifacts",
                return_value=False,
            ), patch(
                "oasis.experiment_runner.parity_plot",
                return_value=tmp_path / "plots" / "parity.png",
            ), patch(
                "oasis.experiment_runner.load_sample_atoms_for_wide_df",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.atoms_to_graph_dataset_view",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.load_or_run_learning_curve_results_from_config",
                return_value=LearningCurveResults.empty(),
            ) as mock_learning_curve, patch(
                "oasis.experiment_runner.learning_curve_plot",
                return_value=tmp_path / "plots" / "learning_curve.png",
            ):
                run_experiment(cfg)

        self.assertIs(mock_learning_curve.call_args.args[0], filtered_wide_df)

    def test_run_experiment_suffixes_outputs_and_cache_paths_for_anomaly_aware_mode(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cfg = SimpleNamespace(
                mlip=SimpleNamespace(dataset=str(tmp_path / "mamun_oh.json")),
                probe_features=None,
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        budget_mode="full_remainder_test",
                        results_bundle_path=tmp_path / "results.json",
                        graph_dataset=SimpleNamespace(
                            path=tmp_path / "aligned_graphs.parquet",
                            join_key="reaction",
                        ),
                        mlip_selection=SimpleNamespace(
                            exclude_anomalous=True,
                            label_allowlist=["normal"],
                            strict_inference_anomaly=False,
                        ),
                        models=SimpleNamespace(
                            use_latent=False,
                            probe_gnn=SimpleNamespace(enabled=False),
                        ),
                    )
                ),
                analysis=SimpleNamespace(
                    base_dir=tmp_path / "mlips",
                    comparison_plot_path=tmp_path / "comparison.png",
                ),
                plot=SimpleNamespace(
                    output_dir=tmp_path / "plots",
                ),
            )
            fake_wide_df = _FakeWideFrame()

            with patch(
                "oasis.experiment_runner.find_result_files",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.load_wide_predictions",
                return_value=fake_wide_df,
            ), patch(
                "oasis.experiment_runner.filter_structures_with_insufficient_valid_mlips",
                return_value=fake_wide_df,
            ), patch(
                "oasis.experiment_runner.filter_anomalous_mlip_columns",
                return_value=fake_wide_df,
            ), patch(
                "oasis.experiment_runner.parity_plot",
                return_value=tmp_path / "plots" / "parity.png",
            ) as mock_parity_plot, patch(
                "oasis.experiment_runner.graph_artifact_matches_frame",
                return_value=False,
            ), patch(
                "oasis.experiment_runner.load_sample_atoms_for_wide_df",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.atoms_to_graph_dataset_view",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.save_aligned_graph_dataset_parquet",
                return_value=tmp_path / "aligned_graphs_anomalyaware_on.parquet",
            ), patch(
                "oasis.experiment_runner.load_or_run_learning_curve_results_from_config",
                return_value=LearningCurveResults.empty(),
            ) as mock_results, patch(
                "oasis.experiment_runner.learning_curve_plot",
                return_value=tmp_path / "plots" / "learning_curve_anomalyaware_on.png",
            ) as mock_learning_curve_plot:
                run_experiment(cfg)

        self.assertEqual(
            mock_parity_plot.call_args.kwargs["output_path"],
            tmp_path / "plots" / "mlips_vs_dft_parity_anomalyaware_on.png",
        )
        self.assertEqual(
            mock_learning_curve_plot.call_args.kwargs["output_path"],
            tmp_path / "plots" / "learning_curve_anomalyaware_on.png",
        )
        self.assertEqual(
            mock_results.call_args.args[1].experiment.learning_curve.results_bundle_path,
            tmp_path / "results_anomalyaware_on.json",
        )
        self.assertEqual(
            mock_results.call_args.args[1].experiment.learning_curve.graph_dataset.path,
            tmp_path / "aligned_graphs_anomalyaware_on.parquet",
        )
        self.assertEqual(
            cfg.analysis.comparison_plot_path,
            tmp_path / "comparison_anomalyaware_on.png",
        )

    def test_run_experiment_emits_learning_and_screening_plots_when_both_are_configured(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cfg = SimpleNamespace(
                mlip=SimpleNamespace(dataset=str(tmp_path / "mamun_oh.json")),
                probe_features=None,
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        budget_mode="full_remainder_test",
                        graph_dataset=None,
                        mlip_selection=SimpleNamespace(
                            exclude_anomalous=False,
                            label_allowlist=["normal"],
                            strict_inference_anomaly=False,
                        ),
                        models=SimpleNamespace(
                            use_latent=False,
                            probe_gnn=SimpleNamespace(enabled=False),
                        ),
                    ),
                    screening=SimpleNamespace(
                        budget_mode="screening_fraction",
                        screen_fraction=0.25,
                        min_screen_size=1,
                        validation_fraction=0.2,
                        min_val_size=1,
                        min_tuning_val_size=1,
                        min_inner_train_size=1,
                        results_bundle_path=None,
                        reuse_results=False,
                        force_refresh_methods=[],
                        force_refresh_train_sizes={},
                    ),
                ),
                analysis=SimpleNamespace(base_dir=tmp_path / "mlips"),
                plot=SimpleNamespace(
                    output_dir=tmp_path / "plots",
                ),
            )
            fake_wide_df = _FakeWideFrame()

            with patch(
                "oasis.experiment_runner.find_result_files",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.load_wide_predictions",
                return_value=fake_wide_df,
            ), patch(
                "oasis.experiment_runner.filter_anomalous_mlip_columns",
                return_value=fake_wide_df,
            ), patch(
                "oasis.experiment_runner.parity_plot",
                return_value=tmp_path / "plots" / "parity.png",
            ), patch(
                "oasis.experiment_runner.load_sample_atoms_for_wide_df",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.atoms_to_graph_dataset_view",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.load_or_run_learning_curve_results_from_config",
                return_value=LearningCurveResults.empty(),
            ) as mock_results, patch(
                "oasis.experiment_runner.learning_curve_plot",
                return_value=tmp_path / "plots" / "learning_curve_anomalyaware_off.png",
            ) as mock_learning_curve_plot, patch(
                "oasis.experiment_runner.screening_budget_plot",
                side_effect=[
                    tmp_path / "plots" / "screening_budget_anomalyaware_off.png",
                    tmp_path / "tmp" / "screening_budget_panel_anomalyaware_off.png",
                ],
            ) as mock_screening_plot, patch(
                "oasis.experiment_runner.learning_screening_figure",
                return_value=tmp_path / "plots" / "learning_screening_figure_anomalyaware_off.png",
            ) as mock_learning_screening_figure:
                run_experiment(cfg)

        self.assertEqual(mock_results.call_count, 2)
        mock_learning_curve_plot.assert_called_once()
        self.assertEqual(mock_screening_plot.call_count, 2)
        self.assertEqual(
            mock_screening_plot.call_args_list[0].kwargs["output_path"],
            tmp_path / "plots" / "screening_budget_anomalyaware_off.png",
        )
        self.assertEqual(
            mock_screening_plot.call_args_list[1].kwargs["show_legend"],
            False,
        )
        self.assertEqual(
            mock_learning_screening_figure.call_args.kwargs["screening_curve_path"],
            tmp_path / "tmp" / "screening_budget_panel_anomalyaware_off.png",
        )
        self.assertEqual(
            mock_learning_screening_figure.call_args.kwargs["learning_curve_path"],
            tmp_path / "plots" / "learning_curve_anomalyaware_off.png",
        )

    def test_run_experiment_wires_time_accuracy_plots_from_saved_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cfg = SimpleNamespace(
                mlip=SimpleNamespace(dataset=str(tmp_path / "mamun_oh.json")),
                probe_features=None,
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        graph_dataset=None,
                        models=SimpleNamespace(
                            use_latent=False,
                            probe_gnn=SimpleNamespace(enabled=False),
                        ),
                    )
                ),
                analysis=SimpleNamespace(base_dir=tmp_path / "mlips"),
                plot=SimpleNamespace(
                    output_dir=tmp_path / "plots",
                    fixed_split=SimpleNamespace(train_fraction=0.5),
                ),
            )
            fake_wide_df = _FakeWideFrame([f"r{i}" for i in range(10)])
            learning_curve_results = self._fixed_split_timed_learning_curve_results()
            generation_timing = {
                "model_a": MlipGenerationTimingSummary(
                    model_name="model_a",
                    reaction_count=2,
                    generation_time_total_s=10.0,
                    generation_time_slab_s=4.0,
                    generation_time_adslab_s=6.0,
                    generation_steps_total=20,
                    generation_steps_slab=8,
                    generation_steps_adslab=12,
                    time_per_step_s=0.5,
                ),
                "model_b": MlipGenerationTimingSummary(
                    model_name="model_b",
                    reaction_count=2,
                    generation_time_total_s=12.0,
                    generation_time_slab_s=5.0,
                    generation_time_adslab_s=7.0,
                    generation_steps_total=24,
                    generation_steps_slab=10,
                    generation_steps_adslab=14,
                    time_per_step_s=0.5,
                ),
            }
            result_files = [
                tmp_path / "model_a_result.json",
                tmp_path / "model_b_result.json",
            ]

            with patch(
                "oasis.experiment_runner.find_result_files",
                return_value=result_files,
            ), patch(
                "oasis.experiment_runner.load_wide_predictions",
                return_value=fake_wide_df,
            ), patch(
                "oasis.experiment_runner.parity_plot",
                return_value=tmp_path / "plots" / "parity.png",
            ), patch(
                "oasis.experiment_runner.load_sample_atoms_for_wide_df",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.atoms_to_graph_dataset_view",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.load_or_run_learning_curve_results_from_config",
                return_value=learning_curve_results,
            ), patch(
                "oasis.experiment_runner.learning_curve_plot",
                return_value=tmp_path / "plots" / "learning_curve_anomalyaware_off.png",
            ), patch(
                "oasis.experiment_runner.load_generation_timing_summaries",
                return_value=generation_timing,
            ) as mock_load_generation_timing, patch(
                "oasis.experiment_runner.generation_time_accuracy_plot",
                return_value=tmp_path / "plots" / "generation_time_accuracy_anomalyaware_off.png",
            ) as mock_generation_plot, patch(
                "oasis.experiment_runner.training_time_accuracy_plot",
                return_value=tmp_path / "plots" / "training_time_accuracy_anomalyaware_off.png",
            ) as mock_training_plot, patch(
                "oasis.experiment_runner.total_time_accuracy_plot",
                return_value=tmp_path / "plots" / "total_time_accuracy_anomalyaware_off.png",
            ) as mock_total_plot, patch(
                "oasis.experiment_runner.fixed_split_training_time_accuracy_plot",
                return_value=tmp_path / "plots" / "fixed_split_training_time_accuracy_anomalyaware_off.png",
            ) as mock_fixed_training_plot, patch(
                "oasis.experiment_runner.fixed_split_total_time_accuracy_plot",
                return_value=tmp_path / "plots" / "fixed_split_total_time_accuracy_anomalyaware_off.png",
            ) as mock_fixed_total_plot:
                run_experiment(cfg)

        self.assertEqual(mock_load_generation_timing.call_count, 2)
        for call in mock_load_generation_timing.call_args_list:
            self.assertEqual(call.args[0], result_files)
        for mock_plot, expected_path in (
            (
                mock_generation_plot,
                tmp_path / "plots" / "generation_time_accuracy_anomalyaware_off.png",
            ),
            (
                mock_training_plot,
                tmp_path / "plots" / "training_time_accuracy_anomalyaware_off.png",
            ),
            (
                mock_total_plot,
                tmp_path / "plots" / "total_time_accuracy_anomalyaware_off.png",
            ),
        ):
            mock_plot.assert_called_once()
            self.assertIs(mock_plot.call_args.kwargs["results"], learning_curve_results)
            self.assertIs(
                mock_plot.call_args.kwargs["generation_timing_by_mlip"],
                generation_timing,
            )
            self.assertEqual(
                mock_plot.call_args.kwargs["mlip_feature_names"],
                ("model_a", "model_b"),
            )
            self.assertIsNone(
                mock_plot.call_args.kwargs["generation_timing_by_method"]
            )
            self.assertEqual(mock_plot.call_args.kwargs["output_path"], expected_path)
        for mock_plot, expected_path in (
            (
                mock_fixed_training_plot,
                tmp_path
                / "plots"
                / "fixed_split_training_time_accuracy_anomalyaware_off.png",
            ),
            (
                mock_fixed_total_plot,
                tmp_path
                / "plots"
                / "fixed_split_total_time_accuracy_anomalyaware_off.png",
            ),
        ):
            mock_plot.assert_called_once()
            self.assertIs(mock_plot.call_args.kwargs["results"], learning_curve_results)
            self.assertIs(
                mock_plot.call_args.kwargs["generation_timing_by_mlip"],
                generation_timing,
            )
            self.assertEqual(mock_plot.call_args.kwargs["dataset_size"], 10)
            self.assertEqual(
                mock_plot.call_args.kwargs["mlip_feature_names"],
                ("model_a", "model_b"),
            )
            self.assertIsNone(
                mock_plot.call_args.kwargs["generation_timing_by_method"]
            )
            self.assertEqual(mock_plot.call_args.kwargs["train_fraction"], 0.5)
            self.assertEqual(mock_plot.call_args.kwargs["output_path"], expected_path)

    def test_run_experiment_wires_probe_generation_timing_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cfg = SimpleNamespace(
                mlip=SimpleNamespace(dataset=str(tmp_path / "mamun_oh.json")),
                probe_features=SimpleNamespace(
                    dataset_path=tmp_path / "probe_dataset.json",
                    mlip_results_dir=tmp_path / "probe_results",
                ),
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        graph_dataset=None,
                        models=SimpleNamespace(
                            use_latent=False,
                            use_probe_gnn=True,
                            probe_gnn=SimpleNamespace(enabled=True),
                        ),
                    )
                ),
                analysis=SimpleNamespace(base_dir=tmp_path / "mlips"),
                plot=SimpleNamespace(
                    output_dir=tmp_path / "plots",
                    fixed_split=SimpleNamespace(train_fraction=0.5),
                ),
            )
            fake_wide_df = _FakeWideFrame([f"r{i}" for i in range(10)])
            learning_curve_results = self._fixed_split_timed_learning_curve_results()
            generation_timing = {
                "model_a": MlipGenerationTimingSummary(
                    model_name="model_a",
                    reaction_count=2,
                    generation_time_total_s=10.0,
                    generation_time_slab_s=4.0,
                    generation_time_adslab_s=6.0,
                    generation_steps_total=20,
                    generation_steps_slab=8,
                    generation_steps_adslab=12,
                    time_per_step_s=0.5,
                ),
                "model_b": MlipGenerationTimingSummary(
                    model_name="model_b",
                    reaction_count=2,
                    generation_time_total_s=12.0,
                    generation_time_slab_s=5.0,
                    generation_time_adslab_s=7.0,
                    generation_steps_total=24,
                    generation_steps_slab=10,
                    generation_steps_adslab=14,
                    time_per_step_s=0.5,
                ),
            }
            probe_generation_timing = {
                "model_a": MlipGenerationTimingSummary(
                    model_name="model_a",
                    reaction_count=3,
                    generation_time_total_s=3.0,
                    generation_time_slab_s=1.0,
                    generation_time_adslab_s=2.0,
                    generation_steps_total=6,
                    generation_steps_slab=2,
                    generation_steps_adslab=4,
                    time_per_step_s=0.5,
                ),
                "model_b": MlipGenerationTimingSummary(
                    model_name="model_b",
                    reaction_count=3,
                    generation_time_total_s=4.5,
                    generation_time_slab_s=1.5,
                    generation_time_adslab_s=3.0,
                    generation_steps_total=9,
                    generation_steps_slab=3,
                    generation_steps_adslab=6,
                    time_per_step_s=0.5,
                ),
            }
            result_files = [
                tmp_path / "model_a_result.json",
                tmp_path / "model_b_result.json",
            ]
            probe_result_files = [
                tmp_path / "probe_results" / "model_a_result.json",
                tmp_path / "probe_results" / "model_b_result.json",
            ]

            def fake_find_result_files(base_dir, **kwargs):
                if base_dir == cfg.analysis.base_dir:
                    return result_files
                if base_dir == cfg.probe_features.mlip_results_dir:
                    self.assertEqual(kwargs.get("pattern"), "*_result.json")
                    return probe_result_files
                raise AssertionError(f"Unexpected base_dir: {base_dir!r}")

            with patch(
                "oasis.experiment_runner.ensure_probe_artifacts",
                return_value=True,
            ), patch(
                "oasis.experiment_runner.find_result_files",
                side_effect=fake_find_result_files,
            ), patch(
                "oasis.experiment_runner.load_wide_predictions",
                return_value=fake_wide_df,
            ), patch(
                "oasis.experiment_runner.parity_plot",
                return_value=tmp_path / "plots" / "parity.png",
            ), patch(
                "oasis.experiment_runner.load_sample_atoms_for_wide_df",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.atoms_to_graph_dataset_view",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.load_or_run_learning_curve_results_from_config",
                return_value=learning_curve_results,
            ), patch(
                "oasis.experiment_runner.learning_curve_plot",
                return_value=tmp_path / "plots" / "learning_curve_anomalyaware_off.png",
            ), patch(
                "oasis.experiment_runner.load_generation_timing_summaries",
                return_value=generation_timing,
            ), patch(
                "oasis.experiment_runner.load_probe_generation_timing_summaries",
                return_value=probe_generation_timing,
            ) as mock_probe_generation_timing, patch(
                "oasis.experiment_runner.generation_time_accuracy_plot",
                return_value=tmp_path / "plots" / "generation_time_accuracy_anomalyaware_off.png",
            ) as mock_generation_plot, patch(
                "oasis.experiment_runner.training_time_accuracy_plot",
                return_value=tmp_path / "plots" / "training_time_accuracy_anomalyaware_off.png",
            ) as mock_training_plot, patch(
                "oasis.experiment_runner.total_time_accuracy_plot",
                return_value=tmp_path / "plots" / "total_time_accuracy_anomalyaware_off.png",
            ), patch(
                "oasis.experiment_runner.fixed_split_training_time_accuracy_plot",
                return_value=tmp_path / "plots" / "fixed_split_training_time_accuracy_anomalyaware_off.png",
            ), patch(
                "oasis.experiment_runner.fixed_split_total_time_accuracy_plot",
                return_value=tmp_path / "plots" / "fixed_split_total_time_accuracy_anomalyaware_off.png",
            ):
                run_experiment(cfg)

        mock_probe_generation_timing.assert_called_once_with(probe_result_files)
        expected_override = mock_generation_plot.call_args.kwargs[
            "generation_timing_by_method"
        ]["probe_gnn"]
        self.assertEqual(expected_override.generation_time_s, 7.5)
        self.assertEqual(expected_override.generation_time_slab_s, 2.5)
        self.assertEqual(expected_override.generation_time_adslab_s, 5.0)
        self.assertEqual(expected_override.generation_steps_total, 15)
        self.assertEqual(expected_override.mlip_feature_names, ("model_a", "model_b"))
        self.assertEqual(
            mock_training_plot.call_args.kwargs["generation_timing_by_method"][
                "probe_gnn"
            ],
            expected_override,
        )

    def test_run_experiment_wires_latent_generation_timing_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            latent_csv_path = tmp_path / "latent.csv"
            latent_timing_path = tmp_path / "latent.csv.timing.json"
            latent_timing_path.write_text(
                '{"generation_time_s": 12.5}\n',
                encoding="utf-8",
            )
            cfg = SimpleNamespace(
                mlip=SimpleNamespace(dataset=str(tmp_path / "mamun_oh.json")),
                probe_features=None,
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        graph_dataset=None,
                        models=SimpleNamespace(
                            use_latent=True,
                            latent=SimpleNamespace(
                                csv_path=latent_csv_path,
                                timing_path=latent_timing_path,
                            ),
                            probe_gnn=SimpleNamespace(enabled=False),
                        ),
                    )
                ),
                analysis=SimpleNamespace(base_dir=tmp_path / "mlips"),
                plot=SimpleNamespace(
                    output_dir=tmp_path / "plots",
                    fixed_split=SimpleNamespace(train_fraction=0.5),
                ),
            )
            fake_wide_df = _FakeWideFrame([f"r{i}" for i in range(10)])
            learning_curve_results = self._fixed_split_timed_learning_curve_results()
            generation_timing = {
                "model_a": MlipGenerationTimingSummary(
                    model_name="model_a",
                    reaction_count=2,
                    generation_time_total_s=10.0,
                    generation_time_slab_s=4.0,
                    generation_time_adslab_s=6.0,
                    generation_steps_total=20,
                    generation_steps_slab=8,
                    generation_steps_adslab=12,
                    time_per_step_s=0.5,
                ),
                "model_b": MlipGenerationTimingSummary(
                    model_name="model_b",
                    reaction_count=2,
                    generation_time_total_s=12.0,
                    generation_time_slab_s=5.0,
                    generation_time_adslab_s=7.0,
                    generation_steps_total=24,
                    generation_steps_slab=10,
                    generation_steps_adslab=14,
                    time_per_step_s=0.5,
                ),
            }
            result_files = [
                tmp_path / "model_a_result.json",
                tmp_path / "model_b_result.json",
            ]

            with patch(
                "oasis.experiment_runner.find_result_files",
                return_value=result_files,
            ), patch(
                "oasis.experiment_runner.load_wide_predictions",
                return_value=fake_wide_df,
            ), patch(
                "oasis.experiment_runner.build_auxiliary_views",
                return_value=(fake_wide_df, {}),
            ), patch(
                "oasis.experiment_runner.ensure_probe_artifacts",
                return_value=False,
            ), patch(
                "oasis.experiment_runner.parity_plot",
                return_value=tmp_path / "plots" / "parity.png",
            ), patch(
                "oasis.experiment_runner.load_sample_atoms_for_wide_df",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.atoms_to_graph_dataset_view",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.load_or_run_learning_curve_results_from_config",
                return_value=learning_curve_results,
            ), patch(
                "oasis.experiment_runner.learning_curve_plot",
                return_value=tmp_path / "plots" / "learning_curve_anomalyaware_off.png",
            ), patch(
                "oasis.experiment_runner.load_generation_timing_summaries",
                return_value=generation_timing,
            ), patch(
                "oasis.experiment_runner.generation_time_accuracy_plot",
                return_value=tmp_path / "plots" / "generation_time_accuracy_anomalyaware_off.png",
            ) as mock_generation_plot, patch(
                "oasis.experiment_runner.training_time_accuracy_plot",
                return_value=tmp_path / "plots" / "training_time_accuracy_anomalyaware_off.png",
            ):
                run_experiment(cfg)

        latent_override = mock_generation_plot.call_args.kwargs[
            "generation_timing_by_method"
        ]["latent"]
        self.assertEqual(latent_override.generation_time_s, 12.5)
        self.assertEqual(latent_override.mlip_feature_names, ("latent_csv",))

    def test_run_experiment_skips_latent_generation_timing_override_when_sidecar_missing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            latent_csv_path = tmp_path / "latent.csv"
            cfg = SimpleNamespace(
                mlip=SimpleNamespace(dataset=str(tmp_path / "mamun_oh.json")),
                probe_features=None,
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        graph_dataset=None,
                        models=SimpleNamespace(
                            use_latent=True,
                            latent=SimpleNamespace(
                                csv_path=latent_csv_path,
                                timing_path=None,
                            ),
                            probe_gnn=SimpleNamespace(enabled=False),
                        ),
                    )
                ),
                analysis=SimpleNamespace(base_dir=tmp_path / "mlips"),
                plot=SimpleNamespace(
                    output_dir=tmp_path / "plots",
                    fixed_split=SimpleNamespace(train_fraction=0.5),
                ),
            )
            fake_wide_df = _FakeWideFrame([f"r{i}" for i in range(10)])
            learning_curve_results = self._fixed_split_timed_learning_curve_results()
            generation_timing = {
                "model_a": MlipGenerationTimingSummary(
                    model_name="model_a",
                    reaction_count=2,
                    generation_time_total_s=10.0,
                    generation_time_slab_s=4.0,
                    generation_time_adslab_s=6.0,
                    generation_steps_total=20,
                    generation_steps_slab=8,
                    generation_steps_adslab=12,
                    time_per_step_s=0.5,
                ),
                "model_b": MlipGenerationTimingSummary(
                    model_name="model_b",
                    reaction_count=2,
                    generation_time_total_s=12.0,
                    generation_time_slab_s=5.0,
                    generation_time_adslab_s=7.0,
                    generation_steps_total=24,
                    generation_steps_slab=10,
                    generation_steps_adslab=14,
                    time_per_step_s=0.5,
                ),
            }
            result_files = [
                tmp_path / "model_a_result.json",
                tmp_path / "model_b_result.json",
            ]

            with patch(
                "oasis.experiment_runner.find_result_files",
                return_value=result_files,
            ), patch(
                "oasis.experiment_runner.load_wide_predictions",
                return_value=fake_wide_df,
            ), patch(
                "oasis.experiment_runner.build_auxiliary_views",
                return_value=(fake_wide_df, {}),
            ), patch(
                "oasis.experiment_runner.ensure_probe_artifacts",
                return_value=False,
            ), patch(
                "oasis.experiment_runner.parity_plot",
                return_value=tmp_path / "plots" / "parity.png",
            ), patch(
                "oasis.experiment_runner.load_sample_atoms_for_wide_df",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.atoms_to_graph_dataset_view",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.load_or_run_learning_curve_results_from_config",
                return_value=learning_curve_results,
            ), patch(
                "oasis.experiment_runner.learning_curve_plot",
                return_value=tmp_path / "plots" / "learning_curve_anomalyaware_off.png",
            ), patch(
                "oasis.experiment_runner.load_generation_timing_summaries",
                return_value=generation_timing,
            ), patch(
                "oasis.experiment_runner.generation_time_accuracy_plot",
                return_value=tmp_path / "plots" / "generation_time_accuracy_anomalyaware_off.png",
            ) as mock_generation_plot, patch(
                "oasis.experiment_runner.training_time_accuracy_plot",
                return_value=tmp_path / "plots" / "training_time_accuracy_anomalyaware_off.png",
            ):
                run_experiment(cfg)

        self.assertIsNone(
            mock_generation_plot.call_args.kwargs["generation_timing_by_method"]
        )

    def test_run_experiment_rebuilds_stale_graph_artifact_when_reactions_do_not_match(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            graph_artifact_path = tmp_path / "aligned_graphs.parquet"
            graph_artifact_path.write_text("stale", encoding="utf-8")
            cfg = SimpleNamespace(
                mlip=SimpleNamespace(dataset=str(tmp_path / "mamun_oh.json")),
                probe_features=None,
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        graph_dataset=SimpleNamespace(
                            path=graph_artifact_path,
                            join_key="reaction",
                        ),
                        models=SimpleNamespace(
                            use_latent=False,
                            probe_gnn=SimpleNamespace(enabled=False),
                        ),
                    )
                ),
                analysis=SimpleNamespace(base_dir=tmp_path / "mlips"),
                plot=SimpleNamespace(
                    output_dir=tmp_path / "plots",
                ),
            )
            fake_wide_df = _FakeWideFrame(["r0", "r1", "r2"])

            with patch(
                "oasis.experiment_runner.find_result_files",
                return_value=[],
            ):
                with patch(
                    "oasis.experiment_runner.load_wide_predictions",
                    return_value=fake_wide_df,
                ):
                        with patch(
                            "oasis.experiment_runner.parity_plot",
                            return_value=tmp_path / "plots" / "parity.png",
                        ):
                            with patch(
                                "oasis.experiment_runner.graph_artifact_matches_frame",
                                return_value=False,
                            ):
                                with patch(
                                    "oasis.experiment_runner.load_sample_atoms_for_wide_df",
                                    return_value=["a0", "a1", "a2"],
                                ):
                                    with patch(
                                        "oasis.experiment_runner.atoms_to_graph_dataset_view",
                                        return_value=["g0", "g1", "g2"],
                                    ) as mock_build_graph_view:
                                        with patch(
                                            "oasis.experiment_runner.save_aligned_graph_dataset_parquet",
                                            return_value=graph_artifact_path,
                                        ) as mock_save_graph_artifact:
                                            with patch(
                                                "oasis.experiment_runner.load_or_run_learning_curve_results_from_config",
                                                return_value=LearningCurveResults.empty(),
                                            ):
                                                with patch(
                                                    "oasis.experiment_runner.learning_curve_plot",
                                                    return_value=tmp_path
                                                    / "plots"
                                                    / "learning_curve.png",
                                                ):
                                                    run_experiment(cfg)

            self.assertTrue(mock_build_graph_view.called)
            self.assertTrue(mock_save_graph_artifact.called)

    def test_run_experiment_rebuilds_graphs_without_persisting_when_graph_cache_is_omitted(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cfg = SimpleNamespace(
                mlip=SimpleNamespace(dataset=str(tmp_path / "mamun_oh.json")),
                probe_features=None,
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        graph_dataset=None,
                        models=SimpleNamespace(
                            use_latent=False,
                            probe_gnn=SimpleNamespace(enabled=False),
                        ),
                    )
                ),
                analysis=SimpleNamespace(base_dir=tmp_path / "mlips"),
                plot=SimpleNamespace(
                    output_dir=tmp_path / "plots",
                ),
            )
            fake_wide_df = _FakeWideFrame(["r0", "r1", "r2"])

            with patch(
                "oasis.experiment_runner.find_result_files",
                return_value=[],
            ):
                with patch(
                    "oasis.experiment_runner.load_wide_predictions",
                    return_value=fake_wide_df,
                ):
                        with patch(
                            "oasis.experiment_runner.parity_plot",
                            return_value=tmp_path / "plots" / "parity.png",
                        ):
                            with patch(
                                "oasis.experiment_runner.load_sample_atoms_for_wide_df",
                                return_value=["a0", "a1", "a2"],
                            ):
                                with patch(
                                    "oasis.experiment_runner.atoms_to_graph_dataset_view",
                                    return_value=["g0", "g1", "g2"],
                                ) as mock_build_graph_view:
                                    with patch(
                                        "oasis.experiment_runner.save_aligned_graph_dataset_parquet",
                                    ) as mock_save_graph_artifact:
                                        with patch(
                                            "oasis.experiment_runner.load_or_run_learning_curve_results_from_config",
                                            return_value=LearningCurveResults.empty(),
                                        ) as mock_learning_curve_results:
                                            with patch(
                                                "oasis.experiment_runner.learning_curve_plot",
                                                return_value=tmp_path
                                                / "plots"
                                                / "learning_curve.png",
                                            ):
                                                run_experiment(cfg)

            self.assertTrue(mock_build_graph_view.called)
            self.assertFalse(mock_save_graph_artifact.called)
            self.assertEqual(
                mock_learning_curve_results.call_args.kwargs["graph_view"],
                ["g0", "g1", "g2"],
            )

    def test_run_experiment_passes_separate_screening_config_to_second_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cfg = SimpleNamespace(
                mlip=SimpleNamespace(dataset=str(tmp_path / "mamun_oh.json")),
                probe_features=None,
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        budget_mode="full_remainder_test",
                        graph_dataset=None,
                        results_bundle_path=tmp_path / "learning.json",
                        models=SimpleNamespace(
                            use_latent=False,
                            probe_gnn=SimpleNamespace(enabled=False),
                        ),
                    ),
                    screening=SimpleNamespace(
                        budget_mode="screening_fraction",
                        screen_fraction=0.25,
                        min_screen_size=2,
                        validation_fraction=0.3,
                        min_val_size=2,
                        min_tuning_val_size=3,
                        min_inner_train_size=4,
                        results_bundle_path=tmp_path / "screening.json",
                        reuse_results=True,
                        force_refresh_methods=["ridge"],
                        force_refresh_train_sizes={"lasso": [8]},
                    ),
                ),
                analysis=SimpleNamespace(base_dir=tmp_path / "mlips"),
                plot=SimpleNamespace(
                    output_dir=tmp_path / "plots",
                ),
            )
            fake_wide_df = _FakeWideFrame()

            with patch(
                "oasis.experiment_runner.find_result_files",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.load_wide_predictions",
                return_value=fake_wide_df,
            ), patch(
                "oasis.experiment_runner.parity_plot",
                return_value=tmp_path / "plots" / "parity.png",
            ), patch(
                "oasis.experiment_runner.load_sample_atoms_for_wide_df",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.atoms_to_graph_dataset_view",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.load_or_run_learning_curve_results_from_config",
                return_value=LearningCurveResults.empty(),
            ) as mock_results, patch(
                "oasis.experiment_runner.learning_curve_plot",
                return_value=tmp_path / "plots" / "learning_curve.png",
            ) as mock_learning_curve_plot, patch(
                "oasis.experiment_runner.screening_budget_plot",
                side_effect=[
                    tmp_path / "plots" / "screening_budget.png",
                    tmp_path / "tmp" / "screening_budget_panel.png",
                ],
            ) as mock_screening_plot, patch(
                "oasis.experiment_runner.learning_screening_figure",
                return_value=tmp_path / "plots" / "learning_screening_figure.png",
            ) as mock_learning_screening_figure:
                run_experiment(cfg)

        first_cfg = mock_results.call_args_list[0].args[1]
        second_cfg = mock_results.call_args_list[1].args[1]
        self.assertEqual(
            first_cfg.experiment.learning_curve.results_bundle_path,
            tmp_path / "learning_anomalyaware_off.json",
        )
        self.assertEqual(
            second_cfg.experiment.learning_curve.results_bundle_path,
            tmp_path / "screening_anomalyaware_off.json",
        )
        self.assertEqual(
            second_cfg.experiment.learning_curve.budget_mode,
            "screening_fraction",
        )
        self.assertEqual(second_cfg.experiment.learning_curve.min_screen_size, 2)
        self.assertTrue(second_cfg.experiment.learning_curve.reuse_results)
        self.assertEqual(
            second_cfg.experiment.learning_curve.force_refresh_methods,
            ["ridge"],
        )
        mock_learning_curve_plot.assert_called_once()
        self.assertEqual(mock_screening_plot.call_count, 2)
        self.assertEqual(
            mock_screening_plot.call_args_list[0].kwargs["output_path"],
            tmp_path / "plots" / "screening_budget_anomalyaware_off.png",
        )

    def test_run_experiment_forwards_curve_window_to_learning_curve_plot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cfg = SimpleNamespace(
                mlip=SimpleNamespace(dataset=str(tmp_path / "mamun_oh.json")),
                probe_features=None,
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        budget_mode="full_remainder_test",
                        graph_dataset=None,
                        models=SimpleNamespace(
                            use_latent=False,
                            probe_gnn=SimpleNamespace(enabled=False),
                        ),
                    )
                ),
                analysis=SimpleNamespace(base_dir=tmp_path / "mlips"),
                plot=SimpleNamespace(
                    output_dir=tmp_path / "plots",
                    curve_window=SimpleNamespace(min_x=10, max_x=50, include_x=[10, 30]),
                ),
            )
            fake_wide_df = _FakeWideFrame()

            with patch(
                "oasis.experiment_runner.find_result_files",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.load_wide_predictions",
                return_value=fake_wide_df,
            ), patch(
                "oasis.experiment_runner.parity_plot",
                return_value=tmp_path / "plots" / "parity.png",
            ), patch(
                "oasis.experiment_runner.load_sample_atoms_for_wide_df",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.atoms_to_graph_dataset_view",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.load_or_run_learning_curve_results_from_config",
                return_value=LearningCurveResults.empty(),
            ), patch(
                "oasis.experiment_runner.learning_curve_plot",
                return_value=tmp_path / "plots" / "learning_curve.png",
            ) as mock_learning_curve_plot:
                run_experiment(cfg)

        self.assertEqual(mock_learning_curve_plot.call_args.kwargs["min_x"], 10)
        self.assertEqual(mock_learning_curve_plot.call_args.kwargs["max_x"], 50)
        self.assertEqual(mock_learning_curve_plot.call_args.kwargs["include_x"], [10, 30])
        self.assertEqual(
            mock_learning_curve_plot.call_args.kwargs["output_path"],
            tmp_path / "plots" / "learning_curve_anomalyaware_off.png",
        )

    def test_run_experiment_resolves_curve_window_include_fractions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cfg = SimpleNamespace(
                mlip=SimpleNamespace(dataset=str(tmp_path / "mamun_oh.json")),
                probe_features=None,
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        budget_mode="full_remainder_test",
                        graph_dataset=None,
                        models=SimpleNamespace(
                            use_latent=False,
                            probe_gnn=SimpleNamespace(enabled=False),
                        ),
                    )
                ),
                analysis=SimpleNamespace(base_dir=tmp_path / "mlips"),
                plot=SimpleNamespace(
                    output_dir=tmp_path / "plots",
                    curve_window=SimpleNamespace(
                        min_x=None,
                        max_x=None,
                        include_x=None,
                        include_fractions=[0.5, 1.0],
                    ),
                ),
            )
            fake_wide_df = _FakeWideFrame()

            with patch(
                "oasis.experiment_runner.find_result_files",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.load_wide_predictions",
                return_value=fake_wide_df,
            ), patch(
                "oasis.experiment_runner.parity_plot",
                return_value=tmp_path / "plots" / "parity.png",
            ), patch(
                "oasis.experiment_runner.load_sample_atoms_for_wide_df",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.atoms_to_graph_dataset_view",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.load_or_run_learning_curve_results_from_config",
                return_value=LearningCurveResults.empty(),
            ), patch(
                "oasis.experiment_runner.learning_curve_plot",
                return_value=tmp_path / "plots" / "learning_curve.png",
            ) as mock_learning_curve_plot:
                run_experiment(cfg)

        self.assertEqual(mock_learning_curve_plot.call_args.kwargs["include_x"], [1, 2])

    def test_run_experiment_merges_curve_window_include_x_and_fractions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cfg = SimpleNamespace(
                mlip=SimpleNamespace(dataset=str(tmp_path / "mamun_oh.json")),
                probe_features=None,
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        budget_mode="full_remainder_test",
                        graph_dataset=None,
                        models=SimpleNamespace(
                            use_latent=False,
                            probe_gnn=SimpleNamespace(enabled=False),
                        ),
                    )
                ),
                analysis=SimpleNamespace(base_dir=tmp_path / "mlips"),
                plot=SimpleNamespace(
                    output_dir=tmp_path / "plots",
                    curve_window=SimpleNamespace(
                        min_x=None,
                        max_x=None,
                        include_x=[3],
                        include_fractions=[0.5, 1.0],
                    ),
                ),
            )
            fake_wide_df = _FakeWideFrame()

            with patch(
                "oasis.experiment_runner.find_result_files",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.load_wide_predictions",
                return_value=fake_wide_df,
            ), patch(
                "oasis.experiment_runner.parity_plot",
                return_value=tmp_path / "plots" / "parity.png",
            ), patch(
                "oasis.experiment_runner.load_sample_atoms_for_wide_df",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.atoms_to_graph_dataset_view",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.load_or_run_learning_curve_results_from_config",
                return_value=LearningCurveResults.empty(),
            ), patch(
                "oasis.experiment_runner.learning_curve_plot",
                return_value=tmp_path / "plots" / "learning_curve.png",
            ) as mock_learning_curve_plot:
                run_experiment(cfg)

        self.assertEqual(mock_learning_curve_plot.call_args.kwargs["include_x"], [1, 2, 3])

    def test_run_experiment_full_dataset_window_disables_only_min_max(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cfg = SimpleNamespace(
                mlip=SimpleNamespace(dataset=str(tmp_path / "mamun_oh.json")),
                probe_features=None,
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        budget_mode="full_remainder_test",
                        graph_dataset=None,
                        models=SimpleNamespace(
                            use_latent=False,
                            probe_gnn=SimpleNamespace(enabled=False),
                        ),
                    )
                ),
                analysis=SimpleNamespace(base_dir=tmp_path / "mlips"),
                plot=SimpleNamespace(
                    output_dir=tmp_path / "plots",
                    curve_window=SimpleNamespace(
                        full_dataset_window=True,
                        min_x=10,
                        max_x=50,
                        include_x=[10, 30],
                    ),
                ),
            )
            fake_wide_df = _FakeWideFrame()

            with patch(
                "oasis.experiment_runner.find_result_files",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.load_wide_predictions",
                return_value=fake_wide_df,
            ), patch(
                "oasis.experiment_runner.parity_plot",
                return_value=tmp_path / "plots" / "parity.png",
            ), patch(
                "oasis.experiment_runner.load_sample_atoms_for_wide_df",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.atoms_to_graph_dataset_view",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.load_or_run_learning_curve_results_from_config",
                return_value=LearningCurveResults.empty(),
            ), patch(
                "oasis.experiment_runner.learning_curve_plot",
                return_value=tmp_path / "plots" / "learning_curve.png",
            ) as mock_learning_curve_plot:
                run_experiment(cfg)

        self.assertIsNone(mock_learning_curve_plot.call_args.kwargs["min_x"])
        self.assertIsNone(mock_learning_curve_plot.call_args.kwargs["max_x"])
        self.assertEqual(mock_learning_curve_plot.call_args.kwargs["include_x"], [10, 30])

    def test_run_experiment_curve_window_all_remains_backward_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cfg = SimpleNamespace(
                mlip=SimpleNamespace(dataset=str(tmp_path / "mamun_oh.json")),
                probe_features=None,
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        budget_mode="full_remainder_test",
                        graph_dataset=None,
                        models=SimpleNamespace(
                            use_latent=False,
                            probe_gnn=SimpleNamespace(enabled=False),
                        ),
                    )
                ),
                analysis=SimpleNamespace(base_dir=tmp_path / "mlips"),
                plot=SimpleNamespace(
                    output_dir=tmp_path / "plots",
                    curve_window=SimpleNamespace(
                        all=True,
                        min_x=10,
                        max_x=50,
                        include_x=[10, 30],
                    ),
                ),
            )
            fake_wide_df = _FakeWideFrame()

            with patch(
                "oasis.experiment_runner.find_result_files",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.load_wide_predictions",
                return_value=fake_wide_df,
            ), patch(
                "oasis.experiment_runner.parity_plot",
                return_value=tmp_path / "plots" / "parity.png",
            ), patch(
                "oasis.experiment_runner.load_sample_atoms_for_wide_df",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.atoms_to_graph_dataset_view",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.load_or_run_learning_curve_results_from_config",
                return_value=LearningCurveResults.empty(),
            ), patch(
                "oasis.experiment_runner.learning_curve_plot",
                return_value=tmp_path / "plots" / "learning_curve.png",
            ) as mock_learning_curve_plot:
                run_experiment(cfg)

        self.assertIsNone(mock_learning_curve_plot.call_args.kwargs["min_x"])
        self.assertIsNone(mock_learning_curve_plot.call_args.kwargs["max_x"])
        self.assertEqual(mock_learning_curve_plot.call_args.kwargs["include_x"], [10, 30])

    def test_run_experiment_full_dataset_window_preserves_include_fractions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cfg = SimpleNamespace(
                mlip=SimpleNamespace(dataset=str(tmp_path / "mamun_oh.json")),
                probe_features=None,
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        budget_mode="full_remainder_test",
                        graph_dataset=None,
                        models=SimpleNamespace(
                            use_latent=False,
                            probe_gnn=SimpleNamespace(enabled=False),
                        ),
                    )
                ),
                analysis=SimpleNamespace(base_dir=tmp_path / "mlips"),
                plot=SimpleNamespace(
                    output_dir=tmp_path / "plots",
                    curve_window=SimpleNamespace(
                        full_dataset_window=True,
                        min_x=10,
                        max_x=50,
                        include_fractions=[0.5, 1.0],
                    ),
                ),
            )
            fake_wide_df = _FakeWideFrame()

            with patch(
                "oasis.experiment_runner.find_result_files",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.load_wide_predictions",
                return_value=fake_wide_df,
            ), patch(
                "oasis.experiment_runner.parity_plot",
                return_value=tmp_path / "plots" / "parity.png",
            ), patch(
                "oasis.experiment_runner.load_sample_atoms_for_wide_df",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.atoms_to_graph_dataset_view",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.load_or_run_learning_curve_results_from_config",
                return_value=LearningCurveResults.empty(),
            ), patch(
                "oasis.experiment_runner.learning_curve_plot",
                return_value=tmp_path / "plots" / "learning_curve.png",
            ) as mock_learning_curve_plot:
                run_experiment(cfg)

        self.assertIsNone(mock_learning_curve_plot.call_args.kwargs["min_x"])
        self.assertIsNone(mock_learning_curve_plot.call_args.kwargs["max_x"])
        self.assertEqual(mock_learning_curve_plot.call_args.kwargs["include_x"], [1, 2])

    def test_run_experiment_emits_uq_summary_figure_when_learning_curve_results_have_uq(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cfg = SimpleNamespace(
                mlip=SimpleNamespace(dataset=str(tmp_path / "mamun_oh.json")),
                probe_features=None,
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        budget_mode="full_remainder_test",
                        graph_dataset=None,
                        models=SimpleNamespace(
                            use_latent=False,
                            probe_gnn=SimpleNamespace(enabled=False),
                        ),
                    )
                ),
                analysis=SimpleNamespace(base_dir=tmp_path / "mlips"),
                plot=SimpleNamespace(
                    output_dir=tmp_path / "plots",
                    curve_window=SimpleNamespace(min_x=5, max_x=10, include_x=[5, 10]),
                ),
            )
            fake_wide_df = _FakeWideFrame()
            results = self._uq_results()

            with patch(
                "oasis.experiment_runner.find_result_files",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.load_wide_predictions",
                return_value=fake_wide_df,
            ), patch(
                "oasis.experiment_runner.parity_plot",
                return_value=tmp_path / "plots" / "parity.png",
            ), patch(
                "oasis.experiment_runner.load_sample_atoms_for_wide_df",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.atoms_to_graph_dataset_view",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.load_or_run_learning_curve_results_from_config",
                return_value=results,
            ), patch(
                "oasis.experiment_runner.learning_curve_plot",
                return_value=tmp_path / "plots" / "learning_curve.png",
            ), patch(
                "oasis.experiment_runner.miscalibration_area_plot",
                return_value=tmp_path / "tmp" / "miscalibration.png",
            ) as mock_miscalibration_plot, patch(
                "oasis.experiment_runner.sharpness_plot",
                return_value=tmp_path / "tmp" / "sharpness.png",
            ) as mock_sharpness_plot, patch(
                "oasis.experiment_runner.dispersion_plot",
                return_value=tmp_path / "tmp" / "dispersion.png",
            ) as mock_dispersion_plot, patch(
                "oasis.experiment_runner.uq_summary_figure",
                return_value=tmp_path / "plots" / "uq_summary_figure.png",
            ) as mock_uq_summary_figure:
                run_experiment(cfg)

        mock_miscalibration_plot.assert_called_once()
        self.assertEqual(
            Path(mock_miscalibration_plot.call_args.kwargs["output_path"]).name,
            "miscalibration_area_panel_anomalyaware_off.png",
        )
        self.assertEqual(mock_miscalibration_plot.call_args.kwargs["min_x"], 5)
        self.assertEqual(mock_miscalibration_plot.call_args.kwargs["max_x"], 10)
        self.assertEqual(mock_miscalibration_plot.call_args.kwargs["include_x"], [5, 10])
        self.assertEqual(mock_miscalibration_plot.call_args.kwargs["show_xlabel"], False)
        self.assertAlmostEqual(
            mock_miscalibration_plot.call_args.kwargs["zero_shot_value"],
            0.4,
            places=12,
        )
        self.assertEqual(
            Path(mock_sharpness_plot.call_args.kwargs["output_path"]).name,
            "sharpness_panel_anomalyaware_off.png",
        )
        self.assertEqual(mock_sharpness_plot.call_args.kwargs["show_legend"], False)
        self.assertEqual(mock_sharpness_plot.call_args.kwargs["show_xlabel"], False)
        self.assertAlmostEqual(
            mock_sharpness_plot.call_args.kwargs["zero_shot_value"],
            0.1,
            places=12,
        )
        self.assertEqual(
            Path(mock_dispersion_plot.call_args.kwargs["output_path"]).name,
            "dispersion_panel_anomalyaware_off.png",
        )
        self.assertEqual(mock_dispersion_plot.call_args.kwargs["show_legend"], False)
        self.assertAlmostEqual(
            mock_dispersion_plot.call_args.kwargs["zero_shot_value"],
            0.0,
            places=12,
        )
        self.assertEqual(
            mock_uq_summary_figure.call_args.kwargs["output_path"],
            tmp_path / "plots" / "uq_summary_figure_anomalyaware_off.png",
        )
        self.assertEqual(
            mock_uq_summary_figure.call_args.kwargs["miscalibration_area_path"],
            tmp_path / "tmp" / "miscalibration.png",
        )
        self.assertEqual(
            mock_uq_summary_figure.call_args.kwargs["sharpness_path"],
            tmp_path / "tmp" / "sharpness.png",
        )
        self.assertEqual(
            mock_uq_summary_figure.call_args.kwargs["dispersion_path"],
            tmp_path / "tmp" / "dispersion.png",
        )
