from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl

from oasis.experiment_runner import (
    _apply_dev_run_curve_overrides,
    _apply_dev_run_frame_cap,
    _configured_policy_fixed_method_baselines,
    _load_all_datasets_oracle_uq_rows,
    _load_oracle_learning_curve_rows_for_dataset,
    _load_policy_regret_rows_for_dataset,
    _policy_selection_diagnostic_persistence_context,
    _load_zero_shot_stage_rows_for_dataset,
    _zero_shot_stage_artifact_path,
    _build_zero_shot_stage_rows,
    _write_policy_selection_diagnostic,
    BudgetSpanVariant,
    PolicyDiagnosticBuildOutputs,
    configured_budget_span_variants,
    load_all_datasets_policy_regret_rows,
    load_all_datasets_oracle_learning_curve_rows,
    load_filtered_wide_predictions,
    render_budget_span_variants,
    run_experiment,
    run_experiment_from_config,
    write_all_datasets_uq_oracle_plots,
    write_all_datasets_policy_regret_plot,
    write_all_datasets_zero_shot_rmse_stage_plot,
    write_all_datasets_oracle_learning_curve_plot,
    write_zero_shot_rmse_stage_plot,
    write_zero_shot_stage_parity_plots,
)
from oasis.experiment.policy_diagnostic import PolicySelectionDiagnosticResults
from oasis.tune import OptunaTuningConfig
from oasis.mlip.timing import MlipGenerationTimingSummary
from oasis.sweep import LearningCurveResults, SweepDataset


class _FakeColumn:
    def __init__(self, values):
        self._values = values

    def to_list(self):
        return list(self._values)

    def to_numpy(self):
        return np.asarray(self._values)

    def is_in(self, candidates):
        candidate_set = set(candidates)
        return _FakeMask([value in candidate_set for value in self._values])


class _FakeMask:
    def __init__(self, values):
        self._values = list(values)


class _FakeWideFrame:
    def __init__(self, reactions=None, reference_ads_eng=None) -> None:
        reactions = ["r0", "r1"] if reactions is None else list(reactions)
        reference_ads_eng = (
            [float(i + 1) for i in range(len(reactions))]
            if reference_ads_eng is None
            else list(reference_ads_eng)
        )
        self._columns = {
            "reaction": _FakeColumn(reactions),
            "reference_ads_eng": _FakeColumn(reference_ads_eng),
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

    def filter(self, mask):
        if not isinstance(mask, _FakeMask):
            raise TypeError(f"Unsupported mask type: {type(mask)!r}")
        filtered_columns = {
            name: [value for value, keep in zip(column.to_list(), mask._values) if keep]
            for name, column in self._columns.items()
        }
        return _FakeWideFrame(
            reactions=filtered_columns["reaction"],
            reference_ads_eng=filtered_columns["reference_ads_eng"],
        )

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
    def _kernel_ridge_only_uq_results() -> LearningCurveResults:
        return LearningCurveResults(
            kernel_ridge_uq_df=pd.DataFrame(
                {
                    "n_train": [5, 10],
                    "miscalibration_area": [0.18, 0.11],
                    "sharpness": [0.24, 0.19],
                    "dispersion": [0.34, 0.27],
                    "uncertainty_kind": ["calibrated", "calibrated"],
                }
            )
        )

    @staticmethod
    def _sparse_linear_only_uq_results() -> LearningCurveResults:
        uq_frame = pd.DataFrame(
            {
                "n_train": [5, 10],
                "miscalibration_area": [0.2, 0.12],
                "sharpness": [0.3, 0.22],
                "dispersion": [0.4, 0.32],
                "uncertainty_kind": ["calibrated", "calibrated"],
            }
        )
        return LearningCurveResults(
            lasso_uq_df=uq_frame,
            elastic_uq_df=uq_frame,
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

    def test_configured_policy_fixed_method_baselines_honors_toggles_and_labels(self) -> None:
        cfg = SimpleNamespace(
            experiment=SimpleNamespace(
                screening=SimpleNamespace(
                    plot_baselines=SimpleNamespace(
                        low_data_domain=SimpleNamespace(
                            enabled=False,
                            method_name="residual",
                            label="Few-shot (residual)",
                        ),
                        high_data_domain=SimpleNamespace(
                            enabled=True,
                            method_name="ridge",
                            label="Late ridge",
                        ),
                    )
                )
            )
        )

        self.assertEqual(
            _configured_policy_fixed_method_baselines(cfg),
            (("ridge", "Late ridge"),),
        )

    def test_configured_budget_span_variants_uses_explicit_sizes_and_fractions(self) -> None:
        cfg = SimpleNamespace(
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(
                    sweep_sizes=[1, 2, 3, 20],
                    sweep_fractions=[0.05, 0.1, 0.2],
                    min_train=None,
                    max_train=None,
                    step=1,
                )
            )
        )

        variants = configured_budget_span_variants(cfg)

        self.assertEqual([variant.key for variant in variants], ["absolute", "fraction"])
        self.assertEqual(variants[0].resolved_include_x(n_samples=100), [1, 2, 3, 20])
        self.assertEqual(variants[1].resolved_include_x(n_samples=100), [5, 10, 20])

    def test_configured_budget_span_variants_falls_back_to_integer_range(self) -> None:
        cfg = SimpleNamespace(
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(
                    sweep_sizes=[],
                    sweep_fractions=[],
                    min_train=2,
                    max_train=6,
                    step=2,
                )
            )
        )

        variants = configured_budget_span_variants(cfg)

        self.assertEqual(len(variants), 1)
        self.assertEqual(variants[0].key, "absolute")
        self.assertEqual(variants[0].resolved_include_x(n_samples=100), [2, 4, 6])

    def test_configured_budget_span_variants_can_return_fraction_only(self) -> None:
        cfg = SimpleNamespace(
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(
                    sweep_sizes=[],
                    sweep_fractions=[0.25, 0.5],
                    min_train=None,
                    max_train=None,
                    step=1,
                )
            )
        )

        variants = configured_budget_span_variants(cfg)

        self.assertEqual([variant.key for variant in variants], ["fraction"])
        self.assertEqual(variants[0].resolved_include_x(n_samples=20), [5, 10])

    def test_render_budget_span_variants_applies_suffixes_and_returns_absolute_path(
        self,
    ) -> None:
        cfg = SimpleNamespace(
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(
                    sweep_sizes=[1, 2],
                    sweep_fractions=[0.5, 1.0],
                    min_train=None,
                    max_train=None,
                    step=1,
                )
            )
        )
        rendered: list[tuple[str | None, Path]] = []

        def _render(span_variant, output_path):
            rendered.append(
                (
                    None if span_variant is None else span_variant.key,
                    output_path,
                )
            )
            return output_path

        saved_path = render_budget_span_variants(
            cfg,
            base_output_path=Path("plots/learning_curve.png"),
            render_variant=_render,
        )

        self.assertEqual(
            rendered,
            [
                ("absolute", Path("plots/learning_curve_absolute.png")),
                ("fraction", Path("plots/learning_curve_fraction.png")),
            ],
        )
        self.assertEqual(saved_path, Path("plots/learning_curve_absolute.png"))

    def test_write_policy_selection_diagnostic_saves_artifact_and_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_dir = tmp_path / "plots"
            output_dir.mkdir(parents=True, exist_ok=True)
            cfg = SimpleNamespace(
                seed=23,
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        min_train=4,
                        max_train=8,
                        step=2,
                        n_repeats=2,
                        min_test_size=1,
                        validation_fraction=0.2,
                        min_val_size=1,
                        min_tuning_val_size=1,
                        calibration_enabled=False,
                        calibration_fraction=0.2,
                        min_cal_size=1,
                        min_inner_train_size=1,
                        sweep_sizes=[],
                        sweep_fractions=[],
                        models=SimpleNamespace(
                            use_ridge=True,
                            use_kernel_ridge=False,
                            use_lasso=False,
                            use_elastic_net=False,
                            use_residual=False,
                            use_weighted_linear=False,
                            use_weighted_simplex=False,
                            use_graph_mean=False,
                            use_latent=False,
                            moe=SimpleNamespace(enabled=False),
                            probe_gnn=SimpleNamespace(enabled=False),
                            gnn_direct=SimpleNamespace(enabled=False),
                        ),
                    ),
                    screening=SimpleNamespace(
                        screen_fraction=0.25,
                        min_screen_size=1,
                        validation_fraction=0.3,
                        min_val_size=1,
                        min_tuning_val_size=1,
                        calibration_enabled=False,
                        calibration_fraction=0.2,
                        min_cal_size=1,
                        min_inner_train_size=1,
                        policy_names=[
                            "min_screening_rmse",
                            "combined_screening_rmse_miscalibration",
                        ],
                        combined_miscalibration_lambda=2.0,
                    ),
                ),
                plot=SimpleNamespace(output_dir=output_dir),
            )
            wide_df = pd.DataFrame(
                {
                    "reference_ads_eng": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    "ridge_mlip_ads_eng_median": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
                }
            )
            diagnostic_results = PolicySelectionDiagnosticResults(
                detail_df=pd.DataFrame(
                    {
                        "policy_name": ["min_screening_rmse", "min_screening_rmse"],
                        "budget": [4, 4],
                        "repeat": [0, 1],
                        "oracle_method": ["ridge", "ridge"],
                        "screening_selected_method": ["ridge", "ridge"],
                        "oracle_outer_rmse": [0.2, 0.25],
                        "screening_selected_outer_rmse": [0.2, 0.25],
                        "regret": [0.0, 0.0],
                        "screening_cv_rmse": [0.1, 0.11],
                        "screening_miscalibration_area": [0.05, 0.04],
                        "agreement": [True, True],
                    }
                ),
                outer_metrics_df=pd.DataFrame(
                    {
                        "budget": [4, 4],
                        "repeat": [0, 1],
                        "method": ["ridge", "ridge"],
                        "outer_test_rmse": [0.2, 0.25],
                    }
                ),
                summary_df=pd.DataFrame(
                    {
                        "policy_name": ["min_screening_rmse"],
                        "budget": [4],
                        "mean_regret": [0.0],
                        "std_regret": [0.0],
                        "se_regret": [0.0],
                        "ci95_low": [0.0],
                        "ci95_high": [0.0],
                        "agreement_rate": [1.0],
                        "oracle_outer_rmse_mean": [0.225],
                        "screening_selected_outer_rmse_mean": [0.225],
                    }
                ),
            )
            screening_rows_df = pd.DataFrame(
                {
                    "method": ["ridge", "ridge"],
                    "budget": [4, 4],
                    "repeat": [0, 1],
                    "split_fingerprint": ["fp-0", "fp-1"],
                    "screening_cv_rmse": [0.1, 0.11],
                    "screening_miscalibration_area": [0.05, 0.04],
                }
            )
            artifact_path = tmp_path / "policy_selection_diagnostic.json"
            screening_rows_path = tmp_path / "policy_selection_screening_rows.json"
            with patch(
                "oasis.experiment_runner.policy_selection_diagnostic_bundle_path",
                return_value=artifact_path,
            ), patch(
                "oasis.experiment_runner.policy_selection_screening_rows_bundle_path",
                return_value=screening_rows_path,
            ), patch(
                "oasis.experiment_runner._build_policy_selection_diagnostic_results_for_cfg",
                return_value=PolicyDiagnosticBuildOutputs(
                    results=diagnostic_results,
                    screening_rows_df=screening_rows_df,
                ),
            ) as mock_build_diagnostic:
                artifact_path = _write_policy_selection_diagnostic(
                    cfg=cfg,
                    wide_df=wide_df,
                    graph_view=None,
                    auxiliary_views=None,
                    output_dir=output_dir,
                    run_suffix="anomalyaware_off",
                    min_x=None,
                    max_x=None,
                    include_x=None,
                )
                detail_path = (
                    output_dir / "policy_selection_diagnostic_detail_anomalyaware_off.csv"
                )
                summary_path = (
                    output_dir / "policy_selection_diagnostic_summary_anomalyaware_off.csv"
                )
                oracle_plot_path = (
                    output_dir / "policy_selected_vs_oracle_anomalyaware_off_absolute.png"
                )
                regret_plot_path = output_dir / "policy_regret_anomalyaware_off_absolute.png"
                artifact_exists = artifact_path is not None and artifact_path.is_file()
                detail_exists = detail_path.is_file()
                summary_exists = summary_path.is_file()
                screening_rows_exists = screening_rows_path.is_file()
                oracle_plot_exists = oracle_plot_path.is_file()
                regret_plot_exists = regret_plot_path.is_file()

        assert artifact_path is not None
        self.assertEqual(
            artifact_path.name,
            "policy_selection_diagnostic.json",
        )
        self.assertTrue(artifact_exists)
        self.assertTrue(detail_exists)
        self.assertTrue(summary_exists)
        self.assertTrue(screening_rows_exists)
        self.assertTrue(oracle_plot_exists)
        self.assertTrue(regret_plot_exists)
        mock_build_diagnostic.assert_called_once()

    def test_write_policy_selection_diagnostic_emits_dual_selected_vs_oracle_outputs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_dir = tmp_path / "plots"
            output_dir.mkdir(parents=True, exist_ok=True)
            cfg = SimpleNamespace(
                seed=23,
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        min_train=None,
                        max_train=None,
                        step=1,
                        n_repeats=2,
                        min_test_size=1,
                        validation_fraction=0.2,
                        min_val_size=1,
                        min_tuning_val_size=1,
                        calibration_enabled=False,
                        calibration_fraction=0.2,
                        min_cal_size=1,
                        min_inner_train_size=1,
                        sweep_sizes=[1, 2],
                        sweep_fractions=[0.5, 1.0],
                        models=SimpleNamespace(
                            use_ridge=True,
                            use_kernel_ridge=False,
                            use_lasso=False,
                            use_elastic_net=False,
                            use_residual=False,
                            use_weighted_linear=False,
                            use_weighted_simplex=False,
                            use_graph_mean=False,
                            use_latent=False,
                            moe=SimpleNamespace(enabled=False),
                            probe_gnn=SimpleNamespace(enabled=False),
                            gnn_direct=SimpleNamespace(enabled=False),
                        ),
                    ),
                    screening=SimpleNamespace(
                        screen_fraction=0.25,
                        min_screen_size=1,
                        validation_fraction=0.3,
                        min_val_size=1,
                        min_tuning_val_size=1,
                        calibration_enabled=False,
                        calibration_fraction=0.2,
                        min_cal_size=1,
                        min_inner_train_size=1,
                        policy_names=["min_screening_rmse"],
                        combined_miscalibration_lambda=1.0,
                    ),
                ),
                plot=SimpleNamespace(output_dir=output_dir),
            )
            wide_df = pd.DataFrame(
                {
                    "reference_ads_eng": [1.0, 2.0],
                    "ridge_mlip_ads_eng_median": [1.1, 2.1],
                }
            )
            diagnostic_results = PolicySelectionDiagnosticResults(
                detail_df=pd.DataFrame(
                    {
                        "policy_name": ["min_screening_rmse"],
                        "budget": [1],
                        "repeat": [0],
                        "oracle_method": ["ridge"],
                        "screening_selected_method": ["ridge"],
                        "oracle_outer_rmse": [0.2],
                        "screening_selected_outer_rmse": [0.2],
                        "regret": [0.0],
                        "screening_cv_rmse": [0.1],
                        "screening_miscalibration_area": [0.05],
                        "agreement": [True],
                    }
                ),
                outer_metrics_df=pd.DataFrame(
                    {
                        "budget": [1],
                        "repeat": [0],
                        "method": ["ridge"],
                        "outer_test_rmse": [0.2],
                    }
                ),
                summary_df=pd.DataFrame(
                    {
                        "policy_name": ["min_screening_rmse"],
                        "budget": [1],
                        "mean_regret": [0.0],
                        "std_regret": [0.0],
                        "se_regret": [0.0],
                        "ci95_low": [0.0],
                        "ci95_high": [0.0],
                        "agreement_rate": [1.0],
                        "oracle_outer_rmse_mean": [0.2],
                        "screening_selected_outer_rmse_mean": [0.2],
                    }
                ),
            )
            screening_rows_df = pd.DataFrame(
                {
                    "method": ["ridge"],
                    "budget": [1],
                    "repeat": [0],
                    "split_fingerprint": ["fp-0"],
                    "screening_cv_rmse": [0.1],
                    "screening_miscalibration_area": [0.05],
                }
            )

            with patch(
                "oasis.experiment_runner._build_policy_selection_diagnostic_results_for_cfg",
                return_value=PolicyDiagnosticBuildOutputs(
                    results=diagnostic_results,
                    screening_rows_df=screening_rows_df,
                ),
            ), patch(
                "oasis.experiment_runner.policy_selected_vs_oracle_plot",
            ) as mock_selected_plot, patch(
                "oasis.experiment_runner.policy_regret_plot",
            ) as mock_regret_plot:
                _write_policy_selection_diagnostic(
                    cfg=cfg,
                    wide_df=wide_df,
                    graph_view=None,
                    auxiliary_views=None,
                    output_dir=output_dir,
                    run_suffix="anomalyaware_off",
                    min_x=None,
                    max_x=None,
                    include_x=None,
                )

        self.assertEqual(mock_selected_plot.call_count, 2)
        self.assertEqual(
            mock_selected_plot.call_args_list[0].kwargs["output_path"],
            output_dir / "policy_selected_vs_oracle_anomalyaware_off_absolute.png",
        )
        self.assertEqual(
            mock_selected_plot.call_args_list[1].kwargs["output_path"],
            output_dir / "policy_selected_vs_oracle_anomalyaware_off_fraction.png",
        )
        self.assertEqual(
            mock_selected_plot.call_args_list[0].kwargs["include_x"],
            [1, 2],
        )
        self.assertEqual(
            mock_selected_plot.call_args_list[1].kwargs["include_x"],
            [1, 2],
        )
        self.assertEqual(mock_regret_plot.call_count, 2)
        self.assertEqual(
            mock_regret_plot.call_args_list[0].kwargs["output_path"],
            output_dir / "policy_regret_anomalyaware_off_absolute.png",
        )
        self.assertEqual(
            mock_regret_plot.call_args_list[1].kwargs["output_path"],
            output_dir / "policy_regret_anomalyaware_off_fraction.png",
        )

    def test_write_policy_selection_diagnostic_reuses_cached_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_dir = tmp_path / "plots"
            output_dir.mkdir(parents=True, exist_ok=True)
            cfg = SimpleNamespace(
                seed=23,
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        min_train=4,
                        max_train=8,
                        step=2,
                        n_repeats=2,
                        min_test_size=1,
                        validation_fraction=0.2,
                        min_val_size=1,
                        min_tuning_val_size=1,
                        calibration_enabled=False,
                        calibration_fraction=0.2,
                        min_cal_size=1,
                        min_inner_train_size=1,
                        sweep_sizes=[],
                        sweep_fractions=[],
                        models=SimpleNamespace(
                            use_ridge=True,
                            use_kernel_ridge=False,
                            use_lasso=False,
                            use_elastic_net=False,
                            use_residual=False,
                            use_weighted_linear=False,
                            use_weighted_simplex=False,
                            use_graph_mean=False,
                            use_latent=False,
                            moe=SimpleNamespace(enabled=False),
                            probe_gnn=SimpleNamespace(enabled=False),
                            gnn_direct=SimpleNamespace(enabled=False),
                        ),
                    ),
                    screening=SimpleNamespace(
                        screen_fraction=0.25,
                        min_screen_size=1,
                        validation_fraction=0.3,
                        min_val_size=1,
                        min_tuning_val_size=1,
                        calibration_enabled=False,
                        calibration_fraction=0.2,
                        min_cal_size=1,
                        min_inner_train_size=1,
                        policy_names=["min_screening_rmse"],
                        combined_miscalibration_lambda=1.0,
                    ),
                ),
                plot=SimpleNamespace(output_dir=output_dir),
            )
            wide_df = pd.DataFrame(
                {
                    "reference_ads_eng": [1.0, 2.0, 3.0, 4.0],
                    "ridge_mlip_ads_eng_median": [1.1, 2.1, 3.1, 4.1],
                }
            )
            diagnostic_results = PolicySelectionDiagnosticResults(
                detail_df=pd.DataFrame(
                    {
                        "policy_name": ["min_screening_rmse"],
                        "budget": [4],
                        "repeat": [0],
                        "oracle_method": ["ridge"],
                        "screening_selected_method": ["ridge"],
                        "oracle_outer_rmse": [0.2],
                        "screening_selected_outer_rmse": [0.2],
                        "regret": [0.0],
                        "screening_cv_rmse": [0.1],
                        "screening_miscalibration_area": [0.05],
                        "agreement": [True],
                    }
                ),
                outer_metrics_df=pd.DataFrame(
                    {
                        "budget": [4],
                        "repeat": [0],
                        "method": ["ridge"],
                        "outer_test_rmse": [0.2],
                    }
                ),
                summary_df=pd.DataFrame(
                    {
                        "policy_name": ["min_screening_rmse"],
                        "budget": [4],
                        "mean_regret": [0.0],
                        "std_regret": [0.0],
                        "se_regret": [0.0],
                        "ci95_low": [0.0],
                        "ci95_high": [0.0],
                        "agreement_rate": [1.0],
                        "oracle_outer_rmse_mean": [0.2],
                        "screening_selected_outer_rmse_mean": [0.2],
                    }
                ),
            )

            artifact_path = tmp_path / "policy_selection_diagnostic.json"
            artifact_path.write_text("cached", encoding="utf-8")

            with patch(
                "oasis.experiment_runner.policy_selection_diagnostic_bundle_path",
                return_value=artifact_path,
            ), patch(
                "oasis.experiment_runner.policy_selection_screening_rows_bundle_path",
                return_value=tmp_path / "policy_selection_screening_rows.json",
            ), patch(
                "oasis.experiment_runner.load_policy_selection_diagnostic_artifact",
                return_value=SimpleNamespace(results=diagnostic_results),
            ) as mock_load, patch(
                "oasis.experiment_runner._build_policy_selection_diagnostic_results_for_cfg",
            ) as mock_build:
                saved_path = _write_policy_selection_diagnostic(
                    cfg=cfg,
                    wide_df=wide_df,
                    graph_view=None,
                    auxiliary_views=None,
                    output_dir=output_dir,
                    run_suffix="anomalyaware_off",
                    min_x=None,
                    max_x=None,
                    include_x=None,
                )

        self.assertEqual(saved_path, artifact_path)
        mock_load.assert_called_once()
        mock_build.assert_not_called()

    def test_write_policy_selection_diagnostic_rebuilds_when_cached_artifact_is_incompatible(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_dir = tmp_path / "plots"
            output_dir.mkdir(parents=True, exist_ok=True)
            cfg = SimpleNamespace(
                seed=23,
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        min_train=4,
                        max_train=8,
                        step=2,
                        n_repeats=2,
                        min_test_size=1,
                        validation_fraction=0.2,
                        min_val_size=1,
                        min_tuning_val_size=1,
                        calibration_enabled=False,
                        calibration_fraction=0.2,
                        min_cal_size=1,
                        min_inner_train_size=1,
                        sweep_sizes=[],
                        sweep_fractions=[],
                        models=SimpleNamespace(
                            use_ridge=True,
                            use_kernel_ridge=False,
                            use_lasso=False,
                            use_elastic_net=False,
                            use_residual=False,
                            use_weighted_linear=False,
                            use_weighted_simplex=False,
                            use_graph_mean=False,
                            use_latent=False,
                            moe=SimpleNamespace(enabled=False),
                            probe_gnn=SimpleNamespace(enabled=False),
                            gnn_direct=SimpleNamespace(enabled=False),
                        ),
                    ),
                    screening=SimpleNamespace(
                        screen_fraction=0.25,
                        min_screen_size=1,
                        validation_fraction=0.3,
                        min_val_size=1,
                        min_tuning_val_size=1,
                        calibration_enabled=False,
                        calibration_fraction=0.2,
                        min_cal_size=1,
                        min_inner_train_size=1,
                        policy_names=["min_screening_rmse"],
                        combined_miscalibration_lambda=1.0,
                    ),
                ),
                plot=SimpleNamespace(output_dir=output_dir),
            )
            wide_df = pd.DataFrame(
                {
                    "reference_ads_eng": [1.0, 2.0, 3.0, 4.0],
                    "ridge_mlip_ads_eng_median": [1.1, 2.1, 3.1, 4.1],
                }
            )
            diagnostic_results = PolicySelectionDiagnosticResults(
                detail_df=pd.DataFrame(
                    {
                        "policy_name": ["min_screening_rmse"],
                        "budget": [4],
                        "repeat": [0],
                        "oracle_method": ["ridge"],
                        "screening_selected_method": ["ridge"],
                        "oracle_outer_rmse": [0.2],
                        "screening_selected_outer_rmse": [0.2],
                        "regret": [0.0],
                        "screening_cv_rmse": [0.1],
                        "screening_miscalibration_area": [0.05],
                        "agreement": [True],
                    }
                ),
                outer_metrics_df=pd.DataFrame(
                    {
                        "budget": [4],
                        "repeat": [0],
                        "method": ["ridge"],
                        "outer_test_rmse": [0.2],
                    }
                ),
                summary_df=pd.DataFrame(
                    {
                        "policy_name": ["min_screening_rmse"],
                        "budget": [4],
                        "mean_regret": [0.0],
                        "std_regret": [0.0],
                        "se_regret": [0.0],
                        "ci95_low": [0.0],
                        "ci95_high": [0.0],
                        "agreement_rate": [1.0],
                        "oracle_outer_rmse_mean": [0.2],
                        "screening_selected_outer_rmse_mean": [0.2],
                    }
                ),
            )

            artifact_path = tmp_path / "policy_selection_diagnostic.json"
            artifact_path.write_text("cached", encoding="utf-8")

            with patch(
                "oasis.experiment_runner.policy_selection_diagnostic_bundle_path",
                return_value=artifact_path,
            ), patch(
                "oasis.experiment_runner.policy_selection_screening_rows_bundle_path",
                return_value=tmp_path / "policy_selection_screening_rows.json",
            ), patch(
                "oasis.experiment_runner.load_policy_selection_diagnostic_artifact",
                side_effect=ValueError("incompatible"),
            ) as mock_load, patch(
                "oasis.experiment_runner._build_policy_selection_diagnostic_results_for_cfg",
                return_value=PolicyDiagnosticBuildOutputs(
                    results=diagnostic_results,
                    screening_rows_df=pd.DataFrame(
                        {
                            "method": ["ridge"],
                            "budget": [4],
                            "repeat": [0],
                            "split_fingerprint": ["fp-0"],
                            "screening_cv_rmse": [0.1],
                            "screening_miscalibration_area": [0.05],
                        }
                    ),
                ),
            ) as mock_build:
                saved_path = _write_policy_selection_diagnostic(
                    cfg=cfg,
                    wide_df=wide_df,
                    graph_view=None,
                    auxiliary_views=None,
                    output_dir=output_dir,
                    run_suffix="anomalyaware_off",
                    min_x=None,
                    max_x=None,
                    include_x=None,
                )

        self.assertEqual(saved_path, artifact_path)
        self.assertEqual(mock_load.call_count, 2)
        mock_build.assert_called_once()

    def test_run_experiment_from_config_loads_config_then_runs(self) -> None:
        cfg = SimpleNamespace()

        with patch("oasis.experiment_runner.get_config", return_value=cfg) as mock_get:
            with patch(
                "oasis.experiment_runner.run_experiment",
                return_value="results",
            ) as mock_run:
                result = run_experiment_from_config(["experiment.toml"])

        mock_get.assert_called_once_with(["experiment.toml"])
        mock_run.assert_called_once_with(cfg)
        self.assertEqual(result, "results")

    def test_build_zero_shot_stage_rows_uses_strict_per_structure_mask_for_stage_three(
        self,
    ) -> None:
        cfg = SimpleNamespace(
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(
                    mlip_selection=SimpleNamespace(
                        exclude_anomalous=True,
                        strict_inference_anomaly=True,
                    )
                )
            )
        )
        frame = pl.DataFrame(
            {
                "reaction": ["r0", "r1"],
                "reference_ads_eng": [0.0, 0.0],
                "a_mlip_ads_eng_median": [0.0, 0.0],
                "b_mlip_ads_eng_median": [10.0, 10.0],
                "a_slab_conv": [0, 0],
                "a_ads_conv": [0, 0],
                "a_slab_move": [0, 0],
                "a_ads_move": [0, 0],
                "a_slab_seed": [0, 0],
                "a_ads_seed": [0, 0],
                "a_ads_eng_seed": [0, 0],
                "a_adsorbate_migration": [0, 0],
                "b_slab_conv": [1, 1],
                "b_ads_conv": [0, 0],
                "b_slab_move": [0, 0],
                "b_ads_move": [0, 0],
                "b_slab_seed": [0, 0],
                "b_ads_seed": [0, 0],
                "b_ads_eng_seed": [0, 0],
                "b_adsorbate_migration": [0, 0],
            }
        )

        stage_rows = _build_zero_shot_stage_rows(
            cfg=cfg,
            dataset_tag="example",
            raw_wide_df=frame,
            selected_wide_df=frame,
        )

        self.assertEqual(stage_rows[1]["rmse"], 5.0)
        self.assertEqual(stage_rows[2]["rmse"], 0.0)
        lone_mlip_rows = [row for row in stage_rows if row.get("mlip") is not None]
        self.assertEqual(len(lone_mlip_rows), 2)
        self.assertEqual(
            {row["mlip"]: row["rmse"] for row in lone_mlip_rows},
            {"a": 0.0, "b": 10.0},
        )

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

    def test_run_experiment_requires_external_probe_dataset_when_probe_gnn_enabled(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cfg = SimpleNamespace(
                mlip=SimpleNamespace(dataset=str(tmp_path / "mamun_oh.json")),
                probe_features=SimpleNamespace(
                    dataset_path=tmp_path / "missing_probe_dataset.json",
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
                ),
            )

            with self.assertRaisesRegex(
                FileNotFoundError,
                "external probe dataset",
            ):
                run_experiment(cfg)

    def test_run_experiment_skips_graph_view_for_non_graph_models(self) -> None:
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
                            use_graph_mean=False,
                            use_gnn_direct=False,
                            gnn_direct=SimpleNamespace(enabled=False),
                            use_probe_gnn=False,
                            probe_gnn=SimpleNamespace(enabled=False),
                            moe=SimpleNamespace(enabled=False, gate_type="mlip_baseline"),
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
                "oasis.experiment_runner.find_result_files",
                return_value=[],
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
                "oasis.experiment_runner.prepare_graph_view",
                side_effect=AssertionError(
                    "prepare_graph_view should not run for non-graph models"
                ),
            ), patch(
                "oasis.experiment_runner.load_or_run_learning_curve_results_from_config",
                return_value=LearningCurveResults.empty(),
            ), patch(
                "oasis.experiment_runner.learning_curve_plot",
                return_value=tmp_path / "plots" / "learning_curve.png",
            ):
                run_experiment(cfg)

    def test_run_experiment_requires_external_probe_results_dir_when_probe_gnn_enabled(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            probe_dataset_path = tmp_path / "probe_dataset.json"
            probe_dataset_path.write_text("{}", encoding="utf-8")
            cfg = SimpleNamespace(
                mlip=SimpleNamespace(dataset=str(tmp_path / "mamun_oh.json")),
                probe_features=SimpleNamespace(
                    dataset_path=probe_dataset_path,
                    mlip_results_dir=tmp_path / "missing_probe_results",
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
                ),
            )

            with self.assertRaisesRegex(
                FileNotFoundError,
                "external probe MLIP results directory",
            ):
                run_experiment(cfg)

    def test_load_filtered_wide_predictions_applies_anomaly_aware_mlip_selection(
        self,
    ) -> None:
        cfg = SimpleNamespace(
            analysis=SimpleNamespace(base_dir=Path("data/mlips/OH-BMA")),
            resolved_mlip_results_dir=Path("data/mlips/Tol-KHLOHC"),
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(
                    mlip_selection=SimpleNamespace(
                        enabled=["mace", "uma"],
                        exclude_anomalous_mlips=True,
                        minimum_quorum=3,
                    )
                )
            ),
        )
        fake_wide_df = _FakeWideFrame()

        with patch(
            "oasis.experiment_runner.find_result_files",
            return_value=[],
        ) as mock_find_result_files:
            with patch(
                "oasis.experiment_runner.load_wide_predictions",
                return_value=fake_wide_df,
            ):
                with patch(
                    "oasis.experiment_runner.filter_structures_with_insufficient_valid_mlips",
                    return_value=fake_wide_df,
                ) as mock_filter_structures:
                    wide_df, *_ = load_filtered_wide_predictions(cfg)

        self.assertIs(wide_df, fake_wide_df)
        mock_find_result_files.assert_called_once_with(
            Path("data/mlips/Tol-KHLOHC"),
            enabled_models=["mace", "uma"],
        )
        mock_filter_structures.assert_called_once_with(
            fake_wide_df,
            enabled=True,
            label_allowlist=None,
            strict_inference_anomaly=True,
            min_valid_mlips=3,
        )

    def test_apply_dev_run_frame_cap_truncates_rows_when_enabled(self) -> None:
        cfg = SimpleNamespace(dev_run=True)
        wide_df = pd.DataFrame(
            {
                "reaction": [f"r{i}" for i in range(40)],
                "reference_ads_eng": np.arange(40, dtype=float),
                "model_a_mlip_ads_eng_median": np.arange(40, dtype=float),
            }
        )

        capped = _apply_dev_run_frame_cap(cfg, wide_df)

        self.assertEqual(len(capped), 24)
        self.assertEqual(capped["reaction"].tolist()[0], "r0")
        self.assertEqual(capped["reaction"].tolist()[-1], "r23")

    def test_apply_dev_run_curve_overrides_sets_single_smoke_sweep(self) -> None:
        learning_curve_cfg = SimpleNamespace(
            n_repeats=30,
            sweep_sizes=[],
            sweep_fractions=[0.1, 0.2],
            min_train=5,
            max_train=50,
            step=1,
            models=SimpleNamespace(
                moe=SimpleNamespace(
                    tuning=SimpleNamespace(optuna=OptunaTuningConfig(n_trials=50))
                ),
                probe_gnn=SimpleNamespace(
                    tuning=SimpleNamespace(optuna=OptunaTuningConfig(n_trials=5))
                ),
                gnn_direct=SimpleNamespace(
                    tuning=SimpleNamespace(optuna=OptunaTuningConfig(n_trials=2))
                ),
            ),
        )
        screening_cfg = SimpleNamespace(
            screen_fraction=0.2,
            min_screen_size=1,
        )
        cfg = SimpleNamespace(
            dev_run=True,
            experiment=SimpleNamespace(
                learning_curve=learning_curve_cfg,
                screening=screening_cfg,
            ),
        )

        _apply_dev_run_curve_overrides(cfg, n_samples=24)

        self.assertEqual(learning_curve_cfg.n_repeats, 1)
        self.assertEqual(learning_curve_cfg.sweep_sizes, [8])
        self.assertEqual(learning_curve_cfg.sweep_fractions, [])
        self.assertEqual(learning_curve_cfg.min_train, 8)
        self.assertEqual(learning_curve_cfg.max_train, 8)
        self.assertEqual(learning_curve_cfg.models.moe.tuning.optuna.n_trials, 3)
        self.assertEqual(learning_curve_cfg.models.probe_gnn.tuning.optuna.n_trials, 3)
        self.assertEqual(learning_curve_cfg.models.gnn_direct.tuning.optuna.n_trials, 2)
        self.assertEqual(screening_cfg.screen_fraction, 0.2)
        self.assertEqual(screening_cfg.min_screen_size, 1)

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
                dataset_profile=SimpleNamespace(tag="mamun_oh"),
                datasets={
                    "mamun_oh": SimpleNamespace(),
                    "khlohc": SimpleNamespace(),
                    "rodrigo": SimpleNamespace(),
                },
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
            ) as mock_learning_curve_plot, patch(
                "oasis.experiment_runner.write_zero_shot_rmse_stage_plot",
                return_value=tmp_path / "plots" / "zero_shot_rmse_stage_anomalyaware_on.png",
            ) as mock_zero_shot_plot, patch(
                "oasis.experiment_runner.write_zero_shot_stage_parity_plots",
                return_value=[],
            ), patch(
                "oasis.experiment_runner.write_all_datasets_zero_shot_rmse_stage_plot",
                return_value=(
                    tmp_path
                    / "plots"
                    / "zero_shot_rmse_stage_all_datasets_anomalyaware_on.png"
                ),
            ) as mock_all_datasets_zero_shot_plot, patch(
                "oasis.experiment_runner.write_all_datasets_oracle_learning_curve_plot",
                return_value=(
                    tmp_path
                    / "plots"
                    / "learning_curve_oracle_all_datasets_anomalyaware_on.png"
                ),
            ) as mock_all_datasets_oracle_plot:
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
            mock_zero_shot_plot.call_args.kwargs["output_dir"],
            tmp_path / "plots",
        )
        self.assertEqual(
            mock_zero_shot_plot.call_args.kwargs["run_suffix"],
            "anomalyaware_on",
        )
        self.assertEqual(
            mock_all_datasets_oracle_plot.call_args.kwargs["output_dir"],
            tmp_path / "plots",
        )
        self.assertEqual(
            mock_all_datasets_oracle_plot.call_args.kwargs["run_suffix"],
            "anomalyaware_on",
        )
        self.assertEqual(
            mock_all_datasets_oracle_plot.call_args.kwargs["enabled_method_names"],
            [],
        )
        self.assertIsNone(mock_all_datasets_oracle_plot.call_args.kwargs["min_x"])
        self.assertIsNone(mock_all_datasets_oracle_plot.call_args.kwargs["max_x"])
        self.assertIsNone(mock_all_datasets_oracle_plot.call_args.kwargs["include_x"])
        self.assertEqual(
            mock_results.call_args.args[1].experiment.learning_curve.results_bundle_path,
            tmp_path / "results_anomalyaware_on_latent_off_n2.json",
        )
        self.assertEqual(
            mock_results.call_args.args[1].experiment.learning_curve.graph_dataset.path,
            tmp_path / "aligned_graphs_anomalyaware_on_latent_off_n2.parquet",
        )
        self.assertEqual(
            cfg.analysis.comparison_plot_path,
            tmp_path / "comparison_anomalyaware_on.png",
        )

    def test_load_oracle_learning_curve_rows_for_dataset_filters_to_enabled_methods(
        self,
    ) -> None:
        cfg = SimpleNamespace(
            dataset_profile=SimpleNamespace(tag="bio_mass"),
            datasets={
                "bio_mass": SimpleNamespace(
                    mlip_run_dirname_or_default=lambda tag: "Bio-Mass"
                )
            },
            probe_features=None,
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(
                    models=SimpleNamespace(
                        use_ridge=True,
                        use_kernel_ridge=False,
                        use_lasso=False,
                        use_elastic_net=False,
                        use_residual=False,
                        use_weighted_linear=False,
                        use_weighted_simplex=False,
                        use_graph_mean=False,
                        use_gnn_direct=False,
                        use_probe_gnn=False,
                        use_latent=False,
                        moe=SimpleNamespace(enabled=False),
                        probe_gnn=SimpleNamespace(enabled=False),
                        gnn_direct=SimpleNamespace(enabled=False),
                    )
                )
            ),
        )
        fake_wide_df = _FakeWideFrame()
        results = LearningCurveResults(
            ridge_df=pd.DataFrame(
                {
                    "n_train": [2, 4],
                    "rmse_mean": [0.35, 0.30],
                    "rmse_std": [0.02, 0.02],
                }
            ),
            latent_df=pd.DataFrame(
                {
                    "n_train": [2, 4],
                    "rmse_mean": [0.10, 0.09],
                    "rmse_std": [0.01, 0.01],
                }
            ),
        )
        enabled_method_names = ["ridge"]

        with patch(
            "oasis.experiment_runner.ensure_probe_artifacts",
            return_value=False,
        ), patch(
            "oasis.experiment_runner.load_filtered_wide_predictions",
            return_value=(fake_wide_df, [], fake_wide_df),
        ), patch(
            "oasis.experiment_runner.build_auxiliary_views",
            return_value=(fake_wide_df, {}),
        ), patch(
            "oasis.experiment_runner.prepare_graph_view",
            return_value=None,
        ), patch(
            "oasis.experiment_runner._apply_persistent_output_suffixes",
            return_value="anomalyaware_off_latent_off_n2",
        ), patch(
            "oasis.experiment_runner.load_or_run_learning_curve_results_from_config",
            return_value=results,
        ):
            rows = _load_oracle_learning_curve_rows_for_dataset(
                cfg,
                dataset_tag="bio_mass",
                enabled_method_names=enabled_method_names,
            )

        self.assertEqual(
            rows,
            [
                {
                    "dataset": "bio_mass",
                    "dataset_label": "Bio-Mass",
                    "n_train": 2,
                    "oracle_rmse": 0.35,
                    "oracle_method": "ridge",
                },
                {
                    "dataset": "bio_mass",
                    "dataset_label": "Bio-Mass",
                    "n_train": 4,
                    "oracle_rmse": 0.30,
                    "oracle_method": "ridge",
                },
            ],
        )

    def test_load_oracle_learning_curve_rows_for_dataset_resolves_fraction_span_per_dataset(
        self,
    ) -> None:
        cfg = SimpleNamespace(
            dataset_profile=SimpleNamespace(tag="bio_mass"),
            datasets={
                "bio_mass": SimpleNamespace(
                    mlip_run_dirname_or_default=lambda tag: "Bio-Mass"
                ),
                "khlohc": SimpleNamespace(
                    mlip_run_dirname_or_default=lambda tag: "KHLOHC-TOL"
                ),
            },
            probe_features=None,
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(models=SimpleNamespace())
            ),
        )
        small_wide_df = _FakeWideFrame(reactions=["r0", "r1", "r2", "r3"])
        large_wide_df = _FakeWideFrame(reactions=["r0", "r1", "r2", "r3", "r4", "r5"])
        results = LearningCurveResults(
            ridge_df=pd.DataFrame(
                {
                    "n_train": [2, 3, 4, 6],
                    "rmse_mean": [0.40, 0.35, 0.30, 0.25],
                    "rmse_std": [0.01, 0.01, 0.01, 0.01],
                }
            )
        )

        def _wide_df_for_cfg(dataset_cfg):
            tag = dataset_cfg.dataset_profile.tag
            wide_df = small_wide_df if tag == "bio_mass" else large_wide_df
            return (wide_df, [], wide_df)

        def _resolved_sizes(n_samples, **kwargs):
            if n_samples == 4:
                return [2, 4]
            if n_samples == 6:
                return [3, 6]
            raise AssertionError(f"Unexpected n_samples={n_samples}")

        with patch(
            "oasis.experiment_runner.ensure_probe_artifacts",
            return_value=False,
        ), patch(
            "oasis.experiment_runner.load_filtered_wide_predictions",
            side_effect=_wide_df_for_cfg,
        ), patch(
            "oasis.experiment_runner.build_auxiliary_views",
            side_effect=lambda dataset_cfg, wide_df, probe_gnn_enabled: (wide_df, {}),
        ), patch(
            "oasis.experiment_runner.prepare_graph_view",
            return_value=None,
        ), patch(
            "oasis.experiment_runner._apply_persistent_output_suffixes",
            return_value="anomalyaware_off_latent_off_n2",
        ), patch(
            "oasis.experiment_runner.load_or_run_learning_curve_results_from_config",
            return_value=results,
        ), patch(
            "oasis.experiment_runner.resolve_configured_sweep_sizes",
            side_effect=_resolved_sizes,
        ):
            small_rows = _load_oracle_learning_curve_rows_for_dataset(
                cfg,
                dataset_tag="bio_mass",
                enabled_method_names=["ridge"],
                span_variant=BudgetSpanVariant(
                    key="fraction",
                    output_suffix="fraction",
                    sweep_fractions=(0.5, 1.0),
                ),
            )
            large_rows = _load_oracle_learning_curve_rows_for_dataset(
                cfg,
                dataset_tag="khlohc",
                enabled_method_names=["ridge"],
                span_variant=BudgetSpanVariant(
                    key="fraction",
                    output_suffix="fraction",
                    sweep_fractions=(0.5, 1.0),
                ),
            )

        self.assertEqual([row["n_train"] for row in small_rows], [2, 4])
        self.assertEqual([row["n_train"] for row in large_rows], [3, 6])

    def test_load_all_datasets_oracle_learning_curve_rows_preserves_dataset_order(
        self,
    ) -> None:
        cfg = SimpleNamespace(
            dataset_profile=SimpleNamespace(tag="rodrigo"),
            datasets={
                "mamun_oh": SimpleNamespace(),
                "khlohc": SimpleNamespace(),
            },
        )

        with patch(
            "oasis.experiment_runner._load_oracle_learning_curve_rows_for_dataset",
            side_effect=[
                [{"dataset": "rodrigo", "n_train": 2, "oracle_rmse": 0.3}],
                [{"dataset": "mamun_oh", "n_train": 2, "oracle_rmse": 0.2}],
                [{"dataset": "khlohc", "n_train": 2, "oracle_rmse": 0.1}],
            ],
        ) as mock_load_rows:
            rows = load_all_datasets_oracle_learning_curve_rows(
                cfg=cfg,
                enabled_method_names=["ridge"],
            )

        self.assertEqual(
            [call.kwargs["dataset_tag"] for call in mock_load_rows.call_args_list],
            ["rodrigo", "mamun_oh", "khlohc"],
        )
        self.assertEqual(
            [call.kwargs["enabled_method_names"] for call in mock_load_rows.call_args_list],
            [["ridge"], ["ridge"], ["ridge"]],
        )
        self.assertEqual(
            [call.kwargs["span_variant"] for call in mock_load_rows.call_args_list],
            [None, None, None],
        )
        self.assertEqual(
            [call.kwargs["cache_only"] for call in mock_load_rows.call_args_list],
            [True, True, True],
        )
        self.assertEqual(
            [row["dataset"] for row in rows],
            ["rodrigo", "mamun_oh", "khlohc"],
        )

    def test_load_all_datasets_oracle_learning_curve_rows_uses_explicit_enabled_methods(
        self,
    ) -> None:
        cfg = SimpleNamespace(
            dataset_profile=SimpleNamespace(tag="bio_mass"),
            datasets={
                "bio_mass": SimpleNamespace(
                    mlip_run_dirname_or_default=lambda tag: "Bio-Mass"
                ),
                "khlohc": SimpleNamespace(
                    mlip_run_dirname_or_default=lambda tag: "KHLOHC-TOL"
                ),
            },
            probe_features=None,
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(
                    models=SimpleNamespace(
                        use_ridge=False,
                        use_kernel_ridge=False,
                        use_lasso=False,
                        use_elastic_net=False,
                        use_residual=False,
                        use_weighted_linear=False,
                        use_weighted_simplex=False,
                        use_graph_mean=False,
                        use_gnn_direct=False,
                        use_probe_gnn=False,
                        use_latent=False,
                        moe=SimpleNamespace(enabled=False),
                        probe_gnn=SimpleNamespace(enabled=False),
                        gnn_direct=SimpleNamespace(enabled=False),
                    )
                )
            ),
        )

        fake_wide_df = _FakeWideFrame()
        results = LearningCurveResults(
            ridge_df=pd.DataFrame(
                {
                    "n_train": [2, 4],
                    "rmse_mean": [0.35, 0.30],
                    "rmse_std": [0.02, 0.02],
                }
            ),
            latent_df=pd.DataFrame(
                {
                    "n_train": [2, 4],
                    "rmse_mean": [0.10, 0.09],
                    "rmse_std": [0.01, 0.01],
                }
            ),
        )

        with patch(
            "oasis.experiment_runner.ensure_probe_artifacts",
            return_value=False,
        ), patch(
            "oasis.experiment_runner.load_filtered_wide_predictions",
            return_value=(fake_wide_df, [], fake_wide_df),
        ), patch(
            "oasis.experiment_runner.build_auxiliary_views",
            return_value=(fake_wide_df, {}),
        ), patch(
            "oasis.experiment_runner._apply_persistent_output_suffixes",
            return_value="anomalyaware_off_latent_off_n2",
        ), patch(
            "oasis.experiment_runner._load_cached_learning_curve_results_for_dataset_cfg",
            return_value=results,
        ):
            rows = load_all_datasets_oracle_learning_curve_rows(
                cfg=cfg,
                enabled_method_names=["ridge"],
            )

        self.assertEqual(
            rows,
            [
                {
                    "dataset": "bio_mass",
                    "dataset_label": "Bio-Mass",
                    "n_train": 2,
                    "oracle_rmse": 0.35,
                    "oracle_method": "ridge",
                },
                {
                    "dataset": "bio_mass",
                    "dataset_label": "Bio-Mass",
                    "n_train": 4,
                    "oracle_rmse": 0.30,
                    "oracle_method": "ridge",
                },
                {
                    "dataset": "khlohc",
                    "dataset_label": "KHLOHC-TOL",
                    "n_train": 2,
                    "oracle_rmse": 0.35,
                    "oracle_method": "ridge",
                },
                {
                    "dataset": "khlohc",
                    "dataset_label": "KHLOHC-TOL",
                    "n_train": 4,
                    "oracle_rmse": 0.30,
                    "oracle_method": "ridge",
                },
            ],
        )

    def test_load_all_datasets_oracle_uq_rows_selects_per_dataset_rmse_oracle(
        self,
    ) -> None:
        cfg = SimpleNamespace(
            dataset_profile=SimpleNamespace(tag="bio_mass"),
            datasets={
                "bio_mass": SimpleNamespace(
                    mlip_run_dirname_or_default=lambda tag: "Bio-Mass"
                ),
            },
            probe_features=None,
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(
                    models=SimpleNamespace(
                        use_ridge=False,
                        use_kernel_ridge=False,
                        use_lasso=False,
                        use_elastic_net=False,
                        use_residual=False,
                        use_weighted_linear=False,
                        use_weighted_simplex=False,
                        use_graph_mean=False,
                        use_gnn_direct=False,
                        use_probe_gnn=False,
                        use_latent=False,
                        moe=SimpleNamespace(enabled=False),
                        probe_gnn=SimpleNamespace(enabled=False),
                        gnn_direct=SimpleNamespace(enabled=False),
                    )
                )
            ),
        )
        fake_wide_df = _FakeWideFrame(reactions=["r0", "r1", "r2", "r3"])
        results = LearningCurveResults(
            ridge_df=pd.DataFrame(
                {
                    "n_train": [2, 4],
                    "rmse_mean": [0.10, 0.12],
                    "rmse_std": [0.0, 0.0],
                }
            ),
            ridge_uq_df=pd.DataFrame(
                {
                    "n_train": [2, 4],
                    "miscalibration_area": [0.20, 0.10],
                    "sharpness": [0.30, 0.25],
                    "dispersion": [0.40, 0.35],
                }
            ),
            probe_gnn_df=pd.DataFrame(
                {
                    "n_train": [2, 4],
                    "rmse_mean": [0.11, 0.09],
                    "rmse_std": [0.0, 0.0],
                }
            ),
            probe_gnn_uq_df=pd.DataFrame(
                {
                    "n_train": [2, 4],
                    "miscalibration_area": [0.15, 0.12],
                    "sharpness": [0.45, 0.22],
                    "dispersion": [0.55, 0.32],
                }
            ),
        )

        with patch(
            "oasis.experiment_runner.ensure_probe_artifacts",
            return_value=False,
        ), patch(
            "oasis.experiment_runner.load_filtered_wide_predictions",
            return_value=(fake_wide_df, [], fake_wide_df),
        ), patch(
            "oasis.experiment_runner.build_auxiliary_views",
            return_value=(fake_wide_df, {}),
        ), patch(
            "oasis.experiment_runner._apply_persistent_output_suffixes",
            return_value="anomalyaware_off_latent_off_n4",
        ), patch(
            "oasis.experiment_runner._load_cached_learning_curve_results_for_dataset_cfg",
            return_value=results,
        ):
            rows = _load_all_datasets_oracle_uq_rows(
                cfg,
                dataset_tag="bio_mass",
                enabled_method_names=["ridge", "probe_gnn"],
            )

        self.assertEqual(
            rows,
            [
                {
                    "dataset": "bio_mass",
                    "dataset_label": "Bio-Mass",
                    "n_train": 2,
                    "oracle_miscalibration_area": 0.20,
                    "oracle_sharpness": 0.30,
                    "oracle_dispersion": 0.40,
                    "oracle_method": "ridge",
                },
                {
                    "dataset": "bio_mass",
                    "dataset_label": "Bio-Mass",
                    "n_train": 4,
                    "oracle_miscalibration_area": 0.12,
                    "oracle_sharpness": 0.22,
                    "oracle_dispersion": 0.32,
                    "oracle_method": "probe_gnn",
                },
            ],
        )

    def test_load_policy_regret_rows_for_dataset_applies_fraction_variant_per_dataset(
        self,
    ) -> None:
        cfg = SimpleNamespace(
            dataset_profile=SimpleNamespace(tag="bio_mass"),
            datasets={
                "bio_mass": SimpleNamespace(
                    mlip_run_dirname_or_default=lambda tag: "Bio-Mass"
                ),
                "khlohc": SimpleNamespace(
                    mlip_run_dirname_or_default=lambda tag: "KHLOHC-TOL"
                ),
            },
            probe_features=None,
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(
                    min_train=None,
                    max_train=None,
                    step=1,
                    n_repeats=1,
                    models=SimpleNamespace(),
                ),
                screening=SimpleNamespace(screen_fraction=0.25),
            ),
        )
        small_wide_df = _FakeWideFrame(reactions=["r0", "r1", "r2", "r3"])
        large_wide_df = _FakeWideFrame(reactions=["r0", "r1", "r2", "r3", "r4", "r5"])
        def _wide_df_for_cfg(dataset_cfg):
            return (
                (small_wide_df, [], small_wide_df)
                if dataset_cfg.dataset_profile.tag == "bio_mass"
                else (large_wide_df, [], large_wide_df)
            )

        def _diagnostic_results_for_cfg(*, wide_df, **kwargs):
            budgets = [2, 4] if len(wide_df) == 4 else [3, 6]
            return PolicyDiagnosticBuildOutputs(
                results=PolicySelectionDiagnosticResults(
                    detail_df=pd.DataFrame(
                        {
                            "policy_name": ["min_screening_rmse", "min_screening_rmse"],
                            "budget": budgets,
                            "repeat": [0, 0],
                            "oracle_method": ["ridge", "ridge"],
                            "screening_selected_method": ["ridge", "ridge"],
                            "oracle_outer_rmse": [0.2, 0.2],
                            "screening_selected_outer_rmse": [0.2, 0.2],
                            "regret": [0.0, 0.0],
                            "screening_cv_rmse": [0.1, 0.1],
                            "screening_miscalibration_area": [0.05, 0.05],
                            "agreement": [True, True],
                        }
                    ),
                    outer_metrics_df=pd.DataFrame(
                        {
                            "budget": budgets,
                            "repeat": [0, 0],
                            "method": ["ridge", "ridge"],
                            "outer_test_rmse": [0.2, 0.2],
                        }
                    ),
                    summary_df=pd.DataFrame(
                        {
                            "policy_name": ["min_screening_rmse", "min_screening_rmse"],
                            "budget": budgets,
                            "mean_regret": [0.01, 0.02],
                            "std_regret": [0.0, 0.0],
                            "se_regret": [0.0, 0.0],
                            "ci95_low": [0.01, 0.02],
                            "ci95_high": [0.01, 0.02],
                            "agreement_rate": [1.0, 1.0],
                            "oracle_outer_rmse_mean": [0.2, 0.2],
                            "screening_selected_outer_rmse_mean": [0.21, 0.22],
                        }
                    ),
                ),
                screening_rows_df=pd.DataFrame(
                    {
                        "method": ["ridge", "ridge"],
                        "budget": budgets,
                        "repeat": [0, 0],
                        "split_fingerprint": ["fp-a", "fp-b"],
                        "screening_cv_rmse": [0.1, 0.1],
                        "screening_miscalibration_area": [0.05, 0.05],
                    }
                ),
            )

        def _resolved_sizes(n_samples, **kwargs):
            if n_samples == 4:
                return [2, 4]
            if n_samples == 6:
                return [3, 6]
            raise AssertionError(f"Unexpected n_samples={n_samples}")

        with patch(
            "oasis.experiment_runner.ensure_probe_artifacts",
            return_value=False,
        ), patch(
            "oasis.experiment_runner.load_filtered_wide_predictions",
            side_effect=_wide_df_for_cfg,
        ), patch(
            "oasis.experiment_runner.build_auxiliary_views",
            side_effect=lambda dataset_cfg, wide_df, probe_gnn_enabled: (wide_df, {}),
        ), patch(
            "oasis.experiment_runner.prepare_graph_view",
            return_value=None,
        ), patch(
            "oasis.experiment_runner._apply_persistent_output_suffixes",
            return_value="anomalyaware_off_latent_off_n2",
        ), patch(
            "oasis.experiment_runner._load_cached_policy_selection_results_for_dataset_cfg",
            return_value=None,
        ), patch(
            "oasis.experiment_runner._build_policy_selection_diagnostic_results_for_cfg",
            side_effect=_diagnostic_results_for_cfg,
        ), patch(
            "oasis.experiment_runner.resolve_configured_sweep_sizes",
            side_effect=_resolved_sizes,
        ):
            small_rows = _load_policy_regret_rows_for_dataset(
                cfg,
                dataset_tag="bio_mass",
                span_variant=BudgetSpanVariant(
                    key="fraction",
                    output_suffix="fraction",
                    sweep_fractions=(0.5, 1.0),
                ),
            )
            large_rows = _load_policy_regret_rows_for_dataset(
                cfg,
                dataset_tag="khlohc",
                span_variant=BudgetSpanVariant(
                    key="fraction",
                    output_suffix="fraction",
                    sweep_fractions=(0.5, 1.0),
                ),
            )

        self.assertEqual([row["budget"] for row in small_rows], [2, 4])
        self.assertEqual([row["budget"] for row in large_rows], [3, 6])
        self.assertEqual(
            [row["dataset_label"] for row in large_rows],
            ["KHLOHC-TOL", "KHLOHC-TOL"],
        )

    def test_load_policy_regret_rows_for_dataset_reuses_cached_policy_artifact(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_dir = tmp_path / "plots"
            output_dir.mkdir(parents=True, exist_ok=True)
            cfg = SimpleNamespace(
                dataset_profile=SimpleNamespace(tag="bio_mass"),
                datasets={
                    "bio_mass": SimpleNamespace(
                        mlip_run_dirname_or_default=lambda tag: "Bio-Mass"
                    )
                },
                probe_features=None,
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        min_train=2,
                        max_train=4,
                        step=2,
                        n_repeats=1,
                        min_test_size=1,
                        validation_fraction=0.2,
                        min_val_size=1,
                        min_tuning_val_size=1,
                        calibration_enabled=False,
                        calibration_fraction=0.2,
                        min_cal_size=1,
                        min_inner_train_size=1,
                        sweep_sizes=[],
                        sweep_fractions=[],
                        models=SimpleNamespace(
                            use_ridge=True,
                            use_kernel_ridge=False,
                            use_lasso=False,
                            use_elastic_net=False,
                            use_residual=False,
                            use_weighted_linear=False,
                            use_weighted_simplex=False,
                            use_graph_mean=False,
                            use_latent=False,
                            moe=SimpleNamespace(enabled=False),
                            probe_gnn=SimpleNamespace(enabled=False),
                            gnn_direct=SimpleNamespace(enabled=False),
                        ),
                    ),
                    screening=SimpleNamespace(
                        screen_fraction=0.25,
                        min_screen_size=1,
                        validation_fraction=0.3,
                        min_val_size=1,
                        min_tuning_val_size=1,
                        calibration_enabled=False,
                        calibration_fraction=0.2,
                        min_cal_size=1,
                        min_inner_train_size=1,
                        policy_names=["min_screening_rmse"],
                        combined_miscalibration_lambda=1.0,
                    ),
                ),
                plot=SimpleNamespace(output_dir=output_dir),
            )
            fake_wide_df = _FakeWideFrame(reactions=["r0", "r1", "r2", "r3"])
            artifact_path = tmp_path / "policy_selection_diagnostic.json"
            artifact_path.write_text("cached", encoding="utf-8")
            diagnostic_results = PolicySelectionDiagnosticResults(
                detail_df=pd.DataFrame(
                    {
                        "policy_name": ["min_screening_rmse"],
                        "budget": [2],
                        "repeat": [0],
                        "oracle_method": ["ridge"],
                        "screening_selected_method": ["ridge"],
                        "oracle_outer_rmse": [0.2],
                        "screening_selected_outer_rmse": [0.2],
                        "regret": [0.0],
                        "screening_cv_rmse": [0.1],
                        "screening_miscalibration_area": [0.05],
                        "agreement": [True],
                    }
                ),
                outer_metrics_df=pd.DataFrame(
                    {
                        "budget": [2],
                        "repeat": [0],
                        "method": ["ridge"],
                        "outer_test_rmse": [0.2],
                    }
                ),
                summary_df=pd.DataFrame(
                    {
                        "policy_name": ["min_screening_rmse"],
                        "budget": [2],
                        "mean_regret": [0.0],
                        "std_regret": [0.0],
                        "se_regret": [0.0],
                        "ci95_low": [0.0],
                        "ci95_high": [0.0],
                        "agreement_rate": [1.0],
                        "oracle_outer_rmse_mean": [0.2],
                        "screening_selected_outer_rmse_mean": [0.2],
                    }
                ),
            )

            with patch(
                "oasis.experiment_runner.policy_selection_diagnostic_bundle_path",
                return_value=artifact_path,
            ), patch(
                "oasis.experiment_runner.policy_selection_screening_rows_bundle_path",
                return_value=tmp_path / "policy_selection_screening_rows.json",
            ), patch(
                "oasis.experiment_runner.ensure_probe_artifacts",
                return_value=False,
            ), patch(
                "oasis.experiment_runner.load_filtered_wide_predictions",
                return_value=(fake_wide_df, [], fake_wide_df),
            ), patch(
                "oasis.experiment_runner.build_auxiliary_views",
                return_value=(fake_wide_df, {}),
            ), patch(
                "oasis.experiment_runner._load_cached_policy_selection_results_for_dataset_cfg",
                return_value=diagnostic_results,
            ) as mock_load, patch(
                "oasis.experiment_runner._build_policy_selection_diagnostic_results_for_cfg",
            ) as mock_build:
                rows = _load_policy_regret_rows_for_dataset(
                    cfg,
                    dataset_tag="bio_mass",
                )

        self.assertEqual([row["budget"] for row in rows], [2])
        mock_load.assert_called_once()
        mock_build.assert_not_called()

    def test_load_all_datasets_policy_regret_rows_preserves_dataset_order(self) -> None:
        cfg = SimpleNamespace(
            dataset_profile=SimpleNamespace(tag="rodrigo"),
            datasets={
                "mamun_oh": SimpleNamespace(),
                "khlohc": SimpleNamespace(),
            },
        )

        with patch(
            "oasis.experiment_runner._load_policy_regret_rows_for_dataset",
            side_effect=[
                [{"dataset": "rodrigo", "budget": 2, "mean_regret": 0.3}],
                [{"dataset": "mamun_oh", "budget": 2, "mean_regret": 0.2}],
                [{"dataset": "khlohc", "budget": 2, "mean_regret": 0.1}],
            ],
        ) as mock_load_rows:
            rows = load_all_datasets_policy_regret_rows(cfg=cfg)

        self.assertEqual(
            [call.kwargs["dataset_tag"] for call in mock_load_rows.call_args_list],
            ["rodrigo", "mamun_oh", "khlohc"],
        )
        self.assertEqual(
            [call.kwargs["span_variant"] for call in mock_load_rows.call_args_list],
            [None, None, None],
        )
        self.assertEqual(
            [call.kwargs["cache_only"] for call in mock_load_rows.call_args_list],
            [True, True, True],
        )
        self.assertEqual(
            [row["dataset"] for row in rows],
            ["rodrigo", "mamun_oh", "khlohc"],
        )

    def test_load_all_datasets_oracle_learning_curve_rows_skips_cache_misses(self) -> None:
        cfg = SimpleNamespace(
            dataset_profile=SimpleNamespace(tag="bio_mass"),
            datasets={
                "bio_mass": SimpleNamespace(),
                "khlohc": SimpleNamespace(),
            },
        )

        with patch(
            "oasis.experiment_runner._load_oracle_learning_curve_rows_for_dataset",
            side_effect=[
                [{"dataset": "bio_mass", "n_train": 2, "oracle_rmse": 0.3}],
                [],
            ],
        ) as mock_load_rows:
            rows = load_all_datasets_oracle_learning_curve_rows(
                cfg=cfg,
                enabled_method_names=["ridge"],
            )

        self.assertEqual(rows, [{"dataset": "bio_mass", "n_train": 2, "oracle_rmse": 0.3}])
        self.assertEqual(mock_load_rows.call_count, 2)

    def test_load_all_datasets_policy_regret_rows_skips_cache_misses(self) -> None:
        cfg = SimpleNamespace(
            dataset_profile=SimpleNamespace(tag="bio_mass"),
            datasets={
                "bio_mass": SimpleNamespace(),
                "khlohc": SimpleNamespace(),
            },
        )

        with patch(
            "oasis.experiment_runner._load_policy_regret_rows_for_dataset",
            side_effect=[
                [{"dataset": "bio_mass", "budget": 2, "mean_regret": 0.1}],
                [],
            ],
        ) as mock_load_rows:
            rows = load_all_datasets_policy_regret_rows(cfg=cfg)

        self.assertEqual(rows, [{"dataset": "bio_mass", "budget": 2, "mean_regret": 0.1}])
        self.assertEqual(mock_load_rows.call_count, 2)

    def test_write_all_datasets_oracle_learning_curve_plot_skips_single_dataset(
        self,
    ) -> None:
        cfg = SimpleNamespace(
            dataset_profile=SimpleNamespace(tag="bio_mass"),
            datasets={"bio_mass": SimpleNamespace()},
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(models=SimpleNamespace())
            ),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with patch(
                "oasis.experiment_runner.oracle_learning_curve_plot"
            ) as mock_plot:
                saved_path = write_all_datasets_oracle_learning_curve_plot(
                    cfg=cfg,
                    output_dir=tmp_path,
                    run_suffix="anomalyaware_off",
                    enabled_method_names=["ridge"],
                )

        self.assertIsNone(saved_path)
        mock_plot.assert_not_called()

    def test_write_all_datasets_oracle_learning_curve_plot_forwards_log_x_toggle(
        self,
    ) -> None:
        cfg = SimpleNamespace(
            dataset_profile=SimpleNamespace(tag="bio_mass"),
            datasets={
                "bio_mass": SimpleNamespace(),
                "khlohc": SimpleNamespace(),
            },
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(models=SimpleNamespace())
            ),
            plot=SimpleNamespace(
                curve_window=SimpleNamespace(oracle_all_datasets_log_x=True)
            ),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with patch(
                "oasis.experiment_runner.load_all_datasets_oracle_learning_curve_rows",
                return_value=[
                    {
                        "dataset": "bio_mass",
                        "dataset_label": "Bio-Mass",
                        "n_train": 2,
                        "oracle_rmse": 0.35,
                        "oracle_method": "ridge",
                    },
                    {
                        "dataset": "khlohc",
                        "dataset_label": "KHLOHC-TOL",
                        "n_train": 2,
                        "oracle_rmse": 0.30,
                        "oracle_method": "ridge",
                    },
                ],
            ), patch(
                "oasis.experiment_runner.oracle_learning_curve_plot",
                return_value=tmp_path / "oracle.png",
            ) as mock_plot:
                saved_path = write_all_datasets_oracle_learning_curve_plot(
                    cfg=cfg,
                    output_dir=tmp_path,
                    run_suffix="anomalyaware_off",
                    enabled_method_names=["ridge"],
                )

        self.assertEqual(saved_path, tmp_path / "oracle.png")

    def test_write_all_datasets_oracle_learning_curve_plot_emits_dual_budget_span_outputs(
        self,
    ) -> None:
        cfg = SimpleNamespace(
            dataset_profile=SimpleNamespace(tag="bio_mass"),
            datasets={
                "bio_mass": SimpleNamespace(),
                "khlohc": SimpleNamespace(),
            },
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(
                    sweep_sizes=[1, 2],
                    sweep_fractions=[0.5, 1.0],
                    min_train=None,
                    max_train=None,
                    step=1,
                    models=SimpleNamespace(),
                )
            ),
            plot=SimpleNamespace(
                curve_window=SimpleNamespace(oracle_all_datasets_log_x=True)
            ),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with patch(
                "oasis.experiment_runner.load_all_datasets_oracle_learning_curve_rows",
                return_value=[
                    {
                        "dataset": "bio_mass",
                        "dataset_label": "Bio-Mass",
                        "n_train": 2,
                        "oracle_rmse": 0.35,
                        "oracle_method": "ridge",
                    }
                ],
            ) as mock_load_rows, patch(
                "oasis.experiment_runner.oracle_learning_curve_plot",
                side_effect=[
                    tmp_path / "oracle_absolute.png",
                    tmp_path / "oracle_fraction.png",
                ],
            ) as mock_plot:
                saved_path = write_all_datasets_oracle_learning_curve_plot(
                    cfg=cfg,
                    output_dir=tmp_path,
                    run_suffix="anomalyaware_off",
                    enabled_method_names=["ridge"],
                )

        self.assertEqual(saved_path, tmp_path / "oracle_absolute.png")
        self.assertEqual(mock_load_rows.call_count, 2)
        self.assertEqual(
            mock_plot.call_args_list[0].kwargs["output_path"],
            tmp_path / "learning_curve_oracle_all_datasets_anomalyaware_off_absolute.png",
        )
        self.assertEqual(
            mock_plot.call_args_list[1].kwargs["output_path"],
            tmp_path / "learning_curve_oracle_all_datasets_anomalyaware_off_fraction.png",
        )
        self.assertEqual(
            [call.kwargs["span_variant"].key for call in mock_load_rows.call_args_list],
            ["absolute", "fraction"],
        )
        self.assertFalse(mock_plot.call_args_list[0].kwargs["log_x"])
        self.assertTrue(mock_plot.call_args_list[1].kwargs["log_x"])

    def test_write_all_datasets_policy_regret_plot_emits_dual_budget_span_outputs(
        self,
    ) -> None:
        cfg = SimpleNamespace(
            dataset_profile=SimpleNamespace(tag="bio_mass"),
            datasets={
                "bio_mass": SimpleNamespace(),
                "khlohc": SimpleNamespace(),
            },
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(
                    sweep_sizes=[1, 2],
                    sweep_fractions=[0.5, 1.0],
                    min_train=None,
                    max_train=None,
                    step=1,
                    models=SimpleNamespace(),
                ),
                screening=SimpleNamespace(screen_fraction=0.25),
            ),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with patch(
                "oasis.experiment_runner.load_all_datasets_policy_regret_rows",
                return_value=[
                    {
                        "dataset": "bio_mass",
                        "dataset_label": "Bio-Mass",
                        "policy_name": "min_screening_rmse",
                        "budget": 2,
                        "mean_regret": 0.01,
                        "std_regret": 0.0,
                        "se_regret": 0.0,
                        "ci95_low": 0.01,
                        "ci95_high": 0.01,
                        "agreement_rate": 1.0,
                        "oracle_outer_rmse_mean": 0.2,
                        "screening_selected_outer_rmse_mean": 0.21,
                    }
                ],
            ) as mock_load_rows, patch(
                "oasis.experiment_runner.all_datasets_policy_regret_plot",
                side_effect=[
                    tmp_path / "regret_absolute.png",
                    tmp_path / "regret_fraction.png",
                ],
            ) as mock_plot:
                saved_path = write_all_datasets_policy_regret_plot(
                    cfg=cfg,
                    output_dir=tmp_path,
                    run_suffix="anomalyaware_off",
                )

        self.assertEqual(saved_path, tmp_path / "regret_absolute.png")
        self.assertEqual(mock_load_rows.call_count, 2)
        self.assertEqual(
            mock_plot.call_args_list[0].kwargs["output_path"],
            tmp_path / "policy_regret_all_datasets_anomalyaware_off_absolute.png",
        )
        self.assertEqual(
            mock_plot.call_args_list[1].kwargs["output_path"],
            tmp_path / "policy_regret_all_datasets_anomalyaware_off_fraction.png",
        )
        self.assertEqual(
            [call.kwargs["span_variant"].key for call in mock_load_rows.call_args_list],
            ["absolute", "fraction"],
        )
        self.assertFalse(mock_plot.call_args_list[0].kwargs["log_x"])
        self.assertTrue(mock_plot.call_args_list[1].kwargs["log_x"])

    def test_write_all_datasets_uq_oracle_plots_emit_dual_budget_span_outputs(
        self,
    ) -> None:
        cfg = SimpleNamespace(
            dataset_profile=SimpleNamespace(tag="bio_mass"),
            datasets={
                "bio_mass": SimpleNamespace(),
                "khlohc": SimpleNamespace(),
            },
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(
                    sweep_sizes=[1, 2],
                    sweep_fractions=[0.5, 1.0],
                    min_train=None,
                    max_train=None,
                    step=1,
                    models=SimpleNamespace(),
                )
            ),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with patch(
                "oasis.experiment_runner.load_all_datasets_oracle_uq_rows",
                return_value=[
                    {
                        "dataset": "bio_mass",
                        "dataset_label": "Bio-Mass",
                        "n_train": 2,
                        "oracle_miscalibration_area": 0.15,
                        "oracle_sharpness": 0.45,
                        "oracle_dispersion": 0.55,
                        "oracle_method": "probe_gnn",
                    }
                ],
            ) as mock_load_rows, patch(
                "oasis.experiment_runner.all_datasets_uq_oracle_plot",
                side_effect=[
                    tmp_path / "miscal_absolute.png",
                    tmp_path / "miscal_fraction.png",
                    tmp_path / "sharp_absolute.png",
                    tmp_path / "sharp_fraction.png",
                    tmp_path / "disp_absolute.png",
                    tmp_path / "disp_fraction.png",
                ],
            ) as mock_plot:
                saved_paths = write_all_datasets_uq_oracle_plots(
                    cfg=cfg,
                    output_dir=tmp_path,
                    run_suffix="anomalyaware_off",
                    enabled_method_names=["ridge", "probe_gnn"],
                )

        self.assertEqual(
            saved_paths,
            {
                "miscalibration_area": tmp_path / "miscal_absolute.png",
                "sharpness": tmp_path / "sharp_absolute.png",
                "dispersion": tmp_path / "disp_absolute.png",
            },
        )
        self.assertEqual(mock_load_rows.call_count, 2)
        self.assertEqual(mock_plot.call_count, 6)
        self.assertEqual(
            mock_plot.call_args_list[0].kwargs["output_path"],
            tmp_path / "uq_oracle_all_datasets_miscalibration_area_anomalyaware_off_absolute.png",
        )
        self.assertEqual(
            mock_plot.call_args_list[1].kwargs["output_path"],
            tmp_path / "uq_oracle_all_datasets_miscalibration_area_anomalyaware_off_fraction.png",
        )

    def test_write_zero_shot_stage_parity_plots_writes_matched_and_anomaly_aware_views(
        self,
    ) -> None:
        cfg = SimpleNamespace(
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(
                    mlip_selection=SimpleNamespace(
                        exclude_anomalous_mlips=True,
                        minimum_quorum=2,
                    )
                )
            )
        )
        selected_wide_df = pl.DataFrame(
            {
                "reaction": ["r0", "r1", "r2"],
                "reference_ads_eng": [1.0, 2.0, 3.0],
                "mace_mlip_ads_eng_median": [1.1, 2.1, 3.1],
                "orb_mlip_ads_eng_median": [0.9, 1.9, 2.9],
                "mace_slab_conv": [0, 0, 0],
                "mace_ads_conv": [0, 0, 0],
                "mace_slab_move": [0, 0, 0],
                "mace_ads_move": [0, 0, 0],
                "mace_slab_seed": [0, 0, 0],
                "mace_ads_seed": [0, 0, 0],
                "mace_ads_eng_seed": [0, 0, 0],
                "mace_adsorbate_migration": [0, 0, 0],
                "mace_energy_anomaly": [0, 0, 0],
                "orb_slab_conv": [0, 1, 0],
                "orb_ads_conv": [0, 0, 0],
                "orb_slab_move": [0, 0, 0],
                "orb_ads_move": [0, 0, 0],
                "orb_slab_seed": [0, 0, 0],
                "orb_ads_seed": [0, 0, 0],
                "orb_ads_eng_seed": [0, 0, 0],
                "orb_adsorbate_migration": [0, 0, 0],
                "orb_energy_anomaly": [0, 0, 0],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with patch(
                "oasis.experiment_runner.parity_plot",
                side_effect=lambda *args, **kwargs: kwargs["output_path"],
            ) as mock_parity_plot:
                saved_paths = write_zero_shot_stage_parity_plots(
                    cfg=cfg,
                    selected_wide_df=selected_wide_df,
                    output_dir=tmp_path,
                    run_suffix="anomalyaware_on",
                )

        self.assertEqual(len(saved_paths), 2)
        self.assertEqual(
            mock_parity_plot.call_args_list[0].kwargs["output_path"],
            tmp_path / "mlips_vs_dft_parity_matched_subset_anomalyaware_on.png",
        )
        self.assertEqual(
            mock_parity_plot.call_args_list[0].kwargs["title"],
            "Parity plot (matched subset / all MLIPs)",
        )
        self.assertEqual(
            mock_parity_plot.call_args_list[1].kwargs["output_path"],
            tmp_path / "mlips_vs_dft_parity_anomaly_aware_anomalyaware_on.png",
        )
        self.assertEqual(
            mock_parity_plot.call_args_list[1].kwargs["title"],
            "Parity plot (matched subset / anomaly-aware selection)",
        )
        self.assertTrue(
            mock_parity_plot.call_args_list[1].kwargs["validity_mask_by_prediction"][
                "mace"
            ].all()
        )
        np.testing.assert_array_equal(
            mock_parity_plot.call_args_list[1].kwargs["validity_mask_by_prediction"][
                "orb"
            ],
            np.array([True, False, True]),
        )

    def test_write_zero_shot_rmse_stage_plot_forwards_swarm_toggle_from_config(
        self,
    ) -> None:
        cfg = SimpleNamespace(
            plot=SimpleNamespace(zero_shot_stage_show_lone_mlip_swarm=False),
            dataset_profile=SimpleNamespace(tag="example"),
            datasets={},
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(
                    mlip_selection=SimpleNamespace(
                        exclude_anomalous_mlips=True,
                        minimum_quorum=2,
                    )
                )
            ),
        )
        frame = pl.DataFrame(
            {
                "reaction": ["r0", "r1"],
                "reference_ads_eng": [0.0, 1.0],
                "a_mlip_ads_eng_median": [0.0, 1.0],
                "b_mlip_ads_eng_median": [1.0, 2.0],
                "a_slab_conv": [0, 0],
                "a_ads_conv": [0, 0],
                "a_slab_move": [0, 0],
                "a_ads_move": [0, 0],
                "a_slab_seed": [0, 0],
                "a_ads_seed": [0, 0],
                "a_ads_eng_seed": [0, 0],
                "a_adsorbate_migration": [0, 0],
                "b_slab_conv": [0, 0],
                "b_ads_conv": [0, 0],
                "b_slab_move": [0, 0],
                "b_ads_move": [0, 0],
                "b_slab_seed": [0, 0],
                "b_ads_seed": [0, 0],
                "b_ads_eng_seed": [0, 0],
                "b_adsorbate_migration": [0, 0],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with patch(
                "oasis.experiment_runner.zero_shot_rmse_stage_plot",
                return_value=tmp_path / "zero_shot_stage.png",
            ) as mock_plot:
                saved_path = write_zero_shot_rmse_stage_plot(
                    cfg=cfg,
                    raw_wide_df=frame,
                    selected_wide_df=frame,
                    output_dir=tmp_path,
                    run_suffix="anomalyaware_on",
                )

        self.assertEqual(saved_path, tmp_path / "zero_shot_stage.png")
        self.assertFalse(mock_plot.call_args.kwargs["show_lone_mlip_swarm"])

    def test_write_zero_shot_rmse_stage_plot_uses_cached_stage_rows(self) -> None:
        cfg = SimpleNamespace(
            plot=SimpleNamespace(zero_shot_stage_show_lone_mlip_swarm=False),
            dataset_profile=SimpleNamespace(tag="example"),
            datasets={},
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(
                    mlip_selection=SimpleNamespace(
                        exclude_anomalous_mlips=True,
                        minimum_quorum=2,
                    )
                )
            ),
        )
        frame = pl.DataFrame(
            {
                "reaction": ["r0"],
                "reference_ads_eng": [0.0],
                "a_mlip_ads_eng_median": [0.0],
            }
        )
        cached_stage_rows = [
            {
                "dataset": "example",
                "dataset_label": "example",
                "stage": "Full / all MLIPs",
                "rmse": 0.25,
                "n_samples": 1,
            }
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            result_file = tmp_path / "mlip.json"
            result_file.write_text("{}")
            artifact_path = tmp_path / "zero_shot_cache.json"
            artifact_path.write_text("{}")
            with patch(
                "oasis.experiment_runner._zero_shot_stage_artifact_path",
                return_value=artifact_path,
            ), patch(
                "oasis.experiment_runner._load_zero_shot_stage_rows_artifact",
                return_value=cached_stage_rows,
            ) as mock_load_artifact, patch(
                "oasis.experiment_runner._build_zero_shot_stage_rows",
                side_effect=AssertionError("should not rebuild cached zero-shot rows"),
            ), patch(
                "oasis.experiment_runner.zero_shot_rmse_stage_plot",
                return_value=tmp_path / "zero_shot_stage.png",
            ) as mock_plot:
                saved_path = write_zero_shot_rmse_stage_plot(
                    cfg=cfg,
                    raw_wide_df=frame,
                    selected_wide_df=frame,
                    result_files=[result_file],
                    output_dir=tmp_path,
                    run_suffix="anomalyaware_on",
                )

        self.assertEqual(saved_path, tmp_path / "zero_shot_stage.png")
        mock_load_artifact.assert_called_once()
        pd.testing.assert_frame_equal(
            mock_plot.call_args.args[0].reset_index(drop=True),
            pd.DataFrame(cached_stage_rows),
        )

    def test_write_all_datasets_zero_shot_rmse_stage_plot_forwards_swarm_toggle_from_config(
        self,
    ) -> None:
        cfg = SimpleNamespace(
            plot=SimpleNamespace(zero_shot_stage_show_lone_mlip_swarm=False),
            dataset_profile=SimpleNamespace(tag="bio_mass"),
            datasets={
                "bio_mass": SimpleNamespace(),
                "khlohc": SimpleNamespace(),
            },
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(
                    mlip_selection=SimpleNamespace(
                        exclude_anomalous_mlips=True,
                        minimum_quorum=2,
                    )
                )
            ),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with patch(
                "oasis.experiment_runner._load_zero_shot_stage_rows_for_dataset",
                side_effect=[
                    [
                        {
                            "dataset": "bio_mass",
                            "dataset_label": "Bio-Mass",
                            "stage": "Full / all MLIPs",
                            "rmse": 0.4,
                            "n_samples": 10,
                        }
                    ],
                    [
                        {
                            "dataset": "khlohc",
                            "dataset_label": "KHLOHC-TOL",
                            "stage": "Full / all MLIPs",
                            "rmse": 0.5,
                            "n_samples": 12,
                        }
                    ],
                ],
            ), patch(
                "oasis.experiment_runner.zero_shot_rmse_stage_plot",
                return_value=tmp_path / "zero_shot_stage_all.png",
            ) as mock_plot:
                saved_path = write_all_datasets_zero_shot_rmse_stage_plot(
                    cfg=cfg,
                    output_dir=tmp_path,
                    run_suffix="anomalyaware_on",
                )

        self.assertEqual(saved_path, tmp_path / "zero_shot_stage_all.png")
        self.assertFalse(mock_plot.call_args.kwargs["show_lone_mlip_swarm"])

    def test_write_all_datasets_zero_shot_rmse_stage_plot_uses_cache_only_loads(
        self,
    ) -> None:
        cfg = SimpleNamespace(
            plot=SimpleNamespace(zero_shot_stage_show_lone_mlip_swarm=False),
            dataset_profile=SimpleNamespace(tag="bio_mass"),
            datasets={
                "bio_mass": SimpleNamespace(),
                "khlohc": SimpleNamespace(),
            },
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(
                    mlip_selection=SimpleNamespace(
                        exclude_anomalous_mlips=True,
                        minimum_quorum=2,
                    )
                )
            ),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with patch(
                "oasis.experiment_runner._load_zero_shot_stage_rows_for_dataset",
                side_effect=[
                    [
                        {
                            "dataset": "bio_mass",
                            "dataset_label": "Bio-Mass",
                            "stage": "Full / all MLIPs",
                            "rmse": 0.4,
                            "n_samples": 10,
                        }
                    ],
                    [
                        {
                            "dataset": "khlohc",
                            "dataset_label": "KHLOHC-TOL",
                            "stage": "Full / all MLIPs",
                            "rmse": 0.5,
                            "n_samples": 12,
                        }
                    ],
                ],
            ) as mock_load_rows, patch(
                "oasis.experiment_runner.zero_shot_rmse_stage_plot",
                return_value=tmp_path / "zero_shot_stage_all.png",
            ):
                write_all_datasets_zero_shot_rmse_stage_plot(
                    cfg=cfg,
                    output_dir=tmp_path,
                    run_suffix="anomalyaware_on",
                )

        self.assertEqual(mock_load_rows.call_count, 2)
        self.assertTrue(
            all(call.kwargs["cache_only"] for call in mock_load_rows.call_args_list)
        )

    def test_load_zero_shot_stage_rows_for_dataset_skips_cache_miss_when_cache_only(
        self,
    ) -> None:
        cfg = SimpleNamespace(
            plot=SimpleNamespace(output_dir=Path("unused")),
            datasets={},
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(
                    mlip_selection=SimpleNamespace(
                        exclude_anomalous_mlips=True,
                        minimum_quorum=2,
                    )
                )
            ),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            result_file = tmp_path / "mlip.json"
            result_file.write_text("{}")
            with patch(
                "oasis.experiment_runner._zero_shot_stage_artifact_path",
                return_value=tmp_path / "zero_shot_cache.json",
            ), patch(
                "oasis.experiment_runner.derive_dataset_profile_paths",
                return_value=SimpleNamespace(analysis_base_dir=tmp_path),
            ), patch(
                "oasis.experiment_runner.load_wide_predictions",
                side_effect=AssertionError("cache-only zero-shot load should not rebuild"),
            ), patch(
                "oasis.experiment_runner.find_result_files",
                return_value=[result_file],
            ):
                rows = _load_zero_shot_stage_rows_for_dataset(
                    cfg,
                    dataset_tag="example",
                    cache_only=True,
                )

        self.assertEqual(rows, [])

    def test_load_zero_shot_stage_rows_for_dataset_uses_dataset_specific_artifact_path(
        self,
    ) -> None:
        cfg = SimpleNamespace(
            dataset_profile=SimpleNamespace(tag="bio_mass"),
            plot=SimpleNamespace(output_dir=Path("unused")),
            datasets={"khlohc": SimpleNamespace()},
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(
                    mlip_selection=SimpleNamespace(
                        exclude_anomalous_mlips=True,
                        minimum_quorum=2,
                    )
                )
            ),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            result_file = tmp_path / "mlip_result.json"
            result_file.write_text("{}")
            expected_rows = [
                {
                    "dataset": "khlohc",
                    "dataset_label": "Tol-KHLOHC",
                    "stage": "Full / all MLIPs",
                    "rmse": 0.5,
                    "n_samples": 12,
                }
            ]
            expected_artifact_path = tmp_path / "khlohc_zero_shot.json"
            expected_artifact_path.write_text("{}")
            expected_signature = {"result_files": ["dummy"]}
            dataset_cfg = SimpleNamespace(
                dataset_profile=SimpleNamespace(tag="khlohc"),
                datasets=cfg.datasets,
                experiment=cfg.experiment,
            )

            with patch(
                "oasis.experiment_runner.derive_dataset_profile_paths",
                return_value=SimpleNamespace(analysis_base_dir=tmp_path),
            ), patch(
                "oasis.experiment_runner._dataset_cfg_for_tag",
                return_value=dataset_cfg,
            ) as mock_dataset_cfg, patch(
                "oasis.experiment_runner._zero_shot_stage_artifact_path",
                return_value=expected_artifact_path,
            ) as mock_artifact_path, patch(
                "oasis.experiment_runner.find_result_files",
                return_value=[result_file],
            ), patch(
                "oasis.experiment_runner._zero_shot_stage_cache_signature",
                return_value=expected_signature,
            ), patch(
                "oasis.experiment_runner._load_zero_shot_stage_rows_artifact",
                return_value=expected_rows,
            ) as mock_load_artifact:
                rows = _load_zero_shot_stage_rows_for_dataset(
                    cfg,
                    dataset_tag="khlohc",
                    cache_only=True,
                )

        self.assertEqual(rows, expected_rows)
        mock_dataset_cfg.assert_called_once_with(cfg, dataset_tag="khlohc")
        mock_artifact_path.assert_called_once_with(dataset_cfg)
        mock_load_artifact.assert_called_once_with(
            expected_artifact_path,
            expected_cache_signature=expected_signature,
        )

    def test_zero_shot_stage_artifact_path_uses_zero_shot_bundle_root(self) -> None:
        cfg = SimpleNamespace(
            dataset_profile=SimpleNamespace(tag="bio_mass"),
            datasets={},
            experiment=SimpleNamespace(learning_curve=SimpleNamespace()),
        )

        artifact_path = _zero_shot_stage_artifact_path(cfg)

        self.assertEqual(
            artifact_path,
            Path("data/results/zero_shot/bio_mass_anomalyaware_off.json"),
        )

    def test_policy_selection_diagnostic_persistence_uses_screening_bundle_root(
        self,
    ) -> None:
        cfg = SimpleNamespace(
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(
                    min_train=2,
                    max_train=4,
                    step=2,
                    n_repeats=1,
                    min_test_size=1,
                    validation_fraction=0.2,
                    min_val_size=1,
                    min_tuning_val_size=1,
                    calibration_enabled=False,
                    calibration_fraction=0.2,
                    min_cal_size=1,
                    min_inner_train_size=1,
                    sweep_sizes=[],
                    sweep_fractions=[],
                    models=SimpleNamespace(
                        use_ridge=True,
                        use_kernel_ridge=False,
                        use_lasso=False,
                        use_elastic_net=False,
                        use_residual=False,
                        use_weighted_linear=False,
                        use_weighted_simplex=False,
                        use_graph_mean=False,
                        use_latent=False,
                        moe=SimpleNamespace(enabled=False),
                        probe_gnn=SimpleNamespace(enabled=False),
                        gnn_direct=SimpleNamespace(enabled=False),
                    ),
                ),
                screening=SimpleNamespace(
                    results_bundle_path=Path(
                        "data/results/screening/bio_mass_anomalyaware_off_latent_off_n6.json"
                    ),
                    policy_names=["min_screening_rmse"],
                    combined_miscalibration_lambda=1.0,
                ),
            )
        )

        persistence = _policy_selection_diagnostic_persistence_context(
            cfg,
            wide_df=pd.DataFrame(
                {
                    "reference_ads_eng": [1.0, 2.0],
                    "ridge_mlip_ads_eng_median": [1.1, 2.1],
                }
            ),
            output_dir=Path("unused"),
            run_suffix="anomalyaware_off",
        )

        self.assertEqual(
            persistence.artifact_path,
            Path(
                "data/results/screening/"
                "policy_selection_diagnostic_bio_mass_anomalyaware_off_latent_off_n6.json"
            ),
        )
        self.assertEqual(
            persistence.screening_rows_artifact_path,
            Path(
                "data/results/screening/"
                "policy_selection_screening_rows_bio_mass_anomalyaware_off_latent_off_n6.json"
            ),
        )
        self.assertEqual(
            persistence.diagnostic_cache_signature["screening"]["policy_names"],
            ["min_screening_rmse"],
        )
        self.assertNotIn(
            "policy_names",
            persistence.screening_rows_cache_signature["screening"],
        )

        cfg.experiment.screening.policy_names = [
            "combined_screening_rmse_miscalibration"
        ]
        cfg.experiment.screening.combined_miscalibration_lambda = 2.0
        updated_persistence = _policy_selection_diagnostic_persistence_context(
            cfg,
            wide_df=pd.DataFrame(
                {
                    "reference_ads_eng": [1.0, 2.0],
                    "ridge_mlip_ads_eng_median": [1.1, 2.1],
                }
            ),
            output_dir=Path("unused"),
            run_suffix="anomalyaware_off",
        )
        self.assertNotEqual(
            persistence.diagnostic_cache_signature,
            updated_persistence.diagnostic_cache_signature,
        )
        self.assertEqual(
            persistence.screening_rows_cache_signature,
            updated_persistence.screening_rows_cache_signature,
        )

    def test_policy_diagnostic_reuses_screening_rows_when_policy_artifact_misses(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            artifact_path = tmp_path / "policy_selection_diagnostic.json"
            screening_rows_path = tmp_path / "policy_selection_screening_rows.json"
            artifact_path.write_text("{}", encoding="utf-8")
            screening_rows_path.write_text("{}", encoding="utf-8")

            cached_screening_rows = pd.DataFrame(
                {
                    "method": ["ridge"],
                    "budget": [5],
                    "repeat": [0],
                    "split_fingerprint": ["abc"],
                    "screening_cv_rmse": [0.1],
                    "screening_miscalibration_area": [0.05],
                }
            )
            dummy_results = PolicySelectionDiagnosticResults(
                detail_df=pd.DataFrame(
                    [
                        {
                            "policy_name": "combined_screening_rmse_miscalibration",
                            "budget": 5,
                            "repeat": 0,
                            "oracle_method": "ridge",
                            "screening_selected_method": "ridge",
                            "oracle_outer_rmse": 0.1,
                            "screening_selected_outer_rmse": 0.1,
                            "regret": 0.0,
                            "screening_cv_rmse": 0.1,
                            "screening_miscalibration_area": 0.05,
                            "agreement": True,
                        }
                    ]
                ),
                outer_metrics_df=pd.DataFrame(
                    [{"budget": 5, "repeat": 0, "method": "ridge", "outer_test_rmse": 0.1}]
                ),
                summary_df=pd.DataFrame(
                    [
                        {
                            "policy_name": "combined_screening_rmse_miscalibration",
                            "budget": 5,
                            "mean_regret": 0.0,
                            "std_regret": 0.0,
                            "se_regret": 0.0,
                            "ci95_low": 0.0,
                            "ci95_high": 0.0,
                            "agreement_rate": 1.0,
                            "oracle_outer_rmse_mean": 0.1,
                            "screening_selected_outer_rmse_mean": 0.1,
                        }
                    ]
                ),
            )
            persistence = SimpleNamespace(
                metadata=object(),
                diagnostic_cache_signature={
                    "learning_curve": {"min_train": 5},
                    "screening": {
                        "screen_fraction": 0.2,
                        "policy_names": ["combined_screening_rmse_miscalibration"],
                        "combined_miscalibration_lambda": 1.0,
                    },
                },
                screening_rows_cache_signature={
                    "learning_curve": {"min_train": 5},
                    "screening": {"screen_fraction": 0.2},
                },
                artifact_path=artifact_path,
                screening_rows_artifact_path=screening_rows_path,
            )
            captured: dict[str, pd.DataFrame | None] = {}

            def _fake_build(**kwargs):
                captured["outer_metrics_df"] = kwargs["cached_outer_repeat_metrics_df"]
                captured["screening_rows_df"] = kwargs["cached_screening_rows_df"]
                return PolicyDiagnosticBuildOutputs(
                    results=dummy_results,
                    screening_rows_df=kwargs["cached_screening_rows_df"],
                )

            with patch(
                "oasis.experiment_runner._policy_selection_diagnostic_persistence_context",
                return_value=persistence,
            ), patch(
                "oasis.experiment_runner.load_policy_selection_diagnostic_artifact",
                side_effect=[
                    ValueError("policy mismatch"),
                    SimpleNamespace(
                        cache_signature={
                            "learning_curve": {"min_train": 5},
                            "screening": {"screen_fraction": 0.2},
                        },
                        results=SimpleNamespace(
                            outer_metrics_df=dummy_results.outer_metrics_df
                        ),
                    ),
                ],
            ), patch(
                "oasis.experiment_runner.load_screening_diagnostic_rows_artifact",
                return_value=SimpleNamespace(screening_rows_df=cached_screening_rows),
            ), patch(
                "oasis.experiment_runner._build_policy_selection_diagnostic_results_for_cfg",
                side_effect=_fake_build,
            ), patch(
                "oasis.experiment_runner._write_policy_selection_diagnostic_outputs",
                return_value=tmp_path / "out.png",
            ):
                result = _write_policy_selection_diagnostic(
                    cfg=SimpleNamespace(),
                    wide_df=pd.DataFrame(),
                    graph_view=None,
                    auxiliary_views=None,
                    output_dir=tmp_path,
                    run_suffix="test",
                    min_x=None,
                    max_x=None,
                    include_x=None,
                )

        self.assertEqual(result, tmp_path / "out.png")
        pd.testing.assert_frame_equal(
            captured["outer_metrics_df"],
            dummy_results.outer_metrics_df,
        )
        pd.testing.assert_frame_equal(
            captured["screening_rows_df"],
            cached_screening_rows,
        )

    def test_policy_diagnostic_reuses_superset_method_cache_when_method_disabled(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            artifact_path = tmp_path / "policy_selection_diagnostic.json"
            screening_rows_path = tmp_path / "policy_selection_screening_rows.json"
            artifact_path.write_text("{}", encoding="utf-8")
            screening_rows_path.write_text("{}", encoding="utf-8")

            prior_outer_metrics_df = pd.DataFrame(
                [
                    {"budget": 5, "repeat": 0, "method": "ridge", "outer_test_rmse": 0.1},
                    {
                        "budget": 5,
                        "repeat": 0,
                        "method": "weighted_linear",
                        "outer_test_rmse": 0.12,
                    },
                ]
            )
            prior_screening_rows_df = pd.DataFrame(
                [
                    {
                        "method": "ridge",
                        "budget": 5,
                        "repeat": 0,
                        "split_fingerprint": "ridge-fp",
                        "screening_cv_rmse": 0.1,
                        "screening_miscalibration_area": 0.05,
                    },
                    {
                        "method": "weighted_linear",
                        "budget": 5,
                        "repeat": 0,
                        "split_fingerprint": "weighted-fp",
                        "screening_cv_rmse": 0.09,
                        "screening_miscalibration_area": 0.04,
                    },
                ]
            )
            persistence = SimpleNamespace(
                metadata=object(),
                diagnostic_cache_signature={
                    "learning_curve": {
                        "min_train": 5,
                        "enabled_model_names": ["ridge"],
                    },
                    "screening": {
                        "screen_fraction": 0.2,
                        "policy_names": ["min_screening_rmse"],
                        "combined_miscalibration_lambda": 1.0,
                    },
                },
                screening_rows_cache_signature={
                    "learning_curve": {
                        "min_train": 5,
                        "enabled_model_names": ["ridge"],
                    },
                    "screening": {"screen_fraction": 0.2},
                },
                artifact_path=artifact_path,
                screening_rows_artifact_path=screening_rows_path,
            )
            captured: dict[str, pd.DataFrame | None] = {}

            def _fake_build(**kwargs):
                captured["outer_metrics_df"] = kwargs["cached_outer_repeat_metrics_df"]
                captured["screening_rows_df"] = kwargs["cached_screening_rows_df"]
                return PolicyDiagnosticBuildOutputs(
                    results=PolicySelectionDiagnosticResults(
                        detail_df=pd.DataFrame(
                            [
                                {
                                    "policy_name": "min_screening_rmse",
                                    "budget": 5,
                                    "repeat": 0,
                                    "oracle_method": "ridge",
                                    "screening_selected_method": "ridge",
                                    "oracle_outer_rmse": 0.1,
                                    "screening_selected_outer_rmse": 0.1,
                                    "regret": 0.0,
                                    "screening_cv_rmse": 0.1,
                                    "screening_miscalibration_area": 0.05,
                                    "agreement": True,
                                }
                            ]
                        ),
                        outer_metrics_df=kwargs["cached_outer_repeat_metrics_df"],
                        summary_df=pd.DataFrame(
                            [
                                {
                                    "policy_name": "min_screening_rmse",
                                    "budget": 5,
                                    "mean_regret": 0.0,
                                    "std_regret": 0.0,
                                    "se_regret": 0.0,
                                    "ci95_low": 0.0,
                                    "ci95_high": 0.0,
                                    "agreement_rate": 1.0,
                                    "oracle_outer_rmse_mean": 0.1,
                                    "screening_selected_outer_rmse_mean": 0.1,
                                }
                            ]
                        ),
                    ),
                    screening_rows_df=kwargs["cached_screening_rows_df"],
                )

            with patch(
                "oasis.experiment_runner._policy_selection_diagnostic_persistence_context",
                return_value=persistence,
            ), patch(
                "oasis.experiment_runner.load_policy_selection_diagnostic_artifact",
                side_effect=[
                    ValueError("method mismatch"),
                    SimpleNamespace(
                        cache_signature={
                            "learning_curve": {
                                "min_train": 5,
                                "enabled_model_names": ["ridge", "weighted_linear"],
                            },
                            "screening": {"screen_fraction": 0.2},
                        },
                        results=SimpleNamespace(
                            outer_metrics_df=prior_outer_metrics_df
                        ),
                    ),
                ],
            ), patch(
                "oasis.experiment_runner.load_screening_diagnostic_rows_artifact",
                side_effect=[
                    ValueError("method mismatch"),
                    SimpleNamespace(
                        cache_signature={
                            "learning_curve": {
                                "min_train": 5,
                                "enabled_model_names": ["ridge", "weighted_linear"],
                            },
                            "screening": {"screen_fraction": 0.2},
                        },
                        screening_rows_df=prior_screening_rows_df,
                    ),
                ],
            ), patch(
                "oasis.experiment_runner._build_policy_selection_diagnostic_results_for_cfg",
                side_effect=_fake_build,
            ), patch(
                "oasis.experiment_runner._write_policy_selection_diagnostic_outputs",
                return_value=tmp_path / "out.png",
            ):
                result = _write_policy_selection_diagnostic(
                    cfg=SimpleNamespace(),
                    wide_df=pd.DataFrame(),
                    graph_view=None,
                    auxiliary_views=None,
                    output_dir=tmp_path,
                    run_suffix="test",
                    min_x=None,
                    max_x=None,
                    include_x=None,
                )

        self.assertEqual(result, tmp_path / "out.png")
        pd.testing.assert_frame_equal(
            captured["outer_metrics_df"].reset_index(drop=True),
            prior_outer_metrics_df.loc[
                prior_outer_metrics_df["method"] == "ridge"
            ].reset_index(drop=True),
        )
        pd.testing.assert_frame_equal(
            captured["screening_rows_df"].reset_index(drop=True),
            prior_screening_rows_df.loc[
                prior_screening_rows_df["method"] == "ridge"
            ].reset_index(drop=True),
        )

    def test_run_experiment_separates_persistent_cache_paths_for_latent_filtered_data(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            latent_csv = tmp_path / "latent.csv"
            pd.DataFrame(
                {"adsorption_energy": [1.0, 3.0], "latent_feature": [0.1, 0.2]}
            ).to_csv(latent_csv, index=False)
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
                            exclude_anomalous=False,
                            label_allowlist=["normal"],
                            strict_inference_anomaly=False,
                        ),
                        models=SimpleNamespace(
                            use_latent=True,
                            latent=SimpleNamespace(csv_path=latent_csv),
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
            fake_wide_df = _FakeWideFrame(
                reactions=["r0", "r1", "r2"],
                reference_ads_eng=[1.0, 2.0, 3.0],
            )

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
                "oasis.experiment_runner.parity_plot",
                return_value=tmp_path / "plots" / "parity.png",
            ), patch(
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
                return_value=tmp_path / "aligned_graphs.parquet",
            ), patch(
                "oasis.experiment_runner.load_or_run_learning_curve_results_from_config",
                return_value=LearningCurveResults.empty(),
            ) as mock_results, patch(
                "oasis.experiment_runner.learning_curve_plot",
                return_value=tmp_path / "plots" / "learning_curve.png",
            ):
                run_experiment(cfg)

        self.assertEqual(
            mock_results.call_args.args[1].experiment.learning_curve.results_bundle_path,
            tmp_path / "results_anomalyaware_off_latent_on_n2.json",
        )
        self.assertEqual(
            mock_results.call_args.args[1].experiment.learning_curve.graph_dataset.path,
            tmp_path / "aligned_graphs_anomalyaware_off_latent_on_n2.parquet",
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
            ) as mock_generation_plot:
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
            ) as mock_generation_plot:
                run_experiment(cfg)

        self.assertIsNone(
            mock_generation_plot.call_args.kwargs["generation_timing_by_method"]
        )

    def test_run_experiment_wires_gnn_direct_generation_timing_override(self) -> None:
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
                            use_gnn_direct=True,
                            gnn_direct=SimpleNamespace(enabled=True),
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
            fake_wide_df = _FakeWideFrame([f"r{i}" for i in range(3)])
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
            sample_atoms = [object(), object(), object()]

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
                return_value=sample_atoms,
            ), patch(
                "oasis.experiment_runner.atoms_to_graph_dataset_view",
                return_value=[],
            ) as mock_graph_build, patch(
                "oasis.experiment_runner.perf_counter",
                side_effect=[10.0, 10.6],
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
            ) as mock_generation_plot:
                run_experiment(cfg)

        gnn_direct_override = mock_generation_plot.call_args.kwargs[
            "generation_timing_by_method"
        ]["gnn_direct"]
        self.assertAlmostEqual(gnn_direct_override.generation_time_s, 0.6)
        self.assertEqual(gnn_direct_override.generation_steps_total, 3)
        self.assertAlmostEqual(gnn_direct_override.time_per_step_s, 0.2)
        self.assertEqual(gnn_direct_override.mlip_feature_names, ("atoms_to_graph",))
        self.assertGreaterEqual(mock_graph_build.call_count, 2)

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
            tmp_path / "learning_anomalyaware_off_latent_off_n2.json",
        )
        self.assertEqual(
            second_cfg.experiment.learning_curve.results_bundle_path,
            tmp_path / "screening_anomalyaware_off_latent_off_n2.json",
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
            ) as mock_learning_curve_plot, patch(
                "oasis.experiment_runner.write_all_datasets_oracle_learning_curve_plot",
                return_value=None,
            ) as mock_oracle_plot:
                run_experiment(cfg)

        self.assertEqual(mock_learning_curve_plot.call_args.kwargs["min_x"], 10)
        self.assertEqual(mock_learning_curve_plot.call_args.kwargs["max_x"], 50)
        self.assertEqual(mock_learning_curve_plot.call_args.kwargs["include_x"], [10, 30])
        self.assertEqual(mock_oracle_plot.call_args.kwargs["min_x"], 10)
        self.assertEqual(mock_oracle_plot.call_args.kwargs["max_x"], 50)
        self.assertEqual(mock_oracle_plot.call_args.kwargs["include_x"], [10, 30])
        self.assertEqual(
            mock_learning_curve_plot.call_args.kwargs["output_path"],
            tmp_path / "plots" / "learning_curve_anomalyaware_off.png",
        )

    def test_run_experiment_emits_dual_learning_curve_outputs_for_budget_spans(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cfg = SimpleNamespace(
                mlip=SimpleNamespace(dataset=str(tmp_path / "mamun_oh.json")),
                probe_features=None,
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        budget_mode="full_remainder_test",
                        graph_dataset=None,
                        sweep_sizes=[1, 2],
                        sweep_fractions=[0.5, 1.0],
                        models=SimpleNamespace(
                            use_latent=False,
                            probe_gnn=SimpleNamespace(enabled=False),
                        ),
                    )
                ),
                analysis=SimpleNamespace(base_dir=tmp_path / "mlips"),
                plot=SimpleNamespace(
                    output_dir=tmp_path / "plots",
                    curve_window=SimpleNamespace(min_x=None, max_x=None, include_x=None),
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
                side_effect=[
                    tmp_path / "plots" / "learning_curve_absolute.png",
                    tmp_path / "plots" / "learning_curve_fraction.png",
                ],
            ) as mock_learning_curve_plot:
                run_experiment(cfg)

        self.assertEqual(mock_learning_curve_plot.call_count, 2)
        self.assertEqual(
            mock_learning_curve_plot.call_args_list[0].kwargs["output_path"],
            tmp_path / "plots" / "learning_curve_anomalyaware_off_absolute.png",
        )
        self.assertEqual(
            mock_learning_curve_plot.call_args_list[1].kwargs["output_path"],
            tmp_path / "plots" / "learning_curve_anomalyaware_off_fraction.png",
        )
        self.assertEqual(
            mock_learning_curve_plot.call_args_list[0].kwargs["include_x"],
            [1, 2],
        )
        self.assertEqual(
            mock_learning_curve_plot.call_args_list[1].kwargs["include_x"],
            [1, 2],
        )

    def test_run_experiment_applies_strict_mlip_mask_to_learning_curve_zero_shot_baseline(
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
                            exclude_anomalous_mlips=True,
                            minimum_quorum=0,
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
            fake_wide_df = pl.DataFrame(
                {
                    "reaction": ["r0", "r1"],
                    "reference_ads_eng": [0.0, 0.0],
                    "model_a_mlip_ads_eng_median": [0.0, 0.0],
                    "model_b_mlip_ads_eng_median": [10.0, 10.0],
                    "model_a_slab_conv": [0, 0],
                    "model_a_ads_conv": [0, 0],
                    "model_a_slab_move": [0, 0],
                    "model_a_ads_move": [0, 0],
                    "model_a_slab_seed": [0, 0],
                    "model_a_ads_seed": [0, 0],
                    "model_a_ads_eng_seed": [0, 0],
                    "model_a_adsorbate_migration": [0, 0],
                    "model_b_slab_conv": [1, 1],
                    "model_b_ads_conv": [0, 0],
                    "model_b_slab_move": [0, 0],
                    "model_b_ads_move": [0, 0],
                    "model_b_slab_seed": [0, 0],
                    "model_b_ads_seed": [0, 0],
                    "model_b_ads_eng_seed": [0, 0],
                    "model_b_adsorbate_migration": [0, 0],
                }
            )

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
                "oasis.experiment_runner.write_zero_shot_rmse_stage_plot",
                return_value=tmp_path / "plots" / "zero_shot_stage.png",
            ), patch(
                "oasis.experiment_runner.write_all_datasets_zero_shot_rmse_stage_plot",
                return_value=None,
            ), patch(
                "oasis.experiment_runner.learning_curve_plot",
                return_value=tmp_path / "plots" / "learning_curve.png",
            ) as mock_learning_curve_plot:
                run_experiment(cfg)

        self.assertAlmostEqual(
            mock_learning_curve_plot.call_args.kwargs["zero_shot_rmse"],
            0.0,
            places=12,
        )

    def test_run_experiment_ignores_legacy_curve_window_include_fractions(self) -> None:
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
            ) as mock_learning_curve_plot, patch(
                "oasis.experiment_runner.write_all_datasets_oracle_learning_curve_plot",
                return_value=None,
            ) as mock_oracle_plot:
                run_experiment(cfg)

        self.assertIsNone(mock_learning_curve_plot.call_args.kwargs["include_x"])
        self.assertIsNone(mock_oracle_plot.call_args.kwargs["include_x"])

    def test_run_experiment_preserves_include_x_when_legacy_include_fractions_is_present(
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

        self.assertEqual(mock_learning_curve_plot.call_args.kwargs["include_x"], [3])

    def test_run_experiment_deduplicates_overlapping_curve_window_points(self) -> None:
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
                        include_x=[1, 2],
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

    def test_run_experiment_full_dataset_window_ignores_legacy_include_fractions(
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
        self.assertIsNone(mock_learning_curve_plot.call_args.kwargs["include_x"])

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
        self.assertEqual(mock_sharpness_plot.call_args.kwargs["show_xlabel"], False)
        self.assertNotIn("show_legend", mock_sharpness_plot.call_args.kwargs)
        self.assertAlmostEqual(
            mock_sharpness_plot.call_args.kwargs["zero_shot_value"],
            0.1,
            places=12,
        )
        self.assertEqual(
            Path(mock_dispersion_plot.call_args.kwargs["output_path"]).name,
            "dispersion_panel_anomalyaware_off.png",
        )
        self.assertNotIn("show_legend", mock_dispersion_plot.call_args.kwargs)
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

    def test_run_experiment_emits_uq_summary_figure_for_kernel_ridge_only_uq(
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
                    curve_window=SimpleNamespace(min_x=None, max_x=None, include_x=None),
                ),
            )
            fake_wide_df = _FakeWideFrame()
            results = self._kernel_ridge_only_uq_results()

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
            ), patch(
                "oasis.experiment_runner.dispersion_plot",
                return_value=tmp_path / "tmp" / "dispersion.png",
            ), patch(
                "oasis.experiment_runner.uq_summary_figure",
                return_value=tmp_path / "plots" / "uq_summary_figure.png",
            ) as mock_uq_summary_figure:
                run_experiment(cfg)

        mock_miscalibration_plot.assert_called_once()
        mock_uq_summary_figure.assert_called_once()

    def test_run_experiment_emits_uq_summary_figure_for_sparse_linear_uq(
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
                    curve_window=SimpleNamespace(min_x=None, max_x=None, include_x=None),
                ),
            )
            fake_wide_df = _FakeWideFrame()
            results = self._sparse_linear_only_uq_results()

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
            ), patch(
                "oasis.experiment_runner.dispersion_plot",
                return_value=tmp_path / "tmp" / "dispersion.png",
            ), patch(
                "oasis.experiment_runner.uq_summary_figure",
                return_value=tmp_path / "plots" / "uq_summary_figure.png",
            ) as mock_uq_summary_figure:
                run_experiment(cfg)

        mock_miscalibration_plot.assert_called_once()
        mock_uq_summary_figure.assert_called_once()

    def test_run_experiment_persists_learning_curve_uq_component_plots_for_both_span_variants(
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
                        sweep_sizes=[5, 10],
                        sweep_fractions=[0.5, 1.0],
                        min_train=None,
                        max_train=None,
                        step=1,
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
            fake_wide_df = _FakeWideFrame(reactions=[f"r{i}" for i in range(10)])
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
                side_effect=lambda *args, **kwargs: Path(kwargs["output_path"]),
            ) as mock_miscalibration_plot, patch(
                "oasis.experiment_runner.sharpness_plot",
                side_effect=lambda *args, **kwargs: Path(kwargs["output_path"]),
            ) as mock_sharpness_plot, patch(
                "oasis.experiment_runner.dispersion_plot",
                side_effect=lambda *args, **kwargs: Path(kwargs["output_path"]),
            ) as mock_dispersion_plot, patch(
                "oasis.experiment_runner.uq_summary_figure",
                side_effect=lambda *args, **kwargs: Path(kwargs["output_path"]),
            ) as mock_uq_summary_figure:
                run_experiment(cfg)

        self.assertEqual(mock_miscalibration_plot.call_count, 2)
        self.assertEqual(
            [Path(call.kwargs["output_path"]).name for call in mock_miscalibration_plot.call_args_list],
            [
                "miscalibration_area_panel_anomalyaware_off_absolute.png",
                "miscalibration_area_panel_anomalyaware_off_fraction.png",
            ],
        )
        self.assertEqual(
            [Path(call.kwargs["output_path"]).parent for call in mock_miscalibration_plot.call_args_list],
            [tmp_path / "plots", tmp_path / "plots"],
        )
        self.assertEqual(
            [Path(call.kwargs["output_path"]).name for call in mock_sharpness_plot.call_args_list],
            [
                "sharpness_panel_anomalyaware_off_absolute.png",
                "sharpness_panel_anomalyaware_off_fraction.png",
            ],
        )
        self.assertEqual(
            [Path(call.kwargs["output_path"]).name for call in mock_dispersion_plot.call_args_list],
            [
                "dispersion_panel_anomalyaware_off_absolute.png",
                "dispersion_panel_anomalyaware_off_fraction.png",
            ],
        )
        self.assertEqual(
            [Path(call.kwargs["output_path"]).name for call in mock_uq_summary_figure.call_args_list],
            [
                "uq_summary_figure_anomalyaware_off_absolute.png",
                "uq_summary_figure_anomalyaware_off_fraction.png",
            ],
        )

    def test_run_experiment_emits_screening_uq_summary_figure_when_screening_results_have_uq(
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
                    curve_window=SimpleNamespace(min_x=5, max_x=10, include_x=[5, 10]),
                ),
            )
            fake_wide_df = _FakeWideFrame()
            learning_results = LearningCurveResults.empty()
            screening_results = self._uq_results()

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
                side_effect=[learning_results, screening_results],
            ), patch(
                "oasis.experiment_runner.learning_curve_plot",
                return_value=tmp_path / "plots" / "learning_curve.png",
            ), patch(
                "oasis.experiment_runner.screening_budget_plot",
                return_value=tmp_path / "plots" / "screening_budget.png",
            ), patch(
                "oasis.experiment_runner.learning_screening_figure",
                return_value=tmp_path / "plots" / "learning_screening_figure.png",
            ), patch(
                "oasis.experiment_runner.miscalibration_area_plot",
                return_value=tmp_path / "tmp" / "screening_miscalibration.png",
            ) as mock_miscalibration_plot, patch(
                "oasis.experiment_runner.sharpness_plot",
                return_value=tmp_path / "tmp" / "screening_sharpness.png",
            ) as mock_sharpness_plot, patch(
                "oasis.experiment_runner.dispersion_plot",
                return_value=tmp_path / "tmp" / "screening_dispersion.png",
            ) as mock_dispersion_plot, patch(
                "oasis.experiment_runner.uq_summary_figure",
                return_value=tmp_path / "plots" / "screening_uq_summary_figure.png",
            ) as mock_uq_summary_figure:
                run_experiment(cfg)

        self.assertEqual(mock_miscalibration_plot.call_count, 1)
        self.assertEqual(
            Path(mock_miscalibration_plot.call_args.kwargs["output_path"]).name,
            "screening_miscalibration_area_panel_anomalyaware_off.png",
        )
        self.assertEqual(
            Path(mock_sharpness_plot.call_args.kwargs["output_path"]).name,
            "screening_sharpness_panel_anomalyaware_off.png",
        )
        self.assertEqual(
            Path(mock_dispersion_plot.call_args.kwargs["output_path"]).name,
            "screening_dispersion_panel_anomalyaware_off.png",
        )
        self.assertEqual(
            mock_uq_summary_figure.call_args.kwargs["output_path"],
            tmp_path / "plots" / "screening_uq_summary_figure_anomalyaware_off.png",
        )
        self.assertEqual(
            mock_uq_summary_figure.call_args.kwargs["miscalibration_area_path"],
            tmp_path / "tmp" / "screening_miscalibration.png",
        )
        self.assertEqual(
            mock_uq_summary_figure.call_args.kwargs["sharpness_path"],
            tmp_path / "tmp" / "screening_sharpness.png",
        )
        self.assertEqual(
            mock_uq_summary_figure.call_args.kwargs["dispersion_path"],
            tmp_path / "tmp" / "screening_dispersion.png",
        )

    def test_run_experiment_persists_screening_uq_component_plots_for_both_span_variants(
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
                        sweep_sizes=[5, 10],
                        sweep_fractions=[0.5, 1.0],
                        min_train=None,
                        max_train=None,
                        step=1,
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
                    curve_window=SimpleNamespace(min_x=5, max_x=10, include_x=[5, 10]),
                ),
            )
            fake_wide_df = _FakeWideFrame(reactions=[f"r{i}" for i in range(10)])
            learning_results = LearningCurveResults.empty()
            screening_results = self._uq_results()

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
                side_effect=[learning_results, screening_results],
            ), patch(
                "oasis.experiment_runner.learning_curve_plot",
                return_value=tmp_path / "plots" / "learning_curve.png",
            ), patch(
                "oasis.experiment_runner.screening_budget_plot",
                return_value=tmp_path / "plots" / "screening_budget.png",
            ), patch(
                "oasis.experiment_runner.learning_screening_figure",
                return_value=tmp_path / "plots" / "learning_screening_figure.png",
            ), patch(
                "oasis.experiment_runner.miscalibration_area_plot",
                side_effect=lambda *args, **kwargs: Path(kwargs["output_path"]),
            ) as mock_miscalibration_plot, patch(
                "oasis.experiment_runner.sharpness_plot",
                side_effect=lambda *args, **kwargs: Path(kwargs["output_path"]),
            ) as mock_sharpness_plot, patch(
                "oasis.experiment_runner.dispersion_plot",
                side_effect=lambda *args, **kwargs: Path(kwargs["output_path"]),
            ) as mock_dispersion_plot, patch(
                "oasis.experiment_runner.uq_summary_figure",
                side_effect=lambda *args, **kwargs: Path(kwargs["output_path"]),
            ) as mock_uq_summary_figure:
                run_experiment(cfg)

        self.assertEqual(mock_miscalibration_plot.call_count, 2)
        self.assertEqual(
            [Path(call.kwargs["output_path"]).name for call in mock_miscalibration_plot.call_args_list],
            [
                "screening_miscalibration_area_panel_anomalyaware_off_absolute.png",
                "screening_miscalibration_area_panel_anomalyaware_off_fraction.png",
            ],
        )
        self.assertEqual(
            [Path(call.kwargs["output_path"]).parent for call in mock_miscalibration_plot.call_args_list],
            [tmp_path / "plots", tmp_path / "plots"],
        )
        self.assertEqual(
            [Path(call.kwargs["output_path"]).name for call in mock_sharpness_plot.call_args_list],
            [
                "screening_sharpness_panel_anomalyaware_off_absolute.png",
                "screening_sharpness_panel_anomalyaware_off_fraction.png",
            ],
        )
        self.assertEqual(
            [Path(call.kwargs["output_path"]).name for call in mock_dispersion_plot.call_args_list],
            [
                "screening_dispersion_panel_anomalyaware_off_absolute.png",
                "screening_dispersion_panel_anomalyaware_off_fraction.png",
            ],
        )
        self.assertEqual(
            [Path(call.kwargs["output_path"]).name for call in mock_uq_summary_figure.call_args_list],
            [
                "screening_uq_summary_figure_anomalyaware_off_absolute.png",
                "screening_uq_summary_figure_anomalyaware_off_fraction.png",
            ],
        )
