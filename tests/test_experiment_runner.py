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
    _load_oracle_learning_curve_rows_for_dataset,
    _build_zero_shot_stage_rows,
    _write_policy_selection_diagnostic,
    load_all_datasets_oracle_learning_curve_rows,
    load_filtered_wide_predictions,
    run_experiment,
    run_experiment_from_config,
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
            with patch(
                "oasis.experiment_runner.build_sweep_dataset_from_config",
                return_value=SweepDataset(
                    mlip_features=np.arange(6, dtype=float).reshape(-1, 1),
                    targets=np.linspace(0.0, 1.0, 6),
                ),
            ), patch(
                "oasis.learning_curve.registry.enabled_learning_curve_model_names_from_config",
                return_value=("ridge",),
            ), patch(
                "oasis.learning_curve.registry.default_sweep_model_families",
                return_value=(),
            ), patch(
                "oasis.experiment_runner.build_policy_selection_diagnostic_results",
                return_value=diagnostic_results,
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
                    output_dir / "policy_selected_vs_oracle_anomalyaware_off.png"
                )
                regret_plot_path = output_dir / "policy_regret_anomalyaware_off.png"
                artifact_exists = artifact_path is not None and artifact_path.is_file()
                detail_exists = detail_path.is_file()
                summary_exists = summary_path.is_file()
                oracle_plot_exists = oracle_plot_path.is_file()
                regret_plot_exists = regret_plot_path.is_file()

        assert artifact_path is not None
        self.assertEqual(
            artifact_path.name,
            "policy_selection_diagnostic_anomalyaware_off.json",
        )
        self.assertTrue(artifact_exists)
        self.assertTrue(detail_exists)
        self.assertTrue(summary_exists)
        self.assertTrue(oracle_plot_exists)
        self.assertTrue(regret_plot_exists)
        self.assertEqual(
            mock_build_diagnostic.call_args.kwargs["policy_names"],
            [
                "min_screening_rmse",
                "combined_screening_rmse_miscalibration",
            ],
        )
        self.assertEqual(
            mock_build_diagnostic.call_args.kwargs["combined_miscalibration_lambda"],
            2.0,
        )

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

    def test_load_oracle_learning_curve_rows_for_dataset_resolves_include_fractions_per_dataset(
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
                include_fractions=[0.5, 1.0],
            )
            large_rows = _load_oracle_learning_curve_rows_for_dataset(
                cfg,
                dataset_tag="khlohc",
                enabled_method_names=["ridge"],
                include_fractions=[0.5, 1.0],
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
            [call.kwargs["include_fractions"] for call in mock_load_rows.call_args_list],
            [None, None, None],
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
            "oasis.experiment_runner.prepare_graph_view",
            return_value=None,
        ), patch(
            "oasis.experiment_runner._apply_persistent_output_suffixes",
            return_value="anomalyaware_off_latent_off_n2",
        ), patch(
            "oasis.experiment_runner.load_or_run_learning_curve_results_from_config",
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
        self.assertTrue(mock_plot.call_args.kwargs["log_x"])

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
        self.assertIsNone(mock_oracle_plot.call_args.kwargs["include_fractions"])
        self.assertEqual(
            mock_learning_curve_plot.call_args.kwargs["output_path"],
            tmp_path / "plots" / "learning_curve_anomalyaware_off.png",
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
            ) as mock_learning_curve_plot, patch(
                "oasis.experiment_runner.write_all_datasets_oracle_learning_curve_plot",
                return_value=None,
            ) as mock_oracle_plot:
                run_experiment(cfg)

        self.assertEqual(mock_learning_curve_plot.call_args.kwargs["include_x"], [1, 2])
        self.assertIsNone(mock_oracle_plot.call_args.kwargs["include_x"])
        self.assertEqual(
            mock_oracle_plot.call_args.kwargs["include_fractions"],
            [0.5, 1.0],
        )

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
