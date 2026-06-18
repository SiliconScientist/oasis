from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from oasis.experiment_runner import (
    load_filtered_wide_predictions,
    run_experiment,
    run_experiment_from_config,
)
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
                    filters=SimpleNamespace(
                        adsorbate=None,
                        anomaly_label=None,
                        reaction_contains=None,
                    ),
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
                                "oasis.experiment_runner.filter_wide_predictions",
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
            plot=SimpleNamespace(
                filters=SimpleNamespace(
                    adsorbate=None,
                    anomaly_label=None,
                    reaction_contains=None,
                )
            ),
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
                    "oasis.experiment_runner.filter_wide_predictions",
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
                    filters=SimpleNamespace(
                        adsorbate=None,
                        anomaly_label=None,
                        reaction_contains=None,
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
                "oasis.experiment_runner.filter_wide_predictions",
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
                    filters=SimpleNamespace(
                        adsorbate=None,
                        anomaly_label=None,
                        reaction_contains=None,
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
                "oasis.experiment_runner.filter_wide_predictions",
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
                    filters=SimpleNamespace(
                        adsorbate=None,
                        anomaly_label=None,
                        reaction_contains=None,
                    ),
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
                        "oasis.experiment_runner.filter_wide_predictions",
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
                    filters=SimpleNamespace(
                        adsorbate=None,
                        anomaly_label=None,
                        reaction_contains=None,
                    ),
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
                        "oasis.experiment_runner.filter_wide_predictions",
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
                    filters=SimpleNamespace(
                        adsorbate=None,
                        anomaly_label=None,
                        reaction_contains=None,
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
                "oasis.experiment_runner.filter_wide_predictions",
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
                    filters=SimpleNamespace(
                        adsorbate=None,
                        anomaly_label=None,
                        reaction_contains=None,
                    ),
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
                "oasis.experiment_runner.filter_wide_predictions",
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
