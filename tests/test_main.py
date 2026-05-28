from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from oasis.__main__ import main
from oasis.sweep import LearningCurveResults


class _FakeColumn:
    def __init__(self, values):
        self._values = values

    def to_list(self):
        return list(self._values)


class _FakeWideFrame:
    def __init__(self, reactions=None) -> None:
        reactions = ["r0", "r1"] if reactions is None else list(reactions)
        self._columns = {
            "reaction": _FakeColumn(reactions),
            "reference_ads_eng": _FakeColumn([float(i + 1) for i in range(len(reactions))]),
        }

    def get_column(self, name: str):
        return self._columns[name]

    def __len__(self) -> int:
        return len(self._columns["reaction"].to_list())


class MainTests(unittest.TestCase):
    def test_main_skips_probe_dataset_build_when_probe_gnn_disabled(self) -> None:
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

            with patch("oasis.__main__.get_config", return_value=cfg):
                with patch("oasis.__main__.build_probe_dataset") as mock_build_probe:
                    with patch(
                        "oasis.__main__.add_mlip_feature_matrices_to_dataset"
                    ) as mock_add_probe_features:
                        with patch(
                            "oasis.__main__.find_result_files",
                            return_value=[],
                        ):
                            with patch(
                                "oasis.__main__.load_wide_predictions",
                                return_value=fake_wide_df,
                            ):
                                with patch(
                                    "oasis.__main__.filter_wide_predictions",
                                    return_value=fake_wide_df,
                                ):
                                    with patch(
                                        "oasis.__main__.parity_plot",
                                        return_value=tmp_path / "plots" / "parity.png",
                                    ):
                                        with patch(
                                            "oasis.__main__.load_sample_atoms_for_wide_df",
                                            return_value=[],
                                        ):
                                            with patch(
                                                "oasis.__main__.atoms_to_graph_dataset_view",
                                                return_value=[],
                                            ):
                                                with patch(
                                                    "oasis.__main__.load_or_run_learning_curve_results_from_config",
                                                    return_value=LearningCurveResults.empty(),
                                                ):
                                                    with patch(
                                                        "oasis.__main__.learning_curve_plot",
                                                        return_value=tmp_path
                                                        / "plots"
                                                        / "learning_curve.png",
                                                    ):
                                                        main()

            self.assertFalse(mock_build_probe.called)
            self.assertFalse(mock_add_probe_features.called)

    def test_main_rebuilds_stale_graph_artifact_when_reactions_do_not_match(self) -> None:
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

            with patch("oasis.__main__.get_config", return_value=cfg):
                with patch(
                    "oasis.__main__.find_result_files",
                    return_value=[],
                ):
                    with patch(
                        "oasis.__main__.load_wide_predictions",
                        return_value=fake_wide_df,
                    ):
                        with patch(
                            "oasis.__main__.filter_wide_predictions",
                            return_value=fake_wide_df,
                        ):
                            with patch(
                                "oasis.__main__.parity_plot",
                                return_value=tmp_path / "plots" / "parity.png",
                            ):
                                with patch(
                                    "oasis.__main__.load_graph_dataset_view",
                                    return_value=SimpleNamespace(sample_ids=("r0", "r1")),
                                ):
                                    with patch(
                                        "oasis.__main__.load_sample_atoms_for_wide_df",
                                        return_value=["a0", "a1", "a2"],
                                    ):
                                        with patch(
                                            "oasis.__main__.atoms_to_graph_dataset_view",
                                            return_value=["g0", "g1", "g2"],
                                        ) as mock_build_graph_view:
                                            with patch(
                                                "oasis.__main__.save_aligned_graph_dataset_parquet",
                                                return_value=graph_artifact_path,
                                            ) as mock_save_graph_artifact:
                                                with patch(
                                                    "oasis.__main__.load_or_run_learning_curve_results_from_config",
                                                    return_value=LearningCurveResults.empty(),
                                                ):
                                                    with patch(
                                                        "oasis.__main__.learning_curve_plot",
                                                        return_value=tmp_path
                                                        / "plots"
                                                        / "learning_curve.png",
                                                    ):
                                                        main()

            self.assertTrue(mock_build_graph_view.called)
            self.assertTrue(mock_save_graph_artifact.called)

    def test_main_rebuilds_graphs_without_persisting_when_graph_cache_is_omitted(self) -> None:
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

            with patch("oasis.__main__.get_config", return_value=cfg):
                with patch(
                    "oasis.__main__.find_result_files",
                    return_value=[],
                ):
                    with patch(
                        "oasis.__main__.load_wide_predictions",
                        return_value=fake_wide_df,
                    ):
                        with patch(
                            "oasis.__main__.filter_wide_predictions",
                            return_value=fake_wide_df,
                        ):
                            with patch(
                                "oasis.__main__.parity_plot",
                                return_value=tmp_path / "plots" / "parity.png",
                            ):
                                with patch(
                                    "oasis.__main__.load_sample_atoms_for_wide_df",
                                    return_value=["a0", "a1", "a2"],
                                ):
                                    with patch(
                                        "oasis.__main__.atoms_to_graph_dataset_view",
                                        return_value=["g0", "g1", "g2"],
                                    ) as mock_build_graph_view:
                                        with patch(
                                            "oasis.__main__.save_aligned_graph_dataset_parquet",
                                        ) as mock_save_graph_artifact:
                                            with patch(
                                                "oasis.__main__.load_or_run_learning_curve_results_from_config",
                                                return_value=LearningCurveResults.empty(),
                                            ) as mock_learning_curve_results:
                                                with patch(
                                                    "oasis.__main__.learning_curve_plot",
                                                    return_value=tmp_path
                                                    / "plots"
                                                    / "learning_curve.png",
                                                ):
                                                    main()

            self.assertTrue(mock_build_graph_view.called)
            self.assertFalse(mock_save_graph_artifact.called)
            self.assertEqual(
                mock_learning_curve_results.call_args.kwargs["graph_view"],
                ["g0", "g1", "g2"],
            )
