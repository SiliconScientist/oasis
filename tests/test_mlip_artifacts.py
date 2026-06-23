from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import polars as pl

from oasis.analysis import (
    filter_anomalous_mlip_columns,
    filter_structures_with_insufficient_valid_mlips,
)
from oasis.mlip.artifacts import (
    INFERENCE_DETAIL_COLUMNS,
    find_result_files,
    load_result_json,
    load_wide_predictions,
    mlip_detail_column_name,
    mlip_energy_column_name,
    mlip_label_column_name,
    model_name_from_result_path,
    result_file_name,
)
from oasis.mlip.result_parsing import (
    detect_anomalies_from_result_dict,
    extract_adsorbate,
)


class MlipResultParsingTests(unittest.TestCase):
    def test_extract_adsorbate_parses_product_side(self) -> None:
        self.assertEqual(extract_adsorbate("COgas+*->OH*"), "OH")
        self.assertIsNone(extract_adsorbate("not-a-reaction"))

    def test_detect_anomalies_from_result_dict_returns_expected_fields(self) -> None:
        result = detect_anomalies_from_result_dict(
            {
                "calculation_settings": {
                    "chemical_bond_cutoff": 1.25,
                    "n_crit_relax": 200,
                },
                "rxn-1->OH*": {
                    "reference": {"ads_eng": 1.0},
                    "final": {
                        "median_num": 0,
                        "ads_eng_median": 1.1,
                        "ads_seed_range": 0.0,
                        "ads_eng_seed_range": 0.0,
                    },
                    "0": {
                        "adslab_steps": 50,
                        "substrate_displacement": 0.1,
                        "max_bond_change": 5.0,
                    },
                    "single_calculation": {"ads_eng": 1.15},
                },
            }
        )

        self.assertEqual(
            result["rxn-1->OH*"],
            {
                "dft_ads_eng": 1.0,
                "mlip_ads_eng_median": 1.1,
                "mlip_ads_eng_single": 1.15,
                "label": "normal",
                "labels": [],
                "details": {
                    "slab_conv": 0,
                    "ads_conv": 0,
                    "slab_move": 0,
                    "ads_move": 0,
                    "slab_seed": 0,
                    "ads_seed": 0,
                    "ads_eng_seed": 0,
                    "adsorbate_migration": 0,
                    "energy_anomaly": 0,
                },
            },
        )


class MlipArtifactTests(unittest.TestCase):
    def test_naming_helpers_match_artifact_conventions(self) -> None:
        self.assertEqual(result_file_name("mace"), "mace_result.json")
        self.assertEqual(model_name_from_result_path(Path("x/mace_result.json")), "mace")
        self.assertEqual(mlip_energy_column_name("mace"), "mace_mlip_ads_eng_median")
        self.assertEqual(mlip_label_column_name("mace"), "mace_label")
        self.assertEqual(
            mlip_detail_column_name("mace", "ads_move"),
            "mace_ads_move",
        )

    def test_find_and_load_result_json(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            model_dir = base_dir / "mace"
            model_dir.mkdir()
            result_path = model_dir / "mace_result.json"
            result_path.write_text(json.dumps({"rxn-1": {"final": {}}}), encoding="utf-8")

            files = find_result_files(base_dir)
            payload = load_result_json(result_path)

        self.assertEqual(files, [result_path])
        self.assertEqual(payload, {"rxn-1": {"final": {}}})

    def test_find_result_files_filters_to_enabled_models(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            for model_name in ("mace", "orb_v3"):
                model_dir = base_dir / model_name
                model_dir.mkdir()
                (model_dir / f"{model_name}_result.json").write_text(
                    json.dumps({"rxn-1": {"final": {}}}),
                    encoding="utf-8",
                )

            files = find_result_files(base_dir, enabled_models=["orb_v3"])

        self.assertEqual([path.name for path in files], ["orb_v3_result.json"])

    def test_find_result_files_raises_when_enabled_model_is_missing(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            model_dir = base_dir / "mace"
            model_dir.mkdir()
            (model_dir / "mace_result.json").write_text(
                json.dumps({"rxn-1": {"final": {}}}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                FileNotFoundError,
                "Requested MLIP result files were not found",
            ):
                find_result_files(base_dir, enabled_models=["mace", "uma"])

    def test_load_wide_predictions_builds_expected_columns(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            result_path = base_dir / "mace_result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "calculation_settings": {
                            "chemical_bond_cutoff": 1.25,
                            "n_crit_relax": 200,
                        },
                        "rxn-1->OH*": {
                            "reference": {"ads_eng": 1.0},
                            "final": {
                                "median_num": 0,
                                "ads_eng_median": 1.1,
                                "ads_seed_range": 0.0,
                                "ads_eng_seed_range": 0.0,
                            },
                            "0": {
                                "adslab_steps": 50,
                                "substrate_displacement": 0.1,
                                "max_bond_change": 5.0,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            wide_df = load_wide_predictions([result_path])

        self.assertEqual(wide_df.get_column("reaction").to_list(), ["rxn-1->OH*"])
        self.assertEqual(wide_df.get_column("adsorbate").to_list(), ["OH"])
        self.assertEqual(wide_df.get_column("reference_ads_eng").to_list(), [1.0])
        self.assertEqual(wide_df.get_column("mace_mlip_ads_eng_median").to_list(), [1.1])
        self.assertEqual(wide_df.get_column("mace_label").to_list(), ["normal"])


class AnomalyAwareMlipSelectionTests(unittest.TestCase):
    @staticmethod
    def _mlip_columns(frame: pl.DataFrame) -> list[str]:
        return [col for col in frame.columns if col.endswith("_mlip_ads_eng_median")]

    def _wide_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "reaction": ["rxn-1->OH*", "rxn-2->OH*"],
                "adsorbate": ["OH", "OH"],
                "reference_ads_eng": [1.0, 2.0],
                "mace_mlip_ads_eng_median": [1.1, 2.1],
                "mace_label": ["normal", "normal"],
                "orb_mlip_ads_eng_median": [0.9, 1.9],
                "orb_label": ["normal", "energy_anomaly"],
                "uma_mlip_ads_eng_median": [1.2, 2.2],
                "uma_label": ["normal", "normal"],
                **{
                    f"mace_{detail}": [0, 0]
                    for detail in INFERENCE_DETAIL_COLUMNS
                },
                **{
                    f"orb_{detail}": [0, 0]
                    for detail in INFERENCE_DETAIL_COLUMNS
                },
                **{
                    f"uma_{detail}": [0, 0]
                    for detail in INFERENCE_DETAIL_COLUMNS
                },
            }
        ).with_columns(pl.lit(1).alias("uma_ads_move"))

    def test_filter_anomalous_mlip_columns_keeps_all_mlips_when_disabled(self) -> None:
        wide_df = self._wide_df()

        filtered = filter_anomalous_mlip_columns(wide_df, enabled=False)

        self.assertEqual(filtered.columns, wide_df.columns)
        self.assertEqual(
            self._mlip_columns(filtered),
            [
                "mace_mlip_ads_eng_median",
                "orb_mlip_ads_eng_median",
                "uma_mlip_ads_eng_median",
            ],
        )

    def test_filter_anomalous_mlip_columns_drops_mlips_with_non_normal_labels(
        self,
    ) -> None:
        wide_df = self._wide_df()

        with patch("builtins.print") as mock_print:
            filtered = filter_anomalous_mlip_columns(wide_df, enabled=True)

        self.assertEqual(
            self._mlip_columns(filtered),
            [
                "mace_mlip_ads_eng_median",
                "uma_mlip_ads_eng_median",
            ],
        )
        self.assertNotIn("orb_label", filtered.columns)
        mock_print.assert_called_once()
        printed = mock_print.call_args.args[0]
        self.assertIn("Applied anomaly-aware MLIP selection", printed)
        self.assertIn("removed 1 ['orb']", printed)

    def test_filter_anomalous_mlip_columns_can_use_strict_inference_details(self) -> None:
        wide_df = self._wide_df()

        with patch("builtins.print") as mock_print:
            filtered = filter_anomalous_mlip_columns(
                wide_df,
                enabled=True,
                strict_inference_anomaly=True,
            )

        self.assertEqual(filtered.columns, wide_df.columns)
        mock_print.assert_called_once()
        self.assertIn(
            "Skipped global anomaly-aware MLIP selection in strict inference mode",
            mock_print.call_args.args[0],
        )

    def test_filter_anomalous_mlip_columns_raises_when_all_mlips_removed(self) -> None:
        wide_df = self._wide_df().with_columns(
            pl.lit("energy_anomaly").alias("mace_label"),
            pl.lit("energy_anomaly").alias("orb_label"),
            pl.lit("energy_anomaly").alias("uma_label"),
        )

        with self.assertRaisesRegex(
            ValueError,
            "No MLIP prediction columns remain after anomaly-aware selection",
        ):
            filter_anomalous_mlip_columns(wide_df, enabled=True)

    def test_filter_structures_with_insufficient_valid_mlips_drops_rows_with_one_remaining(
        self,
    ) -> None:
        wide_df = self._wide_df().with_columns(
            pl.Series("orb_label", ["normal", "energy_anomaly"]),
            pl.Series("uma_label", ["normal", "energy_anomaly"]),
        )

        filtered = filter_structures_with_insufficient_valid_mlips(
            wide_df,
            enabled=True,
        )

        self.assertEqual(filtered.get_column("reaction").to_list(), ["rxn-1->OH*"])

    def test_filter_structures_with_insufficient_valid_mlips_uses_inference_details_only(
        self,
    ) -> None:
        wide_df = self._wide_df().with_columns(
            pl.lit("energy_anomaly").alias("orb_label"),
            pl.Series("orb_ads_move", [0, 1]),
            pl.Series("uma_ads_move", [0, 1]),
        )

        filtered = filter_structures_with_insufficient_valid_mlips(
            wide_df,
            enabled=True,
            strict_inference_anomaly=True,
        )

        self.assertEqual(filtered.get_column("reaction").to_list(), ["rxn-1->OH*"])


class DependencyBoundaryTests(unittest.TestCase):
    def test_importing_io_does_not_import_analysis(self) -> None:
        sys.modules.pop("oasis.io", None)
        sys.modules.pop("oasis.analysis", None)
        before_import = set(sys.modules)

        importlib.import_module("oasis.io")

        self.assertNotIn("oasis.analysis", set(sys.modules) - before_import)

    def test_importing_artifacts_does_not_import_runtime_modules(self) -> None:
        for module_name in (
            "oasis.mlip.artifacts",
            "oasis.mlip.runner",
            "oasis.mlip.tasks",
            "oasis.adapters.rootstock_adapter",
        ):
            sys.modules.pop(module_name, None)
        before_import = set(sys.modules)

        importlib.import_module("oasis.mlip.artifacts")

        imported = set(sys.modules) - before_import
        self.assertNotIn("oasis.mlip.runner", imported)
        self.assertNotIn("oasis.mlip.tasks", imported)
        self.assertNotIn("oasis.adapters.rootstock_adapter", imported)

    def test_importing_analysis_does_not_import_workflow_modules(self) -> None:
        for module_name in (
            "oasis.analysis",
            "oasis.analysis_workflows",
            "oasis.plot",
            "catbench.adsorption",
        ):
            sys.modules.pop(module_name, None)
        before_import = set(sys.modules)

        importlib.import_module("oasis.analysis")

        imported = set(sys.modules) - before_import
        self.assertNotIn("oasis.analysis_workflows", imported)
        self.assertNotIn("oasis.plot", imported)
        self.assertNotIn("catbench.adsorption", imported)

    def test_importing_experiment_data_does_not_import_probe_module(self) -> None:
        for module_name in (
            "oasis.experiment_data",
            "oasis.probe",
        ):
            sys.modules.pop(module_name, None)
        before_import = set(sys.modules)

        importlib.import_module("oasis.experiment_data")

        imported = set(sys.modules) - before_import
        self.assertNotIn("oasis.probe", imported)

    def test_experiment_side_modules_do_not_import_mlip_runtime(self) -> None:
        target_modules = (
            "oasis.exp",
            "oasis.experiment_runner",
            "oasis.experiment_data",
            "oasis.learning_curve.registry",
            "oasis.learning_curve.results_io",
            "oasis.learning_curve.runners",
        )
        runtime_modules = (
            "oasis.mlip.cli",
            "oasis.mlip.runner",
            "oasis.mlip.submit",
            "oasis.mlip.tasks",
            "oasis.adapters.rootstock_adapter",
        )
        original_modules = {
            module_name: sys.modules.get(module_name)
            for module_name in (*target_modules, *runtime_modules)
        }
        try:
            for module_name in (*target_modules, *runtime_modules):
                sys.modules.pop(module_name, None)
            before_import = set(sys.modules)

            for module_name in target_modules:
                importlib.import_module(module_name)

            imported = set(sys.modules) - before_import
            for module_name in runtime_modules:
                self.assertNotIn(module_name, imported)
        finally:
            for module_name in (*target_modules, *runtime_modules):
                sys.modules.pop(module_name, None)
            for module_name, module in original_modules.items():
                if module is not None:
                    sys.modules[module_name] = module
