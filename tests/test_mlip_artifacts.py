from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from oasis.mlip.artifacts import (
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
