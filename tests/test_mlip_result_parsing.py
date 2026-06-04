from __future__ import annotations

import importlib
import sys
import unittest

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


class IoDependencyBoundaryTests(unittest.TestCase):
    def test_importing_io_does_not_import_analysis(self) -> None:
        sys.modules.pop("oasis.io", None)
        sys.modules.pop("oasis.analysis", None)
        before_import = set(sys.modules)

        importlib.import_module("oasis.io")

        self.assertNotIn("oasis.analysis", set(sys.modules) - before_import)
