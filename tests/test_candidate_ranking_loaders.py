from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from oasis.candidate_ranking import load_screening_input_records


def _write_result_file(path: Path, *, median_energy: float, parent_slab_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "calculation_settings": {
                    "chemical_bond_cutoff": 1.25,
                    "n_crit_relax": 200,
                },
                "rxn-1->N*": {
                    "reference": {"ads_eng": 1.0},
                    "single_calculation": {"ads_eng": 1.25},
                    "final": {
                        "median_num": 0,
                        "ads_eng_median": median_energy,
                        "ads_seed_range": 0.0,
                        "ads_eng_seed_range": 0.0,
                    },
                    "0": {
                        "adslab_steps": 50,
                        "substrate_displacement": 0.1,
                        "max_bond_change": 5.0,
                    },
                    "metadata": {
                        "reference": {
                            "parent_slab_id": parent_slab_id,
                            "adslab_id": "adslab-000001",
                            "initial_site_label": "bridge",
                            "initial_site_coordinate": [1.0, 2.0, 3.0],
                            "top_layer_motif": "heterodimer",
                            "adsorbate": "N",
                        },
                        "structures": {
                            "slab": {
                                "parent_slab_id": parent_slab_id,
                                "adslab_id": "adslab-000001",
                                "surface_type": "fcc111",
                                "supercell_size": [3, 3, 4],
                            },
                            "adslab": {
                                "parent_slab_id": parent_slab_id,
                                "adslab_id": "adslab-000001",
                                "initial_site_label": "bridge",
                                "initial_site_coordinate": [1.0, 2.0, 3.0],
                                "top_layer_motif": "heterodimer",
                                "adsorbate": "N",
                                "site_family": "terrace",
                            },
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )


class CandidateRankingLoaderTests(unittest.TestCase):
    def test_load_screening_input_records_preserves_metadata_and_anomalies(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            normal_path = base_dir / "mace" / "mace_result.json"
            anomaly_path = base_dir / "orb" / "orb_result.json"
            _write_result_file(normal_path, median_energy=1.1, parent_slab_id="slab-7")
            _write_result_file(anomaly_path, median_energy=4.5, parent_slab_id="slab-7")

            records = load_screening_input_records([normal_path, anomaly_path])

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.parent_slab_id, "slab-7")
        self.assertEqual(record.adslab_id, "adslab-000001")
        self.assertEqual(record.adsorbate, "N")
        self.assertEqual(record.site_metadata["initial_site_label"], "bridge")
        self.assertEqual(record.site_metadata["top_layer_motif"], "heterodimer")
        self.assertEqual(record.slab_metadata["surface_type"], "fcc111")
        self.assertEqual(record.adslab_metadata["site_family"], "terrace")
        self.assertEqual(tuple(p.model_name for p in record.model_predictions), ("mace", "orb"))
        self.assertEqual(record.model_predictions[0].anomaly.label, "normal")
        self.assertEqual(record.model_predictions[1].anomaly.label, "energy_anomaly")
        self.assertEqual(
            record.model_predictions[1].anomaly.details["energy_anomaly"],
            1,
        )

    def test_load_screening_input_records_rejects_inconsistent_parent_metadata(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            first_path = base_dir / "mace" / "mace_result.json"
            second_path = base_dir / "orb" / "orb_result.json"
            _write_result_file(first_path, median_energy=1.1, parent_slab_id="slab-7")
            _write_result_file(second_path, median_energy=1.2, parent_slab_id="slab-9")

            with self.assertRaisesRegex(ValueError, "Inconsistent screening metadata"):
                load_screening_input_records([first_path, second_path])
