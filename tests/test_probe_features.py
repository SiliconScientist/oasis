from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from oasis.probe_features import (
    add_mlip_feature_matrices_to_dataset,
    build_feature_matrix,
    load_mlip_probe_energies,
)


def _write_probe_dataset(path: Path) -> None:
    dataset = {
        "rxn-1": {
            "unique_probe_ids": ["1", "2"],
        },
        "rxn-2": {
            "unique_probe_ids": ["2"],
        },
    }
    path.write_text(json.dumps(dataset), encoding="utf-8")


def _write_probe_results(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "mace_result.json").write_text(
        json.dumps(
            {
                "unique_probe_1": {"final": {"ads_eng_median": 1.5}},
                "unique_probe_2": {"final": {"ads_eng_median": 2.5}},
            }
        ),
        encoding="utf-8",
    )
    (results_dir / "orb_result.json").write_text(
        json.dumps(
            {
                "unique_probe_2": {"final": {"ads_eng_median": 3.5}},
            }
        ),
        encoding="utf-8",
    )


class ProbeFeatureTests(unittest.TestCase):
    def test_load_mlip_probe_energies_with_explicit_path(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            results_dir = Path(tmp_dir) / "probe_results"
            _write_probe_results(results_dir)

            result = load_mlip_probe_energies(results_dir)

        self.assertEqual(set(result), {"mace", "orb"})
        self.assertEqual(result["mace"]["1"], 1.5)
        self.assertNotIn("1", result["orb"])

    def test_add_mlip_feature_matrices_uses_config_paths_when_args_omitted(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_path = tmp_path / "dataset_with_probe_ids.json"
            results_dir = tmp_path / "probe_results"
            _write_probe_dataset(dataset_path)
            _write_probe_results(results_dir)
            cfg = SimpleNamespace(
                probe_features=SimpleNamespace(
                    dataset_path=dataset_path,
                    mlip_results_dir=results_dir,
                )
            )

            with patch("oasis.probe_features.get_config", return_value=cfg):
                updated = add_mlip_feature_matrices_to_dataset()
            saved = json.loads(dataset_path.read_text(encoding="utf-8"))

        self.assertEqual(
            saved["rxn-1"]["mlip_feature_matrix"]["mlip_names"],
            ["mace", "orb"],
        )
        matrix = saved["rxn-1"]["mlip_feature_matrix"]["matrix"]
        self.assertEqual(matrix[0][0], 1.5)
        self.assertTrue(np.isnan(matrix[0][1]))
        self.assertEqual(matrix[1][0], 2.5)
        self.assertEqual(matrix[1][1], 3.5)
        updated_matrix = updated["rxn-1"]["mlip_feature_matrix"]["matrix"]
        self.assertEqual(updated_matrix[0][0], 1.5)
        self.assertTrue(np.isnan(updated_matrix[0][1]))
        self.assertEqual(updated_matrix[1][0], 2.5)
        self.assertEqual(updated_matrix[1][1], 3.5)

    def test_build_feature_matrix_uses_config_paths_when_args_omitted(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_path = tmp_path / "dataset_with_probe_ids.json"
            results_dir = tmp_path / "probe_results"
            _write_probe_dataset(dataset_path)
            _write_probe_results(results_dir)
            cfg = SimpleNamespace(
                probe_features=SimpleNamespace(
                    dataset_path=dataset_path,
                    mlip_results_dir=results_dir,
                )
            )

            with patch("oasis.probe_features.get_config", return_value=cfg):
                matrix = build_feature_matrix("rxn-2")

        self.assertEqual(matrix.reaction, "rxn-2")
        self.assertEqual(matrix.mlip_names, ["mace", "orb"])
        np.testing.assert_allclose(matrix.matrix, np.array([[2.5], [3.5]]))

    def test_missing_probe_feature_config_raises_clear_error(self) -> None:
        cfg = SimpleNamespace(probe_features=None)

        with patch("oasis.probe_features.get_config", return_value=cfg):
            with self.assertRaisesRegex(
                ValueError,
                "probe_features.dataset_path and probe_features.mlip_results_dir",
            ):
                load_mlip_probe_energies()

    def test_probe_feature_loader_ignores_processed_result_files(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            results_dir = Path(tmp_dir) / "probe_results"
            _write_probe_results(results_dir)
            (results_dir / "mace_processed_result.json").write_text(
                json.dumps({"unique_probe_1": {"final": {"ads_eng_median": 9.9}}}),
                encoding="utf-8",
            )

            result = load_mlip_probe_energies(results_dir)

        self.assertEqual(set(result), {"mace", "orb"})
