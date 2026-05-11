from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from ase import Atoms
try:
    import polars as pl
    from oasis.config import Config
    from oasis.io import load_sample_atoms_for_wide_df

    HAS_IO_DEPS = True
except ModuleNotFoundError:
    HAS_IO_DEPS = False


def _row_wrapped_atoms_json(atoms: Atoms) -> str:
    return json.dumps(
        {
            "1": {
                "numbers": atoms.numbers.tolist(),
                "positions": atoms.positions.tolist(),
                "cell": atoms.cell.tolist(),
                "pbc": atoms.pbc.tolist(),
            }
        }
    )


@unittest.skipUnless(HAS_IO_DEPS, "requires io dependencies")
class LoadSampleAtomsForWideDfTests(unittest.TestCase):
    def test_loads_atoms_in_wide_df_row_order(self) -> None:
        first_atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
        second_atoms = Atoms("He", positions=[[1.0, 0.0, 0.0]])
        dataset = {
            "rxn-b": {"raw": {"OHstar": {"atoms_json": _row_wrapped_atoms_json(second_atoms)}}},
            "rxn-a": {"raw": {"OHstar": {"atoms_json": _row_wrapped_atoms_json(first_atoms)}}},
        }
        wide_df = pl.DataFrame({"reaction": ["rxn-a", "rxn-b"]})

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset.json"
            dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
            cfg = Config(
                **{
                    "ingest": {
                        "source": "data/raw_vasp/systems",
                        "dataset_name": "test",
                        "stoich": {
                            "elements": ["H"],
                            "basis_species": ["H2"],
                            "basis_composition": {"H2": {"H": 2}},
                        },
                    },
                    "mlip": {
                        "dev_n": 1,
                        "dev_run": False,
                        "dataset": str(dataset_path),
                        "models": {"enabled": []},
                        "rootstock": {"root": tmp_dir, "models": {}},
                    },
                }
            )

            sample_atoms = load_sample_atoms_for_wide_df(wide_df, cfg)

        self.assertEqual([atoms.get_chemical_formula() for atoms in sample_atoms], ["H", "He"])

    def test_raises_for_missing_reaction(self) -> None:
        dataset = {
            "rxn-a": {"raw": {"OHstar": {"atoms_json": _row_wrapped_atoms_json(Atoms("H"))}}}
        }
        wide_df = pl.DataFrame({"reaction": ["rxn-a", "rxn-missing"]})

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset.json"
            dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
            cfg = Config(
                **{
                    "ingest": {
                        "source": "data/raw_vasp/systems",
                        "dataset_name": "test",
                        "stoich": {
                            "elements": ["H"],
                            "basis_species": ["H2"],
                            "basis_composition": {"H2": {"H": 2}},
                        },
                    },
                    "mlip": {
                        "dev_n": 1,
                        "dev_run": False,
                        "dataset": str(dataset_path),
                        "models": {"enabled": []},
                        "rootstock": {"root": tmp_dir, "models": {}},
                    },
                }
            )

            with self.assertRaisesRegex(
                KeyError,
                "rxn-missing",
            ):
                load_sample_atoms_for_wide_df(wide_df, cfg)
