from __future__ import annotations

from functools import partial
from itertools import islice
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.db.row import AtomsRow
from ase.io import jsonio

from oasis.ingest.site_constraints import (
    atoms_to_atoms_json_like_template,
    extract_adsorbate_indices,
    extract_adsorbed_atom,
    find_adsorption_sites_on_slab,
    fix_binding_atom_xy,
    index_by_layers,
    load_mlip_dataset,
    plane_from_lowest_atoms,
    rewrap_slab_by_largest_gap,
    strip_adsorbate_from_adslab,
)


DEFAULT_JSON_PATH = Path(__file__).with_name("KHLOHC_origin_adsorption.json")
ADSORPTION_SITE_TOLERANCE = 0.5
GAS_CELL_LENGTH = 15.0


def atoms_from_ase_db_json(atoms_json: str) -> Atoms:
    """Convert an ASE database-style JSON string into an ASE Atoms object."""
    decoded = jsonio.decode(atoms_json)
    row_id = decoded["ids"][0]
    row = decoded[row_id]
    atom_count = len(row["numbers"])
    for key in (
        "momenta",
        "masses",
        "tags",
        "initial_charges",
        "initial_magmoms",
    ):
        values = row.get(key)
        if values is not None and len(values) != atom_count:
            row.pop(key)
    return AtomsRow(row).toatoms()


def unique_probe_output_path(input_dataset_path: Path) -> Path:
    """Build the output path for the unique probe dataset."""
    stem = input_dataset_path.stem
    lower_stem = stem.lower()
    marker = "tolstar"
    if marker in lower_stem:
        start = lower_stem.index(marker)
        stem = f"{stem[:start]}unique_probe{stem[start + len(marker):]}"
    else:
        stem = f"{stem}_unique_probe"
    return input_dataset_path.with_name(f"{stem}{input_dataset_path.suffix}")


def updated_dataset_output_path(input_dataset_path: Path) -> Path:
    """Build the output path for the original dataset with probe ids added."""
    stem = input_dataset_path.stem
    return input_dataset_path.with_name(
        f"{stem}_with_probe_ids{input_dataset_path.suffix}"
    )


def wrap_atoms_json(atoms: Atoms) -> str:
    """Serialize Atoms into the ASE DB-style wrapped payload used by the dataset."""
    row = json.loads(jsonio.encode(atoms))
    return json.dumps({"1": row, "ids": [1], "nextid": 2})


def marker_atoms(
    reference_atoms: Atoms, positions: np.ndarray, marker_symbol: str = "H"
) -> Atoms:
    """Return marker atoms placed in the same cell as a reference structure."""
    positions = np.asarray(positions, dtype=float)
    if positions.size == 0:
        return Atoms(cell=reference_atoms.cell, pbc=reference_atoms.pbc)
    return Atoms(
        symbols=[marker_symbol] * len(positions),
        positions=positions,
        cell=reference_atoms.cell,
        pbc=reference_atoms.pbc,
    )


def append_atoms(base_atoms: Atoms, extra_atoms: Atoms) -> Atoms:
    """Return a copy of `base_atoms` with `extra_atoms` appended."""
    combined = base_atoms.copy()
    if len(extra_atoms) > 0:
        combined.extend(extra_atoms)
    return combined


def gas_reference_atoms(formula: str, cell_length: float = GAS_CELL_LENGTH) -> Atoms:
    """Build a simple gas-phase molecule in a cubic periodic box."""
    gas = molecule(formula)
    gas.set_cell(np.eye(3) * cell_length)
    gas.center()
    gas.set_pbc(True)
    return gas
