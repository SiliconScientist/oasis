import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from ase import Atoms
from ase.constraints import dict2constraint
from ase.io.jsonio import object_hook

from oasis.config import Config, get_config


def load_mlip_dataset_subset(cfg: Config, limit: int = 10) -> dict[str, Any]:
    """
    Load cfg.mlip.dataset JSON as a dictionary and return a small subset.
    """
    dataset_path = cfg.mlip.dataset
    if not dataset_path:
        raise ValueError("cfg.mlip.dataset is not set in config.toml")

    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"MLIP dataset JSON not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not isinstance(dataset, dict):
        raise TypeError(
            f"Expected cfg.mlip.dataset JSON top-level to be an object/dict, got {type(dataset).__name__}"
        )

    return dict(list(dataset.items())[:limit])


def atoms_from_atoms_json(atoms_json: str) -> Atoms:
    """
    Decode ASE DB-style JSON payload and build an ASE Atoms object.
    """
    decoded = json.loads(atoms_json, object_hook=object_hook)
    row_key = next((k for k in decoded if str(k).isdigit()), None)
    if row_key is None:
        raise ValueError("Could not find atom row key in atoms_json payload")

    row = decoded[row_key]
    atoms = Atoms(
        numbers=row["numbers"],
        positions=row["positions"],
        cell=row["cell"],
        pbc=row["pbc"],
    )

    # Preserve constraints if present in serialized payload.
    constraints = row.get("constraints", row.get("constraint"))
    if constraints not in (None, [], ()):
        if isinstance(constraints, dict):
            constraints = dict2constraint(constraints)
        elif isinstance(constraints, list):
            constraints = [
                dict2constraint(c) if isinstance(c, dict) else c for c in constraints
            ]
        atoms.set_constraint(constraints)

    return atoms


def extract_adsorbed_atoms(dataset_subset: dict[str, Any]) -> dict[str, Atoms]:
    """
    Extract adsorbed ASE Atoms objects from reaction entries.

    Expected entry shape:
      entry["raw"] contains a key like "Nstar", "OHstar", etc.
      The bare slab key "star" is ignored.
    """
    adsorbed_atoms: dict[str, Atoms] = {}

    for reaction, entry in dataset_subset.items():
        raw = entry.get("raw", {})
        if not isinstance(raw, dict):
            raise TypeError(f"Entry '{reaction}' has non-dict raw payload")

        adsorbed_keys = [k for k in raw.keys() if k.endswith("star") and k != "star"]
        if not adsorbed_keys:
            raise ValueError(f"No adsorbed '*star' key found in entry '{reaction}'")

        adsorbed_key = adsorbed_keys[0]
        adsorbed_block = raw[adsorbed_key]
        if not isinstance(adsorbed_block, dict) or "atoms_json" not in adsorbed_block:
            raise ValueError(
                f"Entry '{reaction}' key '{adsorbed_key}' is missing 'atoms_json'"
            )

        adsorbed_atoms[reaction] = atoms_from_atoms_json(adsorbed_block["atoms_json"])

    return adsorbed_atoms


def extract_adsorbate_indices(dataset_subset: dict[str, Any]) -> dict[str, list[int]]:
    """
    Extract adsorbate atom indices from each reaction entry.

    Expected entry shape:
      entry["adsorbate_indices"] = [int, int, ...]
    """
    adsorbate_indices_by_reaction: dict[str, list[int]] = {}

    for reaction, entry in dataset_subset.items():
        if "adsorbate_indices" not in entry:
            raise ValueError(f"Entry '{reaction}' is missing 'adsorbate_indices'")

        indices = entry["adsorbate_indices"]
        if not isinstance(indices, list):
            raise TypeError(
                f"Entry '{reaction}' has non-list adsorbate_indices: {type(indices).__name__}"
            )
        if not all(isinstance(i, int) for i in indices):
            raise TypeError(f"Entry '{reaction}' adsorbate_indices must be list[int]")

        adsorbate_indices_by_reaction[reaction] = indices

    return adsorbate_indices_by_reaction


def extract_adsorbate_positions(
    adsorbed_atoms: dict[str, Atoms], adsorbate_indices: dict[str, list[int]]
) -> dict[str, np.ndarray]:
    """
    Get adsorbate atom positions from each adslab.
    Returns positions with shape (n_adsorbate_atoms, 3) per reaction.
    """
    adsorbate_positions: dict[str, np.ndarray] = {}

    for reaction, atoms in adsorbed_atoms.items():
        if reaction not in adsorbate_indices:
            raise KeyError(f"Missing adsorbate indices for reaction '{reaction}'")
        indices = adsorbate_indices[reaction]
        adsorbate_positions[reaction] = atoms.get_positions()[indices]

    return adsorbate_positions


def pick_binding_atom_index(
    adslab: Atoms, adsorbate_indices: list[int], z_tolerance: float = 1e-3
) -> Optional[int]:
    """
    Define binding atom as:
      1) adsorbate atom with minimum z
      2) if tie within z_tolerance, pick atom with minimum distance to nearest slab atom
      3) hydrogen atoms are excluded from candidacy
    Returns the atom index in the adslab, or None for hydrogen-only adsorbates.
    """
    if not adsorbate_indices:
        raise ValueError("adsorbate_indices is empty")

    candidate_indices = [i for i in adsorbate_indices if adslab[i].symbol != "H"]
    if not candidate_indices:
        return None

    adsorbate_index_set = set(adsorbate_indices)
    slab_indices = [i for i in range(len(adslab)) if i not in adsorbate_index_set]
    if not slab_indices:
        raise ValueError("No slab atoms found (all atoms are marked as adsorbate)")

    positions = adslab.get_positions()
    z_values = np.array([positions[i, 2] for i in candidate_indices], dtype=float)
    min_z = float(np.min(z_values))

    z_tied = [
        idx
        for idx in candidate_indices
        if abs(float(positions[idx, 2]) - min_z) <= z_tolerance
    ]

    if len(z_tied) == 1:
        return z_tied[0]

    nearest_dist = {}
    for idx in z_tied:
        dists = adslab.get_distances(idx, slab_indices, mic=True)
        nearest_dist[idx] = float(np.min(dists))

    return min(z_tied, key=lambda idx: (nearest_dist[idx], idx))


def extract_binding_atoms(
    adsorbed_atoms: dict[str, Atoms],
    adsorbate_indices: dict[str, list[int]],
    z_tolerance: float = 1e-3,
) -> dict[str, Optional[int]]:
    """
    Get binding atom index (in adslab indexing) for each reaction.
    Returns None when adsorbate is hydrogen-only.
    """
    binding_atoms: dict[str, Optional[int]] = {}

    for reaction, atoms in adsorbed_atoms.items():
        if reaction not in adsorbate_indices:
            raise KeyError(f"Missing adsorbate indices for reaction '{reaction}'")
        binding_atoms[reaction] = pick_binding_atom_index(
            atoms, adsorbate_indices[reaction], z_tolerance=z_tolerance
        )

    return binding_atoms


def main() -> None:
    from ase.visualize import view

    cfg = get_config()
    dataset_subset = load_mlip_dataset_subset(cfg, limit=10)
    adsorbed_atoms = extract_adsorbed_atoms(dataset_subset)
    adsorbate_indices = extract_adsorbate_indices(dataset_subset)
    adsorbate_positions = extract_adsorbate_positions(adsorbed_atoms, adsorbate_indices)
    binding_atoms = extract_binding_atoms(adsorbed_atoms, adsorbate_indices)
    print(
        f"Loaded {len(dataset_subset)} entries and extracted "
        f"{len(adsorbed_atoms)} adsorbed ASE Atoms objects from {cfg.mlip.dataset}"
    )
    print(f"Extracted adsorbate indices for {len(adsorbate_indices)} entries")
    print(f"Extracted adsorbate positions for {len(adsorbate_positions)} entries")
    print(f"Computed binding atoms for {len(binding_atoms)} entries")


if __name__ == "__main__":
    main()
