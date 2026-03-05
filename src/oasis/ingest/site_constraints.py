from functools import partial
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from ase import Atoms
from ase.constraints import dict2constraint, FixAtoms, FixCartesian
from ase.io.jsonio import object_hook
from ase.geometry import find_mic

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
      3) hydrogen atoms are excluded from candidacy unless adsorbate is single-atom H
    Returns the atom index in the adslab, or None when no binding atom can be defined.
    """
    if not adsorbate_indices:
        raise ValueError("adsorbate_indices is empty")

    candidate_indices = [i for i in adsorbate_indices if adslab[i].symbol != "H"]
    if not candidate_indices:
        if len(adsorbate_indices) == 1 and adslab[adsorbate_indices[0]].symbol == "H":
            return adsorbate_indices[0]
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
    Returns None when no binding atom can be defined under selection rules.
    """
    binding_atoms: dict[str, Optional[int]] = {}

    for reaction, atoms in adsorbed_atoms.items():
        if reaction not in adsorbate_indices:
            raise KeyError(f"Missing adsorbate indices for reaction '{reaction}'")
        binding_atoms[reaction] = pick_binding_atom_index(
            atoms, adsorbate_indices[reaction], z_tolerance=z_tolerance
        )

    return binding_atoms


def strip_adsorbate_from_adslab(adslab: Atoms, adsorbate_indices: list[int]) -> Atoms:
    """
    Return slab-only ASE Atoms by removing atoms listed in adsorbate_indices.
    """
    adsorbate_index_set = set(adsorbate_indices)
    slab_indices = [i for i in range(len(adslab)) if i not in adsorbate_index_set]
    if not slab_indices:
        raise ValueError("No slab atoms found (all atoms are marked as adsorbate)")
    return adslab[slab_indices]


def find_adsorption_sites_on_slab(slab: Atoms) -> np.ndarray:
    """
    Find adsorption sites on an ASE slab using pymatgen's AdsorbateSiteFinder.
    Returns Cartesian site coordinates with shape (n_sites, 3).
    Symmetry/near-site reductions are disabled to keep all sites.
    """
    try:
        from pymatgen.analysis.adsorption import AdsorbateSiteFinder
        from pymatgen.io.ase import AseAtomsAdaptor
    except ImportError as exc:
        raise ImportError(
            "pymatgen is required for adsorption site finding. Install pymatgen."
        ) from exc

    slab_pm = AseAtomsAdaptor.get_structure(slab)
    asf = AdsorbateSiteFinder(slab_pm)
    sites = asf.find_adsorption_sites(symm_reduce=0.0, near_reduce=0.0).get("all", [])
    if not sites:
        raise ValueError("AdsorbateSiteFinder returned no adsorption sites")
    return np.array(sites, dtype=float)


def pick_closest_adsorption_site(
    adslab: Atoms, binding_atom_index: int, adsorption_sites: np.ndarray
) -> np.ndarray:
    """
    Pick the adsorption site nearest to the current adsorbate binding atom.
    Distances are evaluated in-plane (x,y only) with MIC under adslab PBC/cell.
    """
    binding_position = adslab.positions[binding_atom_index]
    displacement = np.array(adsorption_sites - binding_position, dtype=float)
    displacement[:, 2] = 0.0

    pbc_xy = np.array(adslab.pbc, dtype=bool)
    pbc_xy[2] = False
    mic_vectors_xy, mic_lengths_xy = find_mic(displacement, adslab.cell, pbc=pbc_xy)
    closest_idx = int(np.argmin(mic_lengths_xy))

    closest_site = np.array(adsorption_sites[closest_idx], dtype=float)
    closest_site[:2] = binding_position[:2] + mic_vectors_xy[closest_idx, :2]
    return closest_site


def shift_adsorbate_to_site(
    adslab: Atoms,
    adsorbate_indices: list[int],
    target_site: np.ndarray,
    binding_atom_index: int,
) -> Atoms:
    """
    Translate all adsorbate atoms so the binding atom lands on target_site.
    """
    shifted = adslab.copy()
    translation = (
        np.asarray(target_site, dtype=float) - shifted.positions[binding_atom_index]
    )
    shifted.positions[adsorbate_indices] += translation
    return shifted


def snap_adsorbate_to_closest_binding_site(
    adslab: Atoms, adsorbate_indices: list[int], z_tolerance: float = 1e-3
) -> tuple[Atoms, np.ndarray]:
    """
    Snap adsorbate to the ASF adsorption site closest to the current binding atom.
    Returns (shifted_adslab, closest_site_cartesian).
    """
    binding_atom_index = pick_binding_atom_index(
        adslab, adsorbate_indices, z_tolerance=z_tolerance
    )
    if binding_atom_index is None:
        raise ValueError("Cannot identify a binding atom for this adsorbate")

    slab = strip_adsorbate_from_adslab(adslab, adsorbate_indices)
    adsorption_sites = find_adsorption_sites_on_slab(slab)
    closest_site = pick_closest_adsorption_site(
        adslab, binding_atom_index, adsorption_sites
    )
    shifted_adslab = shift_adsorbate_to_site(
        adslab, adsorbate_indices, closest_site, binding_atom_index
    )
    return shifted_adslab, closest_site


def snap_all_adsorbates_to_closest_binding_sites(
    adsorbed_atoms: dict[str, Atoms],
    adsorbate_indices: dict[str, list[int]],
    z_tolerance: float = 1e-3,
) -> dict[str, Atoms]:
    """
    Batch helper that snaps all adslabs to their nearest ASF binding site.
    """
    shifted: dict[str, Atoms] = {}
    for reaction, adslab in adsorbed_atoms.items():
        if reaction not in adsorbate_indices:
            raise KeyError(f"Missing adsorbate indices for reaction '{reaction}'")
        shifted[reaction], _ = snap_adsorbate_to_closest_binding_site(
            adslab, adsorbate_indices[reaction], z_tolerance=z_tolerance
        )
    return shifted


def index_by_height(atoms: Atoms, cutoff: float, below: bool = True) -> list[int]:
    """
    Return atom indices for atoms below a specified height cutoff.
    """
    z_values = atoms.get_positions()[:, 2]
    if below:
        return list(np.where(z_values < cutoff)[0])
    else:
        return list(np.where(z_values >= cutoff)[0])


def fix_atoms(atoms_list: list[Atoms], index_fn: callable) -> list[Atoms]:
    """
    Return new list of ASE Atoms with FixAtoms constraints applied to indices from index_fn.
    """
    fixed_atoms = []
    for atoms in atoms_list:
        indices_to_fix = index_fn(atoms)
        constraint = FixAtoms(indices=indices_to_fix)
        fixed = atoms.copy()
        fixed.set_constraint(constraint)
        fixed_atoms.append(fixed)
    return fixed_atoms


def fix_binding_atoms_xy(
    atoms_by_reaction: dict[str, Atoms], binding_atoms: dict[str, Optional[int]]
) -> dict[str, Atoms]:
    """
    Return structures with binding atoms fixed in x and y only (z remains free).
    Existing constraints are preserved and augmented.
    """
    constrained: dict[str, Atoms] = {}
    for reaction, atoms in atoms_by_reaction.items():
        binding_idx = binding_atoms.get(reaction)
        updated = atoms.copy()

        if binding_idx is None:
            constrained[reaction] = updated
            continue

        existing = updated.constraints
        if existing is None:
            constraints = []
        elif isinstance(existing, (list, tuple)):
            constraints = list(existing)
        else:
            constraints = [existing]

        constraints.append(FixCartesian([binding_idx], mask=(True, True, False)))
        updated.set_constraint(constraints)
        constrained[reaction] = updated

    return constrained


def main() -> None:
    from ase.visualize import view

    cfg = get_config()
    index_fn = partial(index_by_height, cutoff=13.5, below=True)
    dataset_subset = load_mlip_dataset_subset(cfg, limit=10)
    adsorbed_atoms = extract_adsorbed_atoms(dataset_subset)
    adsorbate_indices = extract_adsorbate_indices(dataset_subset)
    adsorbate_positions = extract_adsorbate_positions(adsorbed_atoms, adsorbate_indices)
    binding_atoms = extract_binding_atoms(adsorbed_atoms, adsorbate_indices)
    shifted_adslabs = snap_all_adsorbates_to_closest_binding_sites(
        adsorbed_atoms, adsorbate_indices
    )
    shifted_reactions = list(shifted_adslabs.keys())
    shifted_adslabs_constrained_list = fix_atoms(
        [shifted_adslabs[r] for r in shifted_reactions], index_fn=index_fn
    )
    shifted_adslabs_constrained = dict(
        zip(shifted_reactions, shifted_adslabs_constrained_list, strict=True)
    )
    shifted_adslabs_constrained = fix_binding_atoms_xy(
        shifted_adslabs_constrained, binding_atoms
    )
    print(
        f"Loaded {len(dataset_subset)} entries and extracted "
        f"{len(adsorbed_atoms)} adsorbed ASE Atoms objects from {cfg.mlip.dataset}"
    )
    print(f"Extracted adsorbate indices for {len(adsorbate_indices)} entries")
    print(f"Extracted adsorbate positions for {len(adsorbate_positions)} entries")
    print(f"Computed binding atoms for {len(binding_atoms)} entries")
    print(f"Shifted adsorbates to closest ASF sites for {len(shifted_adslabs)} entries")
    print(
        "Applied index_fn constraints to shifted adslabs for "
        f"{len(shifted_adslabs_constrained)} entries"
    )
    print(
        "Applied x/y FixCartesian constraints to binding atoms for shifted adslabs"
    )


if __name__ == "__main__":
    main()
