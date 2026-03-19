from functools import partial
import json
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from ase import Atoms
from ase.visualize import view
from ase.constraints import FixAtoms, FixCartesian, dict2constraint
from ase.geometry import find_mic
from ase.io.jsonio import encode, object_hook

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


def load_mlip_dataset(cfg: Config) -> dict[str, Any]:
    """
    Load cfg.mlip.dataset JSON as a dictionary.
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

    return dataset


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


def extract_adsorbed_atom(entry: dict[str, Any], reaction: str) -> Atoms:
    """
    Extract adsorbed ASE Atoms objects from reaction entries.

    Expected entry shape:
      entry["raw"] contains a key like "Nstar", "OHstar", etc.
      The bare slab key "star" is ignored.
    """
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

    return atoms_from_atoms_json(adsorbed_block["atoms_json"])


def extract_adsorbate_indices(entry: dict[str, Any], reaction: str) -> list[int]:
    """
    Extract adsorbate atom indices from each reaction entry.

    Expected entry shape:
      entry["adsorbate_indices"] = [int, int, ...]
    """
    if "adsorbate_indices" not in entry:
        raise ValueError(f"Entry '{reaction}' is missing 'adsorbate_indices'")

    indices = entry["adsorbate_indices"]
    if not isinstance(indices, list):
        raise TypeError(
            f"Entry '{reaction}' has non-list adsorbate_indices: {type(indices).__name__}"
        )
    if not all(isinstance(i, int) for i in indices):
        raise TypeError(f"Entry '{reaction}' adsorbate_indices must be list[int]")

    return indices


def extract_adsorbate_positions(
    adsorbed_atom: Atoms, adsorbate_indices: list[int]
) -> np.ndarray:
    """
    Get adsorbate atom positions from each adslab.
    Returns positions with shape (n_adsorbate_atoms, 3) per reaction.
    """
    return adsorbed_atom.get_positions()[adsorbate_indices]


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


def extract_binding_atom(
    adsorbed_atom: Atoms, adsorbate_indices: list[int], z_tolerance: float = 1e-3
) -> Optional[int]:
    """
    Get binding atom index (in adslab indexing) for each reaction.
    Returns None when no binding atom can be defined under selection rules.
    """
    return pick_binding_atom_index(
        adsorbed_atom, adsorbate_indices, z_tolerance=z_tolerance
    )


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

    # Rigid translation placing binding atom at target site
    translation = (
        np.asarray(target_site, dtype=float) - shifted.positions[binding_atom_index]
    )
    shifted.positions[adsorbate_indices] += translation
    return shifted


def add_binding_site_markers(
    adslab: Atoms, adsorption_sites: np.ndarray, marker_symbol: str = "H"
) -> Atoms:
    """
    Return a visualization structure by placing marker atoms at all adsorption sites.
    """
    markers = Atoms(
        symbols=[marker_symbol] * len(adsorption_sites),
        positions=adsorption_sites,
        cell=adslab.cell,
        pbc=adslab.pbc,
    )
    visual = adslab.copy()
    visual.extend(markers)
    return visual


def place_adsorbate_on_closest_binding_site(
    adslab: Atoms,
    adsorbate_indices: list[int],
    z_tolerance: float = 1e-3,
) -> Atoms:
    """
    Place the adsorbate on the ASF adsorption site nearest to the current binding atom.
    """
    binding_atom_index = pick_binding_atom_index(
        adslab, adsorbate_indices, z_tolerance=z_tolerance
    )
    if binding_atom_index is None:
        raise ValueError("Cannot identify a binding atom for this adsorbate")

    slab = strip_adsorbate_from_adslab(adslab, adsorbate_indices)
    adsorption_sites = find_adsorption_sites_on_slab(slab)
    target_site = pick_closest_adsorption_site(
        adslab, binding_atom_index, adsorption_sites
    )
    return shift_adsorbate_to_site(
        adslab, adsorbate_indices, target_site, binding_atom_index
    )


def index_by_height(atoms: Atoms, cutoff: float, below: bool = True) -> list[int]:
    """
    Return atom indices for atoms below a specified height cutoff.
    """
    z_values = atoms.get_positions()[:, 2]
    if below:
        return list(np.where(z_values < cutoff)[0])
    else:
        return list(np.where(z_values >= cutoff)[0])


def index_by_layers(atoms: Atoms, layers: tuple[int, ...] = (1, 2)) -> list[int]:
    layer_indices = get_layer_indices(atoms)
    indices: list[int] = []
    for layer in layers:
        if layer not in layer_indices:
            raise ValueError(
                f"Requested layer {layer}, but only layers 1..{len(layer_indices)} exist"
            )
        indices.extend(layer_indices[layer])
    return sorted(indices)


def fix_atoms(atoms: Atoms, indices: list[int]) -> Atoms:
    """
    Return new ASE Atoms with FixAtoms constraints applied to the selected indices.
    """
    constraint = FixAtoms(indices=indices)
    fixed = atoms.copy()
    fixed.set_constraint(constraint)
    return fixed


def fix_binding_atom_xy(
    atoms: Atoms,
    binding_atom: Optional[int],
) -> Atoms:
    """
    Return a copy with the binding atom fixed in x and y while leaving z free.
    Existing constraints are preserved and augmented.
    """
    updated = atoms.copy()

    if binding_atom is None:
        return updated

    existing = updated.constraints
    if existing is None:
        constraints = []
    elif isinstance(existing, (list, tuple)):
        constraints = list(existing)
    else:
        constraints = [existing]

    constraints.append(FixCartesian(binding_atom, mask=(True, True, False)))
    updated.set_constraint(constraints)
    return updated


def atoms_to_atoms_json_like_template(atoms: Atoms, template_atoms_json: str) -> str:
    """
    Serialize ASE Atoms into the same row-wrapped atoms_json schema as template_atoms_json.
    """
    template = json.loads(template_atoms_json)
    row_key = next((k for k in template if str(k).isdigit()), None)
    if row_key is None:
        raise ValueError("Could not find atom row key in template atoms_json payload")

    template_row = template[row_key]
    if not isinstance(template_row, dict):
        raise TypeError("Template atoms_json row is not a dict")

    encoded = json.loads(encode(atoms))
    row = dict(template_row)
    row["numbers"] = encoded["numbers"]
    row["positions"] = encoded["positions"]
    row["cell"] = encoded["cell"]
    row["pbc"] = encoded["pbc"]
    if "constraints" in encoded:
        row["constraints"] = encoded["constraints"]
    else:
        row.pop("constraints", None)
        row.pop("constraint", None)

    template[row_key] = row
    return json.dumps(template)


def build_shifted_constrained_adsorption_entry(
    entry: dict[str, Any], shifted_atoms: Atoms, reaction: str
) -> dict[str, Any]:
    """
    Return a new dataset where each adsorbed *star atoms_json is replaced with shifted+constrained atoms.
    """
    updated_entry = json.loads(json.dumps(entry))
    raw = updated_entry.get("raw", {})
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

    adsorbed_block["atoms_json"] = atoms_to_atoms_json_like_template(
        shifted_atoms, adsorbed_block["atoms_json"]
    )
    return updated_entry


def shifted_adsorption_output_path(input_dataset_path: Path) -> Path:
    """
    Build output path for shifted adsorption dataset.
    """
    if input_dataset_path.name.endswith("_adsorption.json"):
        new_name = input_dataset_path.name.replace(
            "_adsorption.json", "_shifted_adsorption.json"
        )
    elif input_dataset_path.suffix == ".json":
        new_name = f"{input_dataset_path.stem}_shifted_adsorption.json"
    else:
        new_name = f"{input_dataset_path.name}_shifted_adsorption.json"
    return input_dataset_path.with_name(new_name)


def best_fit_plane(points: np.ndarray):
    """
    points: (N, 3) array of xyz coordinates

    Returns
    -------
    centroid : (3,) array
    normal   : (3,) unit normal vector
    d        : scalar in plane equation ax + by + cz + d = 0
    rms      : RMS perpendicular distance of points to plane
    """
    points = np.asarray(points, dtype=float)
    centroid = points.mean(axis=0)

    Q = points - centroid

    # SVD: normal is the direction with smallest singular value
    _, _, vh = np.linalg.svd(Q, full_matrices=False)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)

    d = -np.dot(normal, centroid)

    distances = Q @ normal
    rms = np.sqrt(np.mean(distances**2))

    return centroid, normal, d, rms


def random_hydrogen_markers_on_plane(
    atoms: Atoms,
    n_markers: int = 200,
    padding: float = 1.0,
    seed: Optional[int] = 0,
    lowest_z_tolerance: float = 0.1,
) -> Atoms:
    """
    Build H marker atoms randomly distributed on the bottom-layer-fit plane.

    Marker extents are inferred from slab atom projections onto two in-plane axes,
    then expanded by `padding` in each axis.
    """
    if n_markers <= 0:
        raise ValueError("n_markers must be > 0")
    positions = atoms.get_positions()
    centroid, normal, _ = plane_from_lowest_atoms(
        atoms, lowest_z_tolerance=lowest_z_tolerance
    )

    helper_axis = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(helper_axis, normal))) > 0.9:
        helper_axis = np.array([0.0, 1.0, 0.0], dtype=float)

    u_axis = np.cross(normal, helper_axis)
    u_axis /= np.linalg.norm(u_axis)
    v_axis = np.cross(normal, u_axis)
    v_axis /= np.linalg.norm(v_axis)

    rel = positions - centroid
    u_coords = rel @ u_axis
    v_coords = rel @ v_axis

    u_min, u_max = float(np.min(u_coords)) - padding, float(np.max(u_coords)) + padding
    v_min, v_max = float(np.min(v_coords)) - padding, float(np.max(v_coords)) + padding

    rng = np.random.default_rng(seed)
    sample_u = rng.uniform(u_min, u_max, size=n_markers)
    sample_v = rng.uniform(v_min, v_max, size=n_markers)
    marker_positions = (
        centroid + np.outer(sample_u, u_axis) + np.outer(sample_v, v_axis)
    )

    return Atoms(
        symbols=["H"] * n_markers,
        positions=marker_positions,
        cell=atoms.cell,
        pbc=atoms.pbc,
    )


def oxygen_markers_along_normal(
    atoms: Atoms,
    n_oxygen_markers: int = 20,
    oxygen_spacing: float = 0.8,
    oxygen_start_offset: float = 0.0,
    lowest_z_tolerance: float = 0.1,
) -> Atoms:
    """
    Build O marker atoms in a line from plane centroid along the oriented normal.
    """
    if n_oxygen_markers <= 0:
        raise ValueError("n_oxygen_markers must be > 0")
    if oxygen_spacing <= 0:
        raise ValueError("oxygen_spacing must be > 0")

    centroid, normal, _ = plane_from_lowest_atoms(
        atoms, lowest_z_tolerance=lowest_z_tolerance
    )
    offsets = oxygen_start_offset + oxygen_spacing * np.arange(
        n_oxygen_markers, dtype=float
    )
    marker_positions = centroid + np.outer(offsets, normal)

    return Atoms(
        symbols=["O"] * n_oxygen_markers,
        positions=marker_positions,
        cell=atoms.cell,
        pbc=atoms.pbc,
    )


def build_plane_visualization(
    atoms: Atoms,
    n_markers: int = 200,
    padding: float = 1.0,
    seed: Optional[int] = 0,
    lowest_z_tolerance: float = 0.1,
    n_oxygen_markers: int = 20,
    oxygen_spacing: float = 0.8,
    oxygen_start_offset: float = 0.0,
) -> Atoms:
    """
    Return a copy of atoms with random H plane markers and an O normal-line overlay.
    """
    visualization = atoms.copy()
    visualization += random_hydrogen_markers_on_plane(
        atoms,
        n_markers=n_markers,
        padding=padding,
        seed=seed,
        lowest_z_tolerance=lowest_z_tolerance,
    )
    visualization += oxygen_markers_along_normal(
        atoms,
        n_oxygen_markers=n_oxygen_markers,
        oxygen_spacing=oxygen_spacing,
        oxygen_start_offset=oxygen_start_offset,
        lowest_z_tolerance=lowest_z_tolerance,
    )
    return visualization


def get_lowest_atom_indices(atoms: Atoms, z_tolerance: float = 0.5) -> list[int]:
    """
    Return indices of atoms belonging to the lowest-z group.

    The lowest atom is identified from the Cartesian z coordinates, and any
    other atom whose z coordinate lies within `z_tolerance` of that minimum
    is included.

    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object.
    z_tolerance : float, default=1e-3
        Maximum allowed difference in Cartesian z coordinate from the minimum-z
        atom for inclusion in the lowest group.

    Returns
    -------
    list[int]
        Indices of atoms in the lowest-z group.
    """
    positions = atoms.get_positions()
    z = positions[:, 2]
    z_min = np.min(z)

    lowest_indices = np.where(np.abs(z - z_min) <= z_tolerance)[0]
    return lowest_indices.tolist()


def plane_from_lowest_atoms(
    atoms: Atoms, lowest_z_tolerance: float = 0.5
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Fit a plane using only the lowest-z atom group.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        (centroid, normal, d) for plane ax + by + cz + d = 0
    """
    repeated_atoms = atoms.repeat(rep=(2, 2, 1))
    lowest_indices = get_lowest_atom_indices(
        repeated_atoms, z_tolerance=lowest_z_tolerance
    )
    if len(lowest_indices) < 3:
        raise ValueError(
            "At least 3 atoms are required in the lowest group to define a plane"
        )
    lowest_points = repeated_atoms.get_positions()[lowest_indices]
    centroid, normal, d, _ = best_fit_plane(lowest_points)

    bottom_centroid = lowest_points.mean(axis=0)
    all_centroid = repeated_atoms.get_positions().mean(axis=0)
    if float(np.dot(normal, all_centroid - bottom_centroid)) < 0.0:
        normal = -normal
        d = -d
    return centroid, normal, d


def get_layer_indices(atoms: Atoms, z_tolerance: float = 0.5) -> dict[int, list[int]]:
    """
    Identify atomic layers in a slab.

    The lowest layer of host atoms is used to define a best-fit surface plane.
    All host atoms are then assigned layers by clustering their heights above
    this plane.

    Returns
    -------
    dict[int, list[int]]
        Mapping from layer index to atom indices, where layer 1 is the
        bottom-most layer.
    """
    _, normal, d = plane_from_lowest_atoms(atoms, lowest_z_tolerance=0.5)

    heights = atoms.get_positions() @ normal + d

    sorted_indices = np.argsort(heights)
    sorted_heights = heights[sorted_indices]

    layers: list[list[int]] = []
    current_layer = [sorted_indices[0]]
    for i in range(1, len(atoms)):
        if abs(sorted_heights[i] - sorted_heights[i - 1]) <= z_tolerance:
            current_layer.append(sorted_indices[i])
        else:
            layers.append(current_layer)
            current_layer = [sorted_indices[i]]
    layers.append(current_layer)

    # Use 1-based layer keys so callers can request (1, 2) for bottom layers.
    layer_indices = {layer_idx + 1: layer for layer_idx, layer in enumerate(layers)}
    return layer_indices


def main() -> None:
    cfg = get_config()
    # index_fn = partial(index_by_height, cutoff=13.5, below=True)
    index_fn = partial(index_by_layers, layers=(1, 2))
    dataset = load_mlip_dataset(cfg)
    updated_dataset: dict[str, Any] = {}
    for reaction, entry in dataset.items():
        adsorbed_atom = extract_adsorbed_atom(entry, reaction)
        indices = extract_adsorbate_indices(entry, reaction)
        binding_atom = extract_binding_atom(adsorbed_atom, indices)
        shifted_adslab = place_adsorbate_on_closest_binding_site(adsorbed_atom, indices)
        bare_surface = strip_adsorbate_from_adslab(adsorbed_atom, indices)
        # # Debug helper: visualize best-fit plane via random H markers.
        # # plane_vis = build_plane_visualization(bare_surface, n_markers=300, seed=0)
        constraint_indices = index_fn(bare_surface)
        constrained_adslab = fix_atoms(shifted_adslab, constraint_indices)
        constrained_adslab = fix_binding_atom_xy(constrained_adslab, binding_atom)
        updated_entry = build_shifted_constrained_adsorption_entry(
            entry, constrained_adslab, reaction
        )
        updated_dataset[reaction] = updated_entry
    input_dataset_path = Path(cfg.mlip.dataset)
    output_path = shifted_adsorption_output_path(input_dataset_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(updated_dataset, f, indent=2)

    print(f"Processed {len(dataset)} entries and wrote {output_path}")


if __name__ == "__main__":
    main()
