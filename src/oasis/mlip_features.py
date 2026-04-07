from functools import partial
from itertools import islice
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.db.row import AtomsRow
from ase.io import jsonio
from ase.visualize import view
from pymatgen.analysis.local_env import JmolNN
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor

from oasis.config import get_config
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


def atoms_from_ase_db_json(atoms_json: str):
    """Convert an ASE database-style JSON string into an ASE Atoms object."""
    decoded = jsonio.decode(atoms_json)
    row_id = decoded["ids"][0]
    return AtomsRow(decoded[row_id]).toatoms()


def unique_probe_output_path(input_dataset_path: Path) -> Path:
    """Build the output path for the unique probe dataset."""
    stem = input_dataset_path.stem
    lower_stem = stem.lower()
    marker = "tolstar"
    if marker in lower_stem:
        start = lower_stem.index(marker)
        stem = f"{stem[:start]}unique_probe{stem[start + len(marker) :]}"
    else:
        stem = f"{stem}_unique_probe"
    return input_dataset_path.with_name(f"{stem}{input_dataset_path.suffix}")


def updated_dataset_output_path(input_dataset_path: Path) -> Path:
    """Build the output path for the original dataset with probe ids added."""
    stem = input_dataset_path.stem
    return input_dataset_path.with_name(
        f"{stem}_with_probe_ids{input_dataset_path.suffix}"
    )


def wrap_atoms_json(atoms: Atoms, unique_id: str) -> str:
    """Serialize Atoms into the ASE DB-style wrapped payload used by the dataset."""
    row = json.loads(jsonio.encode(atoms))
    row["unique_id"] = unique_id
    return json.dumps({"1": row, "ids": [1], "nextid": 2})


def gas_reference_atoms(formula: str, cell_length: float = GAS_CELL_LENGTH) -> Atoms:
    """Build a simple gas-phase molecule in a cubic periodic box."""
    gas = molecule(formula)
    gas.set_cell(np.eye(3) * cell_length)
    gas.center()
    gas.set_pbc(True)
    return gas


def build_unique_probe_entry(
    *,
    unique_id: str,
    bare_surface: Atoms,
    probe_structure: Atoms,
    star_template_atoms_json: str,
    probe_template_atoms_json: str,
    ch4_gas: Atoms,
    h2_gas: Atoms,
) -> dict[str, object]:
    """Build one dataset entry for a newly discovered unique probe system."""
    slab_atom_count = len(bare_surface)
    probe_indices = list(range(slab_atom_count, len(probe_structure)))
    constrained_probe_structure = fix_binding_atom_xy(probe_structure, slab_atom_count)

    return {
        "unique_id": unique_id,
        "raw": {
            "star": {
                "stoi": -1,
                "energy_ref": None,
                "atoms_json": atoms_to_atoms_json_like_template(
                    bare_surface, star_template_atoms_json
                ),
            },
            "ch3star": {
                "stoi": 1,
                "energy_ref": None,
                "atoms_json": atoms_to_atoms_json_like_template(
                    constrained_probe_structure, probe_template_atoms_json
                ),
            },
            "ch4gas": {
                "stoi": -1,
                "energy_ref": None,
                "atoms_json": wrap_atoms_json(
                    ch4_gas.copy(), unique_id=f"{unique_id}_ch4gas"
                ),
            },
            "h2gas": {
                "stoi": 0.5,
                "energy_ref": None,
                "atoms_json": wrap_atoms_json(
                    h2_gas.copy(), unique_id=f"{unique_id}_h2gas"
                ),
            },
        },
        "ref_ads_eng": None,
        "adsorbate_indices": probe_indices,
    }


def load_tolstar_atoms(json_path=DEFAULT_JSON_PATH):
    """
    Build a list of ASE Atoms objects from the nested raw["Tolstar"] entries.

    Returns:
        list[ase.Atoms]: One Atoms object per Tolstar entry in the JSON file.
    """
    with Path(json_path).open() as handle:
        data = json.load(handle)

    atoms_list = []
    for _, entry in data.items():
        raw = entry.get("raw", {})
        tolstar = raw.get("Tolstar")
        if tolstar is None:
            continue
        atoms_list.append(atoms_from_ase_db_json(tolstar["atoms_json"]))

    return atoms_list


def project_points_onto_plane(
    points: np.ndarray, plane_centroid: np.ndarray, plane_normal: np.ndarray
) -> np.ndarray:
    """Project Cartesian points onto a plane."""
    displacements = np.asarray(points, dtype=float) - np.asarray(
        plane_centroid, dtype=float
    )
    distances = displacements @ np.asarray(plane_normal, dtype=float)
    return np.asarray(points, dtype=float) - np.outer(distances, plane_normal)


def site_adsorbate_associations(
    adsorption_sites: np.ndarray,
    surface_positions: np.ndarray,
    adsorbate_elements: list[str],
    plane_centroid: np.ndarray,
    plane_normal: np.ndarray,
    tolerance: float = ADSORPTION_SITE_TOLERANCE,
) -> list[tuple[np.ndarray, str]]:
    """Associate each nearby adsorption site with the nearest bound adsorbate element."""
    projected_sites = project_points_onto_plane(
        adsorption_sites, plane_centroid, plane_normal
    )
    projected_surface_positions = project_points_onto_plane(
        surface_positions, plane_centroid, plane_normal
    )

    associated_sites: list[tuple[np.ndarray, str]] = []
    for site, projected_site in zip(adsorption_sites, projected_sites, strict=False):
        distances = np.linalg.norm(projected_surface_positions - projected_site, axis=1)
        closest_index = int(np.argmin(distances))
        if distances[closest_index] <= tolerance:
            associated_sites.append((site, adsorbate_elements[closest_index]))
    return associated_sites


def select_adsorbate_atoms_for_deduplication(
    atoms: Atoms,
    slab_atom_count: int,
    dedup_atom_indices: tuple[int, ...],
) -> Atoms:
    """Return the slab plus selected atoms from the appended adsorbate."""
    adsorbate_atom_count = len(atoms) - slab_atom_count
    if adsorbate_atom_count <= 0:
        raise ValueError("Expected appended adsorbate atoms for deduplication")
    if not dedup_atom_indices:
        raise ValueError("dedup_atom_indices must not be empty")
    if any(index < 0 or index >= adsorbate_atom_count for index in dedup_atom_indices):
        raise ValueError("dedup_atom_indices must refer to atoms within the adsorbate")

    selected_indices = list(range(slab_atom_count))
    selected_indices.extend(slab_atom_count + index for index in dedup_atom_indices)
    return atoms[selected_indices]


def deduplicate_probe_structures(
    probe_atoms_list,
    dedup_atom_indices_list,
    adaptor,
    matcher,
    slab_atom_count: int,
):
    """Remove duplicate probe structures using selected adsorbate atoms only."""
    if len(probe_atoms_list) != len(dedup_atom_indices_list):
        raise ValueError("dedup_atom_indices_list must align with probe_atoms_list")

    unique_atoms = []
    unique_structures = []
    for atoms, dedup_atom_indices in zip(
        probe_atoms_list, dedup_atom_indices_list, strict=False
    ):
        structure = adaptor.get_structure(
            select_adsorbate_atoms_for_deduplication(
                atoms,
                slab_atom_count=slab_atom_count,
                dedup_atom_indices=dedup_atom_indices,
            )
        )
        if any(matcher.fit(structure, other) for other in unique_structures):
            continue
        unique_atoms.append(atoms)
        unique_structures.append(structure)
    return unique_atoms


def probe_matching_structure(
    probe_atoms: Atoms,
    slab_atom_count: int,
    dedup_atom_indices: tuple[int, ...],
) -> Atoms:
    """Return the reduced probe structure used for global matching."""
    return select_adsorbate_atoms_for_deduplication(
        probe_atoms,
        slab_atom_count=slab_atom_count,
        dedup_atom_indices=dedup_atom_indices,
    )


def probe_match_signature(structure) -> tuple[str, int]:
    """Return a cheap signature used to prefilter probe match candidates."""
    return structure.composition.reduced_formula, len(structure)


def orthonormal_basis_from_axis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build two unit vectors orthogonal to a given unit axis."""
    helper_axis = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(helper_axis, axis))) > 0.9:
        helper_axis = np.array([0.0, 1.0, 0.0], dtype=float)

    basis_1 = np.cross(axis, helper_axis)
    basis_1 /= np.linalg.norm(basis_1)
    basis_2 = np.cross(axis, basis_1)
    basis_2 /= np.linalg.norm(basis_2)
    return basis_1, basis_2


def rotation_matrix_from_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a rotation matrix that maps one vector onto another."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    source /= np.linalg.norm(source)
    target /= np.linalg.norm(target)

    cross = np.cross(source, target)
    sin_theta = np.linalg.norm(cross)
    cos_theta = float(np.dot(source, target))

    if sin_theta < 1e-12:
        if cos_theta > 0.0:
            return np.eye(3)
        helper = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(float(np.dot(source, helper))) > 0.9:
            helper = np.array([0.0, 1.0, 0.0], dtype=float)
        axis = np.cross(source, helper)
        axis /= np.linalg.norm(axis)
        return -np.eye(3) + 2.0 * np.outer(axis, axis)

    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=float,
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - cos_theta) / (sin_theta**2))


def methyl_adsorbate_geometry() -> tuple[list[str], np.ndarray, tuple[int, ...]]:
    """
    Return local methyl geometry derived from ASE methane.

    The carbon is at the origin. The removed methane hydrogen defines the
    implicit surface-C bond direction, which is aligned with local -z.
    """
    methane = molecule("CH4")
    symbols = methane.get_chemical_symbols()
    positions = methane.get_positions()

    carbon_index = symbols.index("C")
    carbon_position = positions[carbon_index]
    hydrogen_indices = [i for i, symbol in enumerate(symbols) if symbol == "H"]
    removed_hydrogen_index = min(
        hydrogen_indices,
        key=lambda index: positions[index][2],
    )

    centered_positions = positions - carbon_position
    removed_direction = centered_positions[removed_hydrogen_index]
    rotation = rotation_matrix_from_vectors(
        removed_direction, np.array([0.0, 0.0, -1.0], dtype=float)
    )
    rotated_positions = centered_positions @ rotation.T

    kept_indices = [carbon_index] + [
        index for index in hydrogen_indices if index != removed_hydrogen_index
    ]
    return (
        [symbols[index] for index in kept_indices],
        rotated_positions[kept_indices],
        (0,),
    )


def monatomic_adsorbate_geometry(
    element: str,
) -> tuple[list[str], np.ndarray, tuple[int, ...]]:
    """Return a single-atom adsorbate template anchored at the adsorption site."""
    return [element], np.zeros((1, 3), dtype=float), (0,)


def adsorbate_geometry_template(
    adsorbate_element: str,
) -> tuple[list[str], np.ndarray, tuple[int, ...]]:
    """Return a visualization adsorbate template for a bound adsorbate element."""
    if adsorbate_element == "C":
        return methyl_adsorbate_geometry()
    return monatomic_adsorbate_geometry(adsorbate_element)


def add_adsorbates(
    adslab: Atoms,
    adsorption_sites: np.ndarray,
    adsorbate_symbols: list[str],
    adsorbate_positions: np.ndarray,
    plane_centroid: np.ndarray,
    plane_normal: np.ndarray,
    anchor_index: int = 0,
) -> Atoms:
    """
    Return a visualization structure with a local adsorbate template at each site.

    `adsorbate_positions` are local coordinates. The atom at `anchor_index` is
    translated onto each adsorption site, and the local +z direction is aligned
    with the outward surface normal.
    """
    local_positions = np.asarray(adsorbate_positions, dtype=float)
    if local_positions.ndim != 2 or local_positions.shape[1] != 3:
        raise ValueError("adsorbate_positions must have shape (n_atoms, 3)")
    if len(adsorbate_symbols) != len(local_positions):
        raise ValueError("adsorbate_symbols must match adsorbate_positions")
    if anchor_index < 0 or anchor_index >= len(local_positions):
        raise ValueError("anchor_index is out of range for the adsorbate template")

    axis = np.asarray(plane_normal, dtype=float)
    axis /= np.linalg.norm(axis)
    centered_positions = local_positions - local_positions[anchor_index]

    probe_symbols: list[str] = []
    probe_positions: list[np.ndarray] = []
    for site in np.asarray(adsorption_sites, dtype=float):
        local_axis = axis.copy()
        if float(np.dot(site - plane_centroid, local_axis)) < 0.0:
            local_axis = -local_axis
        local_basis_1, local_basis_2 = orthonormal_basis_from_axis(local_axis)
        rotation = np.column_stack((local_basis_1, local_basis_2, local_axis))
        rotated_positions = centered_positions @ rotation.T
        for symbol, position in zip(
            adsorbate_symbols, site + rotated_positions, strict=False
        ):
            probe_symbols.append(symbol)
            probe_positions.append(position)

    probes = Atoms(
        symbols=probe_symbols,
        positions=np.asarray(probe_positions, dtype=float),
        cell=adslab.cell,
        pbc=adslab.pbc,
    )
    visual = adslab.copy()
    visual.extend(probes)
    return visual


if __name__ == "__main__":
    cfg = get_config()
    dataset_path = Path(cfg.mlip.dataset)
    index_fn = partial(index_by_layers, layers=-1)
    adaptor = AseAtomsAdaptor()
    jmol_nn = JmolNN()
    structure_matcher = StructureMatcher()
    dataset = load_mlip_dataset(cfg)
    dataset_items = islice(dataset.items(), 50) if cfg.mlip.dev_run else dataset.items()
    updated_dataset: dict[str, dict[str, object]] = {}
    unique_probe_structures: dict[str, Atoms] = {}
    unique_probe_match_structures = {}
    unique_probe_buckets: dict[tuple[str, int], list[str]] = {}
    unique_probe_dataset: dict[str, dict[str, object]] = {}
    next_unique_probe_id = 0
    ch4_gas = gas_reference_atoms("CH4")
    h2_gas = gas_reference_atoms("H2")
    for reaction, entry in dataset_items:
        adsorbed_atoms = extract_adsorbed_atom(entry, reaction)
        adsorbate_indices = extract_adsorbate_indices(entry, reaction)
        bare_surface = rewrap_slab_by_largest_gap(
            strip_adsorbate_from_adslab(adsorbed_atoms, adsorbate_indices)
        )
        adsorption_sites = find_adsorption_sites_on_slab(bare_surface)
        top_layer_indices = index_fn(bare_surface)
        adsorbate_index_set = set(adsorbate_indices)
        slab_indices = [
            i for i in range(len(adsorbed_atoms)) if i not in adsorbate_index_set
        ]
        adsorbed_top_layer_indices = [slab_indices[i] for i in top_layer_indices]
        structure = adaptor.get_structure(adsorbed_atoms)
        saturated_atoms = []
        bound_surface_indices = []
        for surface_index in adsorbed_top_layer_indices:
            adsorbate_neighbors = [
                int(neighbor["site_index"])
                for neighbor in jmol_nn.get_nn_info(structure, surface_index)
                if neighbor["site_index"] in adsorbate_index_set
            ]
            if not adsorbate_neighbors:
                continue
            adsorbate_index = int(adsorbate_neighbors[0])
            saturated_atoms.append(
                {
                    "surface_index": int(surface_index),
                    "adsorbate_index": adsorbate_index,
                    "adsorbate_element": adsorbed_atoms[adsorbate_index].symbol,
                }
            )
            bound_surface_indices.append(slab_indices.index(surface_index))
        plane_centroid, plane_normal, _ = plane_from_lowest_atoms(bare_surface)
        surface_positions = bare_surface.positions[bound_surface_indices]
        adsorbate_elements = [
            saturated_atom["adsorbate_element"] for saturated_atom in saturated_atoms
        ]
        nearby_site_adsorbates = site_adsorbate_associations(
            adsorption_sites=adsorption_sites,
            surface_positions=surface_positions,
            adsorbate_elements=adsorbate_elements,
            plane_centroid=plane_centroid,
            plane_normal=plane_normal,
        )
        entry_unique_probe_ids: list[str] = []
        for adsorption_site, adsorbate_element in nearby_site_adsorbates:
            adsorbate_symbols, adsorbate_positions, dedup_atom_indices = (
                adsorbate_geometry_template(adsorbate_element)
            )
            probe_structure = add_adsorbates(
                bare_surface,
                np.array([adsorption_site]),
                adsorbate_symbols=adsorbate_symbols,
                adsorbate_positions=adsorbate_positions,
                plane_centroid=plane_centroid,
                plane_normal=plane_normal,
            )
            match_structure = adaptor.get_structure(
                probe_matching_structure(
                    probe_structure,
                    slab_atom_count=len(bare_surface),
                    dedup_atom_indices=dedup_atom_indices,
                )
            )
            signature = probe_match_signature(match_structure)

            matching_unique_id = None
            for unique_id in unique_probe_buckets.get(signature, []):
                if structure_matcher.fit(
                    match_structure, unique_probe_match_structures[unique_id]
                ):
                    matching_unique_id = unique_id
                    break

            if matching_unique_id is None:
                matching_unique_id = str(next_unique_probe_id)
                next_unique_probe_id += 1
                unique_probe_structures[matching_unique_id] = probe_structure
                unique_probe_match_structures[matching_unique_id] = match_structure
                unique_probe_buckets.setdefault(signature, []).append(
                    matching_unique_id
                )
                unique_probe_dataset[f"unique_probe_{matching_unique_id}"] = (
                    build_unique_probe_entry(
                        unique_id=matching_unique_id,
                        bare_surface=bare_surface,
                        probe_structure=probe_structure,
                        star_template_atoms_json=entry["raw"]["star"]["atoms_json"],
                        probe_template_atoms_json=entry["raw"]["Tolstar"]["atoms_json"],
                        ch4_gas=ch4_gas,
                        h2_gas=h2_gas,
                    )
                )

            entry_unique_probe_ids.append(matching_unique_id)
        entry["unique_probe_ids"] = entry_unique_probe_ids
        updated_dataset[reaction] = entry

    output_path = unique_probe_output_path(dataset_path)
    output_path.write_text(json.dumps(unique_probe_dataset, indent=2) + "\n")
    updated_output_path = updated_dataset_output_path(dataset_path)
    updated_output_path.write_text(json.dumps(updated_dataset, indent=2) + "\n")
    print(len(unique_probe_structures))
    print(output_path)
    print(updated_output_path)
