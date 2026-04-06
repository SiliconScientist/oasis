from functools import partial
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.db.row import AtomsRow
from ase.io import jsonio
from ase.visualize import view
from pymatgen.analysis.local_env import JmolNN
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor

from oasis.config import get_config
from oasis.ingest.site_constraints import (
    extract_adsorbate_indices,
    extract_adsorbed_atom,
    find_adsorption_sites_on_slab,
    index_by_layers,
    load_mlip_dataset,
    plane_from_lowest_atoms,
    strip_adsorbate_from_adslab,
)


DEFAULT_JSON_PATH = Path(__file__).with_name("KHLOHC_origin_adsorption.json")
ADSORPTION_SITE_TOLERANCE = 0.5
METHYL_CH_BOND_LENGTH = 1.09


def atoms_from_ase_db_json(atoms_json: str):
    """Convert an ASE database-style JSON string into an ASE Atoms object."""
    decoded = jsonio.decode(atoms_json)
    row_id = decoded["ids"][0]
    return AtomsRow(decoded[row_id]).toatoms()


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


def deduplicate_marker_structures(
    marker_atoms_list,
    dedup_atom_indices_list,
    adaptor,
    matcher,
    slab_atom_count: int,
):
    """Remove duplicate marker structures using selected adsorbate atoms only."""
    if len(marker_atoms_list) != len(dedup_atom_indices_list):
        raise ValueError("dedup_atom_indices_list must align with marker_atoms_list")

    unique_atoms = []
    unique_structures = []
    for atoms, dedup_atom_indices in zip(
        marker_atoms_list, dedup_atom_indices_list, strict=False
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


def methyl_adsorbate_geometry(
    ch_bond_length: float = METHYL_CH_BOND_LENGTH,
) -> tuple[list[str], np.ndarray, tuple[int, ...]]:
    """
    Return local methyl geometry with the carbon at the origin.

    The local +z direction points away from the surface, while the implicit
    surface-C bond is the fourth tetrahedral direction along -z.
    """
    polar_component = 1.0 / 3.0
    radial_component = np.sqrt(1.0 - polar_component**2)
    azimuthal_angles = (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0)

    symbols = ["C"]
    positions = [np.zeros(3, dtype=float)]
    for angle in azimuthal_angles:
        direction = np.array(
            [
                radial_component * np.cos(angle),
                radial_component * np.sin(angle),
                polar_component,
            ],
            dtype=float,
        )
        symbols.append("H")
        positions.append(ch_bond_length * direction)
    return symbols, np.asarray(positions, dtype=float), (0,)


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

    marker_symbols: list[str] = []
    marker_positions: list[np.ndarray] = []
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
            marker_symbols.append(symbol)
            marker_positions.append(position)

    markers = Atoms(
        symbols=marker_symbols,
        positions=np.asarray(marker_positions, dtype=float),
        cell=adslab.cell,
        pbc=adslab.pbc,
    )
    visual = adslab.copy()
    visual.extend(markers)
    return visual


if __name__ == "__main__":
    cfg = get_config()
    index_fn = partial(index_by_layers, layers=-1)
    adaptor = AseAtomsAdaptor()
    jmol_nn = JmolNN()
    structure_matcher = StructureMatcher()
    dataset = load_mlip_dataset(cfg)
    for reaction, entry in dataset.items():
        if "Tolstar" not in entry.get("raw", {}):
            continue
        adsorbed_atoms = extract_adsorbed_atom(entry, reaction)
        adsorbate_indices = extract_adsorbate_indices(entry, reaction)
        bare_surface = strip_adsorbate_from_adslab(adsorbed_atoms, adsorbate_indices)
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
        marker_structures = []
        dedup_atom_indices_list = []
        for adsorption_site, adsorbate_element in nearby_site_adsorbates:
            adsorbate_symbols, adsorbate_positions, dedup_atom_indices = (
                adsorbate_geometry_template(adsorbate_element)
            )
            marker_structures.append(
                add_adsorbates(
                    bare_surface,
                    np.array([adsorption_site]),
                    adsorbate_symbols=adsorbate_symbols,
                    adsorbate_positions=adsorbate_positions,
                    plane_centroid=plane_centroid,
                    plane_normal=plane_normal,
                )
            )
            dedup_atom_indices_list.append(dedup_atom_indices)
        unique_marker_structures = deduplicate_marker_structures(
            marker_structures,
            dedup_atom_indices_list,
            adaptor,
            structure_matcher,
            slab_atom_count=len(bare_surface),
        )
        if len(unique_marker_structures) > 1:
            print("Stop here for visualization")
