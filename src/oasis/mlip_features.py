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


def find_nearby_adsorption_sites(
    adsorption_sites: np.ndarray,
    surface_positions: np.ndarray,
    plane_centroid: np.ndarray,
    plane_normal: np.ndarray,
    tolerance: float = ADSORPTION_SITE_TOLERANCE,
) -> np.ndarray:
    """Keep adsorption sites whose in-plane distance to a bound surface atom is within tolerance."""
    projected_sites = project_points_onto_plane(
        adsorption_sites, plane_centroid, plane_normal
    )
    projected_surface_positions = project_points_onto_plane(
        surface_positions, plane_centroid, plane_normal
    )
    matched_sites = []
    for site, projected_site in zip(adsorption_sites, projected_sites, strict=False):
        distances = np.linalg.norm(projected_surface_positions - projected_site, axis=1)
        if np.any(distances <= tolerance):
            matched_sites.append(site)
    return np.array(matched_sites, dtype=float)


def carbon_only_marker_structure(
    atoms: Atoms, slab_atom_count: int, atoms_per_marker: int = 4
) -> Atoms:
    """Return the slab plus marker carbons, omitting marker hydrogens."""
    marker_atom_count = len(atoms) - slab_atom_count
    if marker_atom_count < 0 or marker_atom_count % atoms_per_marker != 0:
        raise ValueError("Marker atoms do not match the expected methyl-group layout")

    marker_indices = range(slab_atom_count, len(atoms), atoms_per_marker)
    selected_indices = list(range(slab_atom_count)) + list(marker_indices)
    return atoms[selected_indices]


def deduplicate_marker_structures(
    marker_atoms_list, adaptor, matcher, slab_atom_count: int
):
    """Remove duplicate marker structures using slab atoms and marker carbons only."""
    unique_atoms = []
    unique_structures = []
    for atoms in marker_atoms_list:
        structure = adaptor.get_structure(
            carbon_only_marker_structure(atoms, slab_atom_count=slab_atom_count)
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


def add_methyl_site_markers(
    adslab: Atoms,
    adsorption_sites: np.ndarray,
    plane_centroid: np.ndarray,
    plane_normal: np.ndarray,
    ch_bond_length: float = METHYL_CH_BOND_LENGTH,
) -> Atoms:
    """
    Return a visualization structure with a methyl group placed at each adsorption site.

    The carbon atom is placed at the adsorption site. The three hydrogens are
    oriented tetrahedrally so they point away from the slab, treating the
    surface-C bond as the fourth tetrahedral direction.
    """
    axis = np.asarray(plane_normal, dtype=float)
    axis /= np.linalg.norm(axis)

    # If a site lies on the opposite side of the plane, flip the local outward axis.
    polar_component = 1.0 / 3.0
    radial_component = np.sqrt(1.0 - polar_component**2)
    azimuthal_angles = (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0)

    marker_symbols: list[str] = []
    marker_positions: list[np.ndarray] = []
    for site in np.asarray(adsorption_sites, dtype=float):
        local_axis = axis.copy()
        if float(np.dot(site - plane_centroid, local_axis)) < 0.0:
            local_axis = -local_axis
        local_basis_1, local_basis_2 = orthonormal_basis_from_axis(local_axis)

        marker_symbols.append("C")
        marker_positions.append(site)
        for angle in azimuthal_angles:
            direction = (
                polar_component * local_axis
                + radial_component
                * (
                    np.cos(angle) * local_basis_1
                    + np.sin(angle) * local_basis_2
                )
            )
            marker_symbols.append("H")
            marker_positions.append(site + ch_bond_length * direction)

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
        nearby_adsorption_sites = find_nearby_adsorption_sites(
            adsorption_sites=adsorption_sites,
            surface_positions=bare_surface.positions[bound_surface_indices],
            plane_centroid=plane_centroid,
            plane_normal=plane_normal,
        )
        bare_surface_with_marker_sites = [
            add_methyl_site_markers(
                bare_surface,
                np.array([adsorption_site]),
                plane_centroid=plane_centroid,
                plane_normal=plane_normal,
            )
            for adsorption_site in nearby_adsorption_sites
        ]
        unique_marker_structures = deduplicate_marker_structures(
            bare_surface_with_marker_sites,
            adaptor,
            structure_matcher,
            slab_atom_count=len(bare_surface),
        )
        print(f"Reaction: {reaction}")
        print(f"Saturated atoms: {saturated_atoms}")
        print(f"Nearby adsorption sites: {nearby_adsorption_sites.tolist()}")
        print(f"Marker structures: {len(unique_marker_structures)}")
