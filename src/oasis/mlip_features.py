from functools import partial
import json
from pathlib import Path

import numpy as np
from ase.db.row import AtomsRow
from ase.io import jsonio
from ase.visualize import view
from pymatgen.analysis.local_env import JmolNN
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor

from oasis.config import get_config
from oasis.ingest.site_constraints import (
    add_binding_site_markers,
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
    displacements = np.asarray(points, dtype=float) - np.asarray(plane_centroid, dtype=float)
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


def deduplicate_marker_structures(marker_atoms_list, adaptor, matcher):
    """Remove duplicate marker structures using pymatgen StructureMatcher."""
    unique_atoms = []
    unique_structures = []
    for atoms in marker_atoms_list:
        structure = adaptor.get_structure(atoms)
        if any(matcher.fit(structure, other) for other in unique_structures):
            continue
        unique_atoms.append(atoms)
        unique_structures.append(structure)
    return unique_atoms


if __name__ == "__main__":
    cfg = get_config()
    index_fn = partial(index_by_layers, layers=-1)
    adaptor = AseAtomsAdaptor()
    jmol_nn = JmolNN()
    structure_matcher = StructureMatcher()
    dataset = load_mlip_dataset(cfg)
    for reaction, entry in dataset.items():
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
            add_binding_site_markers(bare_surface, np.array([adsorption_site]))
            for adsorption_site in nearby_adsorption_sites
        ]
        unique_marker_structures = deduplicate_marker_structures(
            bare_surface_with_marker_sites, adaptor, structure_matcher
        )
        print(f"Reaction: {reaction}")
        print(f"Saturated atoms: {saturated_atoms}")
        print(f"Nearby adsorption sites: {nearby_adsorption_sites.tolist()}")
        print(f"Marker structures: {len(unique_marker_structures)}")
