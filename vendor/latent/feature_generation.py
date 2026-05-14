import numpy as np
from pymatgen.core.periodic_table import Element
from ase.neighborlist import primitive_neighbor_list


def get_host_feature(element_property, atoms, host_index, neighbor_indices, power_term):
    weighted_property_list = []
    for neighbor_index in neighbor_indices:
        neighbor_weight = get_weight(atoms, host_index, neighbor_index, power_term)
        neighbor_atom = atoms[neighbor_index]
        neighbor_element_symbol = getattr(neighbor_atom, "symbol")
        neighbor_property = getattr(Element(neighbor_element_symbol), element_property)
        weighted_property = get_weighted_property(
            atom_weight=neighbor_weight,
            atom_property=neighbor_property,
        )
        weighted_property_list.append(weighted_property)
    host_feature = np.sum(weighted_property_list)
    return host_feature


def get_centered_system(
    atoms,
    input_indices,
):
    """
    Get an expanded atoms object, with indices corresponding to the object's central atoms
    TODO: Delete this slapdash function and get findNeighborIndices() to account for periodic BC's
    """
    num_atoms = len(atoms)
    index_shift = 4
    centered_indices = []
    for index in input_indices:
        centered_indices.append(index + num_atoms * index_shift)
    repeated_atoms = atoms.repeat((3, 3, 1))
    return repeated_atoms, centered_indices


def findNeighborIndices(structure, nnDist, host_atom_indices):
    neighbors_indices_list = [None] * len(host_atom_indices)
    nnListi, nnListj = primitive_neighbor_list(
        "ij",
        structure.get_pbc(),
        structure.get_cell(),
        structure.get_positions(),
        nnDist,
    )
    for i, atom in enumerate(host_atom_indices):
        whichToTake = np.where(nnListi == atom)[0]
        neighbors_indices_list[i] = nnListj[whichToTake]
    return neighbors_indices_list


def get_nn_distance(atoms, index=0):
    num_atoms = len(atoms)
    all_distances = atoms.get_distances(index, range(num_atoms), mic=True)
    all_distances_rounded = np.round(all_distances, decimals=4)
    unique_distances = np.unique(all_distances_rounded)
    nn_distance = unique_distances[1]
    return nn_distance


def get_host_feature_list(
    feature, atoms, host_indices, power_term, neighbor_distance_scaling_factor=1.3
):
    """
    For every host binding atom, calculate a host feature and put the values into a list
    TODO: Put scaling_neighbor_distance_factor into configuration file and feed through the function arguments
    """
    num_host_atoms = len(host_indices)
    nn_distance = get_nn_distance(atoms, index=0)
    scaled_neighbor_distance = neighbor_distance_scaling_factor * nn_distance
    atoms, centered_indices = get_centered_system(atoms, host_indices)
    neighbor_atom_indices_list = findNeighborIndices(
        atoms, scaled_neighbor_distance, centered_indices
    )
    host_feature_list = []
    for i in range(num_host_atoms):
        host_index = centered_indices[i]
        neighbor_indices = neighbor_atom_indices_list[i]
        host_feature = get_host_feature(
            element_property=feature,
            atoms=atoms,
            host_index=host_index,
            neighbor_indices=neighbor_indices,
            power_term=power_term,
        )
        host_feature_list.append(host_feature)
    return host_feature_list


def get_weighted_property(atom_weight: float, atom_property: float):
    weighted_property = atom_weight * atom_property
    return weighted_property


def get_weight(atoms, host_index, neighbor_index, power_term):
    """
    Calculate the weight of a neighboring atom on the host atom
    """
    neighbor_distance = atoms.get_distance(host_index, neighbor_index, mic=True)
    neighbor_weight = 1 / neighbor_distance**power_term
    return neighbor_weight


def get_matching_indices(set, subset):
    """
    Find which indices of a subset that's items match the items of the original set.
    Returns them in the order of the subset
    """
    indices_dict = {value: index for index, value in enumerate(set)}
    matching_indices = []
    for i in subset:
        matching_indices.append(indices_dict[i])
    return matching_indices


def findSurfaceAtoms(atoms, tol=0.4, mode="+z"):
    """
    Finds the atoms on the surface of a structure; returns their indices.
    Probably best to wrap the atoms before calling this.
    TODO: Incorporate into G-HIT.
    """
    if mode == "+z":
        maxz = np.max(atoms.get_positions()[:, 2])
    return [
        i for i, atom in enumerate(atoms) if atoms.get_positions()[i, 2] > maxz - tol
    ]
