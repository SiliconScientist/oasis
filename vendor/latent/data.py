from __future__ import annotations

import os
import json
import pickle
import hashlib
import pathlib
import re
import tempfile
from collections.abc import Callable, Sequence
from typing import Any, TypedDict

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
import sys
import pandas as pd
from numpy.typing import NDArray
from tqdm import tqdm
from montegroup.ghit import GHIT
from latent.config import Config, ExperimentConfig
from latent.feature_generation import (
    get_host_feature_list,
    findSurfaceAtoms,
)
from pymatgen.core.periodic_table import Element


AdsorbateMap = Sequence[Sequence[str]]
GuestAtomSymbols = set[str]
HostBindingAtoms = list[int]
HostBindingElements = NDArray[np.str_]
ModelArray = NDArray[np.float64]


class DatasetRecord(TypedDict):
    atoms: Atoms
    adsorption_energy: float | None
    guest_elements: list[str]
    metadata: dict[str, Any]


class UnrelaxedModelData(TypedDict, total=False):
    atoms_per_layer: int | None
    layer2_from_actual: bool
    layer3_xy_from_layer2: bool
    type: str
    size: tuple[int, ...]
    xy: ModelArray


def get_element_dataframe(symbols: list[str], properties: list[str]) -> pd.DataFrame:
    element_properties = []
    for symbol in symbols:
        property_row = {}
        element = Element(symbol)
        for property_str in properties:
            element_property = getattr(element, property_str)
            property_row[property_str] = element_property
        element_properties.append(property_row)
    df = pd.DataFrame(element_properties)
    return df


def get_unique_element_symbols(systems: list[Atoms]) -> list[str]:
    atom_symbols = []
    for atoms in systems:
        symbols = atoms.get_chemical_symbols()
        for symbol in symbols:
            atom_symbols.append(symbol)
    return list(set(atom_symbols))


def filter_features(
    symbol_vocabulary: list[str], element_property_strs: list[str]
) -> list[str]:
    element_df = get_element_dataframe(symbol_vocabulary, element_property_strs)
    # Remove columns containing strings
    df = element_df.select_dtypes(exclude=["object"])
    # Remove columns containing NaN values
    filtered_properties = df.dropna(axis=1)
    return list(filtered_properties.columns)


def _derive_guest_atom_symbols(adsorbate_map: AdsorbateMap) -> GuestAtomSymbols:
    symbols = set()
    for _, star_name in adsorbate_map:
        for elem in re.findall(r"[A-Z][a-z]?", star_name.replace("star", "")):
            symbols.add(elem)
    return symbols


def _star_name_to_guest_name(star_name: str) -> str:
    mol = star_name.replace("star", "")
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", mol)
    expanded = []
    for elem, count in tokens:
        expanded.extend([elem] * (int(count) if count else 1))
    unique_sorted = sorted(set(expanded))
    return "".join(
        elem + (str(expanded.count(elem)) if expanded.count(elem) > 1 else "")
        for elem in unique_sorted
    )


def _guest_name_from_symbols(
    symbols: Sequence[str],
    guest_atom_symbols: GuestAtomSymbols,
) -> tuple[str, list[str]]:
    guest_elements = [symbol for symbol in symbols if symbol in guest_atom_symbols]
    guest_name_parts = []
    guest_elements_unique = []
    for guest in np.unique(guest_elements):
        guest_elements_unique.append(guest)
        subscript = guest_elements.count(guest)
        guest_name_parts.append(guest + (str(subscript) if subscript > 1 else ""))
    return "".join(guest_name_parts), guest_elements_unique


def _parse_host_binding_atoms(raw_value: Any) -> HostBindingAtoms:
    if not raw_value or raw_value == "[]":
        return []
    return [int(match) for match in re.findall(r"\d+", str(raw_value))]


def _is_allowed_structure(
    symbols: Sequence[str],
    host_elements: Sequence[str],
    guest_atom_symbols: GuestAtomSymbols,
) -> bool:
    allowed_symbols = set(host_elements) | set(guest_atom_symbols)
    return set(symbols).issubset(allowed_symbols)


def _write_ghit_database(records: Sequence[DatasetRecord], db_path: str) -> None:
    for record in tqdm(records, desc="Generate database"):
        atoms = record["atoms"]
        energy = record["adsorption_energy"]
        atoms.set_calculator(SinglePointCalculator(atoms, energy=energy))
        GHIT(atoms).new_data_base(
            db_path,
            key_value_pairs=record["metadata"],
            atom_layers=False,
            transition_metal=False,
            guest_elements=record["guest_elements"],
        )


def _canonicalize_ghit_rows(
    db_path: str,
    guest_list: Sequence[str],
    site_list: Sequence[str],
    host_elements: Sequence[str],
    guest_atom_symbols: GuestAtomSymbols,
    site_name_getter: Callable[[Any], str | None],
) -> tuple[pd.DataFrame, list[Atoms]]:
    guest_set = set(guest_list)
    site_set = set(site_list)
    rows = []
    atoms_list = []
    for row in connect(db_path).select():
        guest_name = row.get("guest_name")
        if guest_name not in guest_set:
            continue
        site_name = site_name_getter(row)
        if not site_name or site_name == "[]" or site_name not in site_set:
            continue

        atoms = row.toatoms()
        symbols = np.array(atoms.get_chemical_symbols())
        if not _is_allowed_structure(symbols, host_elements, guest_atom_symbols):
            continue

        host_binding_atoms = _parse_host_binding_atoms(row.get("host_binding_atoms"))
        if not host_binding_atoms:
            continue

        atoms_list.append(atoms)
        rows.append(
            {
                "adsorption_energy": row.get("adsorption_energy"),
                "site_name": site_name,
                "guest_name": guest_name,
                "host_binding_elements": symbols[host_binding_atoms],
                "host_binding_atoms": host_binding_atoms,
            }
        )
    return pd.DataFrame(rows), atoms_list


def _load_mamun_records(
    adsorbate_map: AdsorbateMap,
    guest_list: Sequence[str],
    pickle_path: str,
    data_slice: Sequence[int] | None = None,
    skip_indices: Sequence[int] | None = None,
) -> list[DatasetRecord]:
    if skip_indices is None:
        skip_indices = []

    with open(pickle_path, "rb") as f:
        reactions = pickle.load(f)
    if data_slice is not None:
        reactions = reactions[slice(*data_slice)]
    raw_df = pd.DataFrame(reactions)

    product_to_adsorbed = {product: star_name for product, star_name in adsorbate_map}
    guest_atom_symbols = _derive_guest_atom_symbols(adsorbate_map)
    guest_set = set(guest_list)
    allowed_products = [
        product
        for product, star_name in adsorbate_map
        if _star_name_to_guest_name(star_name) in guest_set
    ]

    filtered_df = raw_df.loc[raw_df["products"].isin(allowed_products)]
    records = []
    for i in tqdm(range(len(filtered_df["reactionSystems"])), desc="Format data"):
        if i in skip_indices:
            continue
        product = filtered_df.iloc[i]["products"]
        reaction_systems = filtered_df.iloc[i]["reactionSystems"]
        adsorbed_key = product_to_adsorbed[product]
        try:
            atoms = reaction_systems[adsorbed_key]
        except KeyError:
            continue
        guest_name, guest_elements = _guest_name_from_symbols(
            atoms.get_chemical_symbols(),
            guest_atom_symbols,
        )
        records.append(
            {
                "atoms": atoms,
                "adsorption_energy": filtered_df.iloc[i]["reactionEnergy"],
                "guest_elements": guest_elements,
                "metadata": {
                    "adsorption_energy": filtered_df.iloc[i]["reactionEnergy"],
                    "guest_name": guest_name,
                    "facet": int(filtered_df.iloc[i]["facet"]),
                },
            }
        )
    return records


def _load_pw91_records(
    adsorbate_map: AdsorbateMap,
    guest_list: Sequence[str],
    site_list: Sequence[str],
    host_elements: Sequence[str],
    db_path: str,
) -> list[DatasetRecord]:
    guest_atom_symbols = _derive_guest_atom_symbols(adsorbate_map)
    guest_set = set(guest_list)
    site_set = set(site_list)
    host_set = set(host_elements)
    source_db = connect(db_path)

    records = []
    for row in tqdm(source_db.select(), total=source_db.count(), desc="Build database"):
        guest_name = row.get("guest_name")
        if guest_name not in guest_set:
            continue
        site_name = row.get("site_name")
        if site_name not in site_set:
            continue

        atoms = row.toatoms()
        symbols = set(atoms.get_chemical_symbols())
        non_guest_symbols = symbols - guest_atom_symbols
        if not non_guest_symbols.issubset(host_set):
            continue

        records.append(
            {
                "atoms": atoms,
                "adsorption_energy": row.get("adsorption_energy"),
                "guest_elements": sorted(symbols & guest_atom_symbols),
                "metadata": {
                    "adsorption_energy": row.get("adsorption_energy"),
                    "guest_name": guest_name,
                    "source_site_name": site_name,
                },
            }
        )
    return records


def _load_raw_dataset(
    dataset_key: str,
    adsorbate_map: AdsorbateMap,
    guest_list: Sequence[str],
    site_list: Sequence[str],
    host_elements: Sequence[str],
    **kwargs: Any,
) -> tuple[pd.DataFrame, list[Atoms]]:
    guest_atom_symbols = _derive_guest_atom_symbols(adsorbate_map)
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        temp_db_path = tmp.name
    os.unlink(temp_db_path)

    if dataset_key == "dataset_mamun":
        records = _load_mamun_records(
            adsorbate_map=adsorbate_map,
            guest_list=guest_list,
            pickle_path=kwargs["pickle_path"],
            data_slice=kwargs.get("data_slice"),
            skip_indices=kwargs.get("skip_indices"),
        )
        site_name_getter = lambda row: str(row.get("binding_sites"))
    elif dataset_key == "dataset_pw91":
        records = _load_pw91_records(
            adsorbate_map=adsorbate_map,
            guest_list=guest_list,
            site_list=site_list,
            host_elements=host_elements,
            db_path=kwargs["db_path"],
        )
        site_name_getter = lambda row: row.get("source_site_name")
    else:
        raise ValueError(f"Unsupported dataset source: {dataset_key}")

    try:
        _write_ghit_database(records, temp_db_path)
        return _canonicalize_ghit_rows(
            db_path=temp_db_path,
            guest_list=guest_list,
            site_list=site_list,
            host_elements=host_elements,
            guest_atom_symbols=guest_atom_symbols,
            site_name_getter=site_name_getter,
        )
    finally:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)


def load_unrelaxed_models(filepath: str) -> dict[str, UnrelaxedModelData]:
    models: dict[str, UnrelaxedModelData] = {}
    current_name = None
    current_meta: UnrelaxedModelData = {}
    current_xy: list[list[float]] = []

    def _flush():
        if current_name is None:
            return
        if current_xy:
            models[current_name] = {**current_meta, "xy": np.array(current_xy)}
        else:
            models[current_name] = {**current_meta}

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                _flush()
                current_name = line[1:-1]
                current_meta = {
                    "atoms_per_layer": None,
                    "layer2_from_actual": False,
                    "layer3_xy_from_layer2": False,
                }
                current_xy = []
            elif "=" in line:
                key, val = [s.strip() for s in line.split("=", 1)]
                if key == "atoms_per_layer":
                    current_meta[key] = int(val)
                elif key in ("layer2_from_actual", "layer3_xy_from_layer2"):
                    current_meta[key] = val.lower() == "true"
                elif key == "type":
                    current_meta[key] = val
                elif key == "size":
                    current_meta[key] = tuple(int(x) for x in val.split())
            else:
                x, y = map(float, line.split())
                current_xy.append([x, y])

    _flush()
    return models


def _snap_to_ideal(
    coords_in: NDArray[np.float64],
    ideal_coords: NDArray[np.float64],
    cell: Any,
) -> NDArray[np.float64]:
    from ase.geometry import get_distances

    coords_new = np.zeros_like(coords_in)
    used = []
    for i in range(len(coords_in)):
        dists = get_distances(coords_in[i : i + 1], ideal_coords, cell=cell, pbc=True)[
            1
        ][0]
        j = int(np.argmin(dists))
        while j in used:
            dists[j] = np.inf
            j = int(np.argmin(dists))
        used.append(j)
        coords_new[i] = ideal_coords[j]
    return coords_new


def _ideal_lattice_ase_builder(atoms: Atoms, model_data: UnrelaxedModelData) -> Atoms:
    import ase.build

    n = model_data["atoms_per_layer"]
    size = model_data["size"]
    surface_type = model_data["type"]

    ghit = GHIT(atoms)
    ghit.determine_layers(use_layer=True, atoms_per_layer=n)
    layer_map = ghit.get_atom_layers()[0]
    layer_keys_by_z = sorted(
        layer_map,
        key=lambda k: np.mean(atoms.get_positions()[list(layer_map[k]), 2]),
    )
    bottom_layer = sorted(layer_map[layer_keys_by_z[0]])
    second_layer = sorted(layer_map[layer_keys_by_z[1]])

    all_distances = np.sort(atoms.get_all_distances(), axis=1)
    coords = atoms.get_positions()
    cell = atoms.get_cell()
    vacuum = np.min(coords[:, 2])
    symbols = atoms.get_chemical_symbols()

    if surface_type == "fcc111":
        nn_dist = all_distances[bottom_layer[-1], 1]
        a = nn_dist * 2 / np.sqrt(2)
        build_atoms = ase.build.fcc111("Cu", size, a=a, vacuum=vacuum)
        ideal_coords = build_atoms.get_positions()

        sort_ideal_l2 = np.sort(ideal_coords[n : 2 * n], axis=0)
        sort_actual_l2 = np.sort(coords[second_layer], axis=0)
        diff_x = sort_actual_l2[:, 0] - sort_ideal_l2[:, 0]
        diff_y = sort_actual_l2[:, 1] - sort_ideal_l2[:, 1]
        ideal_coords[n : 2 * n] = coords[second_layer]
        ideal_coords[2 * n : 3 * n, 0] += diff_x
        ideal_coords[2 * n : 3 * n, 1] -= diff_y
    elif surface_type == "hcp0001":
        side_diameter = all_distances[bottom_layer[-1], 1]
        within_layer = atoms.get_distances(
            bottom_layer[-1], bottom_layer[:-1], mic=False
        )
        within_layer.sort()
        diameter = within_layer[0]
        c = 2 * np.sqrt(2 / 3) * side_diameter
        build_atoms = ase.build.hcp0001(
            "Cu",
            size,
            a=diameter,
            c=c,
            vacuum=vacuum,
            orthogonal=False,
            periodic=False,
        )
        ideal_coords = build_atoms.get_positions()

        diff = coords[bottom_layer] - ideal_coords[0:n]
        ideal_coords[0:n] = coords[bottom_layer]
        ideal_coords[2 * n : 3 * n] += diff
    else:
        raise ValueError(f"Unknown ASE builder type: {surface_type!r}")

    snapped_positions = _snap_to_ideal(coords, ideal_coords, cell)
    return Atoms(
        symbols=symbols,
        positions=snapped_positions,
        cell=cell,
        pbc=atoms.get_pbc(),
    )


def ideal_lattice(atoms: Atoms, model_data: UnrelaxedModelData) -> Atoms:
    if model_data.get("type"):
        return _ideal_lattice_ase_builder(atoms, model_data)

    scaled_positions = atoms.get_scaled_positions()
    first = scaled_positions[0][2]
    second = scaled_positions[4][2]
    third = second + (second - first)
    chemical_symbols = atoms.get_chemical_symbols()
    cell = atoms.get_cell()

    n = model_data["atoms_per_layer"]
    positions = np.zeros((3 * n, 3))
    positions[:, 0:2] = model_data["xy"]
    positions[0:n, 2] = first
    positions[n : 2 * n, 2] = second
    positions[2 * n : 3 * n, 2] = third

    if model_data.get("layer2_from_actual"):
        positions[n : 2 * n] = scaled_positions[n : 2 * n]

    if model_data.get("layer3_xy_from_layer2"):
        l2 = positions[n : 2 * n]
        positions[2 * n, 0:2] = 2 * l2[1, 0:2]
        positions[2 * n + 1, 0:2] = l2[0, 0:2] + l2[1, 0:2]
        positions[2 * n + 2, 0:2] = l2[1, 0:2] + l2[3, 0:2]
        positions[2 * n + 3, 0:2] = l2[1, 0:2] + l2[2, 0:2]
        positions[2 * n : 3 * n, 2] = third

    return Atoms(symbols=chemical_symbols, scaled_positions=positions, cell=cell)


def load_dataset(
    dataset_key: str,
    adsorbate_map: AdsorbateMap,
    guest_list: Sequence[str],
    site_list: Sequence[str],
    host_elements: Sequence[str],
    unrelaxed_models_path: str | None = None,
    apply_ideal_lattice: bool = True,
    **kwargs: Any,
) -> tuple[pd.DataFrame, list[Atoms], list[Atoms | None]]:
    df, atomsList = _load_raw_dataset(
        dataset_key=dataset_key,
        adsorbate_map=adsorbate_map,
        guest_list=guest_list,
        site_list=site_list,
        host_elements=host_elements,
        **kwargs,
    )

    guest_atom_symbols = _derive_guest_atom_symbols(adsorbate_map)
    unrelaxed_models = (
        load_unrelaxed_models(unrelaxed_models_path) if apply_ideal_lattice else None
    )
    new_bare_surface_list = []
    for atoms in tqdm(atomsList, desc="Implement facets"):
        atoms_copy = atoms.copy()
        ghit = GHIT(atoms_copy)
        ghit.find_guest_atoms(guest_elements=list(guest_atom_symbols))
        atoms_copy.set_constraint()
        atoms_copy = atoms_copy[ghit.get_host_atoms()]
        if apply_ideal_lattice:
            actual_distances = atoms_copy.get_all_distances()
            best_diff = float("inf")
            best_reference = None
            for model_data in unrelaxed_models.values():
                n = model_data["atoms_per_layer"]
                expected = (
                    n * model_data["size"][2] if model_data.get("type") else 3 * n
                )
                if len(atoms_copy) != expected:
                    continue
                reference = ideal_lattice(atoms_copy, model_data)
                diff = np.sum(abs(actual_distances - reference.get_all_distances()))
                if diff < best_diff:
                    best_diff = diff
                    best_reference = reference
            new_bare_surface_list.append(best_reference)
        else:
            new_bare_surface_list.append(atoms_copy)

    return df, atomsList, new_bare_surface_list


def make_data(cfg: Config, exp_cfg: ExperimentConfig) -> pd.DataFrame:
    data_cfg = cfg.data_generation
    surfep_path = data_cfg.surfep_path
    sys.path.append(os.path.join(surfep_path, "algorithms/"))
    from surfEP import surfEP

    pure_element_path = data_cfg.pure_element_path
    mathematica_path = data_cfg.mathematica_path
    power_term = data_cfg.power_term

    pureElementDataFrame1 = pd.read_csv(
        pure_element_path,
        index_col=0,
        skiprows=1,
    )
    pureElementDataFrame2 = pd.read_csv(
        mathematica_path,
        index_col=0,
        skiprows=1,
    )
    pureElementDataFrame = pd.concat(
        [pureElementDataFrame1, pureElementDataFrame2], axis=1, sort=True
    )

    df, atomsList, new_bare_surface_list = load_dataset(
        dataset_key=data_cfg.dataset_module,
        adsorbate_map=exp_cfg.adsorbate_map,
        guest_list=exp_cfg.guest_list,
        site_list=exp_cfg.site_list,
        host_elements=exp_cfg.host_elements,
        unrelaxed_models_path=data_cfg.unrelaxed_models_path,
        apply_ideal_lattice=data_cfg.apply_ideal_lattice,
        # Dataset-specific kwargs forwarded to load_raw_data:
        data_slice=data_cfg.data_slice,
        skip_indices=data_cfg.skip_indices,
        **(data_cfg.model_extra or {}),
    )

    element_symbol_vocabulary = get_unique_element_symbols(atomsList)
    compatible_features = filter_features(
        symbol_vocabulary=element_symbol_vocabulary,
        element_property_strs=exp_cfg.low_theory_features,
    )
    print(
        f"Removed incompatible features. There are {len(compatible_features)} remaining:"
    )
    for feature in compatible_features:
        print(feature)
    combined_features = compatible_features + exp_cfg.high_theory_features

    ### Calculates features of bare surface, adds to dataframe (df)
    surfaceIndicesList = [
        findSurfaceAtoms(bareSurf) for bareSurf in new_bare_surface_list
    ]
    binding_indices_list = [df["host_binding_atoms"][index] for index in range(len(df))]

    adsPredictor = surfEP(
        verbose=False,
        jsonDirectory=os.path.join(surfep_path, "data", "JSONFiles/"),
        elementDataDirectory=os.path.join(surfep_path, "data"),
    )
    # features_all
    (
        sd_coupling_all_list,
        d_filling_diff_all_list,
        e_cond_diff_all_list,
    ) = adsPredictor.featuresForES(new_bare_surface_list, surfaceIndicesList)
    d_filling_all_list = []
    for surfaceIndices, bareSurf in zip(surfaceIndicesList, new_bare_surface_list):
        surfaceElements = np.array(bareSurf.get_chemical_symbols())[surfaceIndices]
        dFillingSurf = [
            pureElementDataFrame.loc[element, "dFillingIdealized"]
            for element in surfaceElements
        ]
        d_filling_all_list.append(dFillingSurf)
    # features
    (
        sd_coupling_list,
        d_filling_diff_list,
        e_cond_diff_list,
    ) = adsPredictor.featuresForES(new_bare_surface_list, binding_indices_list)

    d_filling_list = []
    for surfaceIndices, bareSurf in zip(binding_indices_list, new_bare_surface_list):
        surfaceElements = np.array(bareSurf.get_chemical_symbols())[surfaceIndices]
        dFillingSurf = [
            pureElementDataFrame.loc[element, "dFillingIdealized"]
            for element in surfaceElements
        ]
        d_filling_list.append(dFillingSurf)

    df["sd_coupling_all"] = sd_coupling_all_list
    df["d_filling_n_all"] = d_filling_diff_all_list
    df["e_conductivity_n_all"] = e_cond_diff_all_list
    df["d_filling_central_all"] = d_filling_all_list
    df["sd_coupling"] = sd_coupling_list
    df["d_filling_n"] = d_filling_diff_list
    df["e_conductivity_n"] = e_cond_diff_list
    df["d_filling_central"] = d_filling_list
    df["d_filling_mult"] = df["d_filling_central"] * df["d_filling_n"]

    # Calculate lists of host features and store them in a dataframe column
    num_structures = len(new_bare_surface_list)
    new_cols = {}
    for feature in compatible_features:
        host_feature_all_list_column = []
        host_feature_list_column = []
        # Generate the row values in the column
        for i in tqdm(range(num_structures), desc=f"Weighted feature: {feature}"):
            atoms = new_bare_surface_list[i]
            surface_indices = findSurfaceAtoms(atoms)
            binding_indices = df["host_binding_atoms"][i]
            host_feature_all_list = get_host_feature_list(
                feature, atoms, surface_indices, power_term
            )
            host_feature_list = get_host_feature_list(
                feature, atoms, binding_indices, power_term
            )
            host_feature_all_list_column.append(host_feature_all_list)
            host_feature_list_column.append(host_feature_list)
        new_cols[feature + "_all"] = host_feature_all_list_column
        new_cols[feature] = host_feature_list_column
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    all_elements = []
    for element_list in df["host_binding_elements"]:
        for element in element_list:
            all_elements.append(element)
    host_elements_present = np.unique(np.array(all_elements))
    new_cols = {}
    for host_element in tqdm(host_elements_present, desc="Host fractional occurrence"):
        host_fractional_occurrence = df["host_binding_elements"].apply(
            lambda x: list(x).count(host_element) / len(list(x))
        )
        for feature in combined_features:
            element_occurence_averages = []
            for i, row in df.iterrows():
                # Reading lists from
                host_features = row[feature]
                host_symbols = row["host_binding_elements"]
                element_occurence_averages.append(
                    np.mean(
                        np.array(
                            [
                                feat if host_element in el else 0
                                for feat, el in zip(
                                    host_features,
                                    host_symbols,
                                )
                            ]
                        )
                    )
                )
            new_cols[feature + "_" + host_element] = (
                element_occurence_averages * host_fractional_occurrence
            )
        new_cols["constant" + "_" + host_element] = host_fractional_occurrence
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    indicesList = []
    zip_guest, zip_site = get_zippable_arrays(
        array1=exp_cfg.guest_list, array2=exp_cfg.site_list
    )
    new_cols = {}
    for guest, site in zip(zip_guest, zip_site):
        guestConstantColumn = np.zeros(len(df))
        indices = df.loc[(df["guest_name"] == guest) & (df["site_name"] == site)].index
        guestConstantColumn[indices] = 1
        columnName = "guest_constant_" + guest + "_" + site
        new_cols[columnName] = guestConstantColumn
        indicesList.append(indices)
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def _config_hash(cfg: Config, exp_cfg: ExperimentConfig) -> str:
    data_cfg = cfg.data_generation
    params = dict(
        adsorbate_map=exp_cfg.adsorbate_map,
        low_theory_features=exp_cfg.low_theory_features,
        high_theory_features=exp_cfg.high_theory_features,
        guest_list=exp_cfg.guest_list,
        site_list=exp_cfg.site_list,
        host_elements=exp_cfg.host_elements,
        power_term=data_cfg.power_term,
        apply_ideal_lattice=data_cfg.apply_ideal_lattice,
        data_slice=data_cfg.data_slice,
        skip_indices=data_cfg.skip_indices,
        pure_element_path=data_cfg.pure_element_path,
        mathematica_path=data_cfg.mathematica_path,
        unrelaxed_models_path=data_cfg.unrelaxed_models_path,
        dataset_module=data_cfg.dataset_module,
        **(data_cfg.model_extra or {}),
    )
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]


def output_data(
    cfg: Config | None = None,
    exp_cfg: ExperimentConfig | None = None,
    csv_path: str | pathlib.Path | None = None,
) -> pd.DataFrame:
    if cfg is None:
        if csv_path is None:
            raise ValueError("Either cfg or csv_path must be provided.")
        return pd.read_csv(csv_path)

    if exp_cfg is None:
        raise ValueError("exp_cfg must be provided when cfg is provided.")

    data_cfg = cfg.data_generation
    resolved_csv_path = pathlib.Path(
        csv_path if csv_path is not None else data_cfg.data_filepath
    )

    if resolved_csv_path.is_file():
        return pd.read_csv(resolved_csv_path)

    df = make_data(cfg, exp_cfg)
    df.to_csv(resolved_csv_path, index=False)
    return df


def split_data(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    mf = cfg.model_fitting
    train_df = df.sample(frac=mf.train_frac, random_state=mf.random_state)
    test_df = df.drop(train_df.index)
    return train_df, test_df


def get_zippable_arrays(
    array1: Sequence[str],
    array2: Sequence[str],
) -> tuple[list[str], list[str]]:
    # Create zippable lists
    zippable_array1 = []
    zippable_array2 = []
    for item1 in array1:
        for item2 in array2:
            zippable_array1.append(item1)
            zippable_array2.append(item2)
    return zippable_array1, zippable_array2


def pickle_data(object_list: Any, pickled_data_filepath: str) -> None:
    if not pathlib.Path(pickled_data_filepath).is_file():
        with open(pickled_data_filepath, "wb") as f:  # open a text file
            pickle.dump(object_list, f)  # serialize the list
        f.close()
    return


def load_pickled_data(pickled_data_filepath: str) -> Any:
    with open(pickled_data_filepath, "rb") as f:
        object_list = pickle.load(f)  # deserialize using load
    return object_list
