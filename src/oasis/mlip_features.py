from functools import partial
import json
from pathlib import Path
from typing import Any

from ase.db.row import AtomsRow
from ase.io import jsonio
from ase.visualize import view

from oasis.config import get_config
from oasis.ingest.site_constraints import (
    extract_adsorbate_indices,
    extract_adsorbed_atom,
    index_by_layers,
    load_mlip_dataset,
    strip_adsorbate_from_adslab,
)


DEFAULT_JSON_PATH = Path(__file__).with_name("KHLOHC_origin_adsorption.json")


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


if __name__ == "__main__":
    cfg = get_config()
    index_fn = partial(index_by_layers, layers=(-1))
    dataset = load_mlip_dataset(cfg)
    updated_dataset: dict[str, Any] = {}
    for reaction, entry in dataset.items():
        adsorbed_atom = extract_adsorbed_atom(entry, reaction)
        indices = extract_adsorbate_indices(entry, reaction)
        bare_surface = strip_adsorbate_from_adslab(adsorbed_atom, indices)
        top_layer_indices = index_fn(bare_surface)
        print(f"Reaction: {reaction}")
