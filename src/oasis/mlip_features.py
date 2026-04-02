import json
from pathlib import Path

from ase.db.row import AtomsRow
from ase.io import jsonio
from ase.visualize import view


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
    atoms_list = load_tolstar_atoms(
        json_path="data/raw_data/KHLOHC_origin_adsorption.json"
    )
    print(f"Loaded {len(atoms_list)} Tolstar Atoms objects.")
