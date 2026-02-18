import re
import shutil
from typing import Dict
import yaml
from collections import defaultdict
from pathlib import Path
from catbench.adsorption.data import vasp as catbench_vasp

from oasis.config import get_config
from oasis.ingest.stoichiometry import solve_stoichiometry


def tag_to_ads(tag, tag_map):
    label = tag_map[tag]
    return label.replace("*", "")


def copy_selected_files(src_dir: Path, dst_dir: Path, filenames=("CONTCAR", "OSZICAR")):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        src_file = src_dir / name
        if src_file.is_file():
            shutil.copy2(src_file, dst_dir / name)


def read_E0_from_OSZICAR(file_path):
    """
    Read final energy (E0) from VASP OSZICAR file.

    Args:
        file_path: Path to OSZICAR file

    Returns:
        float: Final energy in eV
    """
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()
            last_line = lines[-1]

        energy = None
        for word in last_line.split():
            if word == "E0=":
                energy_index = last_line.split().index(word) + 1
                energy = last_line.split()[energy_index]
                energy = float(energy)
                break

        if energy is None:
            raise ValueError(f"Energy value not found in file: {file_path}")

        return energy

    except Exception as e:
        raise RuntimeError(
            f"An error occurred while reading the file '{file_path}': {str(e)}"
        )


_TOKEN = re.compile(
    r"""
    (?P<elem>[A-Z][a-z]?) |      # Element symbol, e.g. C, O, Au
    (?P<num>\d+)          |      # Number
    (?P<lpar>\()          |      # (
    (?P<rpar>\))          |      # )
    (?P<other>.)                 # any other single char
""",
    re.VERBOSE,
)


def formula_to_composition(formula: str) -> dict[str, int]:
    """
    Convert a chemical formula string to an element->count dict.

    Examples:
      "CH3OH" -> {"C":1,"H":4,"O":1}
      "CH3(CH2)2OH" -> {"C":3,"H":8,"O":1}
      "*CH3OH" -> {"C":1,"H":4,"O":1}
      "OCH3Au" -> {"O":1,"C":1,"H":3,"Au":1}
    """
    # Strip decorations that aren’t chemistry
    s = formula.strip()
    s = s.replace("*", "")
    s = re.sub(r"[\s\-_]", "", s)

    # Stack of composition dicts for parentheses
    stack: list[defaultdict[str, int]] = [defaultdict(int)]
    pending_elem: str | None = None

    tokens = list(_TOKEN.finditer(s))
    i = 0
    while i < len(tokens):
        t = tokens[i]
        kind = t.lastgroup
        val = t.group()

        if kind == "elem":
            pending_elem = val
            # default count is 1 unless a number follows
            # (we’ll apply after seeing next token)
            i += 1
            # if next token is a number, consume it
            count = 1
            if i < len(tokens) and tokens[i].lastgroup == "num":
                count = int(tokens[i].group())
                i += 1
            stack[-1][pending_elem] += count
            pending_elem = None
            continue

        if kind == "lpar":
            stack.append(defaultdict(int))
            i += 1
            continue

        if kind == "rpar":
            # pop group, apply multiplier if present
            group = stack.pop()
            i += 1
            mult = 1
            if i < len(tokens) and tokens[i].lastgroup == "num":
                mult = int(tokens[i].group())
                i += 1
            for e, c in group.items():
                stack[-1][e] += c * mult
            continue

        if kind == "num":
            # A naked number without a preceding element or ')'
            # means the formula is in a form we don't support.
            raise ValueError(f"Unexpected number '{val}' in formula '{formula}'")

        # Ignore other characters (charges, dots, etc.) for now
        i += 1

    if len(stack) != 1:
        raise ValueError(f"Unbalanced parentheses in formula '{formula}'")

    return dict(stack[0])


def build_coeff_setting(cfg, tag_map: Path) -> Dict[str, dict]:
    """
    Returns:
      coeff_setting[formula] = {
          "slab": -1,
          "adslab": 1,
          "CH4gas": c_CH4,
          "O2gas": c_O2,
          "H2gas": c_H2,
      }
    """

    coeff_setting: Dict[str, dict] = {}

    for tag, formula in tag_map.items():
        comp = formula_to_composition(formula)

        # OPTIONAL: restrict to allowed elements
        allowed = set(cfg.ingest.stoich.elements)
        extra = set(comp) - allowed
        if extra:
            raise ValueError(
                f"Tag {tag} -> '{formula}' includes elements {sorted(extra)} "
                f"not in cfg.ingest.stoich.elements={cfg.ingest.stoich.elements}"
            )

        # Solve for [CH4, O2, H2]
        c_CH4, c_O2, c_H2 = solve_stoichiometry(cfg, comp)

        gas_terms = {
            "CH4gas": -c_CH4,
            "O2gas": -c_O2,
            "H2gas": -c_H2,
        }

        gas_terms = {k: v for k, v in gas_terms.items() if v != 0}

        coeff_setting[formula] = {
            "slab": -1,
            "adslab": 1,
            **gas_terms,
        }
    return coeff_setting


def main():
    cfg = get_config()
    source = cfg.ingest.source
    dest = cfg.ingest.catbench_folder
    if dest is None:
        raise ValueError("cfg.ingest.catbench_folder is not initialized")

    if (source / "mapping.yaml").is_file():
        mapping_root = source
    elif (source.parent / "mapping.yaml").is_file():
        mapping_root = source.parent
    else:
        raise FileNotFoundError(
            f"Could not find mapping.yaml in {source} or {source.parent}"
        )

    with (mapping_root / "mapping.yaml").open() as f:
        tag_map = yaml.safe_load(f)

    dest.mkdir(parents=True, exist_ok=True)
    (dest / "gas").mkdir(parents=True, exist_ok=True)

    # --------------------
    # GAS CONVERSION
    # --------------------
    gas_src = mapping_root / "gas"
    gas_dst = dest / "gas"

    if gas_src.is_dir():
        for folder in gas_src.iterdir():
            if not folder.is_dir():
                continue

            tag = folder.name[:4]
            if tag not in tag_map:
                continue

            ads = tag_to_ads(tag, tag_map)
            new_name = f"{ads}gas"

            copy_selected_files(folder, gas_dst / new_name)

    # --------------------
    # SYSTEM CONVERSION
    # --------------------
    test_src = source
    if not test_src.is_dir():
        raise FileNotFoundError(f"System source folder does not exist: {test_src}")

    systems = defaultdict(list)

    for folder in test_src.iterdir():
        if not folder.is_dir():
            continue

        oszicar_path = folder / "OSZICAR"
        try:
            # Filter out directories with missing/corrupt OSZICAR before conversion.
            read_E0_from_OSZICAR(oszicar_path)
        except RuntimeError:
            continue

        parts = folder.name.split("-")
        if len(parts) < 3:
            continue

        tag_part = parts[-1]
        tag = tag_part[:4]
        system = "-".join(parts[:-1])
        systems[system].append((tag, folder.name))

    for system, entries in systems.items():
        system_dir = dest / system
        system_dir.mkdir(parents=True, exist_ok=True)

        config_counter = defaultdict(int)

        for tag, folder in entries:
            src_path = test_src / folder

            if tag == "0000":
                dst_path = system_dir / "slab"
                copy_selected_files(src_path, dst_path)
            else:
                ads = tag_to_ads(tag, tag_map)
                config_counter[ads] += 1
                config_index = config_counter[ads]

                dst_path = system_dir / ads / str(config_index)
                copy_selected_files(src_path, dst_path)

    coeff_setting = build_coeff_setting(cfg, tag_map)

    # CatBench's VASP preprocessor uses one parameter for both:
    # 1) input dataset directory traversal, and 2) output JSON filename stem.
    # Patch its path helpers so we can read from the real dataset path while
    # writing to a stable project-root output file.
    project_root = Path.cwd()
    output_dir = project_root / "data" / "raw_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{cfg.ingest.dataset_name}_adsorption.json"

    catbench_vasp.get_raw_data_directory = lambda: str(output_dir)
    catbench_vasp.get_raw_data_path = lambda _benchmark_name: str(output_path)
    catbench_vasp.vasp_preprocessing(
        dataset_name=str(dest),
        coeff_setting=coeff_setting,
    )


if __name__ == "__main__":
    main()
