import shutil
import yaml
from collections import defaultdict
from pathlib import Path

from oasis.config import get_config


def tag_to_ads(tag, tag_map):
    label = tag_map[tag]
    return label.replace("*", "")


def copy_selected_files(src_dir: Path, dst_dir: Path, filenames=("CONTCAR", "OSZICAR")):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        src_file = src_dir / name
        if src_file.is_file():
            shutil.copy2(src_file, dst_dir / name)


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

    print("Conversion complete.")


if __name__ == "__main__":
    main()
