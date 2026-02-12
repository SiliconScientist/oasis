import os
import shutil
import yaml
from collections import defaultdict

SOURCE = "data/raw_vasp"
DEST = "data/raw_vasp/rodrigo_catbench"

with open(os.path.join(SOURCE, "mapping.yaml")) as f:
    tag_map = yaml.safe_load(f)


def tag_to_ads(tag):
    label = tag_map[tag]
    return label.replace("*", "")


def copy_selected_files(src_dir, dst_dir, filenames=("CONTCAR", "OSZICAR")):
    os.makedirs(dst_dir, exist_ok=True)
    for name in filenames:
        src_file = os.path.join(src_dir, name)
        if os.path.isfile(src_file):
            shutil.copy2(src_file, os.path.join(dst_dir, name))


os.makedirs(DEST, exist_ok=True)
os.makedirs(os.path.join(DEST, "gas"), exist_ok=True)

# --------------------
# GAS CONVERSION
# --------------------
gas_src = os.path.join(SOURCE, "gas")
gas_dst = os.path.join(DEST, "gas")

for folder in os.listdir(gas_src):
    tag = folder[:4]
    if tag not in tag_map:
        continue

    ads = tag_to_ads(tag)
    new_name = f"{ads}gas"

    copy_selected_files(
        os.path.join(gas_src, folder),
        os.path.join(gas_dst, new_name),
    )

# --------------------
# SYSTEM CONVERSION
# --------------------
test_src = os.path.join(SOURCE, "test")

systems = defaultdict(list)

for folder in os.listdir(test_src):
    parts = folder.split("-")
    if len(parts) < 3:
        # optional: skip unexpected folder names
        continue

    tag_part = parts[-1]
    tag = tag_part[:4]
    system = "-".join(parts[:-1])

    systems[system].append((tag, folder))

for system, entries in systems.items():
    system_dir = os.path.join(DEST, system)
    os.makedirs(system_dir, exist_ok=True)

    config_counter = defaultdict(int)

    for tag, folder in entries:
        src_path = os.path.join(test_src, folder)

        if tag == "0000":
            dst_path = os.path.join(system_dir, "slab")
            copy_selected_files(src_path, dst_path)
        else:
            ads = tag_to_ads(tag)
            config_counter[ads] += 1
            config_index = config_counter[ads]

            dst_path = os.path.join(system_dir, ads, str(config_index))
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)

            copy_selected_files(src_path, dst_path)

print("Conversion complete.")
