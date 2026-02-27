import json
from collections import defaultdict
from pathlib import Path

DEFAULT_BASE_DIR = Path("data/mlips")
DEFAULT_OUT_DIR = Path("data/mlips_by_prefix")
DEFAULT_PREFIXES = ("ol", "sa", "ss")


def run(
    base_dir: Path = DEFAULT_BASE_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    prefixes: tuple[str, ...] = DEFAULT_PREFIXES,
) -> None:
    out_dir.mkdir(exist_ok=True)
    prefixes_set = set(prefixes)

    for model_dir in base_dir.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        result_file = model_dir / f"{model_name}_result.json"

        if not result_file.exists():
            continue

        print(f"Processing {model_name}")

        with open(result_file) as f:
            data = json.load(f)

        calculation_settings = data.get("calculation_settings")

        # Partition by prefix
        subsets = defaultdict(dict)

        for key, value in data.items():
            prefix = key.split("-")[0]
            if prefix in prefixes_set:
                subsets[prefix][key] = value

        # Write each subset
        for prefix, subset_dict in subsets.items():
            out_prefix_dir = out_dir / prefix
            out_prefix_dir.mkdir(exist_ok=True)

            out_model_dir = out_prefix_dir / model_name
            out_model_dir.mkdir(exist_ok=True)
            out_file = out_model_dir / f"{model_name}_result.json"

            output_data = {}
            if calculation_settings is not None:
                output_data["calculation_settings"] = calculation_settings
            output_data.update(subset_dict)

            with open(out_file, "w") as f:
                json.dump(output_data, f, indent=2)

            print(f"  Wrote {prefix} -> {len(subset_dict)} entries")
