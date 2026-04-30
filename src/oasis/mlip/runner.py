# src/oasis/mlip/runner.py

import os
import subprocess
import sys
from pathlib import Path

from oasis.mlip.registry import (
    get_catbench_source_path,
    get_rootstock_python,
)


def run_one_task(line: str, config_path: str):
    parts = line.split()
    if len(parts) == 4:
        model, dataset_name, input_path, output_path = parts
    elif len(parts) == 3:
        model, input_path, output_path = parts
        dataset_name = Path(input_path).stem
    else:
        raise ValueError(
            "Task line must have 3 or 4 fields: "
            "<model> <input_path> <output_path> (legacy) or "
            "<model> <dataset_name> <input_path> <output_path>"
        )

    python_exe = get_rootstock_python(config_path) or sys.executable
    catbench_source = get_catbench_source_path(config_path)

    env = os.environ.copy()
    if catbench_source is not None:
        if not catbench_source.exists():
            raise FileNotFoundError(
                f"Configured CatBench source path does not exist: {catbench_source}"
            )
        if not (catbench_source / "catbench").exists():
            raise FileNotFoundError(
                "Configured CatBench source path must contain a 'catbench' package "
                f"directory: {catbench_source}"
            )
        existing_pythonpath = env.get("PYTHONPATH")
        pythonpath_entries = [str(catbench_source)]
        if existing_pythonpath:
            pythonpath_entries.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    cmd = [
        python_exe,
        "-m",
        "oasis.adapters.rootstock_adapter",
        "--model",
        model,
        "--input",
        input_path,
        "--output",
        output_path,
        "--dataset-name",
        dataset_name,
        "--config",
        str(config_path),
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
