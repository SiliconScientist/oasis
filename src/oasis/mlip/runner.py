# src/oasis/mlip/runner.py

import os
import subprocess
from pathlib import Path

from oasis.mlip.registry import (
    get_catbench_source_path,
    get_model_path,
    get_model_python,
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

    python_exe = get_model_python(model, config_path)
    model_path = get_model_path(model, config_path)  # may be None
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
        f"oasis.adapters.{model}_adapter",
        "--input",
        input_path,
        "--output",
        output_path,
        "--dataset-name",
        dataset_name,
        "--config",
        str(config_path),
    ]

    if model_path is not None:
        cmd += ["--model-path", str(model_path)]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
