# src/oasis/mlip/runner.py

import subprocess
from pathlib import Path

from oasis.mlip.registry import get_model_python, get_model_path


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
    model_path = get_model_path(model, config_path)

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
        "--model-path",
        model_path,
        "--config",
        config_path,
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
