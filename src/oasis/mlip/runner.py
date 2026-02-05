# src/oasis/mlip/runner.py

import subprocess
from oasis.mlip.registry import get_model_python, get_model_path


def run_one_task(line: str, config_path: str):
    model, input_path, output_path = line.split()

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
        "--model-path",
        model_path,
        "--config",
        config_path,
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
