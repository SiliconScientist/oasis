# src/oasis/mlip/runner.py

import subprocess
from oasis.mlip.registry import get_model_python


def run_one_task(line: str, config_path: str):
    model, input_path, output_path = line.split()

    python_exe = get_model_python(model, config_path)

    cmd = [
        python_exe,
        "-m",
        f"oasis.adapters.{model}_adapter",
        "--input",
        input_path,
        "--output",
        output_path,
        "--config",
        config_path,
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
