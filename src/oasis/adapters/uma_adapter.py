# src/oasis/adapters/uma_s1p1_adapter.py

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from catbench.adsorption import AdsorptionCalculation
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit

from oasis.mlip.registry import load_config

MLIP_NAME = "uma-s-1p1"
TASK_NAME = "oc20"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run UMA-S-1p1 adsorption predictions")
    parser.add_argument("--input", required=True, help="Input dataset JSON")
    parser.add_argument("--output", required=True, help="Output result JSON")
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Dataset name (passed from task runner)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to UMA-S-1p1 checkpoint (.pt)",
    )
    parser.add_argument("--n-calcs", type=int, default=3)
    args = parser.parse_args()
    config = load_config(args.config)
    optimizer = str(config.get("mlip", {}).get("optimizer", "LBFGS"))

    t0 = time.time()

    # --- Build calculators ---
    calculators = []
    for _ in range(args.n_calcs):
        predict_unit = load_predict_unit(
            path=args.model_path,
            device=args.device,
        )
        calc = FAIRChemCalculator(
            predict_unit=predict_unit,
            task_name=TASK_NAME,
        )
        calculators.append(calc)

    # --- Run CatBench adsorption workflow ---
    adsorption_calc = AdsorptionCalculation(
        calculators,
        mlip_name=MLIP_NAME,
        benchmark=args.dataset_name,
        optimizer=optimizer,
    )
    results = adsorption_calc.run()

    # --- Standardized output ---
    out = {
        "model": "uma",
        "model_version": MLIP_NAME,
        "checkpoint": Path(args.model_path).name,
        "task_name": TASK_NAME,
        "n_calculators": args.n_calcs,
        "device": args.device,
        "optimizer": optimizer,
        "dataset_name": args.dataset_name,
        "input_dataset": Path(args.input).name,
        "wall_time_s": time.time() - t0,
        "results": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
