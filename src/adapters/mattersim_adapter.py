# src/oasis/adapters/mattersim_adapter.py

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from catbench.adsorption import AdsorptionCalculation
from mattersim.forcefield.potential import Potential, MatterSimCalculator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MatterSim adsorption predictions")
    parser.add_argument("--input", required=True, help="Input dataset JSON")
    parser.add_argument("--output", required=True, help="Output result JSON")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--checkpoint-path",
        default="mattersim-v1.0.0-5M.pth",
        help="MatterSim checkpoint file",
    )
    parser.add_argument(
        "--n-calcs",
        type=int,
        default=3,
        help="Number of independent MatterSim calculators",
    )
    args = parser.parse_args()
    t0 = time.time()
    calculators = []
    for _ in range(args.n_calcs):
        potential = Potential.from_checkpoint(
            load_path=args.checkpoint_path,
            device=args.device,
        )
        calculators.append(MatterSimCalculator(potential=potential))

    # --- Run CatBench adsorption workflow ---
    adsorption_calc = AdsorptionCalculation(
        calculators,
        mlip_name="mattersim-v1-5m",
        dataset_path=args.input,
    )
    results = adsorption_calc.run()

    # --- Standardized output ---
    out = {
        "model": "mattersim",
        "model_version": "v1-5m",
        "checkpoint": Path(args.checkpoint_path).name,
        "n_calculators": args.n_calcs,
        "device": args.device,
        "input_dataset": Path(args.input).name,
        "wall_time_s": time.time() - t0,
        "results": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
