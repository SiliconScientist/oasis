# src/oasis/adapters/sevennet_adapter.py

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from catbench.adsorption import AdsorptionCalculation
from sevenn.calculator import SevenNetCalculator

MLIP_NAME = "7net-omni"
MODAL = "mpa"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SevenNet adsorption predictions")
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
        default=None,
        help="Unused for SevenNet preset model IDs (kept for adapter interface compatibility).",
    )
    parser.add_argument("--n-calcs", type=int, default=3)
    args = parser.parse_args()

    t0 = time.time()

    # --- Build calculators ---
    # SevenNetCalculator typically loads by model ID string; device is handled internally
    calculators = [
        SevenNetCalculator(model=MLIP_NAME, modal=MODAL) for _ in range(args.n_calcs)
    ]

    # --- Run CatBench adsorption workflow ---
    adsorption_calc = AdsorptionCalculation(
        calculators,
        mlip_name=MLIP_NAME,
        benchmark=args.dataset_name,
    )
    results = adsorption_calc.run()

    # --- Standardized output ---
    out = {
        "model": "sevennet",
        "model_version": MLIP_NAME,
        "modal": MODAL,
        "checkpoint": Path(args.model_path).name if args.model_path else None,
        "n_calculators": args.n_calcs,
        "device": args.device,
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
