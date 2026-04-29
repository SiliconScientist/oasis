# src/oasis/adapters/mace_adapter.py

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from contextlib import ExitStack
from rootstock import RootstockCalculator
from catbench.adsorption import AdsorptionCalculation
from mace.calculators import mace_mp

from oasis.mlip.registry import load_config

MLIP_NAME = "mace-mh-1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MACE adsorption predictions")
    parser.add_argument("--input", required=True, help="Input dataset JSON")
    parser.add_argument("--output", required=True, help="Output result JSON")
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Dataset name (passed from task runner)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--n-calcs", type=int, default=3)
    args = parser.parse_args()
    config = load_config(args.config)
    optimizer = str(config.get("mlip", {}).get("optimizer", "LBFGS"))

    t0 = time.time()

    # --- Build calculators ---
    with ExitStack() as stack:
        calculators = [
            stack.enter_context(
                RootstockCalculator(
                    root="/projects/bchg/rootstock",
                    model="mace",
                    checkpoint="medium",
                    device=args.device,
                )
            )
            for _ in range(args.n_calcs)
        ]
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
        "model": "mace",
        "model_version": "mace-mh-1",
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
