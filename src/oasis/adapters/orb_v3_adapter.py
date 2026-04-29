# src/oasis/adapters/orb_v3_adapter.py

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from catbench.adsorption import AdsorptionCalculation
from orb_models.forcefield import pretrained
from orb_models.forcefield.inference.calculator import ORBCalculator

from oasis.mlip.registry import load_config

MLIP_NAME = "orb-v3-conservative-inf-omat"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ORB v3 adsorption predictions")
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
        help="Optional local checkpoint. If provided, try to load it; otherwise use ORB's pretrained loader.",
    )
    parser.add_argument("--n-calcs", type=int, default=3)
    args = parser.parse_args()
    config = load_config(args.config)
    optimizer = str(config.get("mlip", {}).get("optimizer", "LBFGS"))

    t0 = time.time()

    # --- Build calculators ---
    calculators = []
    for _ in range(args.n_calcs):
        orbff, atoms_adapter = pretrained.orb_v3_conservative_inf_omat(
            weights_path=args.model_path,  # This arg is ignored by the pretrained loader, but included here for clarity.
            device=args.device,
            precision="float32-high",  # or "float32-highest" / "float64"
        )
        calculators.append(
            ORBCalculator(model=orbff, atoms_adapter=atoms_adapter, device=args.device)
        )

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
        "model": "orb_v3",
        "model_version": "orb-v3-conservative-inf-omat",
        "checkpoint": Path(args.model_path).name if args.model_path else None,
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
