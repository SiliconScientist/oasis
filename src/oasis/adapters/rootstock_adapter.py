# src/oasis/adapters/rootstock_adapter.py

from __future__ import annotations

import argparse
import json
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from catbench.adsorption import AdsorptionCalculation

from oasis.mlip.registry import load_config


def _resolve_checkpoint(checkpoint: str | None, config_path: Path) -> str | None:
    if checkpoint is None:
        return None

    checkpoint_path = Path(checkpoint)
    if (
        not checkpoint_path.is_absolute()
        and any(sep in checkpoint for sep in ("/", "\\"))
    ):
        return str((config_path.parent / checkpoint_path).resolve())
    return checkpoint


def _checkpoint_label(checkpoint: str | None) -> str | None:
    if checkpoint is None:
        return None

    if any(sep in checkpoint for sep in ("/", "\\")):
        return Path(checkpoint).name
    return checkpoint


def _get_rootstock_spec(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    rootstock = config.get("mlip", {}).get("rootstock", {})
    models = rootstock.get("models", {})
    if model_name not in models:
        known = ", ".join(sorted(str(name) for name in models))
        raise KeyError(
            f"No Rootstock config found for model '{model_name}'. "
            f"Known Rootstock models: {known}"
        )

    spec = dict(models[model_name])
    if "model" not in spec:
        raise KeyError(f"mlip.rootstock.models.{model_name}.model is required")
    if "mlip_name" not in spec:
        raise KeyError(f"mlip.rootstock.models.{model_name}.mlip_name is required")
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Rootstock adsorption predictions")
    parser.add_argument("--model", required=True, help="Configured MLIP model key")
    parser.add_argument("--input", required=True, help="Input dataset JSON")
    parser.add_argument("--output", required=True, help="Output result JSON")
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Dataset name (passed from task runner)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--n-calcs", type=int, default=3)
    args = parser.parse_args()

    from rootstock import RootstockCalculator

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    optimizer = str(config.get("mlip", {}).get("optimizer", "LBFGS"))
    rootstock_cfg = config.get("mlip", {}).get("rootstock", {})
    root = rootstock_cfg.get("root", "/projects/bchg/rootstock")
    spec = _get_rootstock_spec(config, args.model)
    configured_checkpoint = spec.get("checkpoint")
    checkpoint = _resolve_checkpoint(configured_checkpoint, config_path)
    metadata = dict(spec.get("metadata", {}))

    t0 = time.time()

    # --- Build calculators ---
    with ExitStack() as stack:
        calculators = [
            stack.enter_context(
                RootstockCalculator(
                    root=root,
                    model=str(spec["model"]),
                    checkpoint=checkpoint,
                    device=args.device,
                )
            )
            for _ in range(args.n_calcs)
        ]

        # --- Run CatBench adsorption workflow ---
        adsorption_calc = AdsorptionCalculation(
            calculators,
            mlip_name=str(spec["mlip_name"]),
            benchmark=args.dataset_name,
            optimizer=optimizer,
        )
        results = adsorption_calc.run()

    # --- Standardized output ---
    out = {
        "model": spec.get("output_model", args.model),
        "model_version": spec.get("model_version", spec["mlip_name"]),
        "rootstock_model": spec["model"],
        "checkpoint": _checkpoint_label(configured_checkpoint),
        "n_calculators": args.n_calcs,
        "device": args.device,
        "optimizer": optimizer,
        "dataset_name": args.dataset_name,
        "input_dataset": Path(args.input).name,
        "wall_time_s": time.time() - t0,
        "results": results,
    }
    out.update(metadata)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
