from __future__ import annotations

import json
from pathlib import Path


def generation_timing_sidecar_path(
    csv_path: str | Path,
) -> Path:
    path = Path(csv_path)
    return path.with_suffix(path.suffix + ".timing.json")


def write_generation_timing_sidecar(
    csv_path: str | Path,
    *,
    generation_time_s: float,
) -> Path:
    path = generation_timing_sidecar_path(csv_path)
    payload = {
        "generation_time_s": float(generation_time_s),
        "output_csv_path": str(Path(csv_path)),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
