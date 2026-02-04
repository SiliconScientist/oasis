# src/oasis/mlip/schema.py

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Prediction:
    id: str
    energy: float
    units: str = "eV"
    # optional extras (forces, uncertainty, etc.)
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class Result:
    model: str
    dataset: str
    n_structures: int
    predictions: list[Prediction]

    model_version: str | None = None
    weights: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    created_unix_s: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Convert Prediction dataclasses
        d["predictions"] = [asdict(p) for p in self.predictions]
        return d


def atomic_write_json(path: str | Path, obj: dict[str, Any], indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, sort_keys=False)
        f.write("\n")
    tmp.replace(path)
