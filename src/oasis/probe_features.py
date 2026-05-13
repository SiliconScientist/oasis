from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATASET_PATH = (
    _REPO_ROOT / "data/raw_data/KHLOHC_origin_tolstar_adsorption_with_probe_ids.json"
)
_DEFAULT_MLIP_RESULTS_DIR = _REPO_ROOT / "data/mlips/khlohc_toluene_unique_probes"


@dataclass(frozen=True)
class SampleMLIPFeatureMatrix:
    reaction: str
    mlip_names: list[str]
    matrix: np.ndarray

    def to_dict(self) -> dict[str, object]:
        return {
            "reaction": self.reaction,
            "mlip_names": self.mlip_names,
            "matrix": self.matrix.tolist(),
        }


def _load_json(path: Path) -> dict[str, object]:
    import json
    with path.open() as handle:
        return json.load(handle)


def _mlip_name_from_result_path(result_path: Path) -> str:
    return result_path.name.removesuffix("_result.json")
