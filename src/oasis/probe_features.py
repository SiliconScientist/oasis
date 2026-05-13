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


def load_mlip_probe_energies(
    mlip_results_dir: Path = _DEFAULT_MLIP_RESULTS_DIR,
) -> dict[str, dict[str, float]]:
    """
    Return:
        {
            "<mlip_name>": {
                "<unique_probe_id>": <ads_eng_median>,
                ...
            },
            ...
        }
    """
    mlip_probe_energies: dict[str, dict[str, float]] = {}

    for result_path in sorted(mlip_results_dir.glob("*_result.json")):
        result_data = _load_json(result_path)
        probe_energies: dict[str, float] = {}

        for probe_key, probe_data in result_data.items():
            if not probe_key.startswith("unique_probe_"):
                continue

            unique_probe_id = probe_key.removeprefix("unique_probe_")
            final_data = probe_data.get("final", {})
            ads_eng_median = final_data.get("ads_eng_median")
            probe_energies[unique_probe_id] = (
                float(ads_eng_median) if ads_eng_median is not None else np.nan
            )

        mlip_probe_energies[_mlip_name_from_result_path(result_path)] = probe_energies

    if not mlip_probe_energies:
        raise FileNotFoundError(f"No MLIP result files found in {mlip_results_dir}")

    return mlip_probe_energies


def build_sample_mlip_feature_matrices(
    dataset_path: Path = _DEFAULT_DATASET_PATH,
    mlip_results_dir: Path = _DEFAULT_MLIP_RESULTS_DIR,
) -> dict[str, SampleMLIPFeatureMatrix]:
    """
    Build one matrix per sample where:
    - rows are MLIPs from `mlip_results_dir`
    - columns are the sample's probe sites in `unique_probe_ids` order
    - values are the MLIP `ads_eng_median` values for those probes
    """
    dataset = _load_json(dataset_path)
    mlip_probe_energies = load_mlip_probe_energies(mlip_results_dir)
    mlip_names = sorted(mlip_probe_energies)

    feature_matrices: dict[str, SampleMLIPFeatureMatrix] = {}

    for reaction, entry in dataset.items():
        unique_probe_ids = list(entry.get("unique_probe_ids", []))
        matrix = np.full((len(mlip_names), len(unique_probe_ids)), np.nan, dtype=float)

        for row_index, mlip_name in enumerate(mlip_names):
            probe_energies = mlip_probe_energies[mlip_name]
            for column_index, unique_probe_id in enumerate(unique_probe_ids):
                matrix[row_index, column_index] = probe_energies.get(
                    unique_probe_id, np.nan
                )

        feature_matrices[reaction] = SampleMLIPFeatureMatrix(
            reaction=reaction,
            mlip_names=mlip_names,
            matrix=matrix,
        )

    return feature_matrices


def build_feature_matrix(
    reaction: str,
    dataset_path: Path = _DEFAULT_DATASET_PATH,
    mlip_results_dir: Path = _DEFAULT_MLIP_RESULTS_DIR,
) -> SampleMLIPFeatureMatrix:
    feature_matrices = build_sample_mlip_feature_matrices(
        dataset_path=dataset_path,
        mlip_results_dir=mlip_results_dir,
    )
    try:
        return feature_matrices[reaction]
    except KeyError as exc:
        raise KeyError(f"Reaction {reaction!r} not found in {dataset_path}") from exc
