from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from oasis.config import get_config
from oasis.mlip.artifacts import (
    find_result_files,
    load_result_json,
    model_name_from_result_path,
)


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

def _resolve_probe_feature_paths(
    dataset_path: Path | None,
    mlip_results_dir: Path | None,
) -> tuple[Path, Path]:
    if dataset_path is not None and mlip_results_dir is not None:
        return dataset_path, mlip_results_dir

    cfg = get_config()
    probe_cfg = cfg.probe_features
    if probe_cfg is None:
        raise ValueError(
            "probe_features.dataset_path and probe_features.mlip_results_dir must "
            "be provided explicitly or configured in experiment.toml"
        )

    resolved_dataset_path = dataset_path or probe_cfg.dataset_path
    resolved_mlip_results_dir = mlip_results_dir or probe_cfg.mlip_results_dir
    return resolved_dataset_path, resolved_mlip_results_dir


def load_mlip_probe_energies(
    mlip_results_dir: Path | None = None,
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
    if mlip_results_dir is None:
        _, resolved_mlip_results_dir = _resolve_probe_feature_paths(None, None)
    else:
        resolved_mlip_results_dir = mlip_results_dir

    for result_path in find_result_files(
        resolved_mlip_results_dir,
        pattern="*_result.json",
    ):
        result_data = load_result_json(result_path)
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

        mlip_probe_energies[model_name_from_result_path(result_path)] = probe_energies

    return mlip_probe_energies


def build_sample_mlip_feature_matrices(
    dataset_path: Path | None = None,
    mlip_results_dir: Path | None = None,
) -> dict[str, SampleMLIPFeatureMatrix]:
    """
    Build one matrix per sample where:
    - rows are MLIPs from `mlip_results_dir`
    - columns are the sample's probe sites in `unique_probe_ids` order
    - values are the MLIP `ads_eng_median` values for those probes
    """
    resolved_dataset_path, resolved_mlip_results_dir = _resolve_probe_feature_paths(
        dataset_path,
        mlip_results_dir,
    )
    dataset = load_result_json(resolved_dataset_path)
    mlip_probe_energies = load_mlip_probe_energies(resolved_mlip_results_dir)
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
    dataset_path: Path | None = None,
    mlip_results_dir: Path | None = None,
) -> SampleMLIPFeatureMatrix:
    feature_matrices = build_sample_mlip_feature_matrices(
        dataset_path=dataset_path,
        mlip_results_dir=mlip_results_dir,
    )
    try:
        return feature_matrices[reaction]
    except KeyError as exc:
        resolved_dataset_path, _ = _resolve_probe_feature_paths(
            dataset_path,
            mlip_results_dir,
        )
        raise KeyError(
            f"Reaction {reaction!r} not found in {resolved_dataset_path}"
        ) from exc


def add_mlip_feature_matrices_to_dataset(
    dataset_path: Path | None = None,
    mlip_results_dir: Path | None = None,
    output_path: Path | None = None,
    field_name: str = "mlip_feature_matrix",
) -> dict[str, object]:
    import json
    resolved_dataset_path, resolved_mlip_results_dir = _resolve_probe_feature_paths(
        dataset_path,
        mlip_results_dir,
    )
    dataset = load_result_json(resolved_dataset_path)
    feature_matrices = build_sample_mlip_feature_matrices(
        dataset_path=resolved_dataset_path,
        mlip_results_dir=resolved_mlip_results_dir,
    )

    for reaction, entry in dataset.items():
        feature_matrix = feature_matrices[reaction]
        entry[field_name] = {
            "mlip_names": feature_matrix.mlip_names,
            "matrix": feature_matrix.matrix.T.tolist(),
        }

    destination = output_path or resolved_dataset_path
    destination.write_text(json.dumps(dataset, indent=2) + "\n")
    return dataset
