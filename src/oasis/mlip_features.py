from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = (
    REPO_ROOT / "data/raw_data/KHLOHC_origin_tolstar_adsorption_with_probe_ids.json"
)
DEFAULT_MLIP_RESULTS_DIR = REPO_ROOT / "data/mlips/khlohc_toluene_unique_probes"


@dataclass(frozen=True)
class SampleMLIPFeatureMatrix:
    reaction: str
    mlip_names: list[str]
    unique_probe_ids: list[str]
    matrix: np.ndarray

    def to_dict(self) -> dict[str, object]:
        return {
            "reaction": self.reaction,
            "mlip_names": self.mlip_names,
            "unique_probe_ids": self.unique_probe_ids,
            "matrix": self.matrix.tolist(),
        }


def _load_json(path: Path) -> dict[str, object]:
    with path.open() as handle:
        return json.load(handle)


def _mlip_name_from_result_path(result_path: Path) -> str:
    return result_path.name.removesuffix("_result.json")


def load_mlip_probe_energies(
    mlip_results_dir: Path = DEFAULT_MLIP_RESULTS_DIR,
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
    dataset_path: Path = DEFAULT_DATASET_PATH,
    mlip_results_dir: Path = DEFAULT_MLIP_RESULTS_DIR,
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
            unique_probe_ids=unique_probe_ids,
            matrix=matrix,
        )

    return feature_matrices


def build_feature_matrix(
    reaction: str,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    mlip_results_dir: Path = DEFAULT_MLIP_RESULTS_DIR,
) -> SampleMLIPFeatureMatrix:
    feature_matrices = build_sample_mlip_feature_matrices(
        dataset_path=dataset_path,
        mlip_results_dir=mlip_results_dir,
    )
    try:
        return feature_matrices[reaction]
    except KeyError as exc:
        raise KeyError(f"Reaction {reaction!r} not found in {dataset_path}") from exc


def add_mlip_feature_matrices_to_dataset(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    mlip_results_dir: Path = DEFAULT_MLIP_RESULTS_DIR,
    output_path: Path | None = None,
    field_name: str = "mlip_feature_matrix",
) -> dict[str, object]:
    dataset = _load_json(dataset_path)
    feature_matrices = build_sample_mlip_feature_matrices(
        dataset_path=dataset_path,
        mlip_results_dir=mlip_results_dir,
    )

    for reaction, entry in dataset.items():
        feature_matrix = feature_matrices[reaction]
        entry[field_name] = {
            "mlip_names": feature_matrix.mlip_names,
            "unique_probe_ids": feature_matrix.unique_probe_ids,
            "matrix": feature_matrix.matrix.tolist(),
        }

    destination = output_path or dataset_path
    destination.write_text(json.dumps(dataset, indent=2) + "\n")
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
    )
    parser.add_argument(
        "--mlip-results-dir",
        type=Path,
        default=DEFAULT_MLIP_RESULTS_DIR,
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--write-dataset",
        action="store_true",
        help="Write MLIP feature matrices into the dataset entries.",
    )
    args = parser.parse_args()

    if args.write_dataset:
        updated_dataset = add_mlip_feature_matrices_to_dataset(
            dataset_path=args.dataset_path,
            mlip_results_dir=args.mlip_results_dir,
            output_path=args.output,
        )
        first_reaction = next(iter(updated_dataset))
        first_entry = updated_dataset[first_reaction]
        print(
            json.dumps(
                {
                    "dataset_path": str(args.output or args.dataset_path),
                    "samples": len(updated_dataset),
                    "example_reaction": first_reaction,
                    "example_shape": [
                        len(first_entry["mlip_feature_matrix"]["mlip_names"]),
                        len(first_entry["mlip_feature_matrix"]["unique_probe_ids"]),
                    ],
                },
                indent=2,
            )
        )
        return

    feature_matrices = build_sample_mlip_feature_matrices(
        dataset_path=args.dataset_path,
        mlip_results_dir=args.mlip_results_dir,
    )

    if args.output is not None:
        serializable = {
            reaction: feature_matrix.to_dict()
            for reaction, feature_matrix in feature_matrices.items()
        }
        args.output.write_text(json.dumps(serializable, indent=2) + "\n")
        print(args.output)
        return

    first_reaction = next(iter(feature_matrices))
    feature_matrix = feature_matrices[first_reaction]
    print(
        json.dumps(
            {
                "samples": len(feature_matrices),
                "mlips": feature_matrix.mlip_names,
                "example_reaction": first_reaction,
                "example_shape": list(feature_matrix.matrix.shape),
                "example_unique_probe_ids": feature_matrix.unique_probe_ids,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
