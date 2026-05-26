from __future__ import annotations

from dataclasses import dataclass
import json
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from oasis.sweep import LearningCurveResults


_RESULTS_ARTIFACT_VERSION = 1
_METHOD_RESULTS_ARTIFACT_VERSION = 1
_METHOD_RESULT_FIELDS = {
    "ridge": "ridge_df",
    "kernel_ridge": "kernel_ridge_df",
    "lasso": "lasso_df",
    "elastic": "elastic_df",
    "residual": "resid_df",
    "weighted_linear": "weighted_linear_df",
    "weighted_simplex": "weighted_simplex_df",
    "graph_mean": "graph_mean_df",
    "moe": "moe_df",
    "probe_gnn": "probe_gnn_df",
    "gnn_direct": "gnn_direct_df",
    "latent": "latent_df",
}
_METHOD_SELECTION_FIELDS = {
    "ridge": "ridge_selection_df",
    "kernel_ridge": "kernel_ridge_selection_df",
    "lasso": "lasso_selection_df",
    "elastic": "elastic_selection_df",
    "graph_mean": "graph_mean_selection_df",
    "moe": "moe_selection_df",
    "probe_gnn": "probe_gnn_selection_df",
    "gnn_direct": "gnn_direct_selection_df",
}


@dataclass(frozen=True, slots=True)
class LearningCurveSweepMetadata:
    seed: int | None
    min_train: int
    max_train: int
    step: int
    n_repeats: int
    enabled_models: tuple[str, ...]
    adsorbate_filter: str | None = None
    anomaly_filter: str | None = None
    reaction_contains_filter: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "enabled_models",
            tuple(sorted(dict.fromkeys(self.enabled_models))),
        )
        reaction_contains_filter = self.reaction_contains_filter
        if reaction_contains_filter is None:
            return
        normalized = tuple(value for value in reaction_contains_filter if value)
        object.__setattr__(
            self,
            "reaction_contains_filter",
            normalized or None,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "min_train": self.min_train,
            "max_train": self.max_train,
            "step": self.step,
            "n_repeats": self.n_repeats,
            "enabled_models": list(self.enabled_models),
            "adsorbate_filter": self.adsorbate_filter,
            "anomaly_filter": self.anomaly_filter,
            "reaction_contains_filter": (
                None
                if self.reaction_contains_filter is None
                else list(self.reaction_contains_filter)
            ),
        }

    @classmethod
    def from_mapping(
        cls,
        payload: dict[str, Any],
    ) -> LearningCurveSweepMetadata:
        return cls(
            seed=payload.get("seed"),
            min_train=int(payload["min_train"]),
            max_train=int(payload["max_train"]),
            step=int(payload["step"]),
            n_repeats=int(payload["n_repeats"]),
            enabled_models=tuple(payload.get("enabled_models", ())),
            adsorbate_filter=payload.get("adsorbate_filter"),
            anomaly_filter=payload.get("anomaly_filter"),
            reaction_contains_filter=(
                None
                if payload.get("reaction_contains_filter") is None
                else tuple(payload["reaction_contains_filter"])
            ),
        )

    def assert_compatible(self, other: LearningCurveSweepMetadata) -> None:
        this_mapping = self.to_mapping()
        other_mapping = other.to_mapping()
        mismatches = [
            f"{key}: expected {this_mapping[key]!r}, got {other_mapping[key]!r}"
            for key in this_mapping
            if this_mapping[key] != other_mapping.get(key)
        ]
        if mismatches:
            raise ValueError(
                "learning-curve sweep metadata is incompatible: "
                + "; ".join(mismatches)
            )


@dataclass(frozen=True, slots=True)
class LearningCurveMethodArtifact:
    method_name: str
    metadata: LearningCurveSweepMetadata
    results: LearningCurveResults


def learning_curve_sweep_metadata_from_config(
    cfg: Any,
) -> LearningCurveSweepMetadata:
    from oasis.learning_curve.registry import enabled_learning_curve_model_names_from_config

    experiment_cfg = cfg.experiment.learning_curve if cfg.experiment else None
    if experiment_cfg is None:
        raise ValueError("config does not define experiment.learning_curve.")
    plot_filters = cfg.plot.filters if cfg.plot else None
    reaction_contains_filter = (
        None
        if plot_filters is None or plot_filters.reaction_contains is None
        else tuple(value for value in plot_filters.reaction_contains if value)
    )
    return LearningCurveSweepMetadata(
        seed=cfg.seed,
        min_train=experiment_cfg.min_train,
        max_train=experiment_cfg.max_train,
        step=experiment_cfg.step,
        n_repeats=experiment_cfg.n_repeats,
        enabled_models=enabled_learning_curve_model_names_from_config(
            experiment_cfg.models
        ),
        adsorbate_filter=plot_filters.adsorbate if plot_filters else None,
        anomaly_filter=plot_filters.anomaly_label if plot_filters else None,
        reaction_contains_filter=reaction_contains_filter,
    )


def dump_learning_curve_results(
    results: LearningCurveResults,
) -> dict[str, Any]:
    return {
        "version": _RESULTS_ARTIFACT_VERSION,
        "frames": {
            field_name: None if frame is None else frame.to_json(orient="table")
            for field_name, frame in results.to_mapping().items()
        },
    }


def load_learning_curve_results_mapping(
    payload: dict[str, Any],
) -> LearningCurveResults:
    version = payload.get("version")
    if version != _RESULTS_ARTIFACT_VERSION:
        raise ValueError(
            "unsupported learning-curve results artifact version: "
            f"{version!r}."
        )

    frames_payload = payload.get("frames")
    if not isinstance(frames_payload, dict):
        raise TypeError("learning-curve results artifact must contain a frames mapping.")

    frames: dict[str, pd.DataFrame | None] = {}
    for field_name, frame_payload in frames_payload.items():
        if frame_payload is None:
            frames[field_name] = None
            continue
        if not isinstance(frame_payload, str):
            raise TypeError(
                "learning-curve frame payloads must be JSON strings or null."
            )
        frames[field_name] = pd.read_json(StringIO(frame_payload), orient="table")
    return LearningCurveResults.from_mapping(frames)


def save_learning_curve_results(
    results: LearningCurveResults,
    path: str | Path,
) -> Path:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(
        json.dumps(dump_learning_curve_results(results), indent=2),
        encoding="utf-8",
    )
    return resolved_path


def load_learning_curve_results(
    path: str | Path,
) -> LearningCurveResults:
    resolved_path = Path(path)
    return load_learning_curve_results_mapping(
        json.loads(resolved_path.read_text(encoding="utf-8"))
    )


def dump_learning_curve_method_artifact(
    method_name: str,
    results: LearningCurveResults,
    metadata: LearningCurveSweepMetadata,
) -> dict[str, Any]:
    method_results = _extract_method_results(results, method_name)
    metrics_field = _method_metrics_field(method_name)
    if getattr(method_results, metrics_field) is None:
        raise ValueError(
            f"learning-curve results do not contain metrics for method {method_name!r}."
        )
    return {
        "version": _METHOD_RESULTS_ARTIFACT_VERSION,
        "method_name": method_name,
        "metadata": metadata.to_mapping(),
        "results": dump_learning_curve_results(method_results),
    }


def load_learning_curve_method_artifact_mapping(
    payload: dict[str, Any],
    *,
    expected_metadata: LearningCurveSweepMetadata | None = None,
) -> LearningCurveMethodArtifact:
    version = payload.get("version")
    if version != _METHOD_RESULTS_ARTIFACT_VERSION:
        raise ValueError(
            "unsupported learning-curve method artifact version: "
            f"{version!r}."
        )

    method_name = payload.get("method_name")
    if not isinstance(method_name, str):
        raise TypeError("learning-curve method artifact must define method_name.")
    if method_name not in _METHOD_RESULT_FIELDS:
        raise ValueError(f"unknown learning-curve method artifact {method_name!r}.")

    metadata_payload = payload.get("metadata")
    if not isinstance(metadata_payload, dict):
        raise TypeError("learning-curve method artifact must contain metadata.")
    metadata = LearningCurveSweepMetadata.from_mapping(metadata_payload)
    if expected_metadata is not None:
        expected_metadata.assert_compatible(metadata)

    results_payload = payload.get("results")
    if not isinstance(results_payload, dict):
        raise TypeError("learning-curve method artifact must contain results.")
    results = load_learning_curve_results_mapping(results_payload)
    return LearningCurveMethodArtifact(
        method_name=method_name,
        metadata=metadata,
        results=results,
    )


def save_learning_curve_method_artifact(
    method_name: str,
    results: LearningCurveResults,
    metadata: LearningCurveSweepMetadata,
    path: str | Path,
) -> Path:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(
        json.dumps(
            dump_learning_curve_method_artifact(
                method_name,
                results,
                metadata,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    return resolved_path


def load_learning_curve_method_artifact(
    path: str | Path,
    *,
    expected_metadata: LearningCurveSweepMetadata | None = None,
) -> LearningCurveMethodArtifact:
    resolved_path = Path(path)
    return load_learning_curve_method_artifact_mapping(
        json.loads(resolved_path.read_text(encoding="utf-8")),
        expected_metadata=expected_metadata,
    )


def save_learning_curve_method_artifacts(
    results: LearningCurveResults,
    metadata: LearningCurveSweepMetadata,
    directory: str | Path,
) -> dict[str, Path]:
    resolved_dir = Path(directory)
    resolved_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: dict[str, Path] = {}
    for method_name, metrics_field in _METHOD_RESULT_FIELDS.items():
        if getattr(results, metrics_field) is None:
            continue
        saved_paths[method_name] = save_learning_curve_method_artifact(
            method_name,
            results,
            metadata,
            resolved_dir / f"{method_name}.json",
        )
    return saved_paths


def load_learning_curve_method_artifacts(
    directory: str | Path,
    *,
    expected_metadata: LearningCurveSweepMetadata | None = None,
    method_names: tuple[str, ...] | None = None,
) -> tuple[LearningCurveMethodArtifact, ...]:
    resolved_dir = Path(directory)
    selected_method_names = (
        tuple(_METHOD_RESULT_FIELDS)
        if method_names is None
        else method_names
    )
    artifacts: list[LearningCurveMethodArtifact] = []
    for method_name in selected_method_names:
        path = resolved_dir / f"{method_name}.json"
        if not path.is_file():
            continue
        artifacts.append(
            load_learning_curve_method_artifact(
                path,
                expected_metadata=expected_metadata,
            )
        )
    return tuple(artifacts)


def load_learning_curve_results_from_method_artifacts(
    directory: str | Path,
    *,
    expected_metadata: LearningCurveSweepMetadata | None = None,
    method_names: tuple[str, ...] | None = None,
) -> LearningCurveResults:
    results = LearningCurveResults.empty()
    for artifact in load_learning_curve_method_artifacts(
        directory,
        expected_metadata=expected_metadata,
        method_names=method_names,
    ):
        results = results.merge(artifact.results)
    return results


def _method_metrics_field(method_name: str) -> str:
    try:
        return _METHOD_RESULT_FIELDS[method_name]
    except KeyError as exc:
        raise ValueError(f"unknown learning-curve method {method_name!r}.") from exc


def _extract_method_results(
    results: LearningCurveResults,
    method_name: str,
) -> LearningCurveResults:
    metrics_field = _method_metrics_field(method_name)
    field_mapping: dict[str, pd.DataFrame | None] = {
        field_name: None for field_name in results.to_mapping()
    }
    field_mapping[metrics_field] = getattr(results, metrics_field)
    selection_field = _METHOD_SELECTION_FIELDS.get(method_name)
    if selection_field is not None:
        field_mapping[selection_field] = getattr(results, selection_field)
    return LearningCurveResults.from_mapping(field_mapping)
