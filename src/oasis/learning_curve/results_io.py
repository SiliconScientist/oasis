from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from io import StringIO
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from oasis.sweep import LearningCurveResults


_RESULTS_ARTIFACT_VERSION = 1
_RESULTS_BUNDLE_ARTIFACT_VERSION = 2
_METHOD_RESULTS_ARTIFACT_VERSION = 2
_SUPPORTED_RESULTS_BUNDLE_ARTIFACT_VERSIONS = {1, 2}
_SUPPORTED_METHOD_RESULTS_ARTIFACT_VERSIONS = {1, 2}
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
_RESULT_FIELD_TO_METHOD = {
    result_field: method_name
    for method_name, result_field in _METHOD_RESULT_FIELDS.items()
}
_PROVENANCE_COLUMNS = [
    "n_train",
    "seed",
    "n_repeats",
    "sweep_min_train",
    "sweep_max_train",
    "sweep_step",
    "run_id",
    "run_timestamp_utc",
]


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

    def assert_compatible(
        self,
        other: LearningCurveSweepMetadata,
        *,
        allow_enabled_model_superset: bool = False,
        ignore_enabled_models: bool = False,
        ignore_train_grid: bool = False,
        ignore_repeat_count: bool = False,
    ) -> None:
        this_mapping = self.to_mapping()
        other_mapping = other.to_mapping()
        ignored_keys = set()
        if ignore_enabled_models:
            ignored_keys.add("enabled_models")
        if ignore_train_grid:
            ignored_keys.update({"min_train", "max_train", "step"})
        if ignore_repeat_count:
            ignored_keys.add("n_repeats")
        mismatches = [
            f"{key}: expected {this_mapping[key]!r}, got {other_mapping[key]!r}"
            for key in this_mapping
            if key not in ignored_keys
            if key != "enabled_models" or not allow_enabled_model_superset
            if this_mapping[key] != other_mapping.get(key)
        ]
        if allow_enabled_model_superset and "enabled_models" not in ignored_keys:
            expected_models = set(self.enabled_models)
            actual_models = set(other.enabled_models)
            if not expected_models.issubset(actual_models):
                mismatches.append(
                    "enabled_models: expected subset "
                    f"{sorted(expected_models)!r}, got {sorted(actual_models)!r}"
                )
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
    point_provenance: dict[str, pd.DataFrame] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LearningCurveResultsArtifact:
    metadata: LearningCurveSweepMetadata
    results: LearningCurveResults
    point_provenance: dict[str, pd.DataFrame] = field(default_factory=dict)


def learning_curve_sweep_metadata_from_config(
    cfg: Any,
) -> LearningCurveSweepMetadata:
    from oasis.learning_curve.registry import enabled_learning_curve_model_names_from_config

    experiment_cfg = cfg.experiment.learning_curve if cfg.experiment else None
    if experiment_cfg is None:
        raise ValueError("config does not define experiment.learning_curve.")
    plot_cfg = getattr(cfg, "plot", None)
    plot_filters = getattr(plot_cfg, "filters", None)
    reaction_contains_filter = (
        None
        if plot_filters is None or plot_filters.reaction_contains is None
        else tuple(value for value in plot_filters.reaction_contains if value)
    )
    return LearningCurveSweepMetadata(
        seed=cfg.seed,
        min_train=experiment_cfg.min_train,
        max_train=experiment_cfg.max_train,
        step=getattr(experiment_cfg, "step", 1),
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


def dump_learning_curve_results_artifact(
    results: LearningCurveResults,
    metadata: LearningCurveSweepMetadata,
    *,
    point_provenance: dict[str, pd.DataFrame] | None = None,
    run_id: str | None = None,
    run_timestamp_utc: str | None = None,
) -> dict[str, Any]:
    resolved_point_provenance = _resolve_point_provenance(
        results,
        metadata,
        point_provenance=point_provenance,
        run_id=run_id,
        run_timestamp_utc=run_timestamp_utc,
    )
    return {
        "version": _RESULTS_BUNDLE_ARTIFACT_VERSION,
        "metadata": metadata.to_mapping(),
        "results": dump_learning_curve_results(results),
        "point_provenance": _dump_frame_mapping(resolved_point_provenance),
    }


def load_learning_curve_results_artifact_mapping(
    payload: dict[str, Any],
    *,
    expected_metadata: LearningCurveSweepMetadata | None = None,
    allow_enabled_model_superset: bool = False,
    ignore_enabled_models: bool = False,
    ignore_train_grid: bool = False,
    ignore_repeat_count: bool = False,
) -> LearningCurveResultsArtifact:
    version = payload.get("version")
    if version not in _SUPPORTED_RESULTS_BUNDLE_ARTIFACT_VERSIONS:
        raise ValueError(
            "unsupported learning-curve results bundle artifact version: "
            f"{version!r}."
        )

    metadata_payload = payload.get("metadata")
    if not isinstance(metadata_payload, dict):
        raise TypeError("learning-curve results bundle artifact must contain metadata.")
    metadata = LearningCurveSweepMetadata.from_mapping(metadata_payload)
    if expected_metadata is not None:
        expected_metadata.assert_compatible(
            metadata,
            allow_enabled_model_superset=allow_enabled_model_superset,
            ignore_enabled_models=ignore_enabled_models,
            ignore_train_grid=ignore_train_grid,
            ignore_repeat_count=ignore_repeat_count,
        )

    results_payload = payload.get("results")
    if not isinstance(results_payload, dict):
        raise TypeError("learning-curve results bundle artifact must contain results.")
    results = load_learning_curve_results_mapping(results_payload)
    point_provenance = (
        build_learning_curve_point_provenance(
            results,
            metadata,
            run_id="legacy-artifact",
            run_timestamp_utc="",
        )
        if payload.get("point_provenance") is None
        else _load_point_provenance_mapping(payload.get("point_provenance"))
    )
    return LearningCurveResultsArtifact(
        metadata=metadata,
        results=results,
        point_provenance=point_provenance,
    )


def save_learning_curve_results_artifact(
    results: LearningCurveResults,
    metadata: LearningCurveSweepMetadata,
    path: str | Path,
    *,
    point_provenance: dict[str, pd.DataFrame] | None = None,
    run_id: str | None = None,
    run_timestamp_utc: str | None = None,
) -> Path:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(
        json.dumps(
            dump_learning_curve_results_artifact(
                results,
                metadata,
                point_provenance=point_provenance,
                run_id=run_id,
                run_timestamp_utc=run_timestamp_utc,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    return resolved_path


def load_learning_curve_results_artifact(
    path: str | Path,
    *,
    expected_metadata: LearningCurveSweepMetadata | None = None,
    allow_enabled_model_superset: bool = False,
    ignore_enabled_models: bool = False,
    ignore_train_grid: bool = False,
    ignore_repeat_count: bool = False,
) -> LearningCurveResultsArtifact:
    resolved_path = Path(path)
    return load_learning_curve_results_artifact_mapping(
        json.loads(resolved_path.read_text(encoding="utf-8")),
        expected_metadata=expected_metadata,
        allow_enabled_model_superset=allow_enabled_model_superset,
        ignore_enabled_models=ignore_enabled_models,
        ignore_train_grid=ignore_train_grid,
        ignore_repeat_count=ignore_repeat_count,
    )


def dump_learning_curve_method_artifact(
    method_name: str,
    results: LearningCurveResults,
    metadata: LearningCurveSweepMetadata,
    *,
    point_provenance: dict[str, pd.DataFrame] | None = None,
    run_id: str | None = None,
    run_timestamp_utc: str | None = None,
) -> dict[str, Any]:
    method_results = _extract_method_results(results, method_name)
    metrics_field = _method_metrics_field(method_name)
    if getattr(method_results, metrics_field) is None:
        raise ValueError(
            f"learning-curve results do not contain metrics for method {method_name!r}."
        )
    resolved_point_provenance = _extract_method_point_provenance(
        _resolve_point_provenance(
            results,
            metadata,
            point_provenance=point_provenance,
            run_id=run_id,
            run_timestamp_utc=run_timestamp_utc,
        ),
        method_name,
    )
    return {
        "version": _METHOD_RESULTS_ARTIFACT_VERSION,
        "method_name": method_name,
        "metadata": metadata.to_mapping(),
        "results": dump_learning_curve_results(method_results),
        "point_provenance": _dump_frame_mapping(resolved_point_provenance),
    }


def load_learning_curve_method_artifact_mapping(
    payload: dict[str, Any],
    *,
    expected_metadata: LearningCurveSweepMetadata | None = None,
    allow_enabled_model_superset: bool = False,
) -> LearningCurveMethodArtifact:
    version = payload.get("version")
    if version not in _SUPPORTED_METHOD_RESULTS_ARTIFACT_VERSIONS:
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
        expected_metadata.assert_compatible(
            metadata,
            allow_enabled_model_superset=allow_enabled_model_superset,
        )

    results_payload = payload.get("results")
    if not isinstance(results_payload, dict):
        raise TypeError("learning-curve method artifact must contain results.")
    results = load_learning_curve_results_mapping(results_payload)
    point_provenance = (
        build_learning_curve_point_provenance(
            results,
            metadata,
            run_id="legacy-artifact",
            run_timestamp_utc="",
        )
        if payload.get("point_provenance") is None
        else _load_point_provenance_mapping(payload.get("point_provenance"))
    )
    return LearningCurveMethodArtifact(
        method_name=method_name,
        metadata=metadata,
        results=results,
        point_provenance=point_provenance,
    )


def save_learning_curve_method_artifact(
    method_name: str,
    results: LearningCurveResults,
    metadata: LearningCurveSweepMetadata,
    path: str | Path,
    *,
    point_provenance: dict[str, pd.DataFrame] | None = None,
    run_id: str | None = None,
    run_timestamp_utc: str | None = None,
) -> Path:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(
        json.dumps(
            dump_learning_curve_method_artifact(
                method_name,
                results,
                metadata,
                point_provenance=point_provenance,
                run_id=run_id,
                run_timestamp_utc=run_timestamp_utc,
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
    allow_enabled_model_superset: bool = False,
) -> LearningCurveMethodArtifact:
    resolved_path = Path(path)
    return load_learning_curve_method_artifact_mapping(
        json.loads(resolved_path.read_text(encoding="utf-8")),
        expected_metadata=expected_metadata,
        allow_enabled_model_superset=allow_enabled_model_superset,
    )


def save_learning_curve_method_artifacts(
    results: LearningCurveResults,
    metadata: LearningCurveSweepMetadata,
    directory: str | Path,
    *,
    point_provenance: dict[str, pd.DataFrame] | None = None,
    run_id: str | None = None,
    run_timestamp_utc: str | None = None,
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
            point_provenance=point_provenance,
            run_id=run_id,
            run_timestamp_utc=run_timestamp_utc,
        )
    return saved_paths


def load_learning_curve_method_artifacts(
    directory: str | Path,
    *,
    expected_metadata: LearningCurveSweepMetadata | None = None,
    method_names: tuple[str, ...] | None = None,
    allow_enabled_model_superset: bool = False,
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
                allow_enabled_model_superset=allow_enabled_model_superset,
            )
        )
    return tuple(artifacts)


def load_learning_curve_results_from_method_artifacts(
    directory: str | Path,
    *,
    expected_metadata: LearningCurveSweepMetadata | None = None,
    method_names: tuple[str, ...] | None = None,
    allow_enabled_model_superset: bool = False,
) -> LearningCurveResults:
    results = LearningCurveResults.empty()
    for artifact in load_learning_curve_method_artifacts(
        directory,
        expected_metadata=expected_metadata,
        method_names=method_names,
        allow_enabled_model_superset=allow_enabled_model_superset,
    ):
        results = results.merge(artifact.results)
    return results


def learning_curve_method_name_for_result_field(result_field: str) -> str | None:
    return _RESULT_FIELD_TO_METHOD.get(result_field)


def learning_curve_result_field_for_method_name(method_name: str) -> str | None:
    return _METHOD_RESULT_FIELDS.get(method_name)


def learning_curve_selection_field_for_method_name(method_name: str) -> str | None:
    return _METHOD_SELECTION_FIELDS.get(method_name)


def learning_curve_method_names() -> tuple[str, ...]:
    return tuple(_METHOD_RESULT_FIELDS)


def learning_curve_results_has_method(
    results: LearningCurveResults,
    method_name: str,
) -> bool:
    result_field = learning_curve_result_field_for_method_name(method_name)
    return result_field is not None and getattr(results, result_field) is not None


def select_learning_curve_results_methods(
    results: LearningCurveResults,
    method_names: tuple[str, ...],
) -> LearningCurveResults:
    selected = LearningCurveResults.empty()
    for method_name in method_names:
        if learning_curve_results_has_method(results, method_name):
            selected = selected.merge(_extract_method_results(results, method_name))
    return selected


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


def build_learning_curve_point_provenance(
    results: LearningCurveResults,
    metadata: LearningCurveSweepMetadata,
    *,
    run_id: str | None = None,
    run_timestamp_utc: str | None = None,
) -> dict[str, pd.DataFrame]:
    resolved_run_id = run_id or uuid4().hex
    resolved_timestamp = run_timestamp_utc or datetime.now(timezone.utc).isoformat()
    point_provenance: dict[str, pd.DataFrame] = {}
    for field_name, frame in results.to_mapping().items():
        if frame is None:
            continue
        if "n_train" not in frame.columns:
            raise ValueError("learning-curve result frames must contain an n_train column.")
        point_provenance[field_name] = pd.DataFrame(
            {
                "n_train": frame["n_train"].tolist(),
                "seed": [metadata.seed] * len(frame),
                "n_repeats": [metadata.n_repeats] * len(frame),
                "sweep_min_train": [metadata.min_train] * len(frame),
                "sweep_max_train": [metadata.max_train] * len(frame),
                "sweep_step": [metadata.step] * len(frame),
                "run_id": [resolved_run_id] * len(frame),
                "run_timestamp_utc": [resolved_timestamp] * len(frame),
            }
        )
    return point_provenance


def merge_learning_curve_point_provenance(
    left: dict[str, pd.DataFrame],
    right: dict[str, pd.DataFrame],
    *,
    overwrite_fields: set[str] | frozenset[str] = frozenset(),
) -> dict[str, pd.DataFrame]:
    merged: dict[str, pd.DataFrame] = {}
    for field_name in set(left) | set(right):
        merged_frame = _merge_provenance_frame(
            field_name,
            left.get(field_name),
            right.get(field_name),
            allow_overlap=field_name in overwrite_fields,
        )
        if merged_frame is not None:
            merged[field_name] = merged_frame
    return merged


def _resolve_point_provenance(
    results: LearningCurveResults,
    metadata: LearningCurveSweepMetadata,
    *,
    point_provenance: dict[str, pd.DataFrame] | None,
    run_id: str | None,
    run_timestamp_utc: str | None,
) -> dict[str, pd.DataFrame]:
    return (
        build_learning_curve_point_provenance(
            results,
            metadata,
            run_id=run_id,
            run_timestamp_utc=run_timestamp_utc,
        )
        if point_provenance is None
        else _normalize_point_provenance(point_provenance)
    )


def _normalize_point_provenance(
    point_provenance: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    return {
        field_name: _normalize_provenance_frame(frame)
        for field_name, frame in point_provenance.items()
    }


def _dump_frame_mapping(frames: dict[str, pd.DataFrame]) -> dict[str, str]:
    return {
        field_name: frame.to_json(orient="table")
        for field_name, frame in frames.items()
    }


def _load_point_provenance_mapping(
    payload: Any,
) -> dict[str, pd.DataFrame]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError("learning-curve point_provenance must be a mapping.")
    frames: dict[str, pd.DataFrame] = {}
    for field_name, frame_payload in payload.items():
        if not isinstance(frame_payload, str):
            raise TypeError(
                "learning-curve provenance frame payloads must be JSON strings."
            )
        frames[field_name] = _normalize_provenance_frame(
            pd.read_json(StringIO(frame_payload), orient="table")
        )
    return frames


def _normalize_provenance_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "n_train" not in frame.columns:
        raise ValueError("learning-curve provenance frames must contain an n_train column.")
    missing_columns = [column for column in _PROVENANCE_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            "learning-curve provenance frames are missing required columns: "
            f"{missing_columns!r}."
        )
    return frame.loc[:, _PROVENANCE_COLUMNS].sort_values("n_train").reset_index(drop=True)


def _merge_provenance_frame(
    field_name: str,
    left: pd.DataFrame | None,
    right: pd.DataFrame | None,
    *,
    allow_overlap: bool,
) -> pd.DataFrame | None:
    if left is None:
        return right
    if right is None:
        return left
    overlapping_train_sizes = sorted(
        set(left["n_train"].tolist()).intersection(right["n_train"].tolist())
    )
    if overlapping_train_sizes and not allow_overlap:
        raise ValueError(
            f"{field_name} contains duplicate provenance rows for n_train: "
            f"{overlapping_train_sizes!r}."
        )
    merged = pd.concat([left, right], ignore_index=True)
    merged = merged.drop_duplicates(subset=["n_train"], keep="last")
    return merged.sort_values("n_train").reset_index(drop=True)


def _extract_method_point_provenance(
    point_provenance: dict[str, pd.DataFrame],
    method_name: str,
) -> dict[str, pd.DataFrame]:
    selected: dict[str, pd.DataFrame] = {}
    metrics_field = _METHOD_RESULT_FIELDS[method_name]
    metrics_provenance = point_provenance.get(metrics_field)
    if metrics_provenance is not None:
        selected[metrics_field] = metrics_provenance
    selection_field = _METHOD_SELECTION_FIELDS.get(method_name)
    if selection_field is not None and selection_field in point_provenance:
        selected[selection_field] = point_provenance[selection_field]
    return selected
