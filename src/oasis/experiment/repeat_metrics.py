from __future__ import annotations

from dataclasses import dataclass
import json
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from oasis.learning_curve.results_io import LearningCurveSweepMetadata


_LEARNING_CURVE_REPEAT_METRICS_ARTIFACT_VERSION = 1
_REPEAT_METRICS_COLUMNS = ["method", "budget", "repeat", "outer_test_rmse"]


@dataclass(frozen=True, slots=True)
class LearningCurveRepeatMetricsArtifact:
    metadata: LearningCurveSweepMetadata
    repeat_metrics_df: pd.DataFrame

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "repeat_metrics_df",
            normalize_learning_curve_repeat_metrics_frame(self.repeat_metrics_df),
        )


def normalize_learning_curve_repeat_metrics_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [
        column for column in _REPEAT_METRICS_COLUMNS if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(
            "learning-curve repeat metrics frame is missing required columns: "
            f"{missing_columns!r}."
        )
    normalized = frame.copy()
    normalized["method"] = normalized["method"].astype("string")
    normalized["budget"] = pd.Series(normalized["budget"], dtype="Int64")
    normalized["repeat"] = pd.Series(normalized["repeat"], dtype="Int64")
    normalized["outer_test_rmse"] = pd.Series(normalized["outer_test_rmse"], dtype="Float64")
    return normalized.loc[:, _REPEAT_METRICS_COLUMNS].sort_values(
        ["method", "budget", "repeat"]
    ).reset_index(drop=True)


def merge_learning_curve_repeat_metrics(
    left: pd.DataFrame | None,
    right: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if left is None:
        return None if right is None else normalize_learning_curve_repeat_metrics_frame(right)
    if right is None:
        return normalize_learning_curve_repeat_metrics_frame(left)
    merged = pd.concat([left, right], ignore_index=True)
    merged = merged.drop_duplicates(subset=["method", "budget", "repeat"], keep="last")
    return normalize_learning_curve_repeat_metrics_frame(merged)


def repeat_metrics_artifact_path(results_bundle_path: str | Path) -> Path:
    bundle_path = Path(results_bundle_path)
    return bundle_path.with_name(f"{bundle_path.stem}_repeat_metrics{bundle_path.suffix}")


def dump_learning_curve_repeat_metrics_artifact(
    artifact: LearningCurveRepeatMetricsArtifact,
) -> dict[str, Any]:
    return {
        "version": _LEARNING_CURVE_REPEAT_METRICS_ARTIFACT_VERSION,
        "metadata": artifact.metadata.to_bundle_mapping(),
        "repeat_metrics_df": artifact.repeat_metrics_df.to_json(orient="table"),
    }


def load_learning_curve_repeat_metrics_artifact_mapping(
    payload: dict[str, Any],
    *,
    expected_metadata: LearningCurveSweepMetadata | None = None,
) -> LearningCurveRepeatMetricsArtifact:
    version = payload.get("version")
    if version != _LEARNING_CURVE_REPEAT_METRICS_ARTIFACT_VERSION:
        raise ValueError(
            "unsupported learning-curve repeat metrics artifact version: "
            f"{version!r}."
        )
    metadata_payload = payload.get("metadata")
    if not isinstance(metadata_payload, dict):
        raise TypeError("learning-curve repeat metrics artifact must contain metadata.")
    metadata = LearningCurveSweepMetadata.from_bundle_mapping(
        metadata_payload,
        fallback=expected_metadata,
    )
    if expected_metadata is not None:
        expected_metadata.assert_compatible(
            metadata,
            ignore_enabled_models=True,
            ignore_train_grid=True,
            ignore_repeat_count=True,
        )
    repeat_metrics_payload = payload.get("repeat_metrics_df")
    if not isinstance(repeat_metrics_payload, str):
        raise TypeError(
            "learning-curve repeat metrics artifact must contain a repeat_metrics_df JSON string."
        )
    return LearningCurveRepeatMetricsArtifact(
        metadata=metadata,
        repeat_metrics_df=pd.read_json(StringIO(repeat_metrics_payload), orient="table"),
    )


def save_learning_curve_repeat_metrics_artifact(
    artifact: LearningCurveRepeatMetricsArtifact,
    path: str | Path,
) -> Path:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(
        json.dumps(dump_learning_curve_repeat_metrics_artifact(artifact), indent=2),
        encoding="utf-8",
    )
    return resolved_path


def load_learning_curve_repeat_metrics_artifact(
    path: str | Path,
    *,
    expected_metadata: LearningCurveSweepMetadata | None = None,
) -> LearningCurveRepeatMetricsArtifact:
    resolved_path = Path(path)
    return load_learning_curve_repeat_metrics_artifact_mapping(
        json.loads(resolved_path.read_text(encoding="utf-8")),
        expected_metadata=expected_metadata,
    )
