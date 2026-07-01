from __future__ import annotations

from dataclasses import dataclass
import json
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from oasis.learning_curve.results_io import LearningCurveSweepMetadata


_POLICY_DIAGNOSTIC_ARTIFACT_VERSION = 1
_DETAIL_COLUMNS = [
    "budget",
    "repeat",
    "oracle_method",
    "screening_selected_method",
    "oracle_outer_rmse",
    "screening_selected_outer_rmse",
    "regret",
    "screening_cv_rmse",
    "agreement",
]
_SUMMARY_COLUMNS = [
    "budget",
    "mean_regret",
    "std_regret",
    "se_regret",
    "ci95_low",
    "ci95_high",
    "agreement_rate",
    "oracle_outer_rmse_mean",
    "screening_selected_outer_rmse_mean",
]


@dataclass(frozen=True, slots=True)
class PolicySelectionDiagnosticResults:
    detail_df: pd.DataFrame
    summary_df: pd.DataFrame

    def __post_init__(self) -> None:
        object.__setattr__(self, "detail_df", normalize_policy_detail_frame(self.detail_df))
        object.__setattr__(
            self,
            "summary_df",
            normalize_policy_summary_frame(self.summary_df),
        )


@dataclass(frozen=True, slots=True)
class PolicySelectionDiagnosticArtifact:
    metadata: LearningCurveSweepMetadata
    results: PolicySelectionDiagnosticResults


def normalize_policy_detail_frame(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, _DETAIL_COLUMNS, frame_name="policy detail")
    normalized = frame.copy()
    normalized["budget"] = pd.Series(normalized["budget"], dtype="Int64")
    normalized["repeat"] = pd.Series(normalized["repeat"], dtype="Int64")
    normalized["oracle_method"] = normalized["oracle_method"].astype("string")
    normalized["screening_selected_method"] = normalized["screening_selected_method"].astype(
        "string"
    )
    for column in (
        "oracle_outer_rmse",
        "screening_selected_outer_rmse",
        "regret",
        "screening_cv_rmse",
    ):
        normalized[column] = pd.Series(normalized[column], dtype="Float64")
    normalized["agreement"] = pd.Series(normalized["agreement"], dtype="boolean")
    return normalized.loc[:, _DETAIL_COLUMNS].sort_values(["budget", "repeat"]).reset_index(
        drop=True
    )


def normalize_policy_summary_frame(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, _SUMMARY_COLUMNS, frame_name="policy summary")
    normalized = frame.copy()
    normalized["budget"] = pd.Series(normalized["budget"], dtype="Int64")
    for column in _SUMMARY_COLUMNS:
        if column == "budget":
            continue
        normalized[column] = pd.Series(normalized[column], dtype="Float64")
    return normalized.loc[:, _SUMMARY_COLUMNS].sort_values("budget").reset_index(drop=True)


def dump_policy_selection_diagnostic_results(
    results: PolicySelectionDiagnosticResults,
) -> dict[str, str]:
    return {
        "detail_df": results.detail_df.to_json(orient="table"),
        "summary_df": results.summary_df.to_json(orient="table"),
    }


def load_policy_selection_diagnostic_results_mapping(
    payload: dict[str, Any],
) -> PolicySelectionDiagnosticResults:
    detail_payload = payload.get("detail_df")
    summary_payload = payload.get("summary_df")
    if not isinstance(detail_payload, str):
        raise TypeError("policy diagnostic results must contain a detail_df JSON string.")
    if not isinstance(summary_payload, str):
        raise TypeError("policy diagnostic results must contain a summary_df JSON string.")
    return PolicySelectionDiagnosticResults(
        detail_df=pd.read_json(StringIO(detail_payload), orient="table"),
        summary_df=pd.read_json(StringIO(summary_payload), orient="table"),
    )


def dump_policy_selection_diagnostic_artifact(
    artifact: PolicySelectionDiagnosticArtifact,
) -> dict[str, Any]:
    return {
        "version": _POLICY_DIAGNOSTIC_ARTIFACT_VERSION,
        "metadata": artifact.metadata.to_bundle_mapping(),
        "results": dump_policy_selection_diagnostic_results(artifact.results),
    }


def load_policy_selection_diagnostic_artifact_mapping(
    payload: dict[str, Any],
    *,
    expected_metadata: LearningCurveSweepMetadata | None = None,
) -> PolicySelectionDiagnosticArtifact:
    version = payload.get("version")
    if version != _POLICY_DIAGNOSTIC_ARTIFACT_VERSION:
        raise ValueError(f"unsupported policy diagnostic artifact version: {version!r}.")
    metadata_payload = payload.get("metadata")
    if not isinstance(metadata_payload, dict):
        raise TypeError("policy diagnostic artifact must contain metadata.")
    metadata = LearningCurveSweepMetadata.from_bundle_mapping(
        metadata_payload,
        fallback=expected_metadata,
    )
    if expected_metadata is not None:
        expected_metadata.assert_compatible(metadata)
    results_payload = payload.get("results")
    if not isinstance(results_payload, dict):
        raise TypeError("policy diagnostic artifact must contain results.")
    return PolicySelectionDiagnosticArtifact(
        metadata=metadata,
        results=load_policy_selection_diagnostic_results_mapping(results_payload),
    )


def save_policy_selection_diagnostic_artifact(
    artifact: PolicySelectionDiagnosticArtifact,
    path: str | Path,
) -> Path:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(
        json.dumps(dump_policy_selection_diagnostic_artifact(artifact), indent=2),
        encoding="utf-8",
    )
    return resolved_path


def load_policy_selection_diagnostic_artifact(
    path: str | Path,
    *,
    expected_metadata: LearningCurveSweepMetadata | None = None,
) -> PolicySelectionDiagnosticArtifact:
    resolved_path = Path(path)
    return load_policy_selection_diagnostic_artifact_mapping(
        json.loads(resolved_path.read_text(encoding="utf-8")),
        expected_metadata=expected_metadata,
    )


def _require_columns(
    frame: pd.DataFrame,
    required_columns: list[str],
    *,
    frame_name: str,
) -> None:
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"{frame_name} frame is missing required columns: {missing_columns!r}.")
