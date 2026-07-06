from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
import json
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from oasis.experiment.core import _family_method_name, _family_requirements
from oasis.experiment.splits import (
    calibration_size_if_sweep_feasible,
    generate_sweep_splits,
    resolve_configured_sweep_sizes,
    validation_size_if_sweep_feasible,
)
from oasis.learning_curve.results_io import (
    LearningCurveSweepMetadata,
    learning_curve_method_name_for_result_field,
    learning_curve_result_field_for_method_name,
    learning_curve_uq_field_for_method_name,
)
from oasis.sweep import SweepDataset, SweepRunPayload, SweepSplit, SweepSplitCollection


_POLICY_DIAGNOSTIC_ARTIFACT_VERSION = 3
_DEFAULT_POLICY_NAMES = ("min_screening_rmse",)
_DETAIL_COLUMNS = [
    "policy_name",
    "budget",
    "repeat",
    "oracle_method",
    "screening_selected_method",
    "oracle_outer_rmse",
    "screening_selected_outer_rmse",
    "regret",
    "screening_cv_rmse",
    "screening_miscalibration_area",
    "agreement",
]
_SUMMARY_COLUMNS = [
    "policy_name",
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
    cache_signature: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class SharedOuterSplit:
    budget: int
    repeat: int
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass(frozen=True, slots=True)
class ScreeningDiagnosticMetrics:
    cv_rmse: float
    miscalibration_area: float | None


def normalize_policy_detail_frame(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, _DETAIL_COLUMNS, frame_name="policy detail")
    normalized = frame.copy()
    normalized["policy_name"] = normalized["policy_name"].astype("string")
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
        "screening_miscalibration_area",
    ):
        normalized[column] = pd.Series(normalized[column], dtype="Float64")
    normalized["agreement"] = pd.Series(normalized["agreement"], dtype="boolean")
    return (
        normalized.loc[:, _DETAIL_COLUMNS]
        .sort_values(["policy_name", "budget", "repeat"])
        .reset_index(drop=True)
    )


def normalize_policy_summary_frame(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, _SUMMARY_COLUMNS, frame_name="policy summary")
    normalized = frame.copy()
    normalized["policy_name"] = normalized["policy_name"].astype("string")
    normalized["budget"] = pd.Series(normalized["budget"], dtype="Int64")
    for column in _SUMMARY_COLUMNS:
        if column in {"policy_name", "budget"}:
            continue
        normalized[column] = pd.Series(normalized[column], dtype="Float64")
    return (
        normalized.loc[:, _SUMMARY_COLUMNS]
        .sort_values(["policy_name", "budget"])
        .reset_index(drop=True)
    )


def summarize_policy_detail_frame(detail_df: pd.DataFrame) -> pd.DataFrame:
    normalized_detail = normalize_policy_detail_frame(detail_df)
    if normalized_detail.empty:
        return normalize_policy_summary_frame(pd.DataFrame(columns=_SUMMARY_COLUMNS))

    rows: list[dict[str, float | int]] = []
    for (policy_name, budget), group in normalized_detail.groupby(
        ["policy_name", "budget"],
        sort=True,
    ):
        regret = group["regret"].astype(float).to_numpy()
        oracle_outer = group["oracle_outer_rmse"].astype(float).to_numpy()
        selected_outer = group["screening_selected_outer_rmse"].astype(float).to_numpy()
        agreement = group["agreement"].astype(bool).to_numpy()
        mean_regret = float(np.mean(regret))
        std_regret = float(np.std(regret))
        se_regret = float(std_regret / np.sqrt(len(regret)))
        rows.append(
            {
                "policy_name": str(policy_name),
                "budget": int(budget),
                "mean_regret": mean_regret,
                "std_regret": std_regret,
                "se_regret": se_regret,
                "ci95_low": mean_regret - 1.96 * se_regret,
                "ci95_high": mean_regret + 1.96 * se_regret,
                "agreement_rate": float(np.mean(agreement)),
                "oracle_outer_rmse_mean": float(np.mean(oracle_outer)),
                "screening_selected_outer_rmse_mean": float(np.mean(selected_outer)),
            }
        )
    return normalize_policy_summary_frame(pd.DataFrame(rows, columns=_SUMMARY_COLUMNS))


def build_policy_selection_diagnostic_results(
    dataset: SweepDataset,
    *,
    min_train: int,
    max_train: int,
    step: int,
    n_repeats: int,
    seed: int,
    model_families: Sequence[Any],
    outer_validation_fraction: float,
    outer_min_val_size: int,
    outer_min_tuning_val_size: int,
    outer_calibration_enabled: bool,
    outer_calibration_fraction: float,
    outer_min_cal_size: int,
    outer_min_inner_train_size: int,
    min_test_size: int,
    screening_fraction: float,
    min_screen_size: int,
    screening_validation_fraction: float,
    screening_min_val_size: int,
    screening_min_tuning_val_size: int,
    screening_calibration_enabled: bool,
    screening_calibration_fraction: float,
    screening_min_cal_size: int,
    screening_min_inner_train_size: int,
    requested_sweep_sizes: Collection[int] | None = None,
    policy_names: Collection[str] = _DEFAULT_POLICY_NAMES,
    combined_miscalibration_lambda: float = 1.0,
) -> PolicySelectionDiagnosticResults:
    detail_df = build_policy_selection_detail_frame(
        dataset,
        min_train=min_train,
        max_train=max_train,
        step=step,
        n_repeats=n_repeats,
        seed=seed,
        model_families=model_families,
        outer_validation_fraction=outer_validation_fraction,
        outer_min_val_size=outer_min_val_size,
        outer_min_tuning_val_size=outer_min_tuning_val_size,
        outer_calibration_enabled=outer_calibration_enabled,
        outer_calibration_fraction=outer_calibration_fraction,
        outer_min_cal_size=outer_min_cal_size,
        outer_min_inner_train_size=outer_min_inner_train_size,
        min_test_size=min_test_size,
        screening_fraction=screening_fraction,
        min_screen_size=min_screen_size,
        screening_validation_fraction=screening_validation_fraction,
        screening_min_val_size=screening_min_val_size,
        screening_min_tuning_val_size=screening_min_tuning_val_size,
        screening_calibration_enabled=screening_calibration_enabled,
        screening_calibration_fraction=screening_calibration_fraction,
        screening_min_cal_size=screening_min_cal_size,
        screening_min_inner_train_size=screening_min_inner_train_size,
        requested_sweep_sizes=requested_sweep_sizes,
        policy_names=policy_names,
        combined_miscalibration_lambda=combined_miscalibration_lambda,
    )
    return PolicySelectionDiagnosticResults(
        detail_df=detail_df,
        summary_df=summarize_policy_detail_frame(detail_df),
    )


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
        "cache_signature": artifact.cache_signature,
    }


def load_policy_selection_diagnostic_artifact_mapping(
    payload: dict[str, Any],
    *,
    expected_metadata: LearningCurveSweepMetadata | None = None,
    expected_cache_signature: dict[str, Any] | None = None,
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
    cache_signature = payload.get("cache_signature")
    if cache_signature is not None and not isinstance(cache_signature, dict):
        raise TypeError("policy diagnostic artifact cache_signature must be a mapping.")
    if expected_cache_signature is not None and cache_signature != expected_cache_signature:
        raise ValueError("policy diagnostic artifact cache signature is incompatible.")
    return PolicySelectionDiagnosticArtifact(
        metadata=metadata,
        results=load_policy_selection_diagnostic_results_mapping(results_payload),
        cache_signature=cache_signature,
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
    expected_cache_signature: dict[str, Any] | None = None,
) -> PolicySelectionDiagnosticArtifact:
    resolved_path = Path(path)
    return load_policy_selection_diagnostic_artifact_mapping(
        json.loads(resolved_path.read_text(encoding="utf-8")),
        expected_metadata=expected_metadata,
        expected_cache_signature=expected_cache_signature,
    )


def generate_shared_outer_splits(
    n_samples: int,
    *,
    min_train: int,
    max_train: int,
    step: int,
    n_repeats: int,
    seed: int,
    min_test_size: int = 1,
    requested_sweep_sizes: Collection[int] | None = None,
) -> tuple[SharedOuterSplit, ...]:
    rng = np.random.default_rng(seed)
    split_repeats_by_budget: dict[int, int] = {}
    shared_splits: list[SharedOuterSplit] = []
    for split in generate_sweep_splits(
        n_samples=n_samples,
        min_train=min_train,
        max_train=max_train,
        n_repeats=n_repeats,
        rng=rng,
        step=step,
        min_test_size=min_test_size,
        requested_sweep_sizes=requested_sweep_sizes,
    ):
        budget = int(split.sweep_size)
        repeat = split_repeats_by_budget.get(budget, 0)
        split_repeats_by_budget[budget] = repeat + 1
        shared_splits.append(
            SharedOuterSplit(
                budget=budget,
                repeat=repeat,
                train_idx=np.asarray(split.train_idx, dtype=int),
                test_idx=np.asarray(split.test_idx, dtype=int),
            )
        )
    return tuple(shared_splits)


def derive_family_split_collection_from_shared_outer_splits(
    shared_splits: Sequence[SharedOuterSplit],
    *,
    family: Any,
    seed: int,
    validation_fraction: float,
    min_val_size: int,
    min_tuning_val_size: int,
    calibration_enabled: bool,
    calibration_fraction: float,
    min_cal_size: int,
    min_inner_train_size: int,
) -> SweepSplitCollection:
    requirements = _family_requirements(family)
    if not calibration_enabled and getattr(requirements, "requires_calibration", False):
        requirements = type(requirements)(
            min_train_size=requirements.min_train_size,
            requires_inner_validation=requirements.requires_inner_validation,
            requires_calibration=False,
        )
    splits: list[SweepSplit] = []
    for shared_split in shared_splits:
        derived = _derive_family_split_from_shared_outer_split(
            shared_split,
            requirements=requirements,
            seed=seed,
            validation_fraction=validation_fraction,
            min_val_size=min_val_size,
            min_tuning_val_size=min_tuning_val_size,
            calibration_enabled=calibration_enabled,
            calibration_fraction=calibration_fraction,
            min_cal_size=min_cal_size,
            min_inner_train_size=min_inner_train_size,
        )
        if derived is not None:
            splits.append(derived)
    return SweepSplitCollection(
        splits=tuple(splits),
        planning_requirements=requirements,
    )


def build_policy_selection_detail_frame(
    dataset: SweepDataset,
    *,
    min_train: int,
    max_train: int,
    step: int,
    n_repeats: int,
    seed: int,
    model_families: Sequence[Any],
    outer_validation_fraction: float,
    outer_min_val_size: int,
    outer_min_tuning_val_size: int,
    outer_calibration_enabled: bool,
    outer_calibration_fraction: float,
    outer_min_cal_size: int,
    outer_min_inner_train_size: int,
    min_test_size: int,
    screening_fraction: float,
    min_screen_size: int,
    screening_validation_fraction: float,
    screening_min_val_size: int,
    screening_min_tuning_val_size: int,
    screening_calibration_enabled: bool,
    screening_calibration_fraction: float,
    screening_min_cal_size: int,
    screening_min_inner_train_size: int,
    requested_sweep_sizes: Collection[int] | None = None,
    policy_names: Collection[str] = _DEFAULT_POLICY_NAMES,
    combined_miscalibration_lambda: float = 1.0,
) -> pd.DataFrame:
    shared_splits = generate_shared_outer_splits(
        dataset.n_samples,
        min_train=min_train,
        max_train=max_train,
        step=step,
        n_repeats=n_repeats,
        seed=seed,
        min_test_size=min_test_size,
        requested_sweep_sizes=requested_sweep_sizes,
    )
    if not shared_splits:
        return normalize_policy_detail_frame(pd.DataFrame(columns=_DETAIL_COLUMNS))

    outer_rows: list[dict[str, Any]] = []
    screening_rows: list[dict[str, Any]] = []
    for family in model_families:
        method_name = _family_method_name(
            family,
            learning_curve_method_name_for_result_field,
        )
        if method_name is None:
            continue
        split_collection = derive_family_split_collection_from_shared_outer_splits(
            shared_splits,
            family=family,
            seed=seed,
            validation_fraction=outer_validation_fraction,
            min_val_size=outer_min_val_size,
            min_tuning_val_size=outer_min_tuning_val_size,
            calibration_enabled=outer_calibration_enabled,
            calibration_fraction=outer_calibration_fraction,
            min_cal_size=outer_min_cal_size,
            min_inner_train_size=outer_min_inner_train_size,
        )
        if not split_collection.splits:
            continue
        if not hasattr(family, "run_with_artifacts"):
            raise TypeError(
                f"family {method_name!r} must implement run_with_artifacts for policy diagnostics."
            )
        family_artifacts = family.run_with_artifacts(
            SweepRunPayload(
                dataset=dataset,
                split_collection=split_collection,
            )
        )
        repeat_metrics = getattr(family_artifacts, "repeat_metrics", None)
        if repeat_metrics is None or repeat_metrics.empty:
            raise ValueError(
                f"family {method_name!r} did not produce repeat_metrics for policy diagnostics."
            )
        for split in shared_splits:
            outer_match = repeat_metrics.loc[
                (repeat_metrics["n_train"] == split.budget)
                & (repeat_metrics["repeat"] == split.repeat),
                "outer_test_rmse",
            ]
            if outer_match.empty:
                continue
            outer_rows.append(
                {
                    "budget": split.budget,
                    "repeat": split.repeat,
                    "method": method_name,
                    "outer_test_rmse": float(outer_match.iloc[0]),
                }
            )

            screening_result = _run_family_screening_diagnostic(
                family,
                dataset=dataset.subset(split.train_idx),
                budget=split.budget,
                seed=_derived_split_seed(seed, split.budget, split.repeat, salt=7919),
                screening_fraction=screening_fraction,
                min_screen_size=min_screen_size,
                validation_fraction=screening_validation_fraction,
                min_val_size=screening_min_val_size,
                min_tuning_val_size=screening_min_tuning_val_size,
                calibration_enabled=screening_calibration_enabled,
                calibration_fraction=screening_calibration_fraction,
                min_cal_size=screening_min_cal_size,
                min_inner_train_size=screening_min_inner_train_size,
            )
            if screening_result is None:
                continue
            screening_rows.append(
                {
                    "budget": split.budget,
                    "repeat": split.repeat,
                    "method": method_name,
                    "screening_cv_rmse": screening_result.cv_rmse,
                    "screening_miscalibration_area": screening_result.miscalibration_area,
                }
            )

    outer_frame = pd.DataFrame(outer_rows)
    screening_frame = pd.DataFrame(screening_rows)
    detail_rows: list[dict[str, Any]] = []
    if not outer_frame.empty and not screening_frame.empty:
        for (budget, repeat), screening_group in screening_frame.groupby(["budget", "repeat"]):
            outer_group = outer_frame.loc[
                (outer_frame["budget"] == budget) & (outer_frame["repeat"] == repeat)
            ]
            if outer_group.empty:
                continue
            oracle_row = outer_group.sort_values(["outer_test_rmse", "method"]).iloc[0]
            for policy_name in policy_names:
                selected_row = _select_screening_policy_row(
                    screening_group,
                    policy_name=policy_name,
                    budget=int(budget),
                    repeat=int(repeat),
                    combined_miscalibration_lambda=combined_miscalibration_lambda,
                )
                selected_outer = outer_group.loc[
                    outer_group["method"] == selected_row["method"],
                    "outer_test_rmse",
                ]
                if selected_outer.empty:
                    continue
                selected_outer_rmse = float(selected_outer.iloc[0])
                oracle_outer_rmse = float(oracle_row["outer_test_rmse"])
                detail_rows.append(
                    {
                        "policy_name": str(policy_name),
                        "budget": int(budget),
                        "repeat": int(repeat),
                        "oracle_method": str(oracle_row["method"]),
                        "screening_selected_method": str(selected_row["method"]),
                        "oracle_outer_rmse": oracle_outer_rmse,
                        "screening_selected_outer_rmse": selected_outer_rmse,
                        "regret": selected_outer_rmse - oracle_outer_rmse,
                        "screening_cv_rmse": float(selected_row["screening_cv_rmse"]),
                        "screening_miscalibration_area": float(
                            selected_row["screening_miscalibration_area"]
                        )
                        if pd.notna(selected_row["screening_miscalibration_area"])
                        else np.nan,
                        "agreement": bool(oracle_row["method"] == selected_row["method"]),
                    }
                )
    return normalize_policy_detail_frame(pd.DataFrame(detail_rows, columns=_DETAIL_COLUMNS))


def _require_columns(
    frame: pd.DataFrame,
    required_columns: list[str],
    *,
    frame_name: str,
) -> None:
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"{frame_name} frame is missing required columns: {missing_columns!r}.")


def _select_screening_policy_row(
    screening_group: pd.DataFrame,
    *,
    policy_name: str,
    budget: int,
    repeat: int,
    combined_miscalibration_lambda: float,
) -> pd.Series:
    if policy_name == "min_screening_rmse":
        return screening_group.sort_values(["screening_cv_rmse", "method"]).iloc[0]
    if policy_name == "min_screening_miscalibration_area":
        _require_miscalibration_for_policy(
            screening_group,
            policy_name=policy_name,
            budget=budget,
            repeat=repeat,
        )
        return screening_group.sort_values(
            ["screening_miscalibration_area", "screening_cv_rmse", "method"]
        ).iloc[0]
    if policy_name == "combined_screening_rmse_miscalibration":
        _require_miscalibration_for_policy(
            screening_group,
            policy_name=policy_name,
            budget=budget,
            repeat=repeat,
        )
        scored = screening_group.copy()
        scored["rmse_norm"] = _min_max_normalize(scored["screening_cv_rmse"])
        scored["miscalibration_norm"] = _min_max_normalize(
            scored["screening_miscalibration_area"]
        )
        scored["combined_score"] = (
            scored["rmse_norm"]
            + float(combined_miscalibration_lambda) * scored["miscalibration_norm"]
        )
        return scored.sort_values(
            [
                "combined_score",
                "screening_cv_rmse",
                "screening_miscalibration_area",
                "method",
            ]
        ).iloc[0]
    raise ValueError(f"unsupported policy_name: {policy_name!r}.")


def _require_miscalibration_for_policy(
    screening_group: pd.DataFrame,
    *,
    policy_name: str,
    budget: int,
    repeat: int,
) -> None:
    missing_methods = screening_group.loc[
        screening_group["screening_miscalibration_area"].isna(),
        "method",
    ].astype(str)
    if missing_methods.empty:
        return
    raise ValueError(
        "screening miscalibration area is required for "
        f"policy {policy_name!r} at budget={budget}, repeat={repeat}; "
        f"missing methods: {sorted(missing_methods.tolist())!r}."
    )


def _min_max_normalize(values: pd.Series) -> pd.Series:
    numeric = values.astype(float)
    min_value = float(numeric.min())
    max_value = float(numeric.max())
    if max_value <= min_value:
        return pd.Series(np.zeros(len(numeric), dtype=float), index=numeric.index)
    return (numeric - min_value) / (max_value - min_value)


def _derive_family_split_from_shared_outer_split(
    shared_split: SharedOuterSplit,
    *,
    requirements: Any,
    seed: int,
    validation_fraction: float,
    min_val_size: int,
    min_tuning_val_size: int,
    calibration_enabled: bool,
    calibration_fraction: float,
    min_cal_size: int,
    min_inner_train_size: int,
) -> SweepSplit | None:
    budget = int(shared_split.budget)
    if budget < int(requirements.min_train_size):
        return None
    outer_train_idx = np.asarray(shared_split.train_idx, dtype=int)
    rng = np.random.default_rng(
        _derived_split_seed(seed, shared_split.budget, shared_split.repeat, salt=104729)
    )
    val_idx = None
    cal_idx = None
    train_idx = outer_train_idx
    use_calibration = bool(calibration_enabled and getattr(requirements, "requires_calibration", False))
    use_validation = bool(getattr(requirements, "requires_inner_validation", False))

    if use_validation and use_calibration:
        n_cal = calibration_size_if_sweep_feasible(
            budget,
            calibration_fraction=calibration_fraction,
            min_cal_size=min_cal_size,
            min_post_calibration_size=max(min_val_size + min_inner_train_size, 1),
        )
        if n_cal is None:
            return None
        n_val = validation_size_if_sweep_feasible(
            budget - n_cal,
            validation_fraction=validation_fraction,
            min_val_size=min_val_size,
            min_tuning_val_size=min_tuning_val_size,
            min_inner_train_size=min_inner_train_size,
        )
        if n_val is None:
            return None
        val_idx = np.asarray(rng.choice(outer_train_idx, size=n_val, replace=False), dtype=int)
        remaining_idx = np.setdiff1d(outer_train_idx, val_idx, assume_unique=False)
        cal_idx = np.asarray(rng.choice(remaining_idx, size=n_cal, replace=False), dtype=int)
        train_idx = np.setdiff1d(remaining_idx, cal_idx, assume_unique=False)
    elif use_validation:
        n_val = validation_size_if_sweep_feasible(
            budget,
            validation_fraction=validation_fraction,
            min_val_size=min_val_size,
            min_tuning_val_size=min_tuning_val_size,
            min_inner_train_size=min_inner_train_size,
        )
        if n_val is None:
            return None
        val_idx = np.asarray(rng.choice(outer_train_idx, size=n_val, replace=False), dtype=int)
        train_idx = np.setdiff1d(outer_train_idx, val_idx, assume_unique=False)
    elif use_calibration:
        n_cal = calibration_size_if_sweep_feasible(
            budget,
            calibration_fraction=calibration_fraction,
            min_cal_size=min_cal_size,
            min_post_calibration_size=min_inner_train_size,
        )
        if n_cal is None:
            return None
        cal_idx = np.asarray(rng.choice(outer_train_idx, size=n_cal, replace=False), dtype=int)
        train_idx = np.setdiff1d(outer_train_idx, cal_idx, assume_unique=False)

    return SweepSplit(
        sweep_size=budget,
        train_idx=np.asarray(train_idx, dtype=int),
        test_idx=np.asarray(shared_split.test_idx, dtype=int),
        val_idx=None if val_idx is None else np.asarray(val_idx, dtype=int),
        cal_idx=None if cal_idx is None else np.asarray(cal_idx, dtype=int),
    )


def _run_family_screening_diagnostic(
    family: Any,
    *,
    dataset: SweepDataset,
    budget: int,
    seed: int,
    screening_fraction: float,
    min_screen_size: int,
    validation_fraction: float,
    min_val_size: int,
    min_tuning_val_size: int,
    calibration_enabled: bool,
    calibration_fraction: float,
    min_cal_size: int,
    min_inner_train_size: int,
) -> ScreeningDiagnosticMetrics | None:
    from oasis.experiment.core import (
        _annotate_screening_results,
        _build_family_split_collection,
    )

    split_collection = _build_family_split_collection(
        dataset,
        family,
        min_train=budget,
        max_train=budget,
        step=1,
        n_repeats=1,
        seed=seed,
        requested_sweep_sizes=(budget,),
        budget_mode="screening_fraction",
        screen_fraction=screening_fraction,
        min_screen_size=min_screen_size,
        validation_fraction=validation_fraction,
        min_val_size=min_val_size,
        min_tuning_val_size=min_tuning_val_size,
        calibration_enabled=calibration_enabled,
        calibration_fraction=calibration_fraction,
        min_cal_size=min_cal_size,
        min_inner_train_size=min_inner_train_size,
        min_test_size=1,
    )
    if not split_collection.splits:
        return None
    results = family.run(
        SweepRunPayload(
            dataset=dataset,
            split_collection=split_collection,
        )
    )
    results = _annotate_screening_results(results, split_collection)
    method_name = _family_method_name(
        family,
        learning_curve_method_name_for_result_field,
    )
    if method_name is None:
        return None
    result_field = learning_curve_result_field_for_method_name(method_name)
    if result_field is None:
        return None
    result_frame = getattr(results, result_field)
    if result_frame is None or result_frame.empty:
        return None
    uq_field = learning_curve_uq_field_for_method_name(method_name)
    uq_frame = getattr(results, uq_field) if uq_field is not None else None
    miscalibration_area = None
    if (
        uq_frame is not None
        and not uq_frame.empty
        and "miscalibration_area" in uq_frame.columns
    ):
        miscalibration_area = float(uq_frame["miscalibration_area"].iloc[0])
    return ScreeningDiagnosticMetrics(
        cv_rmse=float(result_frame["cv_rmse_mean"].iloc[0]),
        miscalibration_area=miscalibration_area,
    )


def _derived_split_seed(seed: int, budget: int, repeat: int, *, salt: int) -> int:
    return int(seed) + int(budget) * 1009 + int(repeat) * 9176 + int(salt)
