from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import pandas as pd

from oasis.learning_curve.results_io import (
    learning_curve_method_names,
    learning_curve_result_field_for_method_name,
)
from oasis.mlip.timing import MlipGenerationTimingSummary
from oasis.sweep import LearningCurveResults


_METHODS_REQUIRING_EXPLICIT_GENERATION_TIMING = frozenset({"latent", "probe_gnn"})
_TIME_ACCURACY_COLUMNS = [
    "method",
    "n_train",
    "rmse_mean",
    "generation_time_s",
    "training_time_s",
    "total_time_s",
]
_FIXED_SPLIT_TIME_ACCURACY_COLUMNS = [
    "method",
    "n_train",
    "rmse_mean",
    "rmse_std",
    "generation_time_s",
    "training_time_mean_s",
    "training_time_std_s",
    "total_time_mean_s",
    "total_time_std_s",
]


@dataclass(frozen=True, slots=True)
class GenerationTimingAggregate:
    generation_time_s: float
    generation_time_slab_s: float
    generation_time_adslab_s: float
    generation_steps_total: int
    generation_steps_slab: int
    generation_steps_adslab: int
    time_per_step_s: float | None
    mlip_feature_names: tuple[str, ...]


def aggregate_generation_timing(
    generation_timing_by_mlip: dict[str, MlipGenerationTimingSummary],
    *,
    mlip_feature_names: tuple[str, ...] | list[str] | None = None,
) -> GenerationTimingAggregate:
    selected_names = (
        tuple(sorted(generation_timing_by_mlip))
        if mlip_feature_names is None
        else tuple(mlip_feature_names)
    )
    missing_names = [
        feature_name
        for feature_name in selected_names
        if feature_name not in generation_timing_by_mlip
    ]
    if missing_names:
        raise KeyError(
            "missing generation timing summaries for MLIP features: "
            f"{sorted(missing_names)!r}"
        )

    selected_summaries = [
        generation_timing_by_mlip[feature_name] for feature_name in selected_names
    ]
    generation_time_s = sum(
        summary.generation_time_total_s for summary in selected_summaries
    )
    generation_time_slab_s = sum(
        summary.generation_time_slab_s for summary in selected_summaries
    )
    generation_time_adslab_s = sum(
        summary.generation_time_adslab_s for summary in selected_summaries
    )
    generation_steps_slab = sum(
        summary.generation_steps_slab for summary in selected_summaries
    )
    generation_steps_adslab = sum(
        summary.generation_steps_adslab for summary in selected_summaries
    )
    generation_steps_total = sum(
        summary.generation_steps_total for summary in selected_summaries
    )
    return GenerationTimingAggregate(
        generation_time_s=generation_time_s,
        generation_time_slab_s=generation_time_slab_s,
        generation_time_adslab_s=generation_time_adslab_s,
        generation_steps_total=generation_steps_total,
        generation_steps_slab=generation_steps_slab,
        generation_steps_adslab=generation_steps_adslab,
        time_per_step_s=(
            None
            if generation_steps_total <= 0
            else generation_time_s / generation_steps_total
        ),
        mlip_feature_names=selected_names,
    )


def _resolve_generation_timing_for_method(
    method_name: str,
    generation_timing_by_mlip: dict[str, MlipGenerationTimingSummary],
    *,
    mlip_feature_names: tuple[str, ...] | list[str] | None = None,
    generation_timing_by_method: dict[str, GenerationTimingAggregate] | None = None,
) -> GenerationTimingAggregate | None:
    if generation_timing_by_method is not None and method_name in generation_timing_by_method:
        return generation_timing_by_method[method_name]
    if method_name in _METHODS_REQUIRING_EXPLICIT_GENERATION_TIMING:
        return None
    return aggregate_generation_timing(
        generation_timing_by_mlip,
        mlip_feature_names=mlip_feature_names,
    )


def build_time_accuracy_table(
    results: LearningCurveResults,
    generation_timing_by_mlip: dict[str, MlipGenerationTimingSummary],
    *,
    mlip_feature_names: tuple[str, ...] | list[str] | None = None,
    generation_timing_by_method: dict[str, GenerationTimingAggregate] | None = None,
    method_names: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    selected_method_names = (
        tuple(learning_curve_method_names())
        if method_names is None
        else tuple(method_names)
    )

    rows: list[dict[str, Any]] = []
    for method_name in selected_method_names:
        result_field = learning_curve_result_field_for_method_name(method_name)
        if result_field is None:
            continue
        frame = getattr(results, result_field)
        if frame is None or frame.empty:
            continue
        required_columns = {"n_train", "rmse_mean"}
        missing_required_columns = required_columns.difference(frame.columns)
        if missing_required_columns:
            missing_list = ", ".join(sorted(missing_required_columns))
            raise ValueError(
                f"{result_field} is missing required columns: {missing_list}"
            )
        if "fit_time_mean_s" not in frame.columns:
            continue
        generation_timing = _resolve_generation_timing_for_method(
            method_name,
            generation_timing_by_mlip,
            mlip_feature_names=mlip_feature_names,
            generation_timing_by_method=generation_timing_by_method,
        )
        if generation_timing is None:
            continue

        for row in frame.itertuples(index=False):
            row_mapping = row._asdict()
            training_time_s = float(row_mapping["fit_time_mean_s"])
            rows.append(
                {
                    "method": method_name,
                    "n_train": int(row_mapping["n_train"]),
                    "rmse_mean": float(row_mapping["rmse_mean"]),
                    "generation_time_s": generation_timing.generation_time_s,
                    "training_time_s": training_time_s,
                    "total_time_s": generation_timing.generation_time_s
                    + training_time_s,
                }
            )

    return pd.DataFrame(rows, columns=_TIME_ACCURACY_COLUMNS)


def fixed_split_train_size(
    dataset_size: int,
    *,
    train_fraction: float = 0.8,
) -> int:
    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive.")
    if train_fraction <= 0.0 or train_fraction >= 1.0:
        raise ValueError("train_fraction must lie strictly between 0 and 1.")
    return max(1, int(math.floor(train_fraction * dataset_size)))


def build_fixed_split_time_accuracy_table(
    results: LearningCurveResults,
    generation_timing_by_mlip: dict[str, MlipGenerationTimingSummary],
    *,
    dataset_size: int,
    train_fraction: float = 0.8,
    mlip_feature_names: tuple[str, ...] | list[str] | None = None,
    generation_timing_by_method: dict[str, GenerationTimingAggregate] | None = None,
    method_names: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    benchmark_n_train = fixed_split_train_size(
        dataset_size,
        train_fraction=train_fraction,
    )
    selected_method_names = (
        tuple(learning_curve_method_names())
        if method_names is None
        else tuple(method_names)
    )
    rows: list[dict[str, Any]] = []
    for method_name in selected_method_names:
        result_field = learning_curve_result_field_for_method_name(method_name)
        if result_field is None:
            continue
        frame = getattr(results, result_field)
        if frame is None or frame.empty:
            continue
        required_columns = {
            "n_train",
            "rmse_mean",
            "rmse_std",
            "fit_time_mean_s",
            "fit_time_std_s",
        }
        missing_required_columns = required_columns.difference(frame.columns)
        if missing_required_columns:
            continue
        generation_timing = _resolve_generation_timing_for_method(
            method_name,
            generation_timing_by_mlip,
            mlip_feature_names=mlip_feature_names,
            generation_timing_by_method=generation_timing_by_method,
        )
        if generation_timing is None:
            continue
        matched = frame.loc[frame["n_train"] == benchmark_n_train]
        if matched.empty:
            continue
        row_mapping = matched.iloc[0].to_dict()
        training_time_mean_s = float(row_mapping["fit_time_mean_s"])
        training_time_std_s = float(row_mapping["fit_time_std_s"])
        rows.append(
            {
                "method": method_name,
                "n_train": benchmark_n_train,
                "rmse_mean": float(row_mapping["rmse_mean"]),
                "rmse_std": float(row_mapping["rmse_std"]),
                "generation_time_s": generation_timing.generation_time_s,
                "training_time_mean_s": training_time_mean_s,
                "training_time_std_s": training_time_std_s,
                "total_time_mean_s": generation_timing.generation_time_s
                + training_time_mean_s,
                "total_time_std_s": training_time_std_s,
            }
        )
    return pd.DataFrame(rows, columns=_FIXED_SPLIT_TIME_ACCURACY_COLUMNS)
