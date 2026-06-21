from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from oasis.learning_curve.results_io import (
    learning_curve_method_names,
    learning_curve_result_field_for_method_name,
)
from oasis.mlip.timing import MlipGenerationTimingSummary
from oasis.sweep import LearningCurveResults


_EXCLUDED_METHOD_NAMES = frozenset({"latent", "probe_gnn"})
_TIME_ACCURACY_COLUMNS = [
    "method",
    "n_train",
    "rmse_mean",
    "generation_time_s",
    "training_time_s",
    "total_time_s",
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


def build_time_accuracy_table(
    results: LearningCurveResults,
    generation_timing_by_mlip: dict[str, MlipGenerationTimingSummary],
    *,
    mlip_feature_names: tuple[str, ...] | list[str] | None = None,
    method_names: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    generation_timing = aggregate_generation_timing(
        generation_timing_by_mlip,
        mlip_feature_names=mlip_feature_names,
    )
    selected_method_names = (
        tuple(learning_curve_method_names())
        if method_names is None
        else tuple(method_names)
    )

    rows: list[dict[str, Any]] = []
    for method_name in selected_method_names:
        if method_name in _EXCLUDED_METHOD_NAMES:
            continue
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
