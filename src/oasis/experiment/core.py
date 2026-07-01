from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Any

from oasis.experiment_config import LearningCurveBudgetMode
from oasis.learning_curve.results_io import (
    learning_curve_method_name_for_result_field,
    learning_curve_result_field_for_method_name,
    learning_curve_selection_field_for_method_name,
    learning_curve_uq_field_for_method_name,
)
from oasis.experiment.splits import build_sweep_split_collection
from oasis.sweep import (
    LearningCurveResults,
    SweepDataset,
    SweepFamilyRequirements,
    SweepRunPayload,
    SweepSplitCollection,
)


class LearningCurveExecutionArtifacts:
    def __init__(
        self,
        *,
        results: LearningCurveResults,
        repeat_metrics_df: Any = None,
    ) -> None:
        self.results = results
        self.repeat_metrics_df = repeat_metrics_df


def _result_frame_budget_column(frame: Any) -> str:
    if frame is None:
        raise ValueError("learning-curve frame is required.")
    if "n_budget" in frame.columns:
        return "n_budget"
    if "n_train" not in frame.columns:
        raise ValueError("learning-curve result frames must contain an n_train column.")
    return "n_train"


def _family_method_name(
    family: Any,
    resolve_result_field: Any,
) -> str | None:
    if hasattr(family, "method_name") and isinstance(family.method_name, str):
        return family.method_name
    spec = getattr(family, "spec", None)
    result_field = getattr(spec, "result_field", None)
    if isinstance(result_field, str):
        return resolve_result_field(result_field)
    result_field = getattr(family, "result_field", None)
    if isinstance(result_field, str):
        return resolve_result_field(result_field)
    return None


def _family_requirements(family: Any) -> SweepFamilyRequirements:
    if hasattr(family, "capabilities"):
        return family.capabilities().to_requirements()
    if hasattr(family, "requirements"):
        return family.requirements()
    return SweepFamilyRequirements()


def _build_family_split_collection(
    dataset: SweepDataset,
    family: Any,
    *,
    min_train: int | None,
    max_train: int | None,
    step: int,
    n_repeats: int,
    seed: int,
    requested_sweep_sizes: Collection[int] | None = None,
    budget_mode: LearningCurveBudgetMode,
    screen_fraction: float | None,
    min_screen_size: int,
    validation_fraction: float,
    min_val_size: int,
    min_tuning_val_size: int,
    calibration_enabled: bool,
    calibration_fraction: float,
    min_cal_size: int,
    min_inner_train_size: int,
    min_test_size: int,
) -> SweepSplitCollection:
    requirements = _family_requirements(family)
    if not calibration_enabled and requirements.requires_calibration:
        requirements = SweepFamilyRequirements(
            min_train_size=requirements.min_train_size,
            requires_inner_validation=requirements.requires_inner_validation,
            requires_calibration=False,
        )
    return build_sweep_split_collection(
        dataset.n_samples,
        min_train=min_train,
        max_train=max_train,
        step=step,
        n_repeats=n_repeats,
        seed=seed,
        requested_sweep_sizes=requested_sweep_sizes,
        requirements=requirements,
        budget_mode=budget_mode,
        screen_fraction=screen_fraction,
        min_screen_size=min_screen_size,
        validation_fraction=validation_fraction,
        min_val_size=min_val_size,
        min_tuning_val_size=min_tuning_val_size,
        calibration_fraction=calibration_fraction,
        min_cal_size=min_cal_size,
        min_inner_train_size=min_inner_train_size,
        min_test_size=min_test_size,
    )


def _unique_sweep_sizes(split_collection: SweepSplitCollection) -> tuple[int, ...]:
    return tuple(sorted({split.sweep_size for split in split_collection.splits}))


def _result_frame_train_sizes(
    results: LearningCurveResults,
    method_name: str,
    resolve_result_field: Any,
) -> set[int]:
    result_field = resolve_result_field(method_name)
    if result_field is None:
        return set()
    frame = getattr(results, result_field)
    if frame is None:
        return set()
    budget_column = _result_frame_budget_column(frame)
    return {int(value) for value in frame[budget_column].tolist()}


def _filter_learning_curve_frame_by_train_sizes(
    frame: Any,
    allowed_train_sizes: Collection[int],
) -> Any:
    if frame is None:
        return None
    allowed_sizes = {int(value) for value in allowed_train_sizes}
    budget_column = _result_frame_budget_column(frame)
    return (
        frame.loc[frame[budget_column].isin(sorted(allowed_sizes))]
        .sort_values(budget_column)
        .reset_index(drop=True)
    )


def _remove_method_train_sizes_from_results(
    results: LearningCurveResults,
    method_name: str,
    train_sizes_to_remove: Collection[int],
) -> LearningCurveResults:
    if not train_sizes_to_remove:
        return results
    allowed_sizes = {int(value) for value in train_sizes_to_remove}
    field_mapping = results.to_mapping()
    metrics_field = learning_curve_result_field_for_method_name(method_name)
    selection_field = learning_curve_selection_field_for_method_name(method_name)
    uq_field = learning_curve_uq_field_for_method_name(method_name)
    for field_name in (metrics_field, selection_field, uq_field):
        if field_name is None:
            continue
        frame = field_mapping[field_name]
        if frame is None:
            continue
        budget_column = _result_frame_budget_column(frame)
        filtered_frame = (
            frame.loc[~frame[budget_column].isin(sorted(allowed_sizes))]
            .sort_values(budget_column)
            .reset_index(drop=True)
        )
        field_mapping[field_name] = None if filtered_frame.empty else filtered_frame
    return LearningCurveResults.from_mapping(field_mapping)


def _screening_split_metadata_frame(
    split_collection: SweepSplitCollection,
) -> Any:
    rows: list[dict[str, float | int]] = []
    budgets = sorted({int(split.sweep_size) for split in split_collection.splits})
    for n_budget in budgets:
        budget_splits = [
            split for split in split_collection.splits if int(split.sweep_size) == n_budget
        ]
        representative = budget_splits[0]
        n_train = int(
            len(representative.train_idx)
            + (0 if representative.val_idx is None else len(representative.val_idx))
            + (0 if representative.cal_idx is None else len(representative.cal_idx))
        )
        n_screen = int(len(representative.test_idx))
        rows.append(
            {
                "n_budget": n_budget,
                "n_train": n_train,
                "n_screen": n_screen,
                "screen_fraction": float(n_screen / n_budget),
                "n_cv_folds": int(len(budget_splits)),
            }
        )
    import pandas as pd

    return pd.DataFrame(rows).sort_values("n_budget").reset_index(drop=True)


def _annotate_screening_frame(
    frame: Any,
    split_collection: SweepSplitCollection,
) -> Any:
    if frame is None:
        return None
    if "n_train" not in frame.columns:
        raise ValueError("learning-curve result frames must contain an n_train column.")
    metadata = _screening_split_metadata_frame(split_collection).rename(
        columns={"n_train": "screening_train_count"}
    )
    annotated = frame.rename(columns={"n_train": "n_budget"}).merge(
        metadata,
        on="n_budget",
        how="left",
        validate="many_to_one",
    )
    annotated = annotated.rename(columns={"screening_train_count": "n_train"})
    annotated = annotated.rename(
        columns={
            "rmse_mean": "cv_rmse_mean",
            "rmse_std": "cv_rmse_std",
        }
    )
    ordered_columns = []
    for column in ("n_budget", "n_train", "n_screen", "screen_fraction", "n_cv_folds"):
        if column in annotated.columns:
            ordered_columns.append(column)
    ordered_columns.extend(
        column for column in annotated.columns if column not in ordered_columns
    )
    return annotated.loc[:, ordered_columns].sort_values("n_budget").reset_index(drop=True)


def _annotate_screening_results(
    results: LearningCurveResults,
    split_collection: SweepSplitCollection,
) -> LearningCurveResults:
    return LearningCurveResults.from_mapping(
        {
            field_name: _annotate_screening_frame(frame, split_collection)
            for field_name, frame in results.to_mapping().items()
        }
    )


def _run_learning_curve_experiments_with_budget_mode(
    dataset: SweepDataset,
    *,
    min_train: int | None,
    max_train: int | None,
    step: int = 1,
    n_repeats: int,
    seed: int = 42,
    requested_sweep_sizes: Collection[int] | None = None,
    enabled_model_names: Sequence[str] | None = None,
    model_cfg: Any | None = None,
    budget_mode: LearningCurveBudgetMode = "full_remainder_test",
    screen_fraction: float | None = None,
    min_screen_size: int = 1,
    validation_fraction: float = 0.2,
    min_val_size: int = 1,
    min_tuning_val_size: int = 1,
    calibration_enabled: bool = True,
    calibration_fraction: float = 0.2,
    min_cal_size: int = 1,
    min_inner_train_size: int = 1,
    min_test_size: int = 1,
    model_families: Sequence[Any] | None = None,
    requested_sweep_sizes_by_method: Mapping[str, Collection[int]] | None = None,
) -> LearningCurveResults:
    return _run_learning_curve_experiments_with_budget_mode_artifacts(
        dataset,
        min_train=min_train,
        max_train=max_train,
        step=step,
        n_repeats=n_repeats,
        seed=seed,
        requested_sweep_sizes=requested_sweep_sizes,
        enabled_model_names=enabled_model_names,
        model_cfg=model_cfg,
        budget_mode=budget_mode,
        screen_fraction=screen_fraction,
        min_screen_size=min_screen_size,
        validation_fraction=validation_fraction,
        min_val_size=min_val_size,
        min_tuning_val_size=min_tuning_val_size,
        calibration_enabled=calibration_enabled,
        calibration_fraction=calibration_fraction,
        min_cal_size=min_cal_size,
        min_inner_train_size=min_inner_train_size,
        min_test_size=min_test_size,
        model_families=model_families,
        requested_sweep_sizes_by_method=requested_sweep_sizes_by_method,
    ).results


def _run_learning_curve_experiments_with_budget_mode_artifacts(
    dataset: SweepDataset,
    *,
    min_train: int | None,
    max_train: int | None,
    step: int = 1,
    n_repeats: int,
    seed: int = 42,
    requested_sweep_sizes: Collection[int] | None = None,
    enabled_model_names: Sequence[str] | None = None,
    model_cfg: Any | None = None,
    budget_mode: LearningCurveBudgetMode = "full_remainder_test",
    screen_fraction: float | None = None,
    min_screen_size: int = 1,
    validation_fraction: float = 0.2,
    min_val_size: int = 1,
    min_tuning_val_size: int = 1,
    calibration_enabled: bool = True,
    calibration_fraction: float = 0.2,
    min_cal_size: int = 1,
    min_inner_train_size: int = 1,
    min_test_size: int = 1,
    model_families: Sequence[Any] | None = None,
    requested_sweep_sizes_by_method: Mapping[str, Collection[int]] | None = None,
) -> LearningCurveExecutionArtifacts:
    families = model_families
    if families is None:
        from oasis.learning_curve.registry import default_sweep_model_families

        families = default_sweep_model_families(enabled_model_names, config=model_cfg)

    results = LearningCurveResults.empty()
    repeat_metric_frames: list[Any] = []
    for family in families:
        split_collection = _build_family_split_collection(
            dataset,
            family,
            min_train=min_train,
            max_train=max_train,
            step=step,
            n_repeats=n_repeats,
            seed=seed,
            requested_sweep_sizes=requested_sweep_sizes,
            budget_mode=budget_mode,
            screen_fraction=screen_fraction,
            min_screen_size=min_screen_size,
            validation_fraction=validation_fraction,
            min_val_size=min_val_size,
            min_tuning_val_size=min_tuning_val_size,
            calibration_enabled=calibration_enabled,
            calibration_fraction=calibration_fraction,
            min_cal_size=min_cal_size,
            min_inner_train_size=min_inner_train_size,
            min_test_size=min_test_size,
        )
        method_name = _family_method_name(
            family,
            learning_curve_method_name_for_result_field,
        )
        method_requested_sweep_sizes = (
            None
            if requested_sweep_sizes_by_method is None or method_name is None
            else {
                int(value)
                for value in requested_sweep_sizes_by_method.get(method_name, ())
            }
        )
        if method_requested_sweep_sizes is not None:
            split_collection = SweepSplitCollection(
                splits=tuple(
                    split
                    for split in split_collection.splits
                    if split.sweep_size in method_requested_sweep_sizes
                ),
                planning_requirements=split_collection.planning_requirements,
            )
            if not split_collection.splits:
                continue
        payload = SweepRunPayload(dataset=dataset, split_collection=split_collection)
        family_artifacts = (
            family.run_with_artifacts(payload)
            if hasattr(family, "run_with_artifacts")
            else LearningCurveExecutionArtifacts(results=family.run(payload))
        )
        family_results = family_artifacts.results
        if budget_mode == "screening_fraction":
            family_results = _annotate_screening_results(
                family_results,
                split_collection,
            )
        repeat_metrics = getattr(family_artifacts, "repeat_metrics_df", None)
        if repeat_metrics is None:
            repeat_metrics = getattr(family_artifacts, "repeat_metrics", None)
        if (
            budget_mode == "full_remainder_test"
            and repeat_metrics is not None
            and method_name is not None
            and not repeat_metrics.empty
        ):
            repeat_metric_frames.append(
                repeat_metrics.rename(columns={"n_train": "budget"}).assign(method=method_name)
            )
        results = results.merge(family_results)
    repeat_metrics_df = None
    if repeat_metric_frames:
        import pandas as pd

        repeat_metrics_df = (
            pd.concat(repeat_metric_frames, ignore_index=True)
            .loc[:, ["method", "budget", "repeat", "outer_test_rmse"]]
            .sort_values(["method", "budget", "repeat"])
            .reset_index(drop=True)
        )
    return LearningCurveExecutionArtifacts(
        results=results,
        repeat_metrics_df=repeat_metrics_df,
    )
