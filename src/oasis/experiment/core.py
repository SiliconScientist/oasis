from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Any

from oasis.experiment_config import LearningCurveBudgetMode
from oasis.learning_curve.results_io import (
    learning_curve_method_name_for_result_field,
    learning_curve_result_field_for_method_name,
    learning_curve_selection_field_for_method_name,
)
from oasis.experiment.splits import build_sweep_split_collection
from oasis.sweep import (
    LearningCurveResults,
    SweepDataset,
    SweepFamilyRequirements,
    SweepRunPayload,
    SweepSplitCollection,
)


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
    min_train: int,
    max_train: int,
    step: int,
    n_repeats: int,
    seed: int,
    budget_mode: LearningCurveBudgetMode,
    screen_fraction: float | None,
    min_screen_size: int,
    validation_fraction: float,
    min_val_size: int,
    min_tuning_val_size: int,
    min_inner_train_size: int,
    min_test_size: int,
) -> SweepSplitCollection:
    return build_sweep_split_collection(
        dataset.n_samples,
        min_train=min_train,
        max_train=max_train,
        step=step,
        n_repeats=n_repeats,
        seed=seed,
        requirements=_family_requirements(family),
        budget_mode=budget_mode,
        screen_fraction=screen_fraction,
        min_screen_size=min_screen_size,
        validation_fraction=validation_fraction,
        min_val_size=min_val_size,
        min_tuning_val_size=min_tuning_val_size,
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
    if frame is None or "n_train" not in frame.columns:
        return set()
    return {int(value) for value in frame["n_train"].tolist()}


def _filter_learning_curve_frame_by_train_sizes(
    frame: Any,
    allowed_train_sizes: Collection[int],
) -> Any:
    if frame is None:
        return None
    allowed_sizes = {int(value) for value in allowed_train_sizes}
    if "n_train" not in frame.columns:
        raise ValueError("learning-curve result frames must contain an n_train column.")
    return (
        frame.loc[frame["n_train"].isin(sorted(allowed_sizes))]
        .sort_values("n_train")
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
    for field_name in (metrics_field, selection_field):
        if field_name is None:
            continue
        frame = field_mapping[field_name]
        if frame is None:
            continue
        filtered_frame = (
            frame.loc[~frame["n_train"].isin(sorted(allowed_sizes))]
            .sort_values("n_train")
            .reset_index(drop=True)
        )
        field_mapping[field_name] = None if filtered_frame.empty else filtered_frame
    return LearningCurveResults.from_mapping(field_mapping)


def _run_learning_curve_experiments_with_budget_mode(
    dataset: SweepDataset,
    *,
    min_train: int,
    max_train: int,
    step: int = 1,
    n_repeats: int,
    seed: int = 42,
    enabled_model_names: Sequence[str] | None = None,
    model_cfg: Any | None = None,
    budget_mode: LearningCurveBudgetMode = "full_remainder_test",
    screen_fraction: float | None = None,
    min_screen_size: int = 1,
    validation_fraction: float = 0.2,
    min_val_size: int = 1,
    min_tuning_val_size: int = 1,
    min_inner_train_size: int = 1,
    min_test_size: int = 1,
    model_families: Sequence[Any] | None = None,
    requested_sweep_sizes_by_method: Mapping[str, Collection[int]] | None = None,
) -> LearningCurveResults:
    families = model_families
    if families is None:
        from oasis.learning_curve.registry import default_sweep_model_families

        families = default_sweep_model_families(enabled_model_names, config=model_cfg)

    results = LearningCurveResults.empty()
    for family in families:
        split_collection = _build_family_split_collection(
            dataset,
            family,
            min_train=min_train,
            max_train=max_train,
            step=step,
            n_repeats=n_repeats,
            seed=seed,
            budget_mode=budget_mode,
            screen_fraction=screen_fraction,
            min_screen_size=min_screen_size,
            validation_fraction=validation_fraction,
            min_val_size=min_val_size,
            min_tuning_val_size=min_tuning_val_size,
            min_inner_train_size=min_inner_train_size,
            min_test_size=min_test_size,
        )
        method_name = _family_method_name(
            family,
            learning_curve_method_name_for_result_field,
        )
        requested_sweep_sizes = (
            None
            if requested_sweep_sizes_by_method is None or method_name is None
            else {
                int(value)
                for value in requested_sweep_sizes_by_method.get(method_name, ())
            }
        )
        if requested_sweep_sizes is not None:
            split_collection = SweepSplitCollection(
                splits=tuple(
                    split
                    for split in split_collection.splits
                    if split.sweep_size in requested_sweep_sizes
                ),
                planning_requirements=split_collection.planning_requirements,
            )
            if not split_collection.splits:
                continue
        payload = SweepRunPayload(dataset=dataset, split_collection=split_collection)
        results = results.merge(family.run(payload))
    return results
