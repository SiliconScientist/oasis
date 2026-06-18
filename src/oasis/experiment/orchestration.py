from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from oasis.experiment.core import (
    _build_family_split_collection,
    _family_method_name,
    _remove_method_train_sizes_from_results,
    _result_frame_train_sizes,
    _unique_sweep_sizes,
)
from oasis.experiment.dataset import (
    _validate_learning_curve_frame,
    assemble_learning_curve_dataset_from_frame,
    build_sweep_dataset_from_config,
    mlip_feature_names,
)
from oasis.experiment.learning_curve import run_standard_learning_curve_experiments
from oasis.experiment.screening import run_screening_learning_curve_experiments
from oasis.experiment_config import LearningCurveBudgetMode
from oasis.learning_curve.results_io import (
    LearningCurveSweepMetadata,
    build_learning_curve_point_provenance,
    learning_curve_method_name_for_result_field,
    learning_curve_method_names,
    learning_curve_result_field_for_method_name,
    learning_curve_results_has_method,
    learning_curve_selection_field_for_method_name,
    learning_curve_uq_field_for_method_name,
    learning_curve_sweep_metadata_from_config,
    load_learning_curve_results_artifact,
    merge_learning_curve_point_provenance,
    save_learning_curve_results_artifact,
    select_learning_curve_results_methods,
)
from oasis.sweep import GraphDatasetView, LearningCurveResults, SweepDataset

if TYPE_CHECKING:
    from oasis.config import Config


def run_learning_curve_experiments(
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
    """Dispatch to the explicit standard or screening learning-curve workflow."""

    if budget_mode == "full_remainder_test":
        return run_standard_learning_curve_experiments(
            dataset,
            min_train=min_train,
            max_train=max_train,
            step=step,
            n_repeats=n_repeats,
            seed=seed,
            enabled_model_names=enabled_model_names,
            model_cfg=model_cfg,
            validation_fraction=validation_fraction,
            min_val_size=min_val_size,
            min_tuning_val_size=min_tuning_val_size,
            min_inner_train_size=min_inner_train_size,
            min_test_size=min_test_size,
            model_families=model_families,
            requested_sweep_sizes_by_method=requested_sweep_sizes_by_method,
        )
    if budget_mode == "screening_fraction":
        if screen_fraction is None:
            raise ValueError(
                "screen_fraction must be provided when budget_mode='screening_fraction'."
            )
        return run_screening_learning_curve_experiments(
            dataset,
            min_train=min_train,
            max_train=max_train,
            step=step,
            n_repeats=n_repeats,
            seed=seed,
            enabled_model_names=enabled_model_names,
            model_cfg=model_cfg,
            screen_fraction=screen_fraction,
            min_screen_size=min_screen_size,
            validation_fraction=validation_fraction,
            min_val_size=min_val_size,
            min_tuning_val_size=min_tuning_val_size,
            min_inner_train_size=min_inner_train_size,
            min_test_size=min_test_size,
            model_families=model_families,
            requested_sweep_sizes_by_method=requested_sweep_sizes_by_method,
        )
    raise ValueError(f"unsupported budget_mode: {budget_mode!r}")


def run_learning_curve_experiments_from_frame(
    df: Any,
    *,
    min_train: int,
    max_train: int,
    n_repeats: int,
    seed: int = 42,
    enabled_model_names: Sequence[str] | None = None,
    graph_view: GraphDatasetView | None = None,
    graph_join_key: str = "reaction",
    budget_mode: LearningCurveBudgetMode = "full_remainder_test",
    screen_fraction: float | None = None,
    min_screen_size: int = 1,
    validation_fraction: float = 0.2,
    min_val_size: int = 1,
    min_tuning_val_size: int = 1,
    min_inner_train_size: int = 1,
    min_test_size: int = 1,
    model_families: Sequence[Any] | None = None,
) -> LearningCurveResults:
    _validate_learning_curve_frame(df)
    dataset = assemble_learning_curve_dataset_from_frame(
        df,
        graph_view=graph_view,
        graph_join_key=graph_join_key,
    )

    return run_learning_curve_experiments(
        dataset,
        min_train=min_train,
        max_train=max_train,
        n_repeats=n_repeats,
        seed=seed,
        enabled_model_names=enabled_model_names,
        budget_mode=budget_mode,
        screen_fraction=screen_fraction,
        min_screen_size=min_screen_size,
        validation_fraction=validation_fraction,
        min_val_size=min_val_size,
        min_tuning_val_size=min_tuning_val_size,
        min_inner_train_size=min_inner_train_size,
        min_test_size=min_test_size,
        model_families=model_families,
    )


def _metadata_for_available_results(
    expected_metadata: Any,
    results: LearningCurveResults,
    existing_metadata: LearningCurveSweepMetadata | None = None,
) -> LearningCurveSweepMetadata:
    candidate_methods = set(learning_curve_method_names())
    if existing_metadata is not None:
        candidate_methods.update(existing_metadata.enabled_models)
    enabled_models = tuple(
        method_name
        for method_name in sorted(candidate_methods)
        if learning_curve_results_has_method(results, method_name)
    )
    return LearningCurveSweepMetadata(
        seed=expected_metadata.seed,
        min_train=expected_metadata.min_train,
        max_train=expected_metadata.max_train,
        step=expected_metadata.step,
        n_repeats=expected_metadata.n_repeats,
        enabled_models=enabled_models,
        budget_mode=expected_metadata.budget_mode,
        screen_fraction=expected_metadata.screen_fraction,
        min_screen_size=expected_metadata.min_screen_size,
        dataset_tag=expected_metadata.dataset_tag,
        dataset_size=expected_metadata.dataset_size,
        mlip_feature_names=expected_metadata.mlip_feature_names,
        adsorbate_filter=expected_metadata.adsorbate_filter,
        anomaly_filter=expected_metadata.anomaly_filter,
        reaction_contains_filter=expected_metadata.reaction_contains_filter,
    )


def run_learning_curve_experiments_from_config(
    df: Any,
    cfg: Config | None,
    *,
    graph_view: GraphDatasetView | None = None,
    model_families: Sequence[Any] | None = None,
    auxiliary_views: dict[str, Any] | None = None,
) -> LearningCurveResults:
    from oasis.learning_curve.registry import (
        default_sweep_model_families,
        enabled_learning_curve_model_names_from_config,
    )

    experiment_cfg = cfg.experiment.learning_curve if cfg and cfg.experiment else None
    model_cfg = experiment_cfg.models if experiment_cfg else None
    _validate_learning_curve_frame(df)
    current_mlip_feature_names = mlip_feature_names(df)
    dataset = build_sweep_dataset_from_config(
        df, cfg, graph_view=graph_view, auxiliary_views=auxiliary_views
    )
    enabled_model_names = enabled_learning_curve_model_names_from_config(model_cfg)
    available_families = tuple(
        model_families
        if model_families is not None
        else default_sweep_model_families(enabled_model_names, config=model_cfg)
    )
    results_bundle_path = (
        getattr(experiment_cfg, "results_bundle_path", None)
        if experiment_cfg is not None
        else None
    )
    reuse_results = (
        bool(getattr(experiment_cfg, "reuse_results", False))
        if experiment_cfg is not None
        else False
    )
    configured_force_refresh_train_sizes = (
        getattr(experiment_cfg, "force_refresh_train_sizes", {})
        if experiment_cfg is not None
        else {}
    )
    force_refresh_methods = {
        method_name
        for method_name in getattr(experiment_cfg, "force_refresh_methods", ())
        if method_name in enabled_model_names
    }
    force_refresh_train_sizes = {
        method_name: {int(value) for value in sweep_sizes}
        for method_name, sweep_sizes in configured_force_refresh_train_sizes.items()
        if method_name in enabled_model_names
    }

    cached_results = LearningCurveResults.empty()
    cached_method_names: set[str] = set()
    families_to_run: Sequence[Any] | None = available_families
    enabled_model_names_to_run = enabled_model_names
    missing_sweep_sizes_by_method: dict[str, tuple[int, ...]] = {}
    if (
        cfg is not None
        and experiment_cfg is not None
        and reuse_results
        and results_bundle_path is not None
    ):
        expected_metadata = learning_curve_sweep_metadata_from_config(
            cfg,
            dataset_size=len(df),
            mlip_feature_names=current_mlip_feature_names,
        )
        if Path(results_bundle_path).is_file():
            try:
                cached_results = select_learning_curve_results_methods(
                    load_learning_curve_results_artifact(
                        results_bundle_path,
                        expected_metadata=expected_metadata,
                        ignore_enabled_models=True,
                        ignore_train_grid=True,
                        ignore_repeat_count=True,
                    ).results,
                    enabled_model_names,
                )
            except ValueError:
                cached_results = LearningCurveResults.empty()
        if force_refresh_methods or force_refresh_train_sizes:
            for method_name in force_refresh_methods:
                cached_results = select_learning_curve_results_methods(
                    cached_results,
                    tuple(
                        enabled_method_name
                        for enabled_method_name in enabled_model_names
                        if enabled_method_name != method_name
                    ),
                )
            for method_name, sweep_sizes in force_refresh_train_sizes.items():
                if method_name in force_refresh_methods:
                    continue
                cached_results = _remove_method_train_sizes_from_results(
                    cached_results,
                    method_name,
                    sweep_sizes,
                )
        if available_families:
            selected_families: list[Any] = []
            for family in available_families:
                method_name = _family_method_name(
                    family,
                    learning_curve_method_name_for_result_field,
                )
                if method_name is None or method_name not in enabled_model_names:
                    selected_families.append(family)
                    continue
                split_collection = _build_family_split_collection(
                    dataset,
                    family,
                    min_train=experiment_cfg.min_train,
                    max_train=experiment_cfg.max_train,
                    step=getattr(experiment_cfg, "step", 1),
                    n_repeats=experiment_cfg.n_repeats,
                    seed=cfg.seed if cfg.seed is not None else 42,
                    budget_mode=getattr(
                        experiment_cfg, "budget_mode", "full_remainder_test"
                    ),
                    screen_fraction=getattr(experiment_cfg, "screen_fraction", None),
                    min_screen_size=getattr(experiment_cfg, "min_screen_size", 1),
                    validation_fraction=getattr(
                        experiment_cfg, "validation_fraction", 0.2
                    ),
                    min_val_size=getattr(experiment_cfg, "min_val_size", 1),
                    min_tuning_val_size=getattr(
                        experiment_cfg, "min_tuning_val_size", 1
                    ),
                    min_inner_train_size=getattr(
                        experiment_cfg, "min_inner_train_size", 1
                    ),
                    min_test_size=getattr(experiment_cfg, "min_test_size", 1),
                )
                requested_sizes = _unique_sweep_sizes(split_collection)
                cached_sizes = _result_frame_train_sizes(
                    cached_results,
                    method_name,
                    learning_curve_result_field_for_method_name,
                )
                missing_sizes = tuple(
                    sweep_size
                    for sweep_size in requested_sizes
                    if sweep_size not in cached_sizes
                )
                refresh_sizes = tuple(
                    sweep_size
                    for sweep_size in requested_sizes
                    if sweep_size in force_refresh_train_sizes.get(method_name, set())
                )
                requested_run_sizes = tuple(
                    sorted(dict.fromkeys((*missing_sizes, *refresh_sizes)))
                )
                if requested_run_sizes:
                    missing_sweep_sizes_by_method[method_name] = requested_run_sizes
                    selected_families.append(family)
                else:
                    cached_method_names.add(method_name)
            families_to_run = tuple(selected_families)
            enabled_model_names_to_run = tuple(
                method_name
                for method_name in enabled_model_names
                if method_name not in cached_method_names
            )

    fresh_results = LearningCurveResults.empty()
    if (families_to_run is None and enabled_model_names_to_run) or families_to_run:
        fresh_results = run_learning_curve_experiments(
            dataset,
            min_train=experiment_cfg.min_train if experiment_cfg else 5,
            max_train=experiment_cfg.max_train if experiment_cfg else 10,
            step=getattr(experiment_cfg, "step", 1) if experiment_cfg else 1,
            n_repeats=experiment_cfg.n_repeats if experiment_cfg else 50,
            seed=cfg.seed if cfg and cfg.seed is not None else 42,
            enabled_model_names=enabled_model_names_to_run,
            model_cfg=model_cfg,
            budget_mode=(
                getattr(experiment_cfg, "budget_mode", "full_remainder_test")
                if experiment_cfg
                else "full_remainder_test"
            ),
            screen_fraction=(
                getattr(experiment_cfg, "screen_fraction", None)
                if experiment_cfg
                else None
            ),
            min_screen_size=(
                getattr(experiment_cfg, "min_screen_size", 1)
                if experiment_cfg
                else 1
            ),
            validation_fraction=(
                getattr(experiment_cfg, "validation_fraction", 0.2)
                if experiment_cfg
                else 0.2
            ),
            min_val_size=(
                getattr(experiment_cfg, "min_val_size", 1) if experiment_cfg else 1
            ),
            min_tuning_val_size=(
                getattr(experiment_cfg, "min_tuning_val_size", 1)
                if experiment_cfg
                else 1
            ),
            min_inner_train_size=(
                getattr(experiment_cfg, "min_inner_train_size", 1)
                if experiment_cfg
                else 1
            ),
            min_test_size=(
                getattr(experiment_cfg, "min_test_size", 1) if experiment_cfg else 1
            ),
            model_families=families_to_run,
            requested_sweep_sizes_by_method=missing_sweep_sizes_by_method or None,
        )

    results = cached_results.merge(fresh_results)
    if cfg is not None and experiment_cfg is not None:
        expected_metadata = learning_curve_sweep_metadata_from_config(
            cfg,
            dataset_size=len(df),
            mlip_feature_names=current_mlip_feature_names,
        )
        if results_bundle_path is not None:
            existing_bundle_results = LearningCurveResults.empty()
            existing_bundle_point_provenance: dict[str, pd.DataFrame] = {}
            existing_bundle_metadata = expected_metadata
            if Path(results_bundle_path).is_file():
                try:
                    existing_bundle = load_learning_curve_results_artifact(
                        results_bundle_path,
                        expected_metadata=expected_metadata,
                        ignore_enabled_models=True,
                        ignore_train_grid=True,
                        ignore_repeat_count=True,
                    )
                    existing_bundle_results = existing_bundle.results
                    existing_bundle_point_provenance = existing_bundle.point_provenance
                    existing_bundle_metadata = existing_bundle.metadata
                except ValueError:
                    pass
            overwrite_fields = {
                field_name
                for method_name in force_refresh_methods
                for field_name in (
                    learning_curve_result_field_for_method_name(method_name),
                    learning_curve_selection_field_for_method_name(method_name),
                    learning_curve_uq_field_for_method_name(method_name),
                )
                if field_name is not None
            }
            overwrite_train_sizes_by_field = {
                field_name: {
                    int(value)
                    for value in force_refresh_train_sizes.get(method_name, set())
                }
                for method_name in force_refresh_train_sizes
                for field_name in (
                    learning_curve_result_field_for_method_name(method_name),
                    learning_curve_selection_field_for_method_name(method_name),
                    learning_curve_uq_field_for_method_name(method_name),
                )
                if field_name is not None
            }
            bundle_results = existing_bundle_results.merge(
                fresh_results,
                overwrite_fields=overwrite_fields,
                overwrite_train_sizes_by_field=overwrite_train_sizes_by_field,
            )
            bundle_point_provenance = merge_learning_curve_point_provenance(
                existing_bundle_point_provenance,
                build_learning_curve_point_provenance(
                    fresh_results,
                    expected_metadata,
                ),
                overwrite_fields=overwrite_fields,
                overwrite_train_sizes_by_field=overwrite_train_sizes_by_field,
            )
            results = select_learning_curve_results_methods(
                bundle_results,
                enabled_model_names,
            )
            save_learning_curve_results_artifact(
                bundle_results,
                _metadata_for_available_results(
                    expected_metadata,
                    bundle_results,
                    existing_bundle_metadata,
                ),
                results_bundle_path,
                point_provenance=bundle_point_provenance,
            )
    return results


def load_or_run_learning_curve_results_from_config(
    df: Any,
    cfg: Config | None,
    *,
    graph_view: GraphDatasetView | None = None,
    model_families: Sequence[Any] | None = None,
    auxiliary_views: dict[str, Any] | None = None,
) -> LearningCurveResults:
    from oasis.learning_curve.registry import (
        default_sweep_model_families,
        enabled_learning_curve_model_names_from_config,
    )

    experiment_cfg = cfg.experiment.learning_curve if cfg and cfg.experiment else None
    results_bundle_path = (
        getattr(experiment_cfg, "results_bundle_path", None)
        if experiment_cfg is not None
        else None
    )
    reuse_results = (
        bool(getattr(experiment_cfg, "reuse_results", False))
        if experiment_cfg is not None
        else False
    )
    force_refresh_train_sizes = (
        getattr(experiment_cfg, "force_refresh_train_sizes", {})
        if experiment_cfg is not None
        else {}
    )
    force_refresh_methods = {
        method_name
        for method_name in getattr(experiment_cfg, "force_refresh_methods", ())
        if method_name in enabled_learning_curve_model_names_from_config(
            experiment_cfg.models
        )
    }
    if (
        cfg is not None
        and experiment_cfg is not None
        and reuse_results
        and not force_refresh_methods
        and not force_refresh_train_sizes
        and results_bundle_path is not None
    ):
        expected_metadata = learning_curve_sweep_metadata_from_config(
            cfg,
            dataset_size=len(df),
            mlip_feature_names=mlip_feature_names(df),
        )
        enabled_model_names = expected_metadata.enabled_models
        available_families = tuple(
            model_families
            if model_families is not None
            else default_sweep_model_families(
                enabled_model_names,
                config=experiment_cfg.models,
            )
        )
        if Path(results_bundle_path).is_file():
            dataset = build_sweep_dataset_from_config(
                df,
                cfg,
                graph_view=graph_view,
                auxiliary_views=auxiliary_views,
            )
            try:
                cached_results = select_learning_curve_results_methods(
                    load_learning_curve_results_artifact(
                        results_bundle_path,
                        expected_metadata=expected_metadata,
                        ignore_enabled_models=True,
                        ignore_train_grid=True,
                        ignore_repeat_count=True,
                    ).results,
                    enabled_model_names,
                )
            except ValueError:
                cached_results = LearningCurveResults.empty()
            all_methods_fully_cached = True
            for family in available_families:
                method_name = _family_method_name(
                    family,
                    learning_curve_method_name_for_result_field,
                )
                if method_name is None or method_name not in enabled_model_names:
                    continue
                requested_sizes = _unique_sweep_sizes(
                    _build_family_split_collection(
                        dataset,
                        family,
                        min_train=experiment_cfg.min_train,
                        max_train=experiment_cfg.max_train,
                        step=getattr(experiment_cfg, "step", 1),
                        n_repeats=experiment_cfg.n_repeats,
                        seed=cfg.seed if cfg.seed is not None else 42,
                        budget_mode=getattr(
                            experiment_cfg, "budget_mode", "full_remainder_test"
                        ),
                        screen_fraction=getattr(experiment_cfg, "screen_fraction", None),
                        min_screen_size=getattr(experiment_cfg, "min_screen_size", 1),
                        validation_fraction=getattr(
                            experiment_cfg, "validation_fraction", 0.2
                        ),
                        min_val_size=getattr(experiment_cfg, "min_val_size", 1),
                        min_tuning_val_size=getattr(
                            experiment_cfg, "min_tuning_val_size", 1
                        ),
                        min_inner_train_size=getattr(
                            experiment_cfg, "min_inner_train_size", 1
                        ),
                        min_test_size=getattr(experiment_cfg, "min_test_size", 1),
                    )
                )
                cached_sizes = _result_frame_train_sizes(
                    cached_results,
                    method_name,
                    learning_curve_result_field_for_method_name,
                )
                if any(size not in cached_sizes for size in requested_sizes):
                    all_methods_fully_cached = False
                    break
            if all_methods_fully_cached:
                return cached_results

    return run_learning_curve_experiments_from_config(
        df,
        cfg,
        graph_view=graph_view,
        model_families=model_families,
        auxiliary_views=auxiliary_views,
    )
