from __future__ import annotations

from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from oasis.learning_curve.results_io import (
    LearningCurveSweepMetadata,
    learning_curve_method_names,
    learning_curve_results_has_method,
)
from oasis.sweep import (
    GraphDatasetView,
    LearningCurveResults,
    SweepDataset,
    SweepDatasetInputs,
    SweepFamilyRequirements,
    SweepRunPayload,
    SweepSplit,
    SweepSplitCollection,
)

if TYPE_CHECKING:
    from oasis.config import Config


@dataclass(frozen=True, slots=True)
class ParityPlotData:
    reference: np.ndarray
    predictions: Mapping[str, np.ndarray]


def inner_validation_size_for_sweep(
    sweep_size: int,
    *,
    validation_fraction: float,
    min_val_size: int,
    min_tuning_val_size: int = 1,
) -> int:
    """Return the inner validation size for one outer-train budget.

    This policy consumes part of the requested sweep budget for validation via
    `max(min_val_size, min_tuning_val_size, floor(validation_fraction * sweep_size))`.
    The remainder stays available for inner training; outer test samples are not
    involved in this calculation.
    """

    if sweep_size <= 0:
        raise ValueError("sweep_size must be positive.")
    if validation_fraction < 0:
        raise ValueError("validation_fraction must be non-negative.")
    if min_val_size <= 0:
        raise ValueError("min_val_size must be positive.")
    if min_tuning_val_size <= 0:
        raise ValueError("min_tuning_val_size must be positive.")
    return max(
        min_val_size,
        min_tuning_val_size,
        math.floor(validation_fraction * sweep_size),
    )


def validation_size_if_sweep_feasible(
    sweep_size: int,
    *,
    validation_fraction: float,
    min_val_size: int,
    min_tuning_val_size: int = 1,
    min_inner_train_size: int = 1,
) -> int | None:
    """Return the validation size for a feasible validation-aware sweep point.

    If the requested sweep budget cannot satisfy both the validation policy and
    the minimum remaining inner-train budget, return `None`.
    """

    if min_inner_train_size <= 0:
        raise ValueError("min_inner_train_size must be positive.")

    n_val = inner_validation_size_for_sweep(
        sweep_size,
        validation_fraction=validation_fraction,
        min_val_size=min_val_size,
        min_tuning_val_size=min_tuning_val_size,
    )
    if n_val + min_inner_train_size > sweep_size:
        return None
    return n_val

def generate_sweep_splits(
    n_samples: int,
    min_train: int,
    max_train: int,
    n_repeats: int,
    rng: np.random.Generator,
    *,
    step: int = 1,
    min_test_size: int = 1,
) -> Iterator[SweepSplit]:
    """Yield repeated outer train/test splits for each sweep size in the range.

    These splits do not include inner validation. The entire `train_idx` budget
    is available for fitting, and `test_idx` is reserved for outer evaluation.
    """

    if min_test_size <= 0:
        raise ValueError("min_test_size must be positive.")

    idx = np.arange(n_samples)
    max_train = min(max_train, n_samples - min_test_size)
    for n_train in range(min_train, max_train + 1, step):
        for _ in range(n_repeats):
            train_idx = rng.choice(idx, size=n_train, replace=False)
            test_idx = np.setdiff1d(idx, train_idx, assume_unique=False)
            yield SweepSplit(
                sweep_size=n_train,
                train_idx=train_idx,
                test_idx=test_idx,
            )


def generate_sweep_splits_with_validation(
    n_samples: int,
    min_train: int,
    max_train: int,
    n_val: int,
    n_repeats: int,
    rng: np.random.Generator,
    *,
    min_inner_train_size: int = 1,
    min_test_size: int = 1,
) -> Iterator[SweepSplit]:
    """Yield repeated outer train/test splits with inner train/val partitions.

    `sweep_size` is the outer training budget. For each split, `test_idx` is the
    outer holdout used only for final evaluation, while `train_idx` and `val_idx`
    partition that outer training budget for fitting and model selection.
    Selection-aware methods therefore see no information from `test_idx` while
    choosing a candidate configuration.
    """

    if n_val <= 0:
        raise ValueError("n_val must be positive.")
    if n_val >= n_samples:
        raise ValueError("n_val must be smaller than n_samples.")
    if min_inner_train_size <= 0:
        raise ValueError("min_inner_train_size must be positive.")
    if min_test_size <= 0:
        raise ValueError("min_test_size must be positive.")

    idx = np.arange(n_samples)
    max_train = min(max_train, n_samples - min_test_size)
    for n_train in range(max(min_train, n_val + min_inner_train_size), max_train + 1):
        for _ in range(n_repeats):
            outer_train_idx = rng.choice(idx, size=n_train, replace=False)
            test_idx = np.setdiff1d(idx, outer_train_idx, assume_unique=False)
            val_idx = rng.choice(outer_train_idx, size=n_val, replace=False)
            train_idx = np.setdiff1d(outer_train_idx, val_idx, assume_unique=False)
            yield SweepSplit(
                sweep_size=n_train,
                train_idx=train_idx,
                test_idx=test_idx,
                val_idx=val_idx,
            )


def generate_inner_validation_sweep_splits(
    n_samples: int,
    min_train: int,
    max_train: int,
    n_repeats: int,
    rng: np.random.Generator,
    *,
    step: int = 1,
    validation_fraction: float = 0.2,
    min_val_size: int = 1,
    min_tuning_val_size: int = 1,
    min_inner_train_size: int = 1,
    min_test_size: int = 1,
) -> Iterator[SweepSplit]:
    """Yield sweep splits with policy-sized inner validation holdouts.

    This is the validation-aware learning-curve path: the outer test split stays
    untouched during model selection, and validation is taken from within the
    requested training budget.

    Sweep sizes that cannot support both validation and at least
    `min_inner_train_size` inner-train samples are skipped.
    """
    if min_inner_train_size <= 0:
        raise ValueError("min_inner_train_size must be positive.")
    if min_test_size <= 0:
        raise ValueError("min_test_size must be positive.")

    max_train = min(max_train, n_samples - min_test_size)
    for sweep_size in range(min_train, max_train + 1, step):
        n_val = validation_size_if_sweep_feasible(
            sweep_size,
            validation_fraction=validation_fraction,
            min_val_size=min_val_size,
            min_tuning_val_size=min_tuning_val_size,
            min_inner_train_size=min_inner_train_size,
        )
        if n_val is None:
            continue
        yield from generate_sweep_splits_with_validation(
            n_samples=n_samples,
            min_train=sweep_size,
            max_train=sweep_size,
            n_val=n_val,
            n_repeats=n_repeats,
            rng=rng,
            min_inner_train_size=min_inner_train_size,
            min_test_size=min_test_size,
        )


def build_sweep_split_collection(
    n_samples: int,
    *,
    min_train: int,
    max_train: int,
    step: int = 1,
    n_repeats: int,
    seed: int,
    requirements: SweepFamilyRequirements | None = None,
    validation_fraction: float = 0.2,
    min_val_size: int = 1,
    min_tuning_val_size: int = 1,
    min_inner_train_size: int = 1,
    min_test_size: int = 1,
) -> SweepSplitCollection:
    """Build the split collection for one family under its split requirements.

    Train/test-only families receive outer train/test splits across the requested
    sweep range. Validation-aware families receive only sweep sizes large enough
    to support inner train/val partitioning while keeping `test_idx` as the
    outer evaluation holdout.

    The effective minimum sweep size is the maximum of:
    - the caller's requested `min_train`
    - the family's declared `min_train_size`
    - for validation-aware families, the smallest sweep size that can satisfy
      `min_val_size` and `min_tuning_val_size` while still leaving at least
      `min_inner_train_size` inner-train samples

    The feasible sweep range is also bounded above by `n_samples - min_test_size`
    so that every emitted split leaves at least `min_test_size` samples in the
    outer test set. If the combined train/validation/test minima leave no
    feasible sweep points, an empty collection is returned deterministically.
    """

    requirements = requirements or SweepFamilyRequirements()
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if min_train <= 0:
        raise ValueError("min_train must be positive.")
    if max_train <= 0:
        raise ValueError("max_train must be positive.")
    if n_repeats <= 0:
        raise ValueError("n_repeats must be positive.")
    if min_test_size <= 0:
        raise ValueError("min_test_size must be positive.")
    if min_val_size <= 0:
        raise ValueError("min_val_size must be positive.")
    if min_tuning_val_size <= 0:
        raise ValueError("min_tuning_val_size must be positive.")
    if min_inner_train_size <= 0:
        raise ValueError("min_inner_train_size must be positive.")

    feasible_max_train = min(max_train, n_samples - min_test_size)
    effective_min_train = max(min_train, requirements.min_train_size)
    if feasible_max_train < effective_min_train:
        return SweepSplitCollection(
            splits=(),
            planning_requirements=requirements,
        )

    rng = np.random.default_rng(seed)
    if requirements.requires_inner_validation:
        first_feasible_train = next(
            (
                sweep_size
                for sweep_size in range(effective_min_train, feasible_max_train + 1, step)
                if validation_size_if_sweep_feasible(
                    sweep_size,
                    validation_fraction=validation_fraction,
                    min_val_size=min_val_size,
                    min_tuning_val_size=min_tuning_val_size,
                    min_inner_train_size=min_inner_train_size,
                )
                is not None
            ),
            None,
        )
        if first_feasible_train is None:
            return SweepSplitCollection(
                splits=(),
                planning_requirements=requirements,
            )
        effective_min_train = first_feasible_train
        splits = tuple(
            generate_inner_validation_sweep_splits(
                n_samples,
                effective_min_train,
                feasible_max_train,
                n_repeats,
                rng,
                step=step,
                validation_fraction=validation_fraction,
                min_val_size=min_val_size,
                min_tuning_val_size=min_tuning_val_size,
                min_inner_train_size=min_inner_train_size,
                min_test_size=min_test_size,
            )
        )
    else:
        splits = tuple(
            generate_sweep_splits(
                n_samples,
                effective_min_train,
                feasible_max_train,
                n_repeats,
                rng,
                step=step,
                min_test_size=min_test_size,
            )
        )
    return SweepSplitCollection(
        splits=splits,
        planning_requirements=requirements,
    )


def mlip_columns(df: Any) -> list[str]:
    return [c for c in df.columns if c.endswith("_mlip_ads_eng_median")]


def column_to_numpy(df: Any, col: str) -> np.ndarray:
    series = df[col]
    if hasattr(series, "to_numpy"):
        return series.to_numpy()
    return np.asarray(series)


def prepare_parity_plot_data(df: Any) -> ParityPlotData:
    mlip_cols = mlip_columns(df)
    if not mlip_cols:
        raise ValueError(
            "No MLIP prediction columns found (expected *_mlip_ads_eng_median)."
        )
    if len(df) == 0:
        raise ValueError("No data available to plot.")

    return ParityPlotData(
        reference=column_to_numpy(df, "reference_ads_eng"),
        predictions={
            col.removesuffix("_mlip_ads_eng_median"): column_to_numpy(df, col)
            for col in mlip_cols
        },
    )


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
        validation_fraction=validation_fraction,
        min_val_size=min_val_size,
        min_tuning_val_size=min_tuning_val_size,
        min_inner_train_size=min_inner_train_size,
        min_test_size=min_test_size,
        model_families=model_families,
    )


def assemble_learning_curve_dataset_from_frame(
    df: Any,
    *,
    graph_view: GraphDatasetView | None = None,
    graph_join_key: str = "reaction",
    auxiliary_views: dict[str, Any] | None = None,
) -> SweepDataset:
    return build_sweep_dataset_from_frame(
        df,
        graph_view=graph_view,
        graph_join_key=graph_join_key,
        auxiliary_views=auxiliary_views,
    )


def build_sweep_dataset_from_frame(
    df: Any,
    *,
    graph_view: GraphDatasetView | None = None,
    graph_join_key: str = "reaction",
    auxiliary_views: dict[str, Any] | None = None,
) -> SweepDataset:
    if graph_view is not None:
        from oasis.graphs import build_graph_sweep_dataset

        return build_graph_sweep_dataset(
            df, graph_view, join_key=graph_join_key, auxiliary_views=auxiliary_views
        )

    feature_cols = mlip_columns(df)
    if hasattr(df, "select"):
        X = df.select(feature_cols).to_numpy()
    else:
        X = np.column_stack([column_to_numpy(df, col) for col in feature_cols])
    y = column_to_numpy(df, "reference_ads_eng")
    sample_ids = (
        column_to_numpy(df, "reaction")
        if "reaction" in getattr(df, "columns", ())
        else None
    )
    return SweepDataset.from_inputs(
        inputs=SweepDatasetInputs(
            mlip_features=X,
        ),
        targets=y,
        sample_ids=sample_ids,
        auxiliary_views=auxiliary_views,
    )


def build_sweep_dataset_from_config(
    df: Any,
    cfg: Config | None,
    *,
    graph_view: GraphDatasetView | None = None,
    auxiliary_views: dict[str, Any] | None = None,
) -> SweepDataset:
    experiment_cfg = cfg.experiment.learning_curve if cfg and cfg.experiment else None
    graph_dataset_cfg = getattr(experiment_cfg, "graph_dataset", None)
    graph_join_key = (
        graph_dataset_cfg.join_key if graph_dataset_cfg is not None else "reaction"
    )
    if (
        graph_view is None
        and graph_dataset_cfg is not None
        and Path(graph_dataset_cfg.path).is_file()
    ):
        from oasis.graphs import load_sweep_dataset_from_graph_artifact

        return load_sweep_dataset_from_graph_artifact(
            graph_dataset_cfg.path,
            join_key=graph_join_key,
            auxiliary_views=auxiliary_views,
            filter_df=df,
        )
    if graph_view is None and experiment_cfg:
        from oasis.graphs import load_configured_graph_dataset_view

        graph_view = load_configured_graph_dataset_view(graph_dataset_cfg)

    return assemble_learning_curve_dataset_from_frame(
        df,
        graph_view=graph_view,
        graph_join_key=graph_join_key,
        auxiliary_views=auxiliary_views,
    )


def run_learning_curve_experiments_from_config(
    df: Any,
    cfg: Config | None,
    *,
    graph_view: GraphDatasetView | None = None,
    model_families: Sequence[Any] | None = None,
    auxiliary_views: dict[str, Any] | None = None,
) -> LearningCurveResults:
    from oasis.learning_curve.registry import enabled_learning_curve_model_names_from_config
    from oasis.learning_curve.results_io import (
        LearningCurveSweepMetadata,
        learning_curve_method_names,
        learning_curve_sweep_metadata_from_config,
        learning_curve_method_name_for_result_field,
        learning_curve_results_has_method,
        load_learning_curve_results_artifact,
        save_learning_curve_results_artifact,
        select_learning_curve_results_methods,
    )

    experiment_cfg = cfg.experiment.learning_curve if cfg and cfg.experiment else None
    model_cfg = experiment_cfg.models if experiment_cfg else None
    _validate_learning_curve_frame(df)
    dataset = build_sweep_dataset_from_config(
        df, cfg, graph_view=graph_view, auxiliary_views=auxiliary_views
    )
    enabled_model_names = enabled_learning_curve_model_names_from_config(model_cfg)
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
    force_refresh_methods = {
        method_name
        for method_name in getattr(experiment_cfg, "force_refresh_methods", ())
        if method_name in enabled_model_names
    }

    cached_results = LearningCurveResults.empty()
    cached_method_names: set[str] = set()
    families_to_run = model_families
    enabled_model_names_to_run = enabled_model_names
    if (
        cfg is not None
        and experiment_cfg is not None
        and reuse_results
        and results_bundle_path is not None
    ):
        expected_metadata = learning_curve_sweep_metadata_from_config(cfg)
        if Path(results_bundle_path).is_file():
            cached_results = select_learning_curve_results_methods(
                load_learning_curve_results_artifact(
                    results_bundle_path,
                    expected_metadata=expected_metadata,
                    ignore_enabled_models=True,
                ).results,
                enabled_model_names,
            )
        cached_method_names = {
            method_name
            for method_name in enabled_model_names
            if method_name not in force_refresh_methods
            if learning_curve_results_has_method(cached_results, method_name)
        }
        enabled_model_names_to_run = tuple(
            method_name
            for method_name in enabled_model_names
            if method_name not in cached_method_names
        )
        if model_families is not None:
            families_to_run = tuple(
                family
                for family in model_families
                if _family_method_name(family, learning_curve_method_name_for_result_field)
                not in cached_method_names
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
        )

    results = cached_results.merge(fresh_results)
    if cfg is not None and experiment_cfg is not None:
        expected_metadata = learning_curve_sweep_metadata_from_config(cfg)
        if results_bundle_path is not None:
            existing_bundle_results = LearningCurveResults.empty()
            existing_bundle_metadata = expected_metadata
            if Path(results_bundle_path).is_file():
                try:
                    existing_bundle = load_learning_curve_results_artifact(
                        results_bundle_path,
                        expected_metadata=expected_metadata,
                        ignore_enabled_models=True,
                    )
                    existing_bundle_results = existing_bundle.results
                    existing_bundle_metadata = existing_bundle.metadata
                except ValueError:
                    pass
            bundle_results = existing_bundle_results.merge(results)
            save_learning_curve_results_artifact(
                bundle_results,
                _metadata_for_available_results(
                    expected_metadata,
                    bundle_results,
                    existing_bundle_metadata,
                ),
                results_bundle_path,
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
    from oasis.learning_curve.registry import enabled_learning_curve_model_names_from_config
    from oasis.learning_curve.results_io import (
        learning_curve_results_has_method,
        load_learning_curve_results_artifact,
        learning_curve_sweep_metadata_from_config,
        select_learning_curve_results_methods,
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
        and results_bundle_path is not None
    ):
        expected_metadata = learning_curve_sweep_metadata_from_config(cfg)
        enabled_model_names = expected_metadata.enabled_models
        if Path(results_bundle_path).is_file():
            cached_results = select_learning_curve_results_methods(
                load_learning_curve_results_artifact(
                    results_bundle_path,
                    expected_metadata=expected_metadata,
                    ignore_enabled_models=True,
                ).results,
                enabled_model_names,
            )
            if all(
                learning_curve_results_has_method(cached_results, method_name)
                for method_name in enabled_model_names
            ):
                return cached_results

    return run_learning_curve_experiments_from_config(
        df,
        cfg,
        graph_view=graph_view,
        model_families=model_families,
        auxiliary_views=auxiliary_views,
    )


def _validate_learning_curve_frame(df: Any) -> None:
    feature_cols = mlip_columns(df)
    if not feature_cols:
        raise ValueError(
            "No MLIP prediction columns found (expected *_mlip_ads_eng_median)."
        )

    n_rows = getattr(df, "height", len(df))
    if n_rows <= 5:
        raise ValueError("Not enough data to evaluate (need >5 samples).")


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
        adsorbate_filter=expected_metadata.adsorbate_filter,
        anomaly_filter=expected_metadata.anomaly_filter,
        reaction_contains_filter=expected_metadata.reaction_contains_filter,
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
    validation_fraction: float = 0.2,
    min_val_size: int = 1,
    min_tuning_val_size: int = 1,
    min_inner_train_size: int = 1,
    min_test_size: int = 1,
    model_families: Sequence[Any] | None = None,
) -> LearningCurveResults:
    families = model_families
    if families is None:
        from oasis.learning_curve.registry import default_sweep_model_families

        families = default_sweep_model_families(enabled_model_names, config=model_cfg)

    results = LearningCurveResults.empty()
    for family in families:
        requirements = (
            family.capabilities().to_requirements()
            if hasattr(family, "capabilities")
            else family.requirements()
            if hasattr(family, "requirements")
            else SweepFamilyRequirements()
        )
        payload = SweepRunPayload(
            dataset=dataset,
            split_collection=build_sweep_split_collection(
                dataset.n_samples,
                min_train=min_train,
                max_train=max_train,
                step=step,
                n_repeats=n_repeats,
                seed=seed,
                requirements=requirements,
                validation_fraction=validation_fraction,
                min_val_size=min_val_size,
                min_tuning_val_size=min_tuning_val_size,
                min_inner_train_size=min_inner_train_size,
                min_test_size=min_test_size,
            ),
        )
        results = results.merge(family.run(payload))
    return results
