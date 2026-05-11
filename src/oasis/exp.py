from __future__ import annotations

from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
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
) -> int:
    """Return the inner validation size for one outer-train budget.

    This policy consumes part of the requested sweep budget for validation via
    `max(min_val_size, floor(validation_fraction * sweep_size))`. The remainder
    stays available for inner training; outer test samples are not involved in
    this calculation.
    """

    if sweep_size <= 0:
        raise ValueError("sweep_size must be positive.")
    if validation_fraction < 0:
        raise ValueError("validation_fraction must be non-negative.")
    if min_val_size <= 0:
        raise ValueError("min_val_size must be positive.")
    return max(min_val_size, math.floor(validation_fraction * sweep_size))

def generate_sweep_splits(
    n_samples: int,
    min_train: int,
    max_train: int,
    n_repeats: int,
    rng: np.random.Generator,
    *,
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
    for n_train in range(min_train, max_train + 1):
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
    if min_test_size <= 0:
        raise ValueError("min_test_size must be positive.")

    idx = np.arange(n_samples)
    max_train = min(max_train, n_samples - min_test_size)
    for n_train in range(max(min_train, n_val + 1), max_train + 1):
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
    validation_fraction: float = 0.2,
    min_val_size: int = 1,
    min_test_size: int = 1,
) -> Iterator[SweepSplit]:
    """Yield sweep splits with policy-sized inner validation holdouts.

    This is the validation-aware learning-curve path: the outer test split stays
    untouched during model selection, and validation is taken from within the
    requested training budget.

    Sweep sizes that cannot support both validation and at least one inner-train
    sample are skipped. For the current fraction-based policy, that means the
    minimum valid selection-aware sweep size is usually 2, subject to any larger
    family-specific `min_train_size` requirement.
    """
    if min_test_size <= 0:
        raise ValueError("min_test_size must be positive.")

    max_train = min(max_train, n_samples - min_test_size)
    for sweep_size in range(min_train, max_train + 1):
        n_val = inner_validation_size_for_sweep(
            sweep_size,
            validation_fraction=validation_fraction,
            min_val_size=min_val_size,
        )
        if n_val >= sweep_size:
            continue
        yield from generate_sweep_splits_with_validation(
            n_samples=n_samples,
            min_train=sweep_size,
            max_train=sweep_size,
            n_val=n_val,
            n_repeats=n_repeats,
            rng=rng,
            min_test_size=min_test_size,
        )


def build_sweep_split_collection(
    n_samples: int,
    *,
    min_train: int,
    max_train: int,
    n_repeats: int,
    seed: int,
    requirements: SweepFamilyRequirements | None = None,
    validation_fraction: float = 0.2,
    min_val_size: int = 1,
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
      `min_val_size` while still leaving at least one inner-train sample

    The feasible sweep range is also bounded above by `n_samples - min_test_size`
    so that every emitted split leaves at least `min_test_size` samples in the
    outer test set.
    """

    requirements = requirements or SweepFamilyRequirements()
    max_train = min(max_train, n_samples - min_test_size)
    effective_min_train = max(min_train, requirements.min_train_size)
    rng = np.random.default_rng(seed)
    if requirements.requires_inner_validation:
        effective_min_train = max(effective_min_train, min_val_size + 1)
        splits = tuple(
            generate_inner_validation_sweep_splits(
                n_samples,
                effective_min_train,
                max_train,
                n_repeats,
                rng,
                validation_fraction=validation_fraction,
                min_val_size=min_val_size,
                min_test_size=min_test_size,
            )
        )
    else:
        splits = tuple(
            generate_sweep_splits(
                n_samples,
                effective_min_train,
                max_train,
                n_repeats,
                rng,
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
        min_test_size=min_test_size,
        model_families=model_families,
    )


def assemble_learning_curve_dataset_from_frame(
    df: Any,
    *,
    graph_view: GraphDatasetView | None = None,
    graph_join_key: str = "reaction",
) -> SweepDataset:
    return build_sweep_dataset_from_frame(
        df,
        graph_view=graph_view,
        graph_join_key=graph_join_key,
    )


def build_sweep_dataset_from_frame(
    df: Any,
    *,
    graph_view: GraphDatasetView | None = None,
    graph_join_key: str = "reaction",
) -> SweepDataset:
    if graph_view is not None:
        from oasis.graphs import build_graph_sweep_dataset

        return build_graph_sweep_dataset(df, graph_view, join_key=graph_join_key)

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
        auxiliary_views={},
    )


def build_sweep_dataset_from_config(
    df: Any,
    cfg: Config | None,
    *,
    graph_view: GraphDatasetView | None = None,
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
        )
    if graph_view is None and experiment_cfg:
        from oasis.graphs import load_configured_graph_dataset_view

        graph_view = load_configured_graph_dataset_view(graph_dataset_cfg)
    return assemble_learning_curve_dataset_from_frame(
        df,
        graph_view=graph_view,
        graph_join_key=graph_join_key,
    )


def run_learning_curve_experiments_from_config(
    df: Any,
    cfg: Config | None,
    *,
    graph_view: GraphDatasetView | None = None,
    model_families: Sequence[Any] | None = None,
) -> LearningCurveResults:
    from oasis.method import enabled_learning_curve_model_names_from_config

    experiment_cfg = cfg.experiment.learning_curve if cfg and cfg.experiment else None
    model_cfg = experiment_cfg.models if experiment_cfg else None
    _validate_learning_curve_frame(df)
    dataset = build_sweep_dataset_from_config(df, cfg, graph_view=graph_view)
    return run_learning_curve_experiments(
        dataset,
        min_train=experiment_cfg.min_train if experiment_cfg else 5,
        max_train=experiment_cfg.max_train if experiment_cfg else 10,
        n_repeats=experiment_cfg.n_repeats if experiment_cfg else 50,
        seed=cfg.seed if cfg and cfg.seed is not None else 42,
        enabled_model_names=enabled_learning_curve_model_names_from_config(model_cfg),
        validation_fraction=(
            getattr(experiment_cfg, "validation_fraction", 0.2)
            if experiment_cfg
            else 0.2
        ),
        min_val_size=getattr(experiment_cfg, "min_val_size", 1) if experiment_cfg else 1,
        min_test_size=(
            getattr(experiment_cfg, "min_test_size", 1) if experiment_cfg else 1
        ),
        model_families=model_families,
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


def run_learning_curve_experiments(
    dataset: SweepDataset,
    *,
    min_train: int,
    max_train: int,
    n_repeats: int,
    seed: int = 42,
    enabled_model_names: Sequence[str] | None = None,
    validation_fraction: float = 0.2,
    min_val_size: int = 1,
    min_test_size: int = 1,
    model_families: Sequence[Any] | None = None,
) -> LearningCurveResults:
    families = model_families
    if families is None:
        from oasis.method import default_sweep_model_families

        families = default_sweep_model_families(enabled_model_names)

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
                n_repeats=n_repeats,
                seed=seed,
                requirements=requirements,
                validation_fraction=validation_fraction,
                min_val_size=min_val_size,
                min_test_size=min_test_size,
            ),
        )
        results = results.merge(family.run(payload))
    return results
