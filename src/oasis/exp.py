from __future__ import annotations

from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any

import numpy as np
from oasis.sweep import (
    LearningCurveResults,
    SweepDataset,
    SweepFamilyRequirements,
    SweepRunPayload,
    SweepSplit,
    SweepSplitCollection,
)

if TYPE_CHECKING:
    from oasis.config import Config


_INNER_VALIDATION_FRACTION = 0.2


@dataclass(frozen=True, slots=True)
class ParityPlotData:
    reference: np.ndarray
    predictions: Mapping[str, np.ndarray]


def inner_validation_size_for_sweep(
    sweep_size: int,
    *,
    frac: float = _INNER_VALIDATION_FRACTION,
) -> int:
    """Return the inner validation size for one outer-train budget.

    This policy consumes part of the requested sweep budget for validation via
    `max(1, floor(frac * sweep_size))`. The remainder stays available for inner
    training; outer test samples are not involved in this calculation.
    """

    if sweep_size <= 0:
        raise ValueError("sweep_size must be positive.")
    if frac <= 0:
        raise ValueError("frac must be positive.")
    return max(1, math.floor(frac * sweep_size))

def generate_sweep_splits(
    n_samples: int,
    min_train: int,
    max_train: int,
    n_repeats: int,
    rng: np.random.Generator,
) -> Iterator[SweepSplit]:
    """Yield repeated outer train/test splits for each sweep size in the range.

    These splits do not include inner validation. The entire `train_idx` budget
    is available for fitting, and `test_idx` is reserved for outer evaluation.
    """

    idx = np.arange(n_samples)
    max_train = min(max_train, n_samples - 1)
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

    idx = np.arange(n_samples)
    max_train = min(max_train, n_samples - 1)
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
    validation_fraction: float = _INNER_VALIDATION_FRACTION,
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
    max_train = min(max_train, n_samples - 1)
    for sweep_size in range(min_train, max_train + 1):
        n_val = inner_validation_size_for_sweep(
            sweep_size,
            frac=validation_fraction,
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
        )


def build_sweep_split_collection(
    n_samples: int,
    *,
    min_train: int,
    max_train: int,
    n_repeats: int,
    seed: int,
    requirements: SweepFamilyRequirements | None = None,
) -> SweepSplitCollection:
    """Build the split collection for one family under its split requirements.

    Train/test-only families receive outer train/test splits across the requested
    sweep range. Validation-aware families receive only sweep sizes large enough
    to support inner train/val partitioning while keeping `test_idx` as the
    outer evaluation holdout.

    The minimum sweep size for a selection-based family is therefore the maximum
    of:
    - the caller's requested `min_train`
    - the family's declared `min_train_size`
    - the smallest sweep size that leaves at least one inner-train sample after
      the validation policy carves out `val_idx`
    """

    requirements = requirements or SweepFamilyRequirements()
    max_train = min(max_train, n_samples - 1)
    split_generator = generate_sweep_splits
    effective_min_train = max(min_train, requirements.min_train_size)
    if requirements.requires_inner_validation:
        split_generator = generate_inner_validation_sweep_splits
    return SweepSplitCollection(
        splits=tuple(
            split_generator(
                n_samples,
                effective_min_train,
                max_train,
                n_repeats,
                np.random.default_rng(seed),
            )
        ),
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
    use_trim: bool = True,
    enabled_model_names: Sequence[str] | None = None,
) -> LearningCurveResults:
    feature_cols = mlip_columns(df)
    if not feature_cols:
        raise ValueError(
            "No MLIP prediction columns found (expected *_mlip_ads_eng_median)."
        )

    n_rows = getattr(df, "height", len(df))
    if n_rows <= 5:
        raise ValueError("Not enough data to evaluate (need >5 samples).")

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

    return run_learning_curve_experiments(
        SweepDataset(
            mlip_features=X,
            targets=y,
            sample_ids=sample_ids,
            auxiliary_views={},
        ),
        min_train=min_train,
        max_train=max_train,
        n_repeats=n_repeats,
        seed=seed,
        use_trim=use_trim,
        enabled_model_names=enabled_model_names,
    )


def run_learning_curve_experiments_from_config(
    df: Any,
    cfg: Config | None,
) -> LearningCurveResults:
    from oasis.method import enabled_learning_curve_model_names_from_config

    plot_cfg = cfg.plot if cfg else None
    return run_learning_curve_experiments_from_frame(
        df,
        min_train=plot_cfg.min_train if plot_cfg else 5,
        max_train=plot_cfg.max_train if plot_cfg else 10,
        n_repeats=plot_cfg.n_repeats if plot_cfg else 50,
        seed=cfg.seed if cfg and cfg.seed is not None else 42,
        use_trim=plot_cfg.trim if plot_cfg else True,
        enabled_model_names=enabled_learning_curve_model_names_from_config(plot_cfg),
    )


def run_learning_curve_experiments(
    dataset: SweepDataset,
    *,
    min_train: int,
    max_train: int,
    n_repeats: int,
    seed: int = 42,
    use_trim: bool = True,
    enabled_model_names: Sequence[str] | None = None,
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
            ),
            use_trim=use_trim,
        )
        results = results.merge(family.run(payload))
    return results
