from __future__ import annotations

from collections.abc import Iterator
import math
from typing import Collection

import numpy as np

from oasis.experiment_config import LearningCurveBudgetMode
from oasis.sweep import SweepFamilyRequirements, SweepSplit, SweepSplitCollection


def normalize_requested_sweep_sizes(
    requested_sweep_sizes: Collection[int],
) -> tuple[int, ...]:
    normalized = tuple(
        sorted({int(sweep_size) for sweep_size in requested_sweep_sizes})
    )
    if any(sweep_size <= 0 for sweep_size in normalized):
        raise ValueError("requested sweep sizes must be positive.")
    return normalized


def resolve_configured_sweep_sizes(
    n_samples: int,
    *,
    min_train: int | None,
    max_train: int | None,
    step: int = 1,
    sweep_sizes: Collection[int] | None = None,
    sweep_fractions: Collection[float] | None = None,
) -> tuple[int, ...]:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if step <= 0:
        raise ValueError("step must be positive.")

    explicit_sweep_sizes = normalize_requested_sweep_sizes(sweep_sizes or ())
    explicit_sweep_fractions = tuple(float(value) for value in (sweep_fractions or ()))
    if explicit_sweep_sizes and explicit_sweep_fractions:
        raise ValueError("sweep_sizes and sweep_fractions cannot both be provided.")
    if explicit_sweep_fractions:
        if any(value <= 0 or value > 1 for value in explicit_sweep_fractions):
            raise ValueError("sweep_fractions must be between 0 and 1.")
        return tuple(
            sorted(
                {
                    max(1, int(math.floor(value * n_samples)))
                    for value in explicit_sweep_fractions
                }
            )
        )
    if explicit_sweep_sizes:
        return explicit_sweep_sizes
    if min_train is None or max_train is None:
        raise ValueError(
            "min_train and max_train must be provided when no explicit sweep sizes are configured."
        )
    if min_train <= 0:
        raise ValueError("min_train must be positive.")
    if max_train <= 0:
        raise ValueError("max_train must be positive.")
    return tuple(range(min_train, max_train + 1, step))


def _screening_cv_test_folds(
    budget_idx: np.ndarray,
    *,
    n_screen: int,
) -> Iterator[np.ndarray]:
    """Yield disjoint fixed-size screening holdouts from one budget sample."""

    if n_screen <= 0:
        raise ValueError("n_screen must be positive.")
    n_folds = len(budget_idx) // n_screen
    if n_folds < 2:
        return
    for fold_idx in range(n_folds):
        start = fold_idx * n_screen
        stop = start + n_screen
        yield budget_idx[start:stop]


def inner_validation_size_for_sweep(
    sweep_size: int,
    *,
    validation_fraction: float,
    min_val_size: int,
    min_tuning_val_size: int = 1,
) -> int:
    """Return the inner validation size for one outer-train budget."""

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
    """Return the validation size for a feasible validation-aware sweep point."""

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


def calibration_size_if_sweep_feasible(
    sweep_size: int,
    *,
    calibration_fraction: float,
    min_cal_size: int,
    min_post_calibration_size: int = 1,
) -> int | None:
    """Return the calibration size for a feasible calibration-aware sweep point."""

    if sweep_size <= 0:
        raise ValueError("sweep_size must be positive.")
    if calibration_fraction < 0:
        raise ValueError("calibration_fraction must be non-negative.")
    if min_cal_size <= 0:
        raise ValueError("min_cal_size must be positive.")
    if min_post_calibration_size <= 0:
        raise ValueError("min_post_calibration_size must be positive.")

    n_cal = max(min_cal_size, math.floor(calibration_fraction * sweep_size))
    if n_cal + min_post_calibration_size > sweep_size:
        return None
    return n_cal


def minimum_training_size_for_requirements(
    requirements: SweepFamilyRequirements,
    *,
    validation_fraction: float = 0.2,
    min_val_size: int = 1,
    min_tuning_val_size: int = 1,
    min_inner_train_size: int = 1,
) -> int:
    """Return the first training size that satisfies shared split-feasibility rules.

    This centralizes the same minimum-data guard used by split planning so
    callers can reason about method availability without reimplementing the
    validation-aware budget thresholds.
    """

    minimum_size = max(int(requirements.min_train_size), 1)
    if not requirements.requires_inner_validation:
        return minimum_size

    sweep_size = minimum_size
    while True:
        if (
            validation_size_if_sweep_feasible(
                sweep_size,
                validation_fraction=validation_fraction,
                min_val_size=min_val_size,
                min_tuning_val_size=min_tuning_val_size,
                min_inner_train_size=min_inner_train_size,
            )
            is not None
        ):
            return sweep_size
        sweep_size += 1


def screening_holdout_size_for_budget(
    sweep_size: int,
    *,
    screen_fraction: float,
    min_screen_size: int = 1,
) -> int:
    """Return the screening holdout size for one total-budget sweep point."""

    if sweep_size <= 0:
        raise ValueError("sweep_size must be positive.")
    if screen_fraction <= 0 or screen_fraction >= 1:
        raise ValueError("screen_fraction must be between 0 and 1.")
    if min_screen_size <= 0:
        raise ValueError("min_screen_size must be positive.")
    return max(
        min_screen_size,
        math.floor(screen_fraction * sweep_size),
    )


def outer_train_size_if_screening_feasible(
    sweep_size: int,
    *,
    screen_fraction: float,
    min_screen_size: int = 1,
    min_outer_train_size: int = 1,
) -> int | None:
    """Return outer-train size for a feasible screening-budget sweep point."""

    if min_outer_train_size <= 0:
        raise ValueError("min_outer_train_size must be positive.")
    n_screen = screening_holdout_size_for_budget(
        sweep_size,
        screen_fraction=screen_fraction,
        min_screen_size=min_screen_size,
    )
    n_outer_train = sweep_size - n_screen
    if n_outer_train < min_outer_train_size:
        return None
    return n_outer_train


def generate_sweep_splits(
    n_samples: int,
    min_train: int,
    max_train: int,
    n_repeats: int,
    rng: np.random.Generator,
    *,
    step: int = 1,
    min_test_size: int = 1,
    requested_sweep_sizes: Collection[int] | None = None,
) -> Iterator[SweepSplit]:
    """Yield repeated outer train/test splits for each sweep size in the range."""

    if min_test_size <= 0:
        raise ValueError("min_test_size must be positive.")

    idx = np.arange(n_samples)
    sweep_sizes = (
        normalize_requested_sweep_sizes(requested_sweep_sizes)
        if requested_sweep_sizes is not None
        else tuple(range(min_train, min(max_train, n_samples - min_test_size) + 1, step))
    )
    for n_train in sweep_sizes:
        for _ in range(n_repeats):
            train_idx = rng.choice(idx, size=n_train, replace=False)
            test_idx = np.setdiff1d(idx, train_idx, assume_unique=False)
            yield SweepSplit(
                sweep_size=n_train,
                train_idx=train_idx,
                test_idx=test_idx,
            )


def generate_screening_sweep_splits(
    n_samples: int,
    min_train: int,
    max_train: int,
    n_repeats: int,
    rng: np.random.Generator,
    *,
    step: int = 1,
    screen_fraction: float,
    min_screen_size: int = 1,
    min_outer_train_size: int = 1,
    requested_sweep_sizes: Collection[int] | None = None,
) -> Iterator[SweepSplit]:
    """Yield screening-mode outer CV splits over fixed total-budget sweep sizes."""

    idx = np.arange(n_samples)
    sweep_sizes = (
        normalize_requested_sweep_sizes(requested_sweep_sizes)
        if requested_sweep_sizes is not None
        else tuple(range(min_train, min(max_train, n_samples) + 1, step))
    )
    for sweep_size in sweep_sizes:
        n_outer_train = outer_train_size_if_screening_feasible(
            sweep_size,
            screen_fraction=screen_fraction,
            min_screen_size=min_screen_size,
            min_outer_train_size=min_outer_train_size,
        )
        if n_outer_train is None:
            continue
        n_screen = sweep_size - n_outer_train
        budget_idx = rng.choice(idx, size=sweep_size, replace=False)
        shuffled_budget_idx = rng.permutation(budget_idx)
        for test_idx in _screening_cv_test_folds(
            shuffled_budget_idx,
            n_screen=n_screen,
        ):
            train_idx = np.setdiff1d(budget_idx, test_idx, assume_unique=False)
            yield SweepSplit(
                sweep_size=sweep_size,
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
    """Yield repeated outer train/test splits with inner train/val partitions."""

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


def generate_sweep_splits_with_calibration(
    n_samples: int,
    min_train: int,
    max_train: int,
    n_cal: int,
    n_repeats: int,
    rng: np.random.Generator,
    *,
    min_inner_train_size: int = 1,
    min_test_size: int = 1,
) -> Iterator[SweepSplit]:
    """Yield repeated outer train/test splits with inner train/cal partitions."""

    if n_cal <= 0:
        raise ValueError("n_cal must be positive.")
    if n_cal >= n_samples:
        raise ValueError("n_cal must be smaller than n_samples.")
    if min_inner_train_size <= 0:
        raise ValueError("min_inner_train_size must be positive.")
    if min_test_size <= 0:
        raise ValueError("min_test_size must be positive.")

    idx = np.arange(n_samples)
    max_train = min(max_train, n_samples - min_test_size)
    for n_train in range(max(min_train, n_cal + min_inner_train_size), max_train + 1):
        for _ in range(n_repeats):
            outer_train_idx = rng.choice(idx, size=n_train, replace=False)
            test_idx = np.setdiff1d(idx, outer_train_idx, assume_unique=False)
            cal_idx = rng.choice(outer_train_idx, size=n_cal, replace=False)
            train_idx = np.setdiff1d(outer_train_idx, cal_idx, assume_unique=False)
            yield SweepSplit(
                sweep_size=n_train,
                train_idx=train_idx,
                test_idx=test_idx,
                cal_idx=cal_idx,
            )


def generate_sweep_splits_with_validation_and_calibration(
    n_samples: int,
    min_train: int,
    max_train: int,
    n_val: int,
    n_cal: int,
    n_repeats: int,
    rng: np.random.Generator,
    *,
    min_inner_train_size: int = 1,
    min_test_size: int = 1,
) -> Iterator[SweepSplit]:
    """Yield repeated outer train/test splits with inner train/val/cal partitions."""

    if n_val <= 0:
        raise ValueError("n_val must be positive.")
    if n_cal <= 0:
        raise ValueError("n_cal must be positive.")
    if min_inner_train_size <= 0:
        raise ValueError("min_inner_train_size must be positive.")
    if min_test_size <= 0:
        raise ValueError("min_test_size must be positive.")

    idx = np.arange(n_samples)
    max_train = min(max_train, n_samples - min_test_size)
    minimum_outer_train = n_val + n_cal + min_inner_train_size
    for n_train in range(max(min_train, minimum_outer_train), max_train + 1):
        for _ in range(n_repeats):
            outer_train_idx = rng.choice(idx, size=n_train, replace=False)
            test_idx = np.setdiff1d(idx, outer_train_idx, assume_unique=False)
            val_idx = rng.choice(outer_train_idx, size=n_val, replace=False)
            remaining_idx = np.setdiff1d(outer_train_idx, val_idx, assume_unique=False)
            cal_idx = rng.choice(remaining_idx, size=n_cal, replace=False)
            train_idx = np.setdiff1d(remaining_idx, cal_idx, assume_unique=False)
            yield SweepSplit(
                sweep_size=n_train,
                train_idx=train_idx,
                test_idx=test_idx,
                val_idx=val_idx,
                cal_idx=cal_idx,
            )


def generate_screening_sweep_splits_with_validation(
    n_samples: int,
    min_train: int,
    max_train: int,
    n_repeats: int,
    rng: np.random.Generator,
    *,
    step: int = 1,
    screen_fraction: float,
    min_screen_size: int = 1,
    validation_fraction: float = 0.2,
    min_val_size: int = 1,
    min_tuning_val_size: int = 1,
    min_inner_train_size: int = 1,
    min_outer_train_size: int = 1,
    requested_sweep_sizes: Collection[int] | None = None,
) -> Iterator[SweepSplit]:
    """Yield screening-mode outer CV splits with nested validation inside outer train."""

    if min_outer_train_size <= 0:
        raise ValueError("min_outer_train_size must be positive.")

    idx = np.arange(n_samples)
    sweep_sizes = (
        normalize_requested_sweep_sizes(requested_sweep_sizes)
        if requested_sweep_sizes is not None
        else tuple(range(min_train, min(max_train, n_samples) + 1, step))
    )
    for sweep_size in sweep_sizes:
        n_outer_train = outer_train_size_if_screening_feasible(
            sweep_size,
            screen_fraction=screen_fraction,
            min_screen_size=min_screen_size,
            min_outer_train_size=min_outer_train_size,
        )
        if n_outer_train is None:
            continue
        n_val = validation_size_if_sweep_feasible(
            n_outer_train,
            validation_fraction=validation_fraction,
            min_val_size=min_val_size,
            min_tuning_val_size=min_tuning_val_size,
            min_inner_train_size=min_inner_train_size,
        )
        if n_val is None:
            continue
        n_screen = sweep_size - n_outer_train
        budget_idx = rng.choice(idx, size=sweep_size, replace=False)
        shuffled_budget_idx = rng.permutation(budget_idx)
        for test_idx in _screening_cv_test_folds(
            shuffled_budget_idx,
            n_screen=n_screen,
        ):
            outer_train_idx = np.setdiff1d(budget_idx, test_idx, assume_unique=False)
            val_idx = rng.choice(outer_train_idx, size=n_val, replace=False)
            train_idx = np.setdiff1d(outer_train_idx, val_idx, assume_unique=False)
            yield SweepSplit(
                sweep_size=sweep_size,
                train_idx=train_idx,
                test_idx=test_idx,
                val_idx=val_idx,
            )


def generate_screening_sweep_splits_with_calibration(
    n_samples: int,
    min_train: int,
    max_train: int,
    n_repeats: int,
    rng: np.random.Generator,
    *,
    step: int = 1,
    screen_fraction: float,
    min_screen_size: int = 1,
    calibration_fraction: float = 0.2,
    min_cal_size: int = 1,
    min_inner_train_size: int = 1,
    min_outer_train_size: int = 1,
    requested_sweep_sizes: Collection[int] | None = None,
) -> Iterator[SweepSplit]:
    """Yield screening-mode outer CV splits with nested calibration inside outer train."""

    if min_outer_train_size <= 0:
        raise ValueError("min_outer_train_size must be positive.")

    idx = np.arange(n_samples)
    sweep_sizes = (
        normalize_requested_sweep_sizes(requested_sweep_sizes)
        if requested_sweep_sizes is not None
        else tuple(range(min_train, min(max_train, n_samples) + 1, step))
    )
    for sweep_size in sweep_sizes:
        n_outer_train = outer_train_size_if_screening_feasible(
            sweep_size,
            screen_fraction=screen_fraction,
            min_screen_size=min_screen_size,
            min_outer_train_size=min_outer_train_size,
        )
        if n_outer_train is None:
            continue
        n_cal = calibration_size_if_sweep_feasible(
            n_outer_train,
            calibration_fraction=calibration_fraction,
            min_cal_size=min_cal_size,
            min_post_calibration_size=min_inner_train_size,
        )
        if n_cal is None:
            continue
        n_screen = sweep_size - n_outer_train
        budget_idx = rng.choice(idx, size=sweep_size, replace=False)
        shuffled_budget_idx = rng.permutation(budget_idx)
        for test_idx in _screening_cv_test_folds(
            shuffled_budget_idx,
            n_screen=n_screen,
        ):
            outer_train_idx = np.setdiff1d(budget_idx, test_idx, assume_unique=False)
            cal_idx = rng.choice(outer_train_idx, size=n_cal, replace=False)
            train_idx = np.setdiff1d(outer_train_idx, cal_idx, assume_unique=False)
            yield SweepSplit(
                sweep_size=sweep_size,
                train_idx=train_idx,
                test_idx=test_idx,
                cal_idx=cal_idx,
            )


def generate_screening_sweep_splits_with_validation_and_calibration(
    n_samples: int,
    min_train: int,
    max_train: int,
    n_repeats: int,
    rng: np.random.Generator,
    *,
    step: int = 1,
    screen_fraction: float,
    min_screen_size: int = 1,
    validation_fraction: float = 0.2,
    min_val_size: int = 1,
    min_tuning_val_size: int = 1,
    calibration_fraction: float = 0.2,
    min_cal_size: int = 1,
    min_inner_train_size: int = 1,
    min_outer_train_size: int = 1,
    requested_sweep_sizes: Collection[int] | None = None,
) -> Iterator[SweepSplit]:
    """Yield screening-mode outer CV splits with nested validation and calibration."""

    if min_outer_train_size <= 0:
        raise ValueError("min_outer_train_size must be positive.")

    idx = np.arange(n_samples)
    sweep_sizes = (
        normalize_requested_sweep_sizes(requested_sweep_sizes)
        if requested_sweep_sizes is not None
        else tuple(range(min_train, min(max_train, n_samples) + 1, step))
    )
    for sweep_size in sweep_sizes:
        n_outer_train = outer_train_size_if_screening_feasible(
            sweep_size,
            screen_fraction=screen_fraction,
            min_screen_size=min_screen_size,
            min_outer_train_size=min_outer_train_size,
        )
        if n_outer_train is None:
            continue
        n_cal = calibration_size_if_sweep_feasible(
            n_outer_train,
            calibration_fraction=calibration_fraction,
            min_cal_size=min_cal_size,
            min_post_calibration_size=max(min_val_size + min_inner_train_size, 1),
        )
        if n_cal is None:
            continue
        remaining_fit_budget = n_outer_train - n_cal
        n_val = validation_size_if_sweep_feasible(
            remaining_fit_budget,
            validation_fraction=validation_fraction,
            min_val_size=min_val_size,
            min_tuning_val_size=min_tuning_val_size,
            min_inner_train_size=min_inner_train_size,
        )
        if n_val is None:
            continue
        n_screen = sweep_size - n_outer_train
        budget_idx = rng.choice(idx, size=sweep_size, replace=False)
        shuffled_budget_idx = rng.permutation(budget_idx)
        for test_idx in _screening_cv_test_folds(
            shuffled_budget_idx,
            n_screen=n_screen,
        ):
            outer_train_idx = np.setdiff1d(budget_idx, test_idx, assume_unique=False)
            cal_idx = rng.choice(outer_train_idx, size=n_cal, replace=False)
            remaining_idx = np.setdiff1d(outer_train_idx, cal_idx, assume_unique=False)
            val_idx = rng.choice(remaining_idx, size=n_val, replace=False)
            train_idx = np.setdiff1d(remaining_idx, val_idx, assume_unique=False)
            yield SweepSplit(
                sweep_size=sweep_size,
                train_idx=train_idx,
                test_idx=test_idx,
                val_idx=val_idx,
                cal_idx=cal_idx,
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
    requested_sweep_sizes: Collection[int] | None = None,
) -> Iterator[SweepSplit]:
    """Yield sweep splits with policy-sized inner validation holdouts."""

    if min_inner_train_size <= 0:
        raise ValueError("min_inner_train_size must be positive.")
    if min_test_size <= 0:
        raise ValueError("min_test_size must be positive.")

    sweep_sizes = (
        normalize_requested_sweep_sizes(requested_sweep_sizes)
        if requested_sweep_sizes is not None
        else tuple(range(min_train, min(max_train, n_samples - min_test_size) + 1, step))
    )
    for sweep_size in sweep_sizes:
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


def generate_inner_calibration_sweep_splits(
    n_samples: int,
    min_train: int,
    max_train: int,
    n_repeats: int,
    rng: np.random.Generator,
    *,
    step: int = 1,
    calibration_fraction: float = 0.2,
    min_cal_size: int = 1,
    min_inner_train_size: int = 1,
    min_test_size: int = 1,
    requested_sweep_sizes: Collection[int] | None = None,
) -> Iterator[SweepSplit]:
    """Yield sweep splits with policy-sized inner calibration holdouts."""

    if min_inner_train_size <= 0:
        raise ValueError("min_inner_train_size must be positive.")
    if min_test_size <= 0:
        raise ValueError("min_test_size must be positive.")

    sweep_sizes = (
        normalize_requested_sweep_sizes(requested_sweep_sizes)
        if requested_sweep_sizes is not None
        else tuple(range(min_train, min(max_train, n_samples - min_test_size) + 1, step))
    )
    for sweep_size in sweep_sizes:
        n_cal = calibration_size_if_sweep_feasible(
            sweep_size,
            calibration_fraction=calibration_fraction,
            min_cal_size=min_cal_size,
            min_post_calibration_size=min_inner_train_size,
        )
        if n_cal is None:
            continue
        yield from generate_sweep_splits_with_calibration(
            n_samples=n_samples,
            min_train=sweep_size,
            max_train=sweep_size,
            n_cal=n_cal,
            n_repeats=n_repeats,
            rng=rng,
            min_inner_train_size=min_inner_train_size,
            min_test_size=min_test_size,
        )


def generate_inner_validation_calibration_sweep_splits(
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
    calibration_fraction: float = 0.2,
    min_cal_size: int = 1,
    min_inner_train_size: int = 1,
    min_test_size: int = 1,
    requested_sweep_sizes: Collection[int] | None = None,
) -> Iterator[SweepSplit]:
    """Yield sweep splits with policy-sized inner validation and calibration."""

    if min_inner_train_size <= 0:
        raise ValueError("min_inner_train_size must be positive.")
    if min_test_size <= 0:
        raise ValueError("min_test_size must be positive.")

    sweep_sizes = (
        normalize_requested_sweep_sizes(requested_sweep_sizes)
        if requested_sweep_sizes is not None
        else tuple(range(min_train, min(max_train, n_samples - min_test_size) + 1, step))
    )
    for sweep_size in sweep_sizes:
        n_cal = calibration_size_if_sweep_feasible(
            sweep_size,
            calibration_fraction=calibration_fraction,
            min_cal_size=min_cal_size,
            min_post_calibration_size=max(min_val_size + min_inner_train_size, 1),
        )
        if n_cal is None:
            continue
        remaining_fit_budget = sweep_size - n_cal
        n_val = validation_size_if_sweep_feasible(
            remaining_fit_budget,
            validation_fraction=validation_fraction,
            min_val_size=min_val_size,
            min_tuning_val_size=min_tuning_val_size,
            min_inner_train_size=min_inner_train_size,
        )
        if n_val is None:
            continue
        yield from generate_sweep_splits_with_validation_and_calibration(
            n_samples=n_samples,
            min_train=sweep_size,
            max_train=sweep_size,
            n_val=n_val,
            n_cal=n_cal,
            n_repeats=n_repeats,
            rng=rng,
            min_inner_train_size=min_inner_train_size,
            min_test_size=min_test_size,
        )


def build_sweep_split_collection(
    n_samples: int,
    *,
    min_train: int | None,
    max_train: int | None,
    step: int = 1,
    n_repeats: int,
    seed: int,
    requested_sweep_sizes: Collection[int] | None = None,
    requirements: SweepFamilyRequirements | None = None,
    budget_mode: LearningCurveBudgetMode = "full_remainder_test",
    screen_fraction: float | None = None,
    min_screen_size: int = 1,
    validation_fraction: float = 0.2,
    min_val_size: int = 1,
    min_tuning_val_size: int = 1,
    calibration_fraction: float = 0.2,
    min_cal_size: int = 1,
    min_inner_train_size: int = 1,
    min_test_size: int = 1,
) -> SweepSplitCollection:
    """Build the split collection for one family under its split requirements."""

    requirements = requirements or SweepFamilyRequirements()
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if n_repeats <= 0:
        raise ValueError("n_repeats must be positive.")
    if step <= 0:
        raise ValueError("step must be positive.")
    if min_screen_size <= 0:
        raise ValueError("min_screen_size must be positive.")
    if min_test_size <= 0:
        raise ValueError("min_test_size must be positive.")
    if min_val_size <= 0:
        raise ValueError("min_val_size must be positive.")
    if min_tuning_val_size <= 0:
        raise ValueError("min_tuning_val_size must be positive.")
    if min_cal_size <= 0:
        raise ValueError("min_cal_size must be positive.")
    if min_inner_train_size <= 0:
        raise ValueError("min_inner_train_size must be positive.")

    candidate_sweep_sizes = (
        normalize_requested_sweep_sizes(requested_sweep_sizes)
        if requested_sweep_sizes is not None
        else resolve_configured_sweep_sizes(
            n_samples,
            min_train=min_train,
            max_train=max_train,
            step=step,
        )
    )
    if not candidate_sweep_sizes:
        return SweepSplitCollection(
            splits=(),
            planning_requirements=requirements,
        )

    if budget_mode == "screening_fraction":
        if screen_fraction is None:
            raise ValueError(
                "screen_fraction must be provided when budget_mode='screening_fraction'."
            )

    rng = np.random.default_rng(seed)
    if budget_mode == "screening_fraction":
        feasible_sweep_sizes = tuple(
            sweep_size
            for sweep_size in candidate_sweep_sizes
            if sweep_size <= n_samples
            if (
                outer_train_size := outer_train_size_if_screening_feasible(
                    sweep_size,
                    screen_fraction=screen_fraction,
                    min_screen_size=min_screen_size,
                    min_outer_train_size=max(requirements.min_train_size, 1),
                )
            )
            is not None
            if (
                not requirements.requires_inner_validation
                and not requirements.requires_calibration
            )
            or (
                requirements.requires_inner_validation
                and not requirements.requires_calibration
                and validation_size_if_sweep_feasible(
                    outer_train_size,
                    validation_fraction=validation_fraction,
                    min_val_size=min_val_size,
                    min_tuning_val_size=min_tuning_val_size,
                    min_inner_train_size=min_inner_train_size,
                )
                is not None
            )
            or (
                requirements.requires_calibration
                and not requirements.requires_inner_validation
                and calibration_size_if_sweep_feasible(
                    outer_train_size,
                    calibration_fraction=calibration_fraction,
                    min_cal_size=min_cal_size,
                    min_post_calibration_size=min_inner_train_size,
                )
                is not None
            )
            or (
                requirements.requires_inner_validation
                and requirements.requires_calibration
                and (
                    cal_size := calibration_size_if_sweep_feasible(
                        outer_train_size,
                        calibration_fraction=calibration_fraction,
                        min_cal_size=min_cal_size,
                        min_post_calibration_size=max(min_val_size + min_inner_train_size, 1),
                    )
                )
                is not None
                and validation_size_if_sweep_feasible(
                    outer_train_size - cal_size,
                    validation_fraction=validation_fraction,
                    min_val_size=min_val_size,
                    min_tuning_val_size=min_tuning_val_size,
                    min_inner_train_size=min_inner_train_size,
                )
                is not None
            )
        )
        if not feasible_sweep_sizes:
            return SweepSplitCollection(
                splits=(),
                planning_requirements=requirements,
            )
        if requirements.requires_inner_validation and requirements.requires_calibration:
            splits = tuple(
                generate_screening_sweep_splits_with_validation_and_calibration(
                    n_samples=n_samples,
                    min_train=1,
                    max_train=1,
                    n_repeats=n_repeats,
                    rng=rng,
                    step=1,
                    screen_fraction=screen_fraction,
                    min_screen_size=min_screen_size,
                    validation_fraction=validation_fraction,
                    min_val_size=min_val_size,
                    min_tuning_val_size=min_tuning_val_size,
                    calibration_fraction=calibration_fraction,
                    min_cal_size=min_cal_size,
                    min_inner_train_size=min_inner_train_size,
                    min_outer_train_size=max(requirements.min_train_size, 1),
                    requested_sweep_sizes=feasible_sweep_sizes,
                )
            )
        elif requirements.requires_inner_validation:
            splits = tuple(
                generate_screening_sweep_splits_with_validation(
                    n_samples=n_samples,
                    min_train=1,
                    max_train=1,
                    n_repeats=n_repeats,
                    rng=rng,
                    step=1,
                    screen_fraction=screen_fraction,
                    min_screen_size=min_screen_size,
                    validation_fraction=validation_fraction,
                    min_val_size=min_val_size,
                    min_tuning_val_size=min_tuning_val_size,
                    min_inner_train_size=min_inner_train_size,
                    min_outer_train_size=max(requirements.min_train_size, 1),
                    requested_sweep_sizes=feasible_sweep_sizes,
                )
            )
        elif requirements.requires_calibration:
            splits = tuple(
                generate_screening_sweep_splits_with_calibration(
                    n_samples=n_samples,
                    min_train=1,
                    max_train=1,
                    n_repeats=n_repeats,
                    rng=rng,
                    step=1,
                    screen_fraction=screen_fraction,
                    min_screen_size=min_screen_size,
                    calibration_fraction=calibration_fraction,
                    min_cal_size=min_cal_size,
                    min_inner_train_size=min_inner_train_size,
                    min_outer_train_size=max(requirements.min_train_size, 1),
                    requested_sweep_sizes=feasible_sweep_sizes,
                )
            )
        else:
            splits = tuple(
                generate_screening_sweep_splits(
                    n_samples=n_samples,
                    min_train=1,
                    max_train=1,
                    n_repeats=n_repeats,
                    rng=rng,
                    step=1,
                    screen_fraction=screen_fraction,
                    min_screen_size=min_screen_size,
                    min_outer_train_size=max(requirements.min_train_size, 1),
                    requested_sweep_sizes=feasible_sweep_sizes,
                )
            )
    elif requirements.requires_inner_validation and requirements.requires_calibration:
        feasible_sweep_sizes = tuple(
            sweep_size
            for sweep_size in candidate_sweep_sizes
            if requirements.min_train_size <= sweep_size <= n_samples - min_test_size
            if (
                n_cal := calibration_size_if_sweep_feasible(
                    sweep_size,
                    calibration_fraction=calibration_fraction,
                    min_cal_size=min_cal_size,
                    min_post_calibration_size=max(min_val_size + min_inner_train_size, 1),
                )
            )
            is not None
            if validation_size_if_sweep_feasible(
                sweep_size - n_cal,
                validation_fraction=validation_fraction,
                min_val_size=min_val_size,
                min_tuning_val_size=min_tuning_val_size,
                min_inner_train_size=min_inner_train_size,
            )
            is not None
        )
        if not feasible_sweep_sizes:
            return SweepSplitCollection(
                splits=(),
                planning_requirements=requirements,
            )
        splits = tuple(
            generate_inner_validation_calibration_sweep_splits(
                n_samples=n_samples,
                min_train=1,
                max_train=1,
                n_repeats=n_repeats,
                rng=rng,
                step=1,
                validation_fraction=validation_fraction,
                min_val_size=min_val_size,
                min_tuning_val_size=min_tuning_val_size,
                calibration_fraction=calibration_fraction,
                min_cal_size=min_cal_size,
                min_inner_train_size=min_inner_train_size,
                min_test_size=min_test_size,
                requested_sweep_sizes=feasible_sweep_sizes,
            )
        )
    elif requirements.requires_inner_validation:
        feasible_sweep_sizes = tuple(
            sweep_size
            for sweep_size in candidate_sweep_sizes
            if requirements.min_train_size <= sweep_size <= n_samples - min_test_size
            if validation_size_if_sweep_feasible(
                sweep_size,
                validation_fraction=validation_fraction,
                min_val_size=min_val_size,
                min_tuning_val_size=min_tuning_val_size,
                min_inner_train_size=min_inner_train_size,
            )
            is not None
        )
        if not feasible_sweep_sizes:
            return SweepSplitCollection(
                splits=(),
                planning_requirements=requirements,
            )
        splits = tuple(
            generate_inner_validation_sweep_splits(
                n_samples=n_samples,
                min_train=1,
                max_train=1,
                n_repeats=n_repeats,
                rng=rng,
                step=1,
                validation_fraction=validation_fraction,
                min_val_size=min_val_size,
                min_tuning_val_size=min_tuning_val_size,
                min_inner_train_size=min_inner_train_size,
                min_test_size=min_test_size,
                requested_sweep_sizes=feasible_sweep_sizes,
            )
        )
    elif requirements.requires_calibration:
        feasible_sweep_sizes = tuple(
            sweep_size
            for sweep_size in candidate_sweep_sizes
            if requirements.min_train_size <= sweep_size <= n_samples - min_test_size
            if calibration_size_if_sweep_feasible(
                sweep_size,
                calibration_fraction=calibration_fraction,
                min_cal_size=min_cal_size,
                min_post_calibration_size=min_inner_train_size,
            )
            is not None
        )
        if not feasible_sweep_sizes:
            return SweepSplitCollection(
                splits=(),
                planning_requirements=requirements,
            )
        splits = tuple(
            generate_inner_calibration_sweep_splits(
                n_samples=n_samples,
                min_train=1,
                max_train=1,
                n_repeats=n_repeats,
                rng=rng,
                step=1,
                calibration_fraction=calibration_fraction,
                min_cal_size=min_cal_size,
                min_inner_train_size=min_inner_train_size,
                min_test_size=min_test_size,
                requested_sweep_sizes=feasible_sweep_sizes,
            )
        )
    else:
        feasible_sweep_sizes = tuple(
            sweep_size
            for sweep_size in candidate_sweep_sizes
            if requirements.min_train_size <= sweep_size <= n_samples - min_test_size
        )
        splits = tuple(
            generate_sweep_splits(
                n_samples=n_samples,
                min_train=1,
                max_train=1,
                n_repeats=n_repeats,
                rng=rng,
                step=1,
                min_test_size=min_test_size,
                requested_sweep_sizes=feasible_sweep_sizes,
            )
        )
    return SweepSplitCollection(
        splits=splits,
        planning_requirements=requirements,
    )
