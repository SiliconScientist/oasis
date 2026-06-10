from __future__ import annotations

from collections.abc import Iterator
import math

import numpy as np

from oasis.experiment_config import LearningCurveBudgetMode
from oasis.sweep import SweepFamilyRequirements, SweepSplit, SweepSplitCollection


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
) -> Iterator[SweepSplit]:
    """Yield repeated outer train/test splits for each sweep size in the range."""

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
) -> Iterator[SweepSplit]:
    """Yield screening-mode outer CV splits over fixed total-budget sweep sizes."""

    idx = np.arange(n_samples)
    max_budget = min(max_train, n_samples)
    for sweep_size in range(min_train, max_budget + 1, step):
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
) -> Iterator[SweepSplit]:
    """Yield screening-mode outer CV splits with nested validation inside outer train."""

    if min_outer_train_size <= 0:
        raise ValueError("min_outer_train_size must be positive.")

    idx = np.arange(n_samples)
    max_budget = min(max_train, n_samples)
    for sweep_size in range(min_train, max_budget + 1, step):
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
    """Yield sweep splits with policy-sized inner validation holdouts."""

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
    budget_mode: LearningCurveBudgetMode = "full_remainder_test",
    screen_fraction: float | None = None,
    min_screen_size: int = 1,
    validation_fraction: float = 0.2,
    min_val_size: int = 1,
    min_tuning_val_size: int = 1,
    min_inner_train_size: int = 1,
    min_test_size: int = 1,
) -> SweepSplitCollection:
    """Build the split collection for one family under its split requirements."""

    requirements = requirements or SweepFamilyRequirements()
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if min_train <= 0:
        raise ValueError("min_train must be positive.")
    if max_train <= 0:
        raise ValueError("max_train must be positive.")
    if n_repeats <= 0:
        raise ValueError("n_repeats must be positive.")
    if min_screen_size <= 0:
        raise ValueError("min_screen_size must be positive.")
    if min_test_size <= 0:
        raise ValueError("min_test_size must be positive.")
    if min_val_size <= 0:
        raise ValueError("min_val_size must be positive.")
    if min_tuning_val_size <= 0:
        raise ValueError("min_tuning_val_size must be positive.")
    if min_inner_train_size <= 0:
        raise ValueError("min_inner_train_size must be positive.")

    if budget_mode == "screening_fraction":
        if screen_fraction is None:
            raise ValueError(
                "screen_fraction must be provided when budget_mode='screening_fraction'."
            )
        feasible_max_train = min(max_train, n_samples)
        effective_min_train = min_train
    else:
        feasible_max_train = min(max_train, n_samples - min_test_size)
        effective_min_train = max(min_train, requirements.min_train_size)
    if feasible_max_train < effective_min_train:
        return SweepSplitCollection(
            splits=(),
            planning_requirements=requirements,
        )

    rng = np.random.default_rng(seed)
    if budget_mode == "screening_fraction":
        first_feasible_budget = next(
            (
                sweep_size
                for sweep_size in range(effective_min_train, feasible_max_train + 1, step)
                if (
                    outer_train_size := outer_train_size_if_screening_feasible(
                        sweep_size,
                        screen_fraction=screen_fraction,
                        min_screen_size=min_screen_size,
                        min_outer_train_size=max(requirements.min_train_size, 1),
                    )
                )
                is not None
                if not requirements.requires_inner_validation
                or validation_size_if_sweep_feasible(
                    outer_train_size,
                    validation_fraction=validation_fraction,
                    min_val_size=min_val_size,
                    min_tuning_val_size=min_tuning_val_size,
                    min_inner_train_size=min_inner_train_size,
                )
                is not None
            ),
            None,
        )
        if first_feasible_budget is None:
            return SweepSplitCollection(
                splits=(),
                planning_requirements=requirements,
            )
        effective_min_train = first_feasible_budget
        if requirements.requires_inner_validation:
            splits = tuple(
                generate_screening_sweep_splits_with_validation(
                    n_samples,
                    effective_min_train,
                    feasible_max_train,
                    n_repeats,
                    rng,
                    step=step,
                    screen_fraction=screen_fraction,
                    min_screen_size=min_screen_size,
                    validation_fraction=validation_fraction,
                    min_val_size=min_val_size,
                    min_tuning_val_size=min_tuning_val_size,
                    min_inner_train_size=min_inner_train_size,
                    min_outer_train_size=max(requirements.min_train_size, 1),
                )
            )
        else:
            splits = tuple(
                generate_screening_sweep_splits(
                    n_samples,
                    effective_min_train,
                    feasible_max_train,
                    n_repeats,
                    rng,
                    step=step,
                    screen_fraction=screen_fraction,
                    min_screen_size=min_screen_size,
                    min_outer_train_size=max(requirements.min_train_size, 1),
                )
            )
    elif requirements.requires_inner_validation:
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
