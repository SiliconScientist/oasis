from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Any

from oasis.experiment.core import _run_learning_curve_experiments_with_budget_mode
from oasis.sweep import LearningCurveResults, SweepDataset


def run_standard_learning_curve_experiments(
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
    requested_sweep_sizes_by_method: Mapping[str, Collection[int]] | None = None,
) -> LearningCurveResults:
    """Run the standard learning-curve workflow with full remainder test sets."""

    return _run_learning_curve_experiments_with_budget_mode(
        dataset,
        min_train=min_train,
        max_train=max_train,
        step=step,
        n_repeats=n_repeats,
        seed=seed,
        enabled_model_names=enabled_model_names,
        model_cfg=model_cfg,
        budget_mode="full_remainder_test",
        screen_fraction=None,
        min_screen_size=1,
        validation_fraction=validation_fraction,
        min_val_size=min_val_size,
        min_tuning_val_size=min_tuning_val_size,
        min_inner_train_size=min_inner_train_size,
        min_test_size=min_test_size,
        model_families=model_families,
        requested_sweep_sizes_by_method=requested_sweep_sizes_by_method,
    )
