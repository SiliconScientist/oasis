from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SweepSplit:
    """One train/test split for a sweep size."""

    sweep_size: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def generate_sweep_splits(
    n_samples: int,
    min_train: int,
    max_train: int,
    n_repeats: int,
    rng: np.random.Generator,
) -> Iterator[SweepSplit]:
    """Yield repeated train/test splits for each sweep size in the range."""

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
