from __future__ import annotations

import numpy as np

from oasis.exp import generate_sweep_splits
from oasis.sweep import SweepDataset, SweepSplit, SweepSplitCollection, SweepRunPayload


def regression_dataset() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [1.0, 1.2, 0.8],
            [1.8, 2.1, 1.9],
            [2.7, 3.0, 2.9],
            [3.9, 4.2, 3.8],
            [5.1, 5.0, 4.9],
            [6.2, 6.0, 5.8],
        ]
    )
    y = np.array([1.1, 2.0, 2.9, 4.0, 5.0, 6.1])
    return X, y


def regression_train_test_payload(seed: int = 13) -> SweepRunPayload:
    X, y = regression_dataset()
    dataset = SweepDataset(mlip_features=X, targets=y)
    return SweepRunPayload(
        dataset=dataset,
        split_collection=SweepSplitCollection(
            splits=tuple(
                generate_sweep_splits(
                    n_samples=dataset.n_samples,
                    min_train=2,
                    max_train=4,
                    n_repeats=2,
                    rng=np.random.default_rng(seed),
                )
            )
        ),
        use_trim=True,
    )


def weighted_toy_dataset() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [3.0, 2.0],
            [4.0, 3.0],
        ]
    )
    y = 1.5 * X[:, 0] - 0.5 * X[:, 1] + 2.0
    return X, y


def weighted_fixed_payload() -> SweepRunPayload:
    X, y = weighted_toy_dataset()
    return SweepRunPayload(
        dataset=SweepDataset(mlip_features=X, targets=y),
        split_collection=SweepSplitCollection(
            splits=(
                SweepSplit(
                    sweep_size=4,
                    train_idx=np.array([0, 1, 2, 3]),
                    test_idx=np.array([4, 5]),
                ),
                SweepSplit(
                    sweep_size=5,
                    train_idx=np.array([0, 1, 2, 4, 5]),
                    test_idx=np.array([3]),
                ),
            )
        ),
        use_trim=False,
    )
