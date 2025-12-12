from __future__ import annotations

import numpy as np


def _allocate_tuning_counts(
    N_work: int, p_max: float, N0: float, n_max: int
) -> tuple[int, int]:
    """
    Compute how many samples to divert to hyperparameter tuning.
    """
    raw = p_max * (1 - np.exp(-N_work / N0)) * N_work
    n_tune = int(round(min(raw, n_max, N_work)))
    n_tune = max(n_tune, 0)
    n_train = max(N_work - n_tune, 0)
    return n_tune, n_train
