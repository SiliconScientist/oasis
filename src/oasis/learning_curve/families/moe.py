from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class MoEModel:
    weights: np.ndarray
    bias: float = 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias
