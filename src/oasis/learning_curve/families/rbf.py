from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class GaussianRBF:
    """Expand scalar distances into a Gaussian RBF feature vector.

    n_rbf Gaussians are centered at equally-spaced points in [r_min, r_max].
    Width σ equals the spacing between adjacent centers so that neighboring
    basis functions cross at exp(-0.5) ≈ 0.6, giving smooth, overlapping coverage.
    """

    n_rbf: int
    r_min: float = 0.0
    r_max: float = 6.0

    def __post_init__(self) -> None:
        if self.n_rbf < 1:
            raise ValueError("n_rbf must be at least 1.")
        if self.r_max <= self.r_min:
            raise ValueError("r_max must be greater than r_min.")

    @property
    def centers(self) -> np.ndarray:
        return np.linspace(self.r_min, self.r_max, self.n_rbf)

    @property
    def width(self) -> float:
        if self.n_rbf == 1:
            return float(self.r_max - self.r_min)
        return float((self.r_max - self.r_min) / (self.n_rbf - 1))

    def __call__(self, distances: np.ndarray) -> np.ndarray:
        """Expand distances into RBF features.

        Args:
            distances: shape (n,)

        Returns:
            shape (n, n_rbf), values in (0, 1]
        """
        d = np.asarray(distances, dtype=float)
        diff = d[:, np.newaxis] - self.centers  # (n, n_rbf)
        return np.exp(-0.5 * (diff / self.width) ** 2)
