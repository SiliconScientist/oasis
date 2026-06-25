from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Protocol

import numpy as np


_STANDARD_NORMAL = NormalDist()
_ONE_SIGMA_COVERAGE = 0.6826894921370859
_IDENTICALLY_ZERO_SPREAD_ERROR = "cannot fit calibration scale when spread is identically zero."
_ZERO_SPREAD_RESIDUAL_ERROR = (
    "cannot fit calibration scale when positive residuals coincide with zero spread."
)


class SpreadCalibrator(Protocol):
    """Post-hoc calibrator for spread-valued uncertainty estimates."""

    method: str

    def apply(self, spread: np.ndarray) -> np.ndarray: ...


def _as_flat_float_array(values: np.ndarray, *, name: str) -> np.ndarray:
    resolved = np.asarray(values, dtype=float).reshape(-1)
    if np.any(~np.isfinite(resolved)):
        raise ValueError(f"{name} must be finite.")
    return resolved


def _validate_spread(spread: np.ndarray) -> np.ndarray:
    resolved = _as_flat_float_array(spread, name="spread")
    if np.any(resolved < 0.0):
        raise ValueError("spread must be nonnegative.")
    return resolved


@dataclass(frozen=True, slots=True)
class ScalarSpreadCalibrator:
    """Scalar rescaling for spread-only uncertainty estimates."""

    scale: float
    method: str = "scalar_scale"
    reference_coverage: float = _ONE_SIGMA_COVERAGE
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not np.isfinite(self.scale):
            raise ValueError("scale must be finite.")
        if self.scale < 0.0:
            raise ValueError("scale must be nonnegative.")
        if not 0.0 < self.reference_coverage < 1.0:
            raise ValueError("reference_coverage must be strictly between 0 and 1.")

    def apply(self, spread: np.ndarray) -> np.ndarray:
        resolved_spread = _validate_spread(spread)
        return resolved_spread * float(self.scale)

    @classmethod
    def fit(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        spread: np.ndarray,
        *,
        reference_coverage: float = _ONE_SIGMA_COVERAGE,
    ) -> ScalarSpreadCalibrator:
        if not 0.0 < reference_coverage < 1.0:
            raise ValueError("reference_coverage must be strictly between 0 and 1.")
        resolved_true = _as_flat_float_array(y_true, name="y_true")
        resolved_pred = _as_flat_float_array(y_pred, name="y_pred")
        resolved_spread = _validate_spread(spread)
        if len(resolved_true) != len(resolved_pred) or len(resolved_true) != len(
            resolved_spread
        ):
            raise ValueError("y_true, y_pred, and spread must have the same number of rows.")

        abs_errors = np.abs(resolved_true - resolved_pred)
        positive_mask = resolved_spread > 0.0
        if not np.any(positive_mask):
            if np.all(abs_errors == 0.0):
                return cls(scale=1.0, reference_coverage=reference_coverage)
            raise ValueError(_IDENTICALLY_ZERO_SPREAD_ERROR)

        if np.any(abs_errors[~positive_mask] > 0.0):
            raise ValueError(_ZERO_SPREAD_RESIDUAL_ERROR)

        z_score = _STANDARD_NORMAL.inv_cdf((1.0 + float(reference_coverage)) / 2.0)
        ratios = abs_errors[positive_mask] / (z_score * resolved_spread[positive_mask])
        scale = float(np.quantile(ratios, reference_coverage))
        return cls(
            scale=scale,
            reference_coverage=reference_coverage,
            metadata={
                "fit_sample_count": int(len(resolved_true)),
                "positive_spread_count": int(np.count_nonzero(positive_mask)),
            },
        )


def fit_scalar_spread_calibrator_with_identity_fallback(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    spread: np.ndarray,
    *,
    reference_coverage: float = _ONE_SIGMA_COVERAGE,
) -> ScalarSpreadCalibrator:
    """Fit a scalar spread calibrator, falling back to identity on degenerate zero spread."""

    try:
        return ScalarSpreadCalibrator.fit(
            y_true=y_true,
            y_pred=y_pred,
            spread=spread,
            reference_coverage=reference_coverage,
        )
    except ValueError as exc:
        if str(exc) not in {
            _IDENTICALLY_ZERO_SPREAD_ERROR,
            _ZERO_SPREAD_RESIDUAL_ERROR,
        }:
            raise
        resolved_true = _as_flat_float_array(y_true, name="y_true")
        resolved_spread = _validate_spread(spread)
        return ScalarSpreadCalibrator(
            scale=1.0,
            reference_coverage=reference_coverage,
            metadata={
                "fit_sample_count": int(len(resolved_true)),
                "positive_spread_count": int(np.count_nonzero(resolved_spread > 0.0)),
                "fallback_strategy": "identity",
                "fallback_reason": str(exc),
            },
        )
