from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

from oasis.calibration import ScalarSpreadCalibrator


_DEFAULT_CALIBRATION_COVERAGES = np.linspace(0.1, 0.9, 9, dtype=float)
_STANDARD_NORMAL = NormalDist()
_TRAPEZOID = getattr(np, "trapezoid", np.trapz)


def _resolve_nominal_coverages(
    nominal_coverages: np.ndarray | None,
) -> np.ndarray:
    resolved_coverages = (
        _DEFAULT_CALIBRATION_COVERAGES
        if nominal_coverages is None
        else np.asarray(nominal_coverages, dtype=float).reshape(-1)
    )
    if len(resolved_coverages) == 0:
        raise ValueError("nominal_coverages must contain at least one value.")
    if np.any((resolved_coverages <= 0.0) | (resolved_coverages >= 1.0)):
        raise ValueError("nominal_coverages must lie strictly between 0 and 1.")
    return resolved_coverages


def calibration_curve_frame(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    spread: np.ndarray,
    *,
    nominal_coverages: np.ndarray | None = None,
) -> pd.DataFrame:
    resolved_coverages = _resolve_nominal_coverages(nominal_coverages)
    abs_errors = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
    resolved_spread = np.asarray(spread, dtype=float).reshape(-1)
    if len(abs_errors) != len(resolved_spread):
        raise ValueError("y_true, y_pred, and spread must have the same number of rows.")
    z_scores = np.asarray(
        [
            _STANDARD_NORMAL.inv_cdf((1.0 + float(coverage)) / 2.0)
            for coverage in resolved_coverages
        ],
        dtype=float,
    )
    observed_coverages = np.asarray(
        [
            float(np.mean(abs_errors <= z_score * resolved_spread))
            for z_score in z_scores
        ],
        dtype=float,
    )
    absolute_gap = np.abs(observed_coverages - resolved_coverages)
    return pd.DataFrame(
        {
            "nominal_coverage": resolved_coverages,
            "observed_coverage": observed_coverages,
            "absolute_gap": absolute_gap,
        }
    )


def miscalibration_area(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    spread: np.ndarray,
    *,
    nominal_coverages: np.ndarray | None = None,
) -> float:
    curve = calibration_curve_frame(
        y_true,
        y_pred,
        spread,
        nominal_coverages=nominal_coverages,
    )
    if len(curve) == 1:
        return float(curve["absolute_gap"].iloc[0])
    return float(
        _TRAPEZOID(
            curve["absolute_gap"].to_numpy(),
            curve["nominal_coverage"].to_numpy(),
        )
    )


def sharpness_from_spread(spread: np.ndarray) -> float:
    resolved_spread = np.abs(np.asarray(spread, dtype=float).reshape(-1))
    if len(resolved_spread) == 0:
        return 0.0
    return float(np.mean(resolved_spread))


def dispersion_from_spread(spread: np.ndarray) -> float:
    resolved_spread = np.abs(np.asarray(spread, dtype=float).reshape(-1))
    if len(resolved_spread) == 0:
        return 0.0
    mean_spread = float(np.mean(resolved_spread))
    if mean_spread <= 0.0:
        return 0.0
    return float(np.std(resolved_spread) / mean_spread)


@dataclass(frozen=True, slots=True)
class SpreadCalibrationMetrics:
    calibrator: ScalarSpreadCalibrator
    raw_miscalibration_area: float
    calibrated_miscalibration_area: float
    raw_sharpness: float
    calibrated_sharpness: float
    raw_dispersion: float
    calibrated_dispersion: float


def fit_calibration_metrics(
    *,
    calibration_y_true: np.ndarray,
    calibration_y_pred: np.ndarray,
    calibration_spread: np.ndarray,
    evaluation_y_true: np.ndarray,
    evaluation_y_pred: np.ndarray,
    evaluation_spread: np.ndarray,
    reference_coverage: float = 0.6826894921370859,
    nominal_coverages: np.ndarray | None = None,
) -> SpreadCalibrationMetrics:
    calibrator = ScalarSpreadCalibrator.fit(
        calibration_y_true,
        calibration_y_pred,
        calibration_spread,
        reference_coverage=reference_coverage,
    )
    calibrated_spread = calibrator.apply(evaluation_spread)
    return SpreadCalibrationMetrics(
        calibrator=calibrator,
        raw_miscalibration_area=float(
            miscalibration_area(
                evaluation_y_true,
                evaluation_y_pred,
                evaluation_spread,
                nominal_coverages=nominal_coverages,
            )
        ),
        calibrated_miscalibration_area=float(
            miscalibration_area(
                evaluation_y_true,
                evaluation_y_pred,
                calibrated_spread,
                nominal_coverages=nominal_coverages,
            )
        ),
        raw_sharpness=float(sharpness_from_spread(evaluation_spread)),
        calibrated_sharpness=float(sharpness_from_spread(calibrated_spread)),
        raw_dispersion=float(dispersion_from_spread(evaluation_spread)),
        calibrated_dispersion=float(dispersion_from_spread(calibrated_spread)),
    )
