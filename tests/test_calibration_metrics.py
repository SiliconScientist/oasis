from __future__ import annotations

import unittest

import numpy as np

from oasis.calibration_metrics import fit_calibration_metrics


class SpreadCalibrationMetricsTests(unittest.TestCase):
    def test_fit_calibration_metrics_reduces_held_out_miscalibration_area(self) -> None:
        rng = np.random.default_rng(11)
        calibration_residuals = rng.normal(loc=0.0, scale=2.5, size=3000)
        evaluation_residuals = rng.normal(loc=0.0, scale=2.5, size=3000)
        spread = np.ones(3000)
        y_pred = np.zeros(3000)

        metrics = fit_calibration_metrics(
            calibration_y_true=calibration_residuals,
            calibration_y_pred=y_pred,
            calibration_spread=spread,
            evaluation_y_true=evaluation_residuals,
            evaluation_y_pred=y_pred,
            evaluation_spread=spread,
        )

        self.assertGreater(metrics.calibrator.scale, 2.0)
        self.assertLess(
            metrics.calibrated_miscalibration_area,
            metrics.raw_miscalibration_area,
        )
        self.assertGreater(metrics.calibrated_sharpness, metrics.raw_sharpness)
        self.assertAlmostEqual(
            metrics.calibrated_dispersion,
            metrics.raw_dispersion,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
