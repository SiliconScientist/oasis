from __future__ import annotations

import unittest

import numpy as np

from oasis.calibration import ScalarSpreadCalibrator
from oasis.learning_curve.execution import calibration_curve_frame


class ScalarSpreadCalibratorTests(unittest.TestCase):
    def test_apply_scales_spread_elementwise(self) -> None:
        calibrator = ScalarSpreadCalibrator(scale=1.5)

        calibrated = calibrator.apply(np.array([0.0, 1.0, 2.0]))

        np.testing.assert_allclose(calibrated, np.array([0.0, 1.5, 3.0]))

    def test_apply_rejects_negative_spread(self) -> None:
        calibrator = ScalarSpreadCalibrator(scale=1.0)

        with self.assertRaisesRegex(ValueError, "nonnegative"):
            calibrator.apply(np.array([1.0, -0.1]))

    def test_fit_recovers_expected_scale_for_constant_ratios(self) -> None:
        y_true = np.array([2.0, -2.0, 2.0, -2.0])
        y_pred = np.zeros(4)
        spread = np.ones(4)

        calibrator = ScalarSpreadCalibrator.fit(
            y_true,
            y_pred,
            spread,
            reference_coverage=0.6826894921370859,
        )

        self.assertAlmostEqual(calibrator.scale, 2.0, places=6)
        self.assertEqual(calibrator.metadata["fit_sample_count"], 4)
        self.assertEqual(calibrator.metadata["positive_spread_count"], 4)

    def test_fit_accepts_zero_spread_when_residuals_are_zero(self) -> None:
        calibrator = ScalarSpreadCalibrator.fit(
            y_true=np.zeros(3),
            y_pred=np.zeros(3),
            spread=np.zeros(3),
        )

        self.assertEqual(calibrator.scale, 1.0)

    def test_fit_rejects_zero_spread_with_positive_residual(self) -> None:
        with self.assertRaisesRegex(ValueError, "identically zero"):
            ScalarSpreadCalibrator.fit(
                y_true=np.array([1.0, 0.0]),
                y_pred=np.zeros(2),
                spread=np.zeros(2),
            )

    def test_fit_improves_empirical_one_sigma_coverage_on_synthetic_data(self) -> None:
        rng = np.random.default_rng(7)
        calibration_residuals = rng.normal(loc=0.0, scale=2.0, size=4000)
        test_residuals = rng.normal(loc=0.0, scale=2.0, size=4000)
        y_pred = np.zeros(4000)
        raw_spread = np.ones(4000)

        calibrator = ScalarSpreadCalibrator.fit(
            y_true=calibration_residuals,
            y_pred=y_pred,
            spread=raw_spread,
        )
        calibrated_spread = calibrator.apply(raw_spread)

        raw_curve = calibration_curve_frame(test_residuals, y_pred, raw_spread)
        calibrated_curve = calibration_curve_frame(
            test_residuals,
            y_pred,
            calibrated_spread,
        )
        target_idx = int(
            np.argmin(np.abs(raw_curve["nominal_coverage"].to_numpy() - calibrator.reference_coverage))
        )
        raw_gap = float(raw_curve["absolute_gap"].iloc[target_idx])
        calibrated_gap = float(calibrated_curve["absolute_gap"].iloc[target_idx])

        self.assertGreater(calibrator.scale, 1.5)
        self.assertLess(calibrated_gap, raw_gap)


if __name__ == "__main__":
    unittest.main()
