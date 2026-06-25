from __future__ import annotations

import unittest

import numpy as np

from oasis.experiment import build_sweep_split_collection
from oasis.learning_curve.execution import weighted_linear_sweep
from oasis.sweep import SweepDataset, SweepFamilyRequirements, SweepRunPayload


class WeightedLinearCalibrationTests(unittest.TestCase):
    def test_weighted_linear_sweep_marks_calibrated_uq_when_calibration_split_is_present(
        self,
    ) -> None:
        X = np.array(
            [
                [0.0, 2.0],
                [1.0, 3.0],
                [2.0, 4.0],
                [3.0, 5.0],
                [4.0, 6.0],
                [5.0, 7.0],
            ],
            dtype=float,
        )
        y = np.array([0.6, 1.6, 2.6, 3.6, 4.6, 5.6], dtype=float)
        payload = SweepRunPayload(
            dataset=SweepDataset(mlip_features=X, targets=y),
            split_collection=build_sweep_split_collection(
                n_samples=len(y),
                min_train=4,
                max_train=4,
                n_repeats=2,
                seed=8,
                requirements=SweepFamilyRequirements(requires_calibration=True),
                calibration_fraction=0.25,
                min_cal_size=1,
                min_inner_train_size=2,
                min_test_size=2,
            ),
        )

        result = weighted_linear_sweep(payload)

        self.assertEqual(result.uq_summary["uncertainty_kind"].tolist(), ["calibrated"])


if __name__ == "__main__":
    unittest.main()
