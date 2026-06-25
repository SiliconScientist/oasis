from __future__ import annotations

import unittest

import numpy as np

from oasis.experiment import build_sweep_split_collection
from oasis.learning_curve.sklearn_specs import sklearn_sweep_model_specs
from oasis.sweep import SweepDataset, SweepFamilyRequirements, SweepRunPayload
from oasis.tune import sweep_supervised_model_selection


class RidgeCalibrationTests(unittest.TestCase):
    def test_ridge_supervised_selection_marks_calibrated_uq_when_calibration_split_is_present(
        self,
    ) -> None:
        specs = {name: spec for name, _, spec in sklearn_sweep_model_specs()}
        ridge_spec = specs["ridge"].hyperparameter_spec
        assert ridge_spec is not None

        X = np.array(
            [
                [0.0, 2.0],
                [1.0, 3.0],
                [2.0, 4.0],
                [3.0, 5.0],
                [4.0, 6.0],
                [5.0, 7.0],
                [6.0, 8.0],
            ],
            dtype=float,
        )
        y = np.array([0.5, 1.4, 2.6, 3.5, 4.7, 5.5, 6.6], dtype=float)
        payload = SweepRunPayload(
            dataset=SweepDataset(mlip_features=X, targets=y),
            split_collection=build_sweep_split_collection(
                n_samples=len(y),
                min_train=5,
                max_train=5,
                n_repeats=2,
                seed=10,
                requirements=SweepFamilyRequirements(
                    requires_inner_validation=True,
                    requires_calibration=True,
                ),
                validation_fraction=0.2,
                min_val_size=1,
                calibration_fraction=0.2,
                min_cal_size=1,
                min_inner_train_size=2,
                min_test_size=2,
            ),
        )

        result = sweep_supervised_model_selection(payload, ridge_spec)

        assert result.uq_summary is not None
        self.assertEqual(result.uq_summary["uncertainty_kind"].tolist(), ["calibrated"])
        self.assertTrue(
            (result.uq_summary["uncertainty_note"] == "post-hoc scalar calibrated spread").all()
        )


if __name__ == "__main__":
    unittest.main()
