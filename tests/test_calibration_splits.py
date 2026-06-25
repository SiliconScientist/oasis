from __future__ import annotations

import unittest

import numpy as np

from oasis.exp import (
    build_sweep_split_collection,
    generate_inner_calibration_sweep_splits,
    generate_inner_validation_calibration_sweep_splits,
)
from oasis.sweep import (
    SweepDataset,
    SweepFamilyRequirements,
    TrainCalTestSweepRunnerInput,
    TrainValCalTestSweepRunnerInput,
    split_to_runner_input,
)


class CalibrationSplitTests(unittest.TestCase):
    def test_generate_inner_calibration_sweep_splits_yields_disjoint_partitions(self) -> None:
        rng = np.random.default_rng(3)
        splits = list(
            generate_inner_calibration_sweep_splits(
                n_samples=8,
                min_train=4,
                max_train=5,
                n_repeats=2,
                rng=rng,
                calibration_fraction=0.25,
                min_cal_size=1,
                min_inner_train_size=2,
                min_test_size=2,
            )
        )

        self.assertEqual([split.sweep_size for split in splits], [4, 4, 5, 5])
        for split in splits:
            self.assertIsNotNone(split.cal_idx)
            self.assertIsNone(split.val_idx)
            self.assertEqual(len(split.train_idx) + len(split.cal_idx), split.sweep_size)
            self.assertEqual(len(np.intersect1d(split.train_idx, split.cal_idx)), 0)
            self.assertEqual(len(np.intersect1d(split.train_idx, split.test_idx)), 0)
            self.assertEqual(len(np.intersect1d(split.cal_idx, split.test_idx)), 0)
            self.assertEqual(
                len(np.union1d(np.union1d(split.train_idx, split.cal_idx), split.test_idx)),
                8,
            )

    def test_generate_inner_validation_calibration_splits_yields_disjoint_partitions(self) -> None:
        rng = np.random.default_rng(5)
        splits = list(
            generate_inner_validation_calibration_sweep_splits(
                n_samples=10,
                min_train=6,
                max_train=6,
                n_repeats=2,
                rng=rng,
                validation_fraction=0.25,
                min_val_size=1,
                calibration_fraction=0.25,
                min_cal_size=1,
                min_inner_train_size=2,
                min_test_size=2,
            )
        )

        self.assertEqual(len(splits), 2)
        for split in splits:
            self.assertIsNotNone(split.val_idx)
            self.assertIsNotNone(split.cal_idx)
            self.assertEqual(
                len(split.train_idx) + len(split.val_idx) + len(split.cal_idx),
                split.sweep_size,
            )
            all_idx = np.concatenate(
                [split.train_idx, split.val_idx, split.cal_idx, split.test_idx]
            )
            self.assertEqual(len(np.unique(all_idx)), 10)

    def test_build_sweep_split_collection_supports_calibration_requirements(self) -> None:
        split_collection = build_sweep_split_collection(
            n_samples=9,
            min_train=4,
            max_train=5,
            n_repeats=1,
            seed=17,
            requirements=SweepFamilyRequirements(requires_calibration=True),
            calibration_fraction=0.25,
            min_cal_size=1,
            min_inner_train_size=2,
            min_test_size=2,
        )

        self.assertEqual(
            split_collection.planning_requirements,
            SweepFamilyRequirements(requires_calibration=True),
        )
        self.assertTrue(all(split.cal_idx is not None for split in split_collection.splits))
        self.assertTrue(all(split.val_idx is None for split in split_collection.splits))

    def test_split_to_runner_input_emits_calibration_aware_runner_inputs(self) -> None:
        dataset = SweepDataset(
            mlip_features=np.arange(24, dtype=float).reshape(8, 3),
            targets=np.arange(8, dtype=float),
        )

        cal_only = split_to_runner_input(
            dataset,
            build_sweep_split_collection(
                n_samples=8,
                min_train=4,
                max_train=4,
                n_repeats=1,
                seed=1,
                requirements=SweepFamilyRequirements(requires_calibration=True),
                calibration_fraction=0.25,
                min_cal_size=1,
                min_inner_train_size=2,
                min_test_size=2,
            ).splits[0],
        )
        self.assertIsInstance(cal_only, TrainCalTestSweepRunnerInput)

        val_cal = split_to_runner_input(
            dataset,
            build_sweep_split_collection(
                n_samples=8,
                min_train=5,
                max_train=5,
                n_repeats=1,
                seed=1,
                requirements=SweepFamilyRequirements(
                    requires_inner_validation=True,
                    requires_calibration=True,
                ),
                validation_fraction=0.25,
                min_val_size=1,
                calibration_fraction=0.25,
                min_cal_size=1,
                min_inner_train_size=2,
                min_test_size=2,
            ).splits[0],
        )
        self.assertIsInstance(val_cal, TrainValCalTestSweepRunnerInput)


if __name__ == "__main__":
    unittest.main()
