from __future__ import annotations

import unittest

import numpy as np

from oasis.sweep import (
    SweepDataset,
    SweepRunnerPayload,
    SweepSplit,
    SweepSplitCollection,
    SweepRunPayload,
    TrainTestSweepRunnerInput,
    TrainValTestSweepRunnerInput,
)


class SweepDatasetTests(unittest.TestCase):
    def test_sweep_dataset_defaults_sample_ids_to_row_indices(self) -> None:
        dataset = SweepDataset(
            mlip_features=np.arange(12, dtype=float).reshape(4, 3),
            targets=np.arange(4, dtype=float),
        )

        np.testing.assert_array_equal(dataset.sample_ids, np.arange(4))
        self.assertIsNone(dataset.auxiliary_views)

    def test_sweep_dataset_rejects_mismatched_target_length(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "targets must have the same length as mlip_features",
        ):
            SweepDataset(
                mlip_features=np.arange(12, dtype=float).reshape(4, 3),
                targets=np.arange(3, dtype=float),
            )

    def test_sweep_dataset_rejects_mismatched_sample_id_length(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "sample_ids must have the same length as mlip_features",
        ):
            SweepDataset(
                mlip_features=np.arange(12, dtype=float).reshape(4, 3),
                targets=np.arange(4, dtype=float),
                sample_ids=np.array(["a", "b", "c"]),
            )

    def test_sweep_dataset_rejects_mismatched_auxiliary_view_length(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "auxiliary view 'graphs' must have the same length as mlip_features",
        ):
            SweepDataset(
                mlip_features=np.arange(12, dtype=float).reshape(4, 3),
                targets=np.arange(4, dtype=float),
                auxiliary_views={"graphs": np.arange(3, dtype=float)},
            )

    def test_sweep_dataset_accepts_aligned_auxiliary_views(self) -> None:
        sample_ids = np.array(["s0", "s1", "s2", "s3"])
        view = np.array([{"nodes": 1}, {"nodes": 2}, {"nodes": 3}, {"nodes": 4}])

        dataset = SweepDataset(
            mlip_features=np.arange(12, dtype=float).reshape(4, 3),
            targets=np.arange(4, dtype=float),
            sample_ids=sample_ids,
            auxiliary_views={"graphs": view},
        )

        np.testing.assert_array_equal(dataset.sample_ids, sample_ids)
        self.assertIs(dataset.auxiliary_views["graphs"], view)

class SweepPayloadTests(unittest.TestCase):
    def test_sweep_run_payload_converts_train_test_and_train_val_test_runner_inputs(
        self,
    ) -> None:
        dataset = SweepDataset(
            mlip_features=np.arange(18, dtype=float).reshape(6, 3),
            targets=np.arange(6, dtype=float),
        )
        payload = SweepRunPayload(
            dataset=dataset,
            split_collection=SweepSplitCollection(
                splits=(
                    SweepSplit(
                        sweep_size=3,
                        train_idx=np.array([0, 1, 2]),
                        test_idx=np.array([3, 4, 5]),
                    ),
                    SweepSplit(
                        sweep_size=4,
                        train_idx=np.array([0, 1, 2]),
                        val_idx=np.array([3]),
                        test_idx=np.array([4, 5]),
                    ),
                )
            ),
            use_trim=True,
        )

        runner_payload = payload.to_runner_payload()

        self.assertIsInstance(runner_payload, SweepRunnerPayload)
        self.assertTrue(runner_payload.use_trim)
        self.assertIsInstance(runner_payload.splits[0], TrainTestSweepRunnerInput)
        self.assertIsInstance(runner_payload.splits[1], TrainValTestSweepRunnerInput)
        self.assertIs(runner_payload.splits[0].dataset, dataset)
        self.assertIs(runner_payload.splits[1].dataset, dataset)
        np.testing.assert_array_equal(dataset.X, dataset.mlip_features)
        np.testing.assert_array_equal(dataset.y, dataset.targets)
        np.testing.assert_array_equal(
            runner_payload.splits[1].val_idx,
            np.array([3]),
        )
