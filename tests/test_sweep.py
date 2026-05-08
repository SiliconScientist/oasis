from __future__ import annotations

import unittest

import numpy as np

from oasis.sweep import (
    GraphDatasetView,
    GraphRecord,
    SweepDataset,
    SweepRunnerPayload,
    SweepSplit,
    SweepSplitCollection,
    SweepRunPayload,
    TrainTestSweepRunnerInput,
    TrainValTestSweepRunnerInput,
)


class GraphRecordTests(unittest.TestCase):
    def test_graph_record_accepts_valid_shapes(self) -> None:
        record = GraphRecord(
            sample_id="s0",
            node_features=np.arange(6, dtype=float).reshape(3, 2),
            edge_index=np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int64),
            edge_features=np.arange(3, dtype=float),
            graph_features=np.array([1.0, 2.0]),
        )

        self.assertEqual(record.sample_id, "s0")
        self.assertEqual(record.n_nodes, 3)
        self.assertEqual(record.n_edges, 3)

    def test_graph_record_requires_sample_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "sample_id is required"):
            GraphRecord(
                sample_id=None,
                node_features=np.arange(6, dtype=float).reshape(3, 2),
                edge_index=np.array([[0], [1]], dtype=np.int64),
            )

    def test_graph_record_rejects_malformed_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "node_features must be a 2D array"):
            GraphRecord(
                sample_id="s0",
                node_features=np.arange(3, dtype=float),
                edge_index=np.array([[0], [0]], dtype=np.int64),
            )

        with self.assertRaisesRegex(ValueError, "edge_index must have shape"):
            GraphRecord(
                sample_id="s0",
                node_features=np.arange(6, dtype=float).reshape(3, 2),
                edge_index=np.array([[0, 1, 2]], dtype=np.int64),
            )

        with self.assertRaisesRegex(
            ValueError,
            "edge_features must have the same number of rows",
        ):
            GraphRecord(
                sample_id="s0",
                node_features=np.arange(6, dtype=float).reshape(3, 2),
                edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
                edge_features=np.arange(3, dtype=float),
            )


class GraphDatasetViewTests(unittest.TestCase):
    def test_graph_dataset_view_supports_lookup_by_sample_id(self) -> None:
        record_a = GraphRecord(
            sample_id="s0",
            node_features=np.arange(4, dtype=float).reshape(2, 2),
            edge_index=np.array([[0], [1]], dtype=np.int64),
        )
        record_b = GraphRecord(
            sample_id="s1",
            node_features=np.arange(6, dtype=float).reshape(3, 2),
            edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
        )

        view = GraphDatasetView.from_records((record_a, record_b))

        self.assertEqual(len(view), 2)
        self.assertEqual(view.sample_ids, ("s0", "s1"))
        self.assertIs(view["s0"], record_a)
        self.assertIs(view.get("s1"), record_b)
        self.assertIsNone(view.get("missing"))

    def test_graph_dataset_view_rejects_duplicate_sample_ids(self) -> None:
        record_a = GraphRecord(
            sample_id="s0",
            node_features=np.arange(4, dtype=float).reshape(2, 2),
            edge_index=np.array([[0], [1]], dtype=np.int64),
        )
        record_b = GraphRecord(
            sample_id="s0",
            node_features=np.arange(6, dtype=float).reshape(3, 2),
            edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
        )

        with self.assertRaisesRegex(ValueError, "duplicate graph record"):
            GraphDatasetView.from_records((record_a, record_b))

    def test_graph_dataset_view_rejects_mismatched_mapping_keys(self) -> None:
        record = GraphRecord(
            sample_id="s0",
            node_features=np.arange(4, dtype=float).reshape(2, 2),
            edge_index=np.array([[0], [1]], dtype=np.int64),
        )

        with self.assertRaisesRegex(
            ValueError,
            r"key='other', record.sample_id='s0'",
        ):
            GraphDatasetView(records_by_sample_id={"other": record})


class SweepDatasetTests(unittest.TestCase):
    def test_sweep_dataset_defaults_sample_ids_to_row_indices(self) -> None:
        dataset = SweepDataset(
            mlip_features=np.arange(12, dtype=float).reshape(4, 3),
            targets=np.arange(4, dtype=float),
        )

        self.assertEqual(dataset.n_samples, 4)
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

    def test_sweep_dataset_accepts_aligned_graph_view(self) -> None:
        sample_ids = np.array(["s0", "s1"])
        graph_view = GraphDatasetView.from_records(
            (
                GraphRecord(
                    sample_id="s0",
                    node_features=np.arange(4, dtype=float).reshape(2, 2),
                    edge_index=np.array([[0], [1]], dtype=np.int64),
                ),
                GraphRecord(
                    sample_id="s1",
                    node_features=np.arange(6, dtype=float).reshape(3, 2),
                    edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
                ),
            )
        )

        dataset = SweepDataset(
            mlip_features=np.arange(6, dtype=float).reshape(2, 3),
            targets=np.arange(2, dtype=float),
            sample_ids=sample_ids,
            graph_view=graph_view,
        )

        self.assertTrue(dataset.has_graphs)
        self.assertIs(dataset.graphs, graph_view)

    def test_sweep_dataset_rejects_duplicate_graph_ids(self) -> None:
        graph_view = GraphDatasetView.from_records(
            (
                GraphRecord(
                    sample_id="s0",
                    node_features=np.arange(4, dtype=float).reshape(2, 2),
                    edge_index=np.array([[0], [1]], dtype=np.int64),
                ),
                GraphRecord(
                    sample_id="s1",
                    node_features=np.arange(6, dtype=float).reshape(3, 2),
                    edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
                ),
            )
        )

        with self.assertRaisesRegex(
            ValueError,
            r"duplicates: 's0'",
        ):
            SweepDataset(
                mlip_features=np.arange(6, dtype=float).reshape(2, 3),
                targets=np.arange(2, dtype=float),
                sample_ids=np.array(["s0", "s0"]),
                graph_view=graph_view,
            )

    def test_sweep_dataset_rejects_missing_graph_ids_with_specific_ids(self) -> None:
        graph_view = GraphDatasetView.from_records(
            (
                GraphRecord(
                    sample_id="s0",
                    node_features=np.arange(4, dtype=float).reshape(2, 2),
                    edge_index=np.array([[0], [1]], dtype=np.int64),
                ),
            )
        )

        with self.assertRaisesRegex(
            ValueError,
            r"missing graph sample_ids: 's1'",
        ):
            SweepDataset(
                mlip_features=np.arange(6, dtype=float).reshape(2, 3),
                targets=np.arange(2, dtype=float),
                sample_ids=np.array(["s0", "s1"]),
                graph_view=graph_view,
            )

    def test_sweep_dataset_rejects_extra_graph_ids_with_specific_ids(self) -> None:
        graph_view = GraphDatasetView.from_records(
            (
                GraphRecord(
                    sample_id="s0",
                    node_features=np.arange(4, dtype=float).reshape(2, 2),
                    edge_index=np.array([[0], [1]], dtype=np.int64),
                ),
                GraphRecord(
                    sample_id="s2",
                    node_features=np.arange(6, dtype=float).reshape(3, 2),
                    edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
                ),
            )
        )

        with self.assertRaisesRegex(
            ValueError,
            r"extra graph sample_ids: 's2'",
        ):
            SweepDataset(
                mlip_features=np.arange(6, dtype=float).reshape(2, 3),
                targets=np.arange(2, dtype=float),
                sample_ids=np.array(["s0", "s1"]),
                graph_view=graph_view,
            )


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
