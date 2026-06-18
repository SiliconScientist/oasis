from __future__ import annotations

import unittest
from unittest.mock import PropertyMock, patch

import numpy as np
import pandas as pd
from oasis.tune import SweepRunnerArtifacts

from oasis.sweep import (
    build_sweep_batches,
    collate_sweep_samples,
    DatasetLoaderAdapter,
    SweepDatasetBatchLoaderAdapter,
    DatasetLoaderFactoryAdapter,
    DatasetLoaderFactory,
    GraphDatasetView,
    GraphRecord,
    LearningCurveResults,
    LoaderAdapterInput,
    LoaderBatching,
    SweepDataset,
    SweepBatch,
    SweepDatasetInputs,
    SweepDatasetModalities,
    TrainEvalLoaderPolicy,
    SweepFamilyRequirements,
    SweepRunnerPayload,
    TrainTestSplitDatasetInputs,
    SweepSampleInputs,
    SweepSampleModalities,
    SweepSplit,
    SweepSplitCollection,
    SweepRunPayload,
    TrainTestSplitDatasets,
    TrainTestSplitLoaderInputs,
    TrainTestSplitLoaders,
    TrainTestSweepRunnerInput,
    TrainValTestSplitDatasetInputs,
    TrainValTestSplitDatasets,
    TrainValTestSplitLoaderInputs,
    TrainValTestSplitLoaders,
    TrainValTestSweepRunnerInput,
    split_to_dataset_subsets,
    split_to_loader_inputs,
    split_to_loaders,
)


class GraphRecordTests(unittest.TestCase):
    def test_graph_record_accepts_valid_shapes(self) -> None:
        record = GraphRecord(
            sample_id="s0",
            node_features=np.arange(6, dtype=float).reshape(3, 2),
            edge_index=np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int64),
            node_positions=np.arange(9, dtype=float).reshape(3, 3),
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

        with self.assertRaisesRegex(ValueError, "node_positions must have shape"):
            GraphRecord(
                sample_id="s0",
                node_features=np.arange(6, dtype=float).reshape(3, 2),
                edge_index=np.array([[0], [1]], dtype=np.int64),
                node_positions=np.arange(6, dtype=float).reshape(3, 2),
            )

        with self.assertRaisesRegex(
            ValueError,
            "node_positions must have the same number of rows",
        ):
            GraphRecord(
                sample_id="s0",
                node_features=np.arange(6, dtype=float).reshape(3, 2),
                edge_index=np.array([[0], [1]], dtype=np.int64),
                node_positions=np.arange(12, dtype=float).reshape(4, 3),
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
    def _graph_view(self) -> GraphDatasetView:
        return GraphDatasetView.from_records(
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
                GraphRecord(
                    sample_id="s2",
                    node_features=np.arange(2, dtype=float).reshape(1, 2),
                    edge_index=np.empty((2, 0), dtype=np.int64),
                ),
                GraphRecord(
                    sample_id="s3",
                    node_features=np.arange(8, dtype=float).reshape(4, 2),
                    edge_index=np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
                ),
            )
        )

    def _dataset_with_graphs_and_auxiliary(self) -> SweepDataset:
        return SweepDataset(
            mlip_features=np.arange(12, dtype=float).reshape(4, 3),
            targets=np.arange(4, dtype=float) + 0.5,
            sample_ids=np.array(["s0", "s1", "s2", "s3"]),
            graph_view=self._graph_view(),
            auxiliary_views={
                "weights": np.array([1.0, 2.0, 3.0, 4.0]),
                "folds": ["train", "val", "test", "holdout"],
            },
        )

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

    def test_sweep_dataset_rejects_non_vector_targets(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "targets must be a 1D array",
        ):
            SweepDataset(
                mlip_features=np.arange(12, dtype=float).reshape(4, 3),
                targets=np.arange(4, dtype=float).reshape(4, 1),
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

    def test_sweep_dataset_rejects_non_vector_sample_ids(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "sample_ids must be a 1D array",
        ):
            SweepDataset(
                mlip_features=np.arange(12, dtype=float).reshape(4, 3),
                targets=np.arange(4, dtype=float),
                sample_ids=np.array([["a"], ["b"], ["c"], ["d"]]),
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

    def test_sweep_dataset_accepts_graph_only_rows_when_graph_modality_is_present(
        self,
    ) -> None:
        dataset = SweepDataset(
            mlip_features=np.empty((2, 0), dtype=float),
            targets=np.array([0.0, 1.0]),
            sample_ids=np.array(["s0", "s1"]),
            graph_view=GraphDatasetView.from_records(
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
            ),
        )

        self.assertEqual(dataset.mlip_features.shape, (2, 0))
        self.assertTrue(dataset.has_graphs)

    def test_sweep_dataset_rejects_missing_all_modalities(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "dataset must provide at least one modality",
        ):
            SweepDataset(
                mlip_features=np.empty((2, 0), dtype=float),
                targets=np.array([0.0, 1.0]),
                sample_ids=np.array(["s0", "s1"]),
            )

    def test_sweep_dataset_exposes_named_modalities_while_preserving_legacy_fields(
        self,
    ) -> None:
        dataset = self._dataset_with_graphs_and_auxiliary()

        modalities = dataset.modalities

        self.assertIsInstance(modalities, SweepDatasetModalities)
        self.assertIs(modalities.mlip_features, dataset.mlip_features)
        self.assertTrue(modalities.has_graphs)
        self.assertIs(modalities.graphs, dataset.graphs)
        self.assertIs(modalities.graph_view, dataset.graph_view)

    def test_sweep_dataset_exposes_input_views_for_learned_family_callers(self) -> None:
        dataset = self._dataset_with_graphs_and_auxiliary()

        inputs = dataset.inputs

        self.assertIsInstance(inputs, SweepDatasetInputs)
        self.assertIs(inputs.mlip_features, dataset.mlip_features)
        self.assertTrue(inputs.has_graphs)
        self.assertIs(inputs.graph_view_required(), dataset.graph_view)
        self.assertIs(dataset.mlip_view(), dataset.mlip_features)
        self.assertIs(dataset.graph_view_required(), dataset.graph_view)

    def test_sweep_dataset_graph_view_required_rejects_missing_graphs(self) -> None:
        dataset = SweepDataset(
            mlip_features=np.arange(6, dtype=float).reshape(2, 3),
            targets=np.arange(2, dtype=float),
        )

        with self.assertRaisesRegex(
            ValueError,
            "graph_view is required for these dataset inputs",
        ):
            dataset.graph_view_required()

    def test_sweep_dataset_sample_returns_aligned_graph_and_auxiliary_views(self) -> None:
        dataset = self._dataset_with_graphs_and_auxiliary()

        sample = dataset.sample(1)

        self.assertEqual(sample.index, 1)
        np.testing.assert_array_equal(sample.mlip_features, np.array([3.0, 4.0, 5.0]))
        self.assertEqual(sample.target, 1.5)
        self.assertEqual(sample.sample_id, "s1")
        self.assertIsNotNone(sample.graph)
        self.assertEqual(sample.graph.sample_id, "s1")
        self.assertEqual(sample.auxiliary, {"weights": 2.0, "folds": "val"})
        np.testing.assert_array_equal(
            sample.graph.node_features,
            dataset.graphs["s1"].node_features,
        )
        modalities = sample.modalities
        self.assertIsInstance(modalities, SweepSampleModalities)
        np.testing.assert_array_equal(modalities.mlip_features, sample.mlip_features)
        self.assertTrue(modalities.has_graph)
        self.assertIs(modalities.graph, sample.graph)
        inputs = sample.inputs
        self.assertIsInstance(inputs, SweepSampleInputs)
        np.testing.assert_array_equal(inputs.mlip_features, sample.mlip_features)
        self.assertTrue(inputs.has_graph)
        self.assertIs(inputs.graph_required(), sample.graph)
        np.testing.assert_array_equal(sample.mlip_view(), sample.mlip_features)
        self.assertIs(sample.graph_required(), sample.graph)

    def test_sweep_dataset_sample_graph_required_rejects_missing_graph(self) -> None:
        dataset = SweepDataset(
            mlip_features=np.arange(6, dtype=float).reshape(2, 3),
            targets=np.arange(2, dtype=float),
        )

        with self.assertRaisesRegex(
            ValueError,
            "graph is required for these sample inputs",
        ):
            dataset.sample(0).graph_required()

    def test_sweep_dataset_sample_uses_sample_id_to_resolve_graph_after_reordering(
        self,
    ) -> None:
        dataset = self._dataset_with_graphs_and_auxiliary().subset(np.array([2, 0, 3]))

        sample = dataset.sample(0)

        self.assertEqual(sample.index, 0)
        self.assertEqual(sample.sample_id, "s2")
        np.testing.assert_array_equal(sample.mlip_features, np.array([6.0, 7.0, 8.0]))
        self.assertEqual(sample.target, 2.5)
        self.assertEqual(sample.auxiliary, {"weights": 3.0, "folds": "test"})
        self.assertIsNotNone(sample.graph)
        self.assertEqual(sample.graph.sample_id, "s2")
        np.testing.assert_array_equal(
            sample.graph.node_features,
            dataset.graphs["s2"].node_features,
        )

    def test_sweep_dataset_subset_keeps_all_views_aligned_for_split_indices(self) -> None:
        dataset = self._dataset_with_graphs_and_auxiliary()

        subset = dataset.subset(np.array([3, 1]))

        np.testing.assert_array_equal(
            subset.mlip_features,
            np.array([[9.0, 10.0, 11.0], [3.0, 4.0, 5.0]]),
        )
        np.testing.assert_array_equal(subset.targets, np.array([3.5, 1.5]))
        np.testing.assert_array_equal(subset.sample_ids, np.array(["s3", "s1"]))
        self.assertEqual(subset.graphs.sample_ids, ("s3", "s1"))
        self.assertEqual(subset.graphs["s3"].sample_id, "s3")
        np.testing.assert_array_equal(
            subset.auxiliary_views["weights"],
            np.array([4.0, 2.0]),
        )
        self.assertEqual(subset.auxiliary_views["folds"], ["holdout", "val"])
        np.testing.assert_array_equal(
            subset.sample(0).mlip_features,
            np.array([9.0, 10.0, 11.0]),
        )
        self.assertEqual(subset.sample(0).target, 3.5)
        self.assertEqual(subset.sample(0).graph.sample_id, "s3")
        self.assertEqual(subset.sample(0).auxiliary, {"weights": 4.0, "folds": "holdout"})

    def test_sweep_dataset_subset_accepts_boolean_masks(self) -> None:
        dataset = self._dataset_with_graphs_and_auxiliary()

        subset = dataset.subset(np.array([True, False, True, False]))

        np.testing.assert_array_equal(subset.sample_ids, np.array(["s0", "s2"]))
        self.assertEqual(subset.graphs.sample_ids, ("s0", "s2"))
        self.assertEqual(subset.auxiliary_views["folds"], ["train", "test"])

    def test_sweep_dataset_subset_accepts_slices_through_shared_alignment_path(
        self,
    ) -> None:
        dataset = self._dataset_with_graphs_and_auxiliary()

        subset = dataset.subset(slice(1, 4, 2))

        np.testing.assert_array_equal(subset.sample_ids, np.array(["s1", "s3"]))
        np.testing.assert_array_equal(
            subset.mlip_features,
            np.array([[3.0, 4.0, 5.0], [9.0, 10.0, 11.0]]),
        )
        np.testing.assert_array_equal(subset.targets, np.array([1.5, 3.5]))
        self.assertEqual(subset.graphs.sample_ids, ("s1", "s3"))
        np.testing.assert_array_equal(
            subset.auxiliary_views["weights"],
            np.array([2.0, 4.0]),
        )
        self.assertEqual(subset.auxiliary_views["folds"], ["val", "holdout"])

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

    def test_sweep_dataset_rejects_duplicate_graph_view_ids(self) -> None:
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

        with patch.object(
            GraphDatasetView,
            "sample_ids",
            new_callable=PropertyMock,
            return_value=("s0", "s0"),
        ):
            with self.assertRaisesRegex(
                ValueError,
                r"graph_view sample_ids must be unique; duplicates: 's0'",
            ):
                SweepDataset(
                    mlip_features=np.arange(6, dtype=float).reshape(2, 3),
                    targets=np.arange(2, dtype=float),
                    sample_ids=np.array(["s0", "s1"]),
                    graph_view=graph_view,
                )


class SweepPayloadTests(unittest.TestCase):
    def _dataset_with_graphs_and_auxiliary(self) -> SweepDataset:
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
                GraphRecord(
                    sample_id="s2",
                    node_features=np.arange(2, dtype=float).reshape(1, 2),
                    edge_index=np.empty((2, 0), dtype=np.int64),
                ),
                GraphRecord(
                    sample_id="s3",
                    node_features=np.arange(8, dtype=float).reshape(4, 2),
                    edge_index=np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
                ),
                GraphRecord(
                    sample_id="s4",
                    node_features=np.arange(10, dtype=float).reshape(5, 2),
                    edge_index=np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64),
                ),
                GraphRecord(
                    sample_id="s5",
                    node_features=np.arange(4, dtype=float).reshape(2, 2),
                    edge_index=np.array([[0], [1]], dtype=np.int64),
                ),
            )
        )
        return SweepDataset(
            mlip_features=np.arange(18, dtype=float).reshape(6, 3),
            targets=np.arange(6, dtype=float) + 0.25,
            sample_ids=np.array(["s0", "s1", "s2", "s3", "s4", "s5"]),
            graph_view=graph_view,
            auxiliary_views={
                "weights": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                "folds": ["a", "b", "c", "d", "e", "f"],
            },
        )

    def _assert_subset_matches_indices(
        self,
        subset: SweepDataset,
        dataset: SweepDataset,
        indices: np.ndarray,
    ) -> None:
        expected_sample_ids = dataset.sample_ids[indices]
        expected_targets = dataset.targets[indices]
        expected_features = dataset.mlip_features[indices]
        expected_weights = dataset.auxiliary_views["weights"][indices]
        expected_folds = [dataset.auxiliary_views["folds"][index] for index in indices]

        np.testing.assert_array_equal(subset.sample_ids, expected_sample_ids)
        np.testing.assert_array_equal(subset.targets, expected_targets)
        np.testing.assert_array_equal(subset.mlip_features, expected_features)
        np.testing.assert_array_equal(subset.auxiliary_views["weights"], expected_weights)
        self.assertEqual(subset.auxiliary_views["folds"], expected_folds)
        self.assertEqual(subset.graphs.sample_ids, tuple(expected_sample_ids.tolist()))

    def _assert_disjoint_sample_ids(self, *subsets: SweepDataset) -> None:
        seen: set[str] = set()
        for subset in subsets:
            subset_ids = set(subset.sample_ids.tolist())
            self.assertTrue(seen.isdisjoint(subset_ids))
            seen.update(subset_ids)

    def _assert_loader_sample_ids(
        self,
        loader_payload: dict[str, object],
        *,
        expected_split_name: str,
        expected_sample_ids: tuple[str, ...],
    ) -> None:
        self.assertEqual(loader_payload["split_name"], expected_split_name)
        self.assertEqual(loader_payload["sample_ids"], expected_sample_ids)
        self.assertEqual(loader_payload["graph_ids"], expected_sample_ids)
        self.assertEqual(loader_payload["sample_id_set"], set(expected_sample_ids))

    def test_sweep_run_payload_converts_train_test_and_train_val_test_runner_inputs(
        self,
    ) -> None:
        dataset = SweepDataset(
            mlip_features=np.arange(18, dtype=float).reshape(6, 3),
            targets=np.arange(6, dtype=float),
        )
        requirements = SweepFamilyRequirements(
            min_train_size=4,
            requires_inner_validation=True,
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
                ),
                planning_requirements=requirements,
            ),
        )

        runner_payload = payload.to_runner_payload()

        self.assertIsInstance(runner_payload, SweepRunnerPayload)
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
        self.assertEqual(runner_payload.planning_requirements, requirements)

    def test_train_test_runner_input_builds_aligned_dataset_subsets(self) -> None:
        dataset = self._dataset_with_graphs_and_auxiliary()
        split = TrainTestSweepRunnerInput(
            dataset=dataset,
            sweep_size=3,
            train_idx=np.array([4, 1, 0]),
            test_idx=np.array([5, 2]),
        )

        subsets = split.dataset_subsets()

        self.assertIsInstance(subsets, TrainTestSplitDatasets)
        np.testing.assert_array_equal(
            subsets.train.sample_ids,
            np.array(["s4", "s1", "s0"]),
        )
        np.testing.assert_array_equal(
            subsets.test.sample_ids,
            np.array(["s5", "s2"]),
        )
        self.assertEqual(subsets.train.graphs.sample_ids, ("s4", "s1", "s0"))
        self.assertEqual(subsets.test.graphs.sample_ids, ("s5", "s2"))
        np.testing.assert_array_equal(
            subsets.train.auxiliary_views["weights"],
            np.array([5.0, 2.0, 1.0]),
        )
        self.assertEqual(subsets.test.auxiliary_views["folds"], ["f", "c"])
        np.testing.assert_array_equal(
            subsets.train.mlip_features,
            np.array(
                [
                    [12.0, 13.0, 14.0],
                    [3.0, 4.0, 5.0],
                    [0.0, 1.0, 2.0],
                ]
            ),
        )
        np.testing.assert_array_equal(subsets.test.targets, np.array([5.25, 2.25]))
        self.assertEqual(subsets.train.sample(0).graph.sample_id, "s4")
        self.assertEqual(subsets.test.sample(1).auxiliary, {"weights": 3.0, "folds": "c"})
        inputs = subsets.inputs
        self.assertIsInstance(inputs, TrainTestSplitDatasetInputs)
        self.assertIsInstance(inputs.train, SweepDatasetInputs)
        self.assertIs(inputs.train.mlip_features, subsets.train.mlip_features)
        self.assertEqual(inputs.train.graph_view_required().sample_ids, ("s4", "s1", "s0"))
        self.assertIs(inputs.test.mlip_features, subsets.test.mlip_features)
        self.assertEqual(inputs.test.graph_view_required().sample_ids, ("s5", "s2"))

    def test_train_test_runner_input_dataset_subsets_follow_declared_split_membership(
        self,
    ) -> None:
        dataset = self._dataset_with_graphs_and_auxiliary()
        train_idx = np.array([5, 1, 3])
        test_idx = np.array([4, 0, 2])
        split = TrainTestSweepRunnerInput(
            dataset=dataset,
            sweep_size=3,
            train_idx=train_idx,
            test_idx=test_idx,
        )

        subsets = split.dataset_subsets()

        self._assert_subset_matches_indices(subsets.train, dataset, train_idx)
        self._assert_subset_matches_indices(subsets.test, dataset, test_idx)
        self._assert_disjoint_sample_ids(subsets.train, subsets.test)
        self.assertEqual(
            set(subsets.train.sample_ids.tolist()) | set(subsets.test.sample_ids.tolist()),
            set(dataset.sample_ids.tolist()),
        )

    def test_train_test_runner_input_preserves_multimodal_identity_for_sample_ids(
        self,
    ) -> None:
        dataset = self._dataset_with_graphs_and_auxiliary()
        split = TrainTestSweepRunnerInput(
            dataset=dataset,
            sweep_size=3,
            train_idx=np.array([4, 1, 0]),
            test_idx=np.array([5, 2]),
        )

        subsets = split.dataset_subsets()
        expected_by_sample_id = {
            sample.sample_id: sample
            for sample in (dataset.sample(i) for i in range(len(dataset)))
        }

        for subset in (subsets.train, subsets.test):
            for row_index, sample_id in enumerate(subset.sample_ids.tolist()):
                subset_sample = subset.sample(row_index)
                expected = expected_by_sample_id[sample_id]
                self.assertEqual(subset_sample.sample_id, expected.sample_id)
                np.testing.assert_array_equal(
                    subset_sample.mlip_features,
                    expected.mlip_features,
                )
                self.assertEqual(subset_sample.target, expected.target)
                self.assertIs(subset_sample.graph, expected.graph)
                self.assertEqual(subset_sample.auxiliary, expected.auxiliary)

    def test_train_val_test_runner_input_builds_non_overlapping_dataset_subsets(
        self,
    ) -> None:
        dataset = self._dataset_with_graphs_and_auxiliary()
        split = TrainValTestSweepRunnerInput(
            dataset=dataset,
            sweep_size=4,
            train_idx=np.array([0, 2, 4]),
            val_idx=np.array([1]),
            test_idx=np.array([3, 5]),
        )

        subsets = split_to_dataset_subsets(split)

        self.assertIsInstance(subsets, TrainValTestSplitDatasets)
        self.assertEqual(set(subsets.train.sample_ids.tolist()), {"s0", "s2", "s4"})
        self.assertEqual(set(subsets.val.sample_ids.tolist()), {"s1"})
        self.assertEqual(set(subsets.test.sample_ids.tolist()), {"s3", "s5"})
        self.assertTrue(
            set(subsets.train.sample_ids.tolist()).isdisjoint(
                subsets.val.sample_ids.tolist()
            )
        )
        self.assertTrue(
            set(subsets.train.sample_ids.tolist()).isdisjoint(
                subsets.test.sample_ids.tolist()
            )
        )
        self.assertTrue(
            set(subsets.val.sample_ids.tolist()).isdisjoint(
                subsets.test.sample_ids.tolist()
            )
        )
        np.testing.assert_array_equal(
            subsets.train.mlip_features,
            np.array(
                [
                    [0.0, 1.0, 2.0],
                    [6.0, 7.0, 8.0],
                    [12.0, 13.0, 14.0],
                ]
            ),
        )
        np.testing.assert_array_equal(subsets.val.targets, np.array([1.25]))
        np.testing.assert_array_equal(
            subsets.test.auxiliary_views["weights"],
            np.array([4.0, 6.0]),
        )
        self.assertEqual(subsets.test.graphs.sample_ids, ("s3", "s5"))
        inputs = subsets.inputs
        self.assertIsInstance(inputs, TrainValTestSplitDatasetInputs)
        self.assertIs(inputs.train.mlip_features, subsets.train.mlip_features)
        self.assertEqual(inputs.train.graph_view_required().sample_ids, ("s0", "s2", "s4"))
        self.assertIs(inputs.val.mlip_features, subsets.val.mlip_features)
        self.assertEqual(inputs.val.graph_view_required().sample_ids, ("s1",))
        self.assertIs(inputs.test.mlip_features, subsets.test.mlip_features)
        self.assertEqual(inputs.test.graph_view_required().sample_ids, ("s3", "s5"))

    def test_split_to_dataset_subsets_preserves_ordered_membership_per_split(self) -> None:
        dataset = self._dataset_with_graphs_and_auxiliary()
        train_idx = np.array([4, 1, 5])
        val_idx = np.array([0])
        test_idx = np.array([3, 2])
        split = TrainValTestSweepRunnerInput(
            dataset=dataset,
            sweep_size=4,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
        )

        subsets = split_to_dataset_subsets(split)

        self._assert_subset_matches_indices(subsets.train, dataset, train_idx)
        self._assert_subset_matches_indices(subsets.val, dataset, val_idx)
        self._assert_subset_matches_indices(subsets.test, dataset, test_idx)
        self._assert_disjoint_sample_ids(subsets.train, subsets.val, subsets.test)
        self.assertEqual(
            (
                set(subsets.train.sample_ids.tolist())
                | set(subsets.val.sample_ids.tolist())
                | set(subsets.test.sample_ids.tolist())
            ),
            set(dataset.sample_ids.tolist()),
        )

    def test_train_val_test_runner_input_preserves_multimodal_identity_for_sample_ids(
        self,
    ) -> None:
        dataset = self._dataset_with_graphs_and_auxiliary()
        split = TrainValTestSweepRunnerInput(
            dataset=dataset,
            sweep_size=4,
            train_idx=np.array([4, 0, 2]),
            val_idx=np.array([5, 1]),
            test_idx=np.array([3]),
        )

        subsets = split.dataset_subsets()
        expected_by_sample_id = {
            sample.sample_id: sample
            for sample in (dataset.sample(i) for i in range(len(dataset)))
        }

        for subset in (subsets.train, subsets.val, subsets.test):
            for row_index, sample_id in enumerate(subset.sample_ids.tolist()):
                subset_sample = subset.sample(row_index)
                expected = expected_by_sample_id[sample_id]
                self.assertEqual(subset_sample.sample_id, expected.sample_id)
                np.testing.assert_array_equal(
                    subset_sample.mlip_features,
                    expected.mlip_features,
                )
                self.assertEqual(subset_sample.target, expected.target)
                self.assertIs(subset_sample.graph, expected.graph)
                self.assertEqual(subset_sample.auxiliary, expected.auxiliary)

    def test_split_to_loaders_uses_thin_adapter_seam(self) -> None:
        dataset = self._dataset_with_graphs_and_auxiliary()
        split = TrainValTestSweepRunnerInput(
            dataset=dataset,
            sweep_size=4,
            train_idx=np.array([0, 1, 2]),
            val_idx=np.array([3]),
            test_idx=np.array([4, 5]),
        )
        calls: list[dict[str, object]] = []

        def loader_factory(dataset: SweepDataset, *, split_name: str) -> dict[str, object]:
            payload = {
                "split_name": split_name,
                "sample_ids": tuple(dataset.sample_ids.tolist()),
                "targets": tuple(dataset.targets.tolist()),
                "weights": tuple(dataset.auxiliary_views["weights"].tolist()),
                "graph_ids": dataset.graphs.sample_ids,
            }
            calls.append(payload)
            return payload

        self.assertIsInstance(loader_factory, DatasetLoaderFactory)

        loaders = split_to_loaders(split, loader_factory)

        self.assertIsInstance(loaders, TrainValTestSplitLoaders)
        self.assertEqual(
            calls,
            [
                {
                    "split_name": "train",
                    "sample_ids": ("s0", "s1", "s2"),
                    "targets": (0.25, 1.25, 2.25),
                    "weights": (1.0, 2.0, 3.0),
                    "graph_ids": ("s0", "s1", "s2"),
                },
                {
                    "split_name": "val",
                    "sample_ids": ("s3",),
                    "targets": (3.25,),
                    "weights": (4.0,),
                    "graph_ids": ("s3",),
                },
                {
                    "split_name": "test",
                    "sample_ids": ("s4", "s5"),
                    "targets": (4.25, 5.25),
                    "weights": (5.0, 6.0),
                    "graph_ids": ("s4", "s5"),
                },
            ],
        )
        self.assertEqual(loaders.train["split_name"], "train")
        self.assertEqual(loaders.val["sample_ids"], ("s3",))
        self.assertEqual(loaders.test["sample_ids"], ("s4", "s5"))
        self.assertEqual(loaders.test["graph_ids"], ("s4", "s5"))

    def test_split_to_loader_inputs_separates_subset_selection_from_batching(self) -> None:
        dataset = self._dataset_with_graphs_and_auxiliary()
        split = TrainValTestSweepRunnerInput(
            dataset=dataset,
            sweep_size=4,
            train_idx=np.array([5, 1, 0]),
            val_idx=np.array([4]),
            test_idx=np.array([3, 2]),
        )

        def train_collate(rows: object) -> object:
            return rows

        def eval_collate(rows: object) -> object:
            return rows

        class FakeLoaderAdapter:
            def batching_for_split(self, *, split_name: str) -> LoaderBatching:
                if split_name == "train":
                    return LoaderBatching(
                        batch_size=2,
                        shuffle=True,
                        collate_fn=train_collate,
                    )
                return LoaderBatching(
                    batch_size=4,
                    shuffle=False,
                    collate_fn=eval_collate,
                )

            def build_loader(self, loader_input: LoaderAdapterInput) -> object:
                raise AssertionError("build_loader should not be called in this test")

        adapter = FakeLoaderAdapter()
        self.assertIsInstance(adapter, DatasetLoaderAdapter)

        loader_inputs = split_to_loader_inputs(split, adapter)

        self.assertIsInstance(loader_inputs, TrainValTestSplitLoaderInputs)
        np.testing.assert_array_equal(
            loader_inputs.train.dataset.sample_ids,
            np.array(["s5", "s1", "s0"]),
        )
        np.testing.assert_array_equal(
            loader_inputs.val.dataset.sample_ids,
            np.array(["s4"]),
        )
        np.testing.assert_array_equal(
            loader_inputs.test.dataset.sample_ids,
            np.array(["s3", "s2"]),
        )
        self.assertEqual(loader_inputs.train.split_name, "train")
        self.assertEqual(loader_inputs.val.split_name, "val")
        self.assertEqual(loader_inputs.test.split_name, "test")
        self.assertEqual(loader_inputs.train.batching.batch_size, 2)
        self.assertTrue(loader_inputs.train.batching.shuffle)
        self.assertIs(loader_inputs.train.batching.collate_fn, train_collate)
        self.assertEqual(loader_inputs.val.batching.batch_size, 4)
        self.assertFalse(loader_inputs.val.batching.shuffle)
        self.assertIs(loader_inputs.val.batching.collate_fn, eval_collate)
        self.assertEqual(loader_inputs.test.batching.batch_size, 4)
        self.assertFalse(loader_inputs.test.batching.shuffle)
        self.assertIs(loader_inputs.test.batching.collate_fn, eval_collate)

    def test_train_test_loader_inputs_wrap_legacy_loader_factory_with_default_batching(
        self,
    ) -> None:
        dataset = self._dataset_with_graphs_and_auxiliary()
        split = TrainTestSweepRunnerInput(
            dataset=dataset,
            sweep_size=3,
            train_idx=np.array([2, 0, 5]),
            test_idx=np.array([4, 1, 3]),
        )

        def loader_factory(dataset: SweepDataset, *, split_name: str) -> dict[str, object]:
            return {
                "split_name": split_name,
                "sample_ids": tuple(dataset.sample_ids.tolist()),
            }

        self.assertIsInstance(loader_factory, DatasetLoaderFactory)

        adapter = DatasetLoaderFactoryAdapter(loader_factory)
        loader_inputs = split.loader_inputs(adapter)

        self.assertIsInstance(loader_inputs, TrainTestSplitLoaderInputs)
        self.assertEqual(loader_inputs.train.batching, LoaderBatching())
        self.assertEqual(loader_inputs.test.batching, LoaderBatching())
        np.testing.assert_array_equal(
            loader_inputs.train.dataset.sample_ids,
            np.array(["s2", "s0", "s5"]),
        )
        np.testing.assert_array_equal(
            loader_inputs.test.dataset.sample_ids,
            np.array(["s4", "s1", "s3"]),
        )

    def test_train_eval_loader_policy_uses_distinct_train_and_eval_semantics(self) -> None:
        def train_collate(samples: object) -> object:
            return samples

        def eval_collate(samples: object) -> object:
            return samples

        policy = TrainEvalLoaderPolicy(
            batch_size=8,
            eval_batch_size=32,
            train_shuffle=True,
            eval_shuffle=False,
            train_collate_fn=train_collate,
            eval_collate_fn=eval_collate,
        )

        train_batching = policy.batching_for_split(split_name="train")
        val_batching = policy.batching_for_split(split_name="val")
        test_batching = policy.batching_for_split(split_name="test")

        self.assertEqual(train_batching.batch_size, 8)
        self.assertTrue(train_batching.shuffle)
        self.assertIs(train_batching.collate_fn, train_collate)
        self.assertEqual(val_batching.batch_size, 32)
        self.assertFalse(val_batching.shuffle)
        self.assertIs(val_batching.collate_fn, eval_collate)
        self.assertEqual(test_batching.batch_size, 32)
        self.assertFalse(test_batching.shuffle)
        self.assertIs(test_batching.collate_fn, eval_collate)

    def test_sweep_dataset_batch_loader_adapter_shuffles_train_and_not_eval(self) -> None:
        dataset = SweepDataset(
            mlip_features=np.arange(18, dtype=float).reshape(6, 3),
            targets=np.arange(6, dtype=float) + 0.5,
            sample_ids=np.array(["s0", "s1", "s2", "s3", "s4", "s5"]),
        )
        split = TrainValTestSweepRunnerInput(
            dataset=dataset,
            sweep_size=4,
            train_idx=np.array([0, 1, 2, 3]),
            val_idx=np.array([4, 5]),
            test_idx=np.array([2, 1]),
        )
        adapter = SweepDatasetBatchLoaderAdapter(
            policy=TrainEvalLoaderPolicy(
                batch_size=2,
                eval_batch_size=1,
                train_shuffle=True,
                eval_shuffle=False,
            )
        )

        first = split_to_loaders(split, adapter)
        second = split.loaders(adapter)

        self.assertIsInstance(adapter, DatasetLoaderAdapter)
        self.assertEqual([batch.sample_ids for batch in first.train], [("s2", "s0"), ("s1", "s3")])
        self.assertEqual([batch.sample_ids for batch in first.val], [("s4",), ("s5",)])
        self.assertEqual([batch.sample_ids for batch in first.test], [("s2",), ("s1",)])
        self.assertEqual(
            [batch.sample_ids for batch in second.train],
            [batch.sample_ids for batch in first.train],
        )
        self.assertEqual(
            [batch.sample_ids for batch in second.val],
            [batch.sample_ids for batch in first.val],
        )
        self.assertEqual(
            [batch.sample_ids for batch in second.test],
            [batch.sample_ids for batch in first.test],
        )

    def test_split_to_loaders_builds_batched_mlip_only_loaders_without_split_leakage(
        self,
    ) -> None:
        dataset = SweepDataset(
            mlip_features=np.arange(18, dtype=float).reshape(6, 3),
            targets=np.arange(6, dtype=float) + 0.5,
            sample_ids=np.array(["s0", "s1", "s2", "s3", "s4", "s5"]),
        )
        split = TrainValTestSweepRunnerInput(
            dataset=dataset,
            sweep_size=4,
            train_idx=np.array([5, 1, 0]),
            val_idx=np.array([4, 2]),
            test_idx=np.array([3]),
        )

        class BatchedMlipsOnlyAdapter:
            def batching_for_split(self, *, split_name: str) -> LoaderBatching:
                if split_name == "train":
                    return LoaderBatching(
                        batch_size=2,
                        shuffle=False,
                    )
                return LoaderBatching(
                    batch_size=1,
                    shuffle=False,
                )

            def build_loader(self, loader_input: LoaderAdapterInput) -> tuple[SweepBatch, ...]:
                return build_sweep_batches(loader_input)

        adapter = BatchedMlipsOnlyAdapter()
        self.assertIsInstance(adapter, DatasetLoaderAdapter)

        loaders = split_to_loaders(split, adapter)

        self.assertEqual(len(loaders.train), 2)
        self.assertEqual(len(loaders.val), 2)
        self.assertEqual(len(loaders.test), 1)
        self.assertEqual(loaders.train[0].split_name, "train")
        self.assertEqual(loaders.val[0].split_name, "val")
        self.assertEqual(loaders.test[0].split_name, "test")
        self.assertEqual(loaders.train[0].sample_ids, ("s5", "s1"))
        np.testing.assert_array_equal(
            loaders.train[0].targets,
            np.array([5.5, 1.5]),
        )
        np.testing.assert_array_equal(
            loaders.train[0].mlip_features,
            np.array([[15.0, 16.0, 17.0], [3.0, 4.0, 5.0]]),
        )
        self.assertEqual(loaders.train[1].sample_ids, ("s0",))
        np.testing.assert_array_equal(
            loaders.train[1].mlip_features,
            np.array([[0.0, 1.0, 2.0]]),
        )
        self.assertEqual(loaders.val[0].sample_ids, ("s4",))
        np.testing.assert_array_equal(
            loaders.val[0].targets,
            np.array([4.5]),
        )
        self.assertEqual(loaders.val[1].sample_ids, ("s2",))
        np.testing.assert_array_equal(
            loaders.val[1].mlip_features,
            np.array([[6.0, 7.0, 8.0]]),
        )
        self.assertEqual(loaders.test[0].sample_ids, ("s3",))
        np.testing.assert_array_equal(
            loaders.test[0].targets,
            np.array([3.5]),
        )
        np.testing.assert_array_equal(
            loaders.test[0].mlip_features,
            np.array([[9.0, 10.0, 11.0]]),
        )

        train_ids = set().union(*(set(batch.sample_ids) for batch in loaders.train))
        val_ids = set().union(*(set(batch.sample_ids) for batch in loaders.val))
        test_ids = set().union(*(set(batch.sample_ids) for batch in loaders.test))
        self.assertEqual(train_ids, {"s5", "s1", "s0"})
        self.assertEqual(val_ids, {"s4", "s2"})
        self.assertEqual(test_ids, {"s3"})
        self.assertTrue(train_ids.isdisjoint(val_ids))
        self.assertTrue(train_ids.isdisjoint(test_ids))
        self.assertTrue(val_ids.isdisjoint(test_ids))

    def test_collate_sweep_samples_preserves_split_identity_in_batch_object(self) -> None:
        dataset = self._dataset_with_graphs_and_auxiliary()

        batch = collate_sweep_samples(
            (dataset.sample(5), dataset.sample(1)),
            split_name="train",
            batch_index=3,
        )

        self.assertIsInstance(batch, SweepBatch)
        self.assertEqual(batch.split_name, "train")
        self.assertEqual(batch.batch_index, 3)
        self.assertEqual(batch.sample_ids, ("s5", "s1"))
        np.testing.assert_array_equal(batch.targets, np.array([5.25, 1.25]))
        np.testing.assert_array_equal(
            batch.mlip_features,
            np.array([[15.0, 16.0, 17.0], [3.0, 4.0, 5.0]]),
        )
        self.assertEqual(tuple(graph.sample_id for graph in batch.graphs if graph is not None), ("s5", "s1"))
        self.assertEqual(batch.auxiliary, ({"weights": 6.0, "folds": "f"}, {"weights": 2.0, "folds": "b"}))

    def test_split_to_loaders_passes_subset_datasets_without_cross_split_leakage(
        self,
    ) -> None:
        dataset = self._dataset_with_graphs_and_auxiliary()
        train_idx = np.array([5, 0, 2])
        val_idx = np.array([4, 1])
        test_idx = np.array([3])
        split = TrainValTestSweepRunnerInput(
            dataset=dataset,
            sweep_size=4,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
        )
        seen_subset_ids: set[int] = set()

        def loader_factory(dataset: SweepDataset, *, split_name: str) -> dict[str, object]:
            sample_ids = tuple(dataset.sample_ids.tolist())
            seen_subset_ids.add(id(dataset))
            return {
                "split_name": split_name,
                "sample_ids": sample_ids,
                "sample_id_set": set(sample_ids),
                "targets": tuple(dataset.targets.tolist()),
                "weights": tuple(dataset.auxiliary_views["weights"].tolist()),
                "graph_ids": dataset.graphs.sample_ids,
            }

        loaders = split_to_loaders(split, loader_factory)

        self.assertIsInstance(loaders, TrainValTestSplitLoaders)
        self.assertEqual(len(seen_subset_ids), 3)
        self.assertEqual(loaders.train["sample_ids"], ("s5", "s0", "s2"))
        self.assertEqual(loaders.val["sample_ids"], ("s4", "s1"))
        self.assertEqual(loaders.test["sample_ids"], ("s3",))
        self.assertEqual(loaders.train["graph_ids"], loaders.train["sample_ids"])
        self.assertEqual(loaders.val["graph_ids"], loaders.val["sample_ids"])
        self.assertEqual(loaders.test["graph_ids"], loaders.test["sample_ids"])
        self.assertTrue(
            loaders.train["sample_id_set"].isdisjoint(loaders.val["sample_id_set"])
        )
        self.assertTrue(
            loaders.train["sample_id_set"].isdisjoint(loaders.test["sample_id_set"])
        )
        self.assertTrue(
            loaders.val["sample_id_set"].isdisjoint(loaders.test["sample_id_set"])
        )
        self.assertEqual(
            (
                loaders.train["sample_id_set"]
                | loaders.val["sample_id_set"]
                | loaders.test["sample_id_set"]
            ),
            set(dataset.sample_ids.tolist()),
        )

    def test_split_to_loaders_preserves_sample_ids_for_non_monotonic_split_indices(
        self,
    ) -> None:
        dataset = self._dataset_with_graphs_and_auxiliary()
        split = TrainValTestSweepRunnerInput(
            dataset=dataset,
            sweep_size=4,
            train_idx=np.array([4, 0, 5]),
            val_idx=np.array([2, 1]),
            test_idx=np.array([3]),
        )

        def loader_factory(dataset: SweepDataset, *, split_name: str) -> dict[str, object]:
            sample_ids = tuple(dataset.sample_ids.tolist())
            return {
                "split_name": split_name,
                "sample_ids": sample_ids,
                "sample_id_set": set(sample_ids),
                "graph_ids": dataset.graphs.sample_ids,
            }

        loaders = split_to_loaders(split, loader_factory)

        self._assert_loader_sample_ids(
            loaders.train,
            expected_split_name="train",
            expected_sample_ids=("s4", "s0", "s5"),
        )
        self._assert_loader_sample_ids(
            loaders.val,
            expected_split_name="val",
            expected_sample_ids=("s2", "s1"),
        )
        self._assert_loader_sample_ids(
            loaders.test,
            expected_split_name="test",
            expected_sample_ids=("s3",),
        )
        self.assertTrue(
            loaders.train["sample_id_set"].isdisjoint(loaders.val["sample_id_set"])
        )
        self.assertTrue(
            loaders.train["sample_id_set"].isdisjoint(loaders.test["sample_id_set"])
        )
        self.assertTrue(
            loaders.val["sample_id_set"].isdisjoint(loaders.test["sample_id_set"])
        )

    def test_split_to_loaders_repeated_construction_preserves_split_membership(
        self,
    ) -> None:
        dataset = self._dataset_with_graphs_and_auxiliary()
        split = TrainValTestSweepRunnerInput(
            dataset=dataset,
            sweep_size=4,
            train_idx=np.array([1, 5, 0]),
            val_idx=np.array([4]),
            test_idx=np.array([3, 2]),
        )
        calls: list[tuple[str, tuple[str, ...], int]] = []

        def loader_factory(dataset: SweepDataset, *, split_name: str) -> dict[str, object]:
            sample_ids = tuple(dataset.sample_ids.tolist())
            calls.append((split_name, sample_ids, dataset))
            return {
                "split_name": split_name,
                "sample_ids": sample_ids,
                "sample_id_set": set(sample_ids),
                "graph_ids": dataset.graphs.sample_ids,
                "dataset_ref": dataset,
            }

        first = split_to_loaders(split, loader_factory)
        second = split.loaders(loader_factory)

        for loaders in (first, second):
            self._assert_loader_sample_ids(
                loaders.train,
                expected_split_name="train",
                expected_sample_ids=("s1", "s5", "s0"),
            )
            self._assert_loader_sample_ids(
                loaders.val,
                expected_split_name="val",
                expected_sample_ids=("s4",),
            )
            self._assert_loader_sample_ids(
                loaders.test,
                expected_split_name="test",
                expected_sample_ids=("s3", "s2"),
            )
            self.assertTrue(
                loaders.train["sample_id_set"].isdisjoint(loaders.val["sample_id_set"])
            )
            self.assertTrue(
                loaders.train["sample_id_set"].isdisjoint(loaders.test["sample_id_set"])
            )
            self.assertTrue(
                loaders.val["sample_id_set"].isdisjoint(loaders.test["sample_id_set"])
            )

        self.assertEqual(
            calls,
            [
                ("train", ("s1", "s5", "s0"), first.train["dataset_ref"]),
                ("val", ("s4",), first.val["dataset_ref"]),
                ("test", ("s3", "s2"), first.test["dataset_ref"]),
                ("train", ("s1", "s5", "s0"), second.train["dataset_ref"]),
                ("val", ("s4",), second.val["dataset_ref"]),
                ("test", ("s3", "s2"), second.test["dataset_ref"]),
            ],
        )
        self.assertIsNot(first.train["dataset_ref"], second.train["dataset_ref"])
        self.assertIsNot(first.val["dataset_ref"], second.val["dataset_ref"])
        self.assertIsNot(first.test["dataset_ref"], second.test["dataset_ref"])

    def test_train_test_runner_input_loaders_delegate_per_subset(self) -> None:
        dataset = self._dataset_with_graphs_and_auxiliary()
        split = TrainTestSweepRunnerInput(
            dataset=dataset,
            sweep_size=3,
            train_idx=np.array([0, 3, 5]),
            test_idx=np.array([1, 2, 4]),
        )

        loaders = split.loaders(
            lambda subset, *, split_name: {
                "split_name": split_name,
                "sample_ids": tuple(subset.sample_ids.tolist()),
                "targets": tuple(subset.targets.tolist()),
                "graph_ids": subset.graphs.sample_ids,
            }
        )

        self.assertIsInstance(loaders, TrainTestSplitLoaders)
        self.assertEqual(
            loaders.train,
            {
                "split_name": "train",
                "sample_ids": ("s0", "s3", "s5"),
                "targets": (0.25, 3.25, 5.25),
                "graph_ids": ("s0", "s3", "s5"),
            },
        )
        self.assertEqual(
            loaders.test,
            {
                "split_name": "test",
                "sample_ids": ("s1", "s2", "s4"),
                "targets": (1.25, 2.25, 4.25),
                "graph_ids": ("s1", "s2", "s4"),
            },
        )

    def test_train_test_runner_input_loaders_preserve_subset_order_and_alignment(
        self,
    ) -> None:
        dataset = self._dataset_with_graphs_and_auxiliary()
        split = TrainTestSweepRunnerInput(
            dataset=dataset,
            sweep_size=3,
            train_idx=np.array([2, 5, 1]),
            test_idx=np.array([4, 0, 3]),
        )

        def loader_factory(subset: SweepDataset, *, split_name: str) -> dict[str, object]:
            return {
                "split_name": split_name,
                "sample_ids": tuple(subset.sample_ids.tolist()),
                "targets": tuple(subset.targets.tolist()),
                "weights": tuple(subset.auxiliary_views["weights"].tolist()),
                "folds": tuple(subset.auxiliary_views["folds"]),
                "graph_ids": subset.graphs.sample_ids,
            }

        loaders = split.loaders(loader_factory)

        self.assertIsInstance(loaders, TrainTestSplitLoaders)
        self.assertEqual(
            loaders.train,
            {
                "split_name": "train",
                "sample_ids": ("s2", "s5", "s1"),
                "targets": (2.25, 5.25, 1.25),
                "weights": (3.0, 6.0, 2.0),
                "folds": ("c", "f", "b"),
                "graph_ids": ("s2", "s5", "s1"),
            },
        )
        self.assertEqual(
            loaders.test,
            {
                "split_name": "test",
                "sample_ids": ("s4", "s0", "s3"),
                "targets": (4.25, 0.25, 3.25),
                "weights": (5.0, 1.0, 4.0),
                "folds": ("e", "a", "d"),
                "graph_ids": ("s4", "s0", "s3"),
            },
        )


class LearningCurveResultsTests(unittest.TestCase):
    def test_to_mapping_exposes_all_result_fields(self) -> None:
        ridge_df = pd.DataFrame(
            {"n_train": [4, 8], "rmse_mean": [0.4, 0.3], "rmse_std": [0.05, 0.02]}
        )
        ridge_uq_df = pd.DataFrame(
            {
                "n_train": [4, 8],
                "miscalibration_area": [0.11, 0.08],
                "miscalibration_area_std": [0.01, 0.01],
                "sharpness": [0.21, 0.18],
                "sharpness_std": [0.02, 0.02],
                "dispersion": [0.31, 0.28],
                "dispersion_std": [0.03, 0.03],
                "uncertainty_kind": ["spread_only", "spread_only"],
            }
        )
        probe_gnn_selection_df = pd.DataFrame(
            {"n_train": [8], "hidden_dim": [32], "best_validation_score": [0.12]}
        )
        results = LearningCurveResults(
            ridge_df=ridge_df,
            ridge_uq_df=ridge_uq_df,
            probe_gnn_selection_df=probe_gnn_selection_df,
        )

        mapping = results.to_mapping()

        self.assertIs(mapping["ridge_df"], ridge_df)
        self.assertIs(mapping["ridge_uq_df"], ridge_uq_df)
        self.assertIs(mapping["probe_gnn_selection_df"], probe_gnn_selection_df)
        self.assertIn("latent_uq_df", mapping)
        self.assertIsNone(mapping["latent_uq_df"])

    def test_merge_unions_uq_frames_by_n_train(self) -> None:
        base = LearningCurveResults(
            ridge_uq_df=pd.DataFrame(
                {
                    "n_train": [1, 10],
                    "miscalibration_area": [0.3, 0.2],
                    "miscalibration_area_std": [0.03, 0.02],
                    "sharpness": [0.4, 0.3],
                    "sharpness_std": [0.04, 0.03],
                    "dispersion": [0.5, 0.4],
                    "dispersion_std": [0.05, 0.04],
                    "uncertainty_kind": ["spread_only", "spread_only"],
                }
            )
        )
        update = LearningCurveResults(
            ridge_uq_df=pd.DataFrame(
                {
                    "n_train": [20, 30],
                    "miscalibration_area": [0.15, 0.1],
                    "miscalibration_area_std": [0.02, 0.01],
                    "sharpness": [0.25, 0.2],
                    "sharpness_std": [0.03, 0.02],
                    "dispersion": [0.35, 0.3],
                    "dispersion_std": [0.04, 0.03],
                    "uncertainty_kind": ["spread_only", "spread_only"],
                }
            )
        )

        merged = base.merge(update)

        pd.testing.assert_frame_equal(
            merged.ridge_uq_df,
            pd.DataFrame(
                {
                    "n_train": [1, 10, 20, 30],
                    "miscalibration_area": [0.3, 0.2, 0.15, 0.1],
                    "miscalibration_area_std": [0.03, 0.02, 0.02, 0.01],
                    "sharpness": [0.4, 0.3, 0.25, 0.2],
                    "sharpness_std": [0.04, 0.03, 0.03, 0.02],
                    "dispersion": [0.5, 0.4, 0.35, 0.3],
                    "dispersion_std": [0.05, 0.04, 0.04, 0.03],
                    "uncertainty_kind": [
                        "spread_only",
                        "spread_only",
                        "spread_only",
                        "spread_only",
                    ],
                }
            ),
        )

    def test_sweep_runner_artifacts_can_hold_uq_summary(self) -> None:
        metrics = pd.DataFrame(
            {"n_train": [4], "rmse_mean": [0.3], "rmse_std": [0.05]}
        )
        uq_summary = pd.DataFrame(
            {
                "n_train": [4],
                "miscalibration_area": [0.08],
                "miscalibration_area_std": [0.01],
                "sharpness": [0.18],
                "sharpness_std": [0.02],
                "dispersion": [0.28],
                "dispersion_std": [0.03],
                "uncertainty_kind": ["calibrated"],
            }
        )

        artifacts = SweepRunnerArtifacts(metrics=metrics, uq_summary=uq_summary)

        self.assertIs(artifacts.metrics, metrics)
        self.assertIsNone(artifacts.selection_metadata)
        self.assertIs(artifacts.uq_summary, uq_summary)

    def test_merge_unions_metric_frames_by_n_train(self) -> None:
        base = LearningCurveResults(
            ridge_df=pd.DataFrame(
                {
                    "n_train": [1, 10, 20],
                    "rmse_mean": [0.8, 0.5, 0.4],
                    "rmse_std": [0.1, 0.07, 0.05],
                }
            )
        )
        update = LearningCurveResults(
            ridge_df=pd.DataFrame(
                {
                    "n_train": [30, 40],
                    "rmse_mean": [0.35, 0.3],
                    "rmse_std": [0.04, 0.03],
                }
            )
        )

        merged = base.merge(update)

        pd.testing.assert_frame_equal(
            merged.ridge_df,
            pd.DataFrame(
                {
                    "n_train": [1, 10, 20, 30, 40],
                    "rmse_mean": [0.8, 0.5, 0.4, 0.35, 0.3],
                    "rmse_std": [0.1, 0.07, 0.05, 0.04, 0.03],
                }
            ),
        )

    def test_merge_unions_selection_frames_by_n_train(self) -> None:
        base = LearningCurveResults(
            ridge_selection_df=pd.DataFrame(
                {
                    "n_train": [1, 10, 20],
                    "alpha": [0.1, 1.0, 10.0],
                }
            )
        )
        update = LearningCurveResults(
            ridge_selection_df=pd.DataFrame(
                {
                    "n_train": [30, 40],
                    "alpha": [7.5, 10.0],
                }
            )
        )

        merged = base.merge(update)

        pd.testing.assert_frame_equal(
            merged.ridge_selection_df,
            pd.DataFrame(
                {
                    "n_train": [1, 10, 20, 30, 40],
                    "alpha": [0.1, 1.0, 10.0, 7.5, 10.0],
                }
            ),
        )

    def test_merge_rejects_duplicate_metric_rows_without_overwrite(self) -> None:
        base = LearningCurveResults(
            ridge_df=pd.DataFrame(
                {
                    "n_train": [10, 20],
                    "rmse_mean": [0.5, 0.4],
                    "rmse_std": [0.07, 0.05],
                }
            )
        )
        update = LearningCurveResults(
            ridge_df=pd.DataFrame(
                {
                    "n_train": [20, 30],
                    "rmse_mean": [0.39, 0.35],
                    "rmse_std": [0.045, 0.04],
                }
            )
        )

        with self.assertRaisesRegex(
            ValueError,
            r"ridge_df contains duplicate n_train rows: \[20\]",
        ):
            base.merge(update)

    def test_merge_allows_duplicate_metric_rows_with_overwrite(self) -> None:
        base = LearningCurveResults(
            ridge_df=pd.DataFrame(
                {
                    "n_train": [10, 20],
                    "rmse_mean": [0.5, 0.4],
                    "rmse_std": [0.07, 0.05],
                }
            )
        )
        update = LearningCurveResults(
            ridge_df=pd.DataFrame(
                {
                    "n_train": [20, 30],
                    "rmse_mean": [0.39, 0.35],
                    "rmse_std": [0.045, 0.04],
                }
            )
        )

        merged = base.merge(update, overwrite_fields={"ridge_df"})

        pd.testing.assert_frame_equal(
            merged.ridge_df,
            pd.DataFrame(
                {
                    "n_train": [10, 20, 30],
                    "rmse_mean": [0.5, 0.39, 0.35],
                    "rmse_std": [0.07, 0.045, 0.04],
                }
            ),
        )

    def test_merge_screening_frames_keys_by_n_budget(self) -> None:
        base = LearningCurveResults(
            ridge_df=pd.DataFrame(
                {
                    "n_budget": [5],
                    "n_train": [3],
                    "n_screen": [2],
                    "screen_fraction": [0.4],
                    "rmse_mean": [0.5],
                    "rmse_std": [0.07],
                }
            )
        )
        update = LearningCurveResults(
            ridge_df=pd.DataFrame(
                {
                    "n_budget": [6],
                    "n_train": [4],
                    "n_screen": [2],
                    "screen_fraction": [1 / 3],
                    "rmse_mean": [0.39],
                    "rmse_std": [0.045],
                }
            )
        )

        merged = base.merge(update)

        pd.testing.assert_frame_equal(
            merged.ridge_df,
            pd.DataFrame(
                {
                    "n_budget": [5, 6],
                    "n_train": [3, 4],
                    "n_screen": [2, 2],
                    "screen_fraction": [0.4, 1 / 3],
                    "rmse_mean": [0.5, 0.39],
                    "rmse_std": [0.07, 0.045],
                }
            ),
        )
