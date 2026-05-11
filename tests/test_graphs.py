from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import PropertyMock, patch

from ase import Atoms
import numpy as np

from oasis.graphs import (
    AtomsToGraphPolicy,
    atoms_to_graph_dataset_view,
    atoms_to_graph_record,
    build_graph_sweep_dataset,
    dump_graph_dataset_view,
    load_graph_dataset_view,
    save_graph_dataset_view,
)
from oasis.sweep import (
    GraphDatasetView,
    GraphRecord,
    TrainValTestSweepRunnerInput,
)


class LoadGraphDatasetViewTests(unittest.TestCase):
    def test_load_graph_dataset_view_reads_valid_records(self) -> None:
        payload = [
            {
                "sample_id": "s0",
                "node_features": [[0.0, 1.0], [1.0, 0.0]],
                "edge_index": [[0], [1]],
                "node_positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                "edge_features": [[0.5, 1.5]],
                "graph_features": [2.0, 3.0],
            },
            {
                "sample_id": "s1",
                "node_features": [[1.0], [2.0], [3.0]],
                "edge_index": [[0, 1], [1, 2]],
            },
        ]

        view = _load_from_payload(payload)

        self.assertEqual(view.sample_ids, ("s0", "s1"))
        self.assertEqual(view["s0"].n_nodes, 2)
        self.assertEqual(view["s1"].n_edges, 2)

    def test_load_graph_dataset_view_rejects_missing_required_fields(self) -> None:
        payload = [
            {
                "sample_id": "s0",
                "node_features": [[0.0, 1.0], [1.0, 0.0]],
            }
        ]

        with self.assertRaisesRegex(ValueError, "missing required fields: edge_index"):
            _load_from_payload(payload)

    def test_load_graph_dataset_view_rejects_duplicate_sample_ids(self) -> None:
        payload = [
            {
                "sample_id": "s0",
                "node_features": [[0.0], [1.0]],
                "edge_index": [[0], [1]],
            },
            {
                "sample_id": "s0",
                "node_features": [[2.0], [3.0]],
                "edge_index": [[0], [1]],
            },
        ]

        with self.assertRaisesRegex(
            ValueError,
            r"duplicate graph record for sample_id='s0'",
        ):
            _load_from_payload(payload)

    def test_load_graph_dataset_view_rejects_malformed_records(self) -> None:
        payload = [
            {
                "sample_id": "s0",
                "node_features": [[0.0], [1.0]],
                "edge_index": [[0, 1, 2]],
            }
        ]

        with self.assertRaisesRegex(ValueError, "edge_index must have shape"):
            _load_from_payload(payload)


def _load_from_payload(payload: object):
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "graphs.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return load_graph_dataset_view(path)


class AtomsToGraphConversionTests(unittest.TestCase):
    def test_atoms_to_graph_policy_requires_positive_cutoff_multiplier(self) -> None:
        with self.assertRaisesRegex(ValueError, "cutoff_multiplier must be positive"):
            AtomsToGraphPolicy(cutoff_multiplier=0.0)

    def test_atoms_to_graph_record_uses_atomic_numbers_and_sorted_directed_edges(
        self,
    ) -> None:
        atoms = Atoms(
            "OHH",
            positions=[
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [0.0, 0.96, 0.0],
            ],
        )

        record = atoms_to_graph_record(atoms, sample_id="water")

        np.testing.assert_array_equal(
            record.node_features,
            np.array([[8.0], [1.0], [1.0]]),
        )
        np.testing.assert_allclose(
            record.node_positions,
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.96, 0.0, 0.0],
                    [0.0, 0.96, 0.0],
                ]
            ),
        )
        np.testing.assert_array_equal(
            record.edge_index,
            np.array([[0, 0, 1, 2], [1, 2, 0, 0]], dtype=np.int64),
        )
        np.testing.assert_allclose(
            record.edge_features,
            np.full((4, 1), 0.96),
        )
        self.assertIsNone(record.graph_features)
        self.assertEqual(record.sample_id, "water")
        self.assertEqual(record.node_features.dtype, float)
        self.assertEqual(record.node_positions.dtype, float)
        self.assertEqual(record.edge_index.dtype, np.int64)
        self.assertEqual(record.edge_features.dtype, float)

    def test_atoms_to_graph_record_handles_structures_with_no_edges(self) -> None:
        atoms = Atoms(
            "He2",
            positions=[
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
            ],
        )

        record = atoms_to_graph_record(atoms, sample_id="isolated")

        self.assertEqual(record.edge_index.shape, (2, 0))
        self.assertEqual(record.edge_features.shape, (0, 1))

    def test_atoms_to_graph_record_respects_cutoff_policy(self) -> None:
        atoms = Atoms(
            "H2",
            positions=[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
        )

        default_record = atoms_to_graph_record(atoms, sample_id="default")
        expanded_record = atoms_to_graph_record(
            atoms,
            sample_id="expanded",
            policy=AtomsToGraphPolicy(cutoff_multiplier=2.0),
        )

        self.assertEqual(default_record.n_edges, 0)
        np.testing.assert_array_equal(
            expanded_record.edge_index,
            np.array([[0, 1], [1, 0]], dtype=np.int64),
        )
        np.testing.assert_allclose(
            expanded_record.edge_features,
            np.full((2, 1), 1.0),
        )

    def test_atoms_to_graph_record_uses_periodic_minimum_image_distances(self) -> None:
        atoms = Atoms(
            "H2",
            positions=[
                [0.0, 0.0, 0.0],
                [0.95, 0.0, 0.0],
            ],
            cell=[1.0, 1.0, 1.0],
            pbc=[True, False, False],
        )

        record = atoms_to_graph_record(atoms, sample_id="periodic")

        np.testing.assert_array_equal(
            record.edge_index,
            np.array([[0, 1], [1, 0]], dtype=np.int64),
        )
        np.testing.assert_allclose(record.edge_features, np.full((2, 1), 0.05))

    def test_atoms_to_graph_dataset_view_preserves_input_order(self) -> None:
        atoms_list = [
            Atoms("H", positions=[[0.0, 0.0, 0.0]]),
            Atoms("He", positions=[[1.0, 0.0, 0.0]]),
        ]

        view = atoms_to_graph_dataset_view(["rxn-b", "rxn-a"], atoms_list)

        self.assertEqual(view.sample_ids, ("rxn-b", "rxn-a"))
        np.testing.assert_array_equal(
            view["rxn-b"].node_features,
            np.array([[1.0]]),
        )
        np.testing.assert_array_equal(
            view["rxn-b"].node_positions,
            np.array([[0.0, 0.0, 0.0]]),
        )
        np.testing.assert_array_equal(
            view["rxn-a"].node_features,
            np.array([[2.0]]),
        )

    def test_atoms_to_graph_dataset_view_rejects_length_mismatch(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "sample_ids and atoms_list must have the same length",
        ):
            atoms_to_graph_dataset_view(
                ["rxn-a"],
                [
                    Atoms("H", positions=[[0.0, 0.0, 0.0]]),
                    Atoms("He", positions=[[1.0, 0.0, 0.0]]),
                ],
            )

    def test_dump_graph_dataset_view_emits_graph_record_artifacts(self) -> None:
        view = atoms_to_graph_dataset_view(
            ["rxn-a"],
            [Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])],
        )

        payload = dump_graph_dataset_view(view)

        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["sample_id"], "rxn-a")
        self.assertEqual(payload[0]["node_features"], [[1.0], [1.0]])
        self.assertEqual(payload[0]["node_positions"], [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
        self.assertEqual(payload[0]["edge_index"], [[0, 1], [1, 0]])
        self.assertEqual(payload[0]["edge_features"], [[0.74], [0.74]])
        self.assertNotIn("graph_features", payload[0])

    def test_save_graph_dataset_view_round_trips_atoms_graph_artifacts(self) -> None:
        view = atoms_to_graph_dataset_view(
            ["rxn-b", "rxn-a"],
            [
                Atoms("H", positions=[[0.0, 0.0, 0.0]]),
                Atoms("He", positions=[[1.0, 0.0, 0.0]]),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "graphs.json"
            saved_path = save_graph_dataset_view(view, path)
            reloaded = load_graph_dataset_view(saved_path)

        self.assertEqual(saved_path, path)
        self.assertEqual(reloaded.sample_ids, ("rxn-b", "rxn-a"))
        np.testing.assert_array_equal(
            reloaded["rxn-b"].node_features,
            view["rxn-b"].node_features,
        )
        np.testing.assert_array_equal(
            reloaded["rxn-a"].node_positions,
            view["rxn-a"].node_positions,
        )
        np.testing.assert_array_equal(
            reloaded["rxn-a"].edge_index,
            view["rxn-a"].edge_index,
        )


class BuildGraphSweepDatasetTests(unittest.TestCase):
    def test_build_graph_sweep_dataset_aligns_rows_one_to_one(self) -> None:
        wide_df = _Frame(
            {
                "reaction": ["rxn-b", "rxn-a"],
                "reference_ads_eng": [2.0, 1.0],
                "model_a_mlip_ads_eng_median": [2.2, 1.1],
                "model_b_mlip_ads_eng_median": [1.8, 0.9],
            }
        )
        graph_view = GraphDatasetView.from_records(
            (
                GraphRecord(
                    sample_id="rxn-a",
                    node_features=np.arange(4, dtype=float).reshape(2, 2),
                    edge_index=np.array([[0], [1]], dtype=np.int64),
                ),
                GraphRecord(
                    sample_id="rxn-b",
                    node_features=np.arange(6, dtype=float).reshape(3, 2),
                    edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
                ),
            )
        )

        dataset = build_graph_sweep_dataset(wide_df, graph_view)

        np.testing.assert_array_equal(dataset.sample_ids, np.array(["rxn-b", "rxn-a"]))
        np.testing.assert_array_equal(
            dataset.mlip_features,
            np.array([[2.2, 1.8], [1.1, 0.9]]),
        )
        np.testing.assert_array_equal(dataset.targets, np.array([2.0, 1.0]))
        self.assertTrue(dataset.has_graphs)
        self.assertEqual(dataset.graphs.sample_ids, ("rxn-b", "rxn-a"))
        sample = dataset.sample(0)
        self.assertEqual(sample.sample_id, "rxn-b")
        np.testing.assert_array_equal(sample.mlip_features, np.array([2.2, 1.8]))
        self.assertEqual(sample.target, 2.0)
        self.assertIsNotNone(sample.graph)
        self.assertEqual(sample.graph.sample_id, "rxn-b")

    def test_build_graph_sweep_dataset_preserves_identity_across_non_monotonic_splits(
        self,
    ) -> None:
        wide_df = _Frame(
            {
                "reaction": ["rxn-a", "rxn-b", "rxn-c", "rxn-d"],
                "reference_ads_eng": [1.0, 2.0, 3.0, 4.0],
                "model_a_mlip_ads_eng_median": [1.1, 2.2, 3.3, 4.4],
                "model_b_mlip_ads_eng_median": [0.9, 1.8, 2.7, 3.6],
            }
        )
        graph_view = GraphDatasetView.from_records(
            (
                GraphRecord(
                    sample_id="rxn-a",
                    node_features=np.arange(4, dtype=float).reshape(2, 2),
                    edge_index=np.array([[0], [1]], dtype=np.int64),
                ),
                GraphRecord(
                    sample_id="rxn-b",
                    node_features=np.arange(6, dtype=float).reshape(3, 2),
                    edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
                ),
                GraphRecord(
                    sample_id="rxn-c",
                    node_features=np.arange(2, dtype=float).reshape(1, 2),
                    edge_index=np.empty((2, 0), dtype=np.int64),
                ),
                GraphRecord(
                    sample_id="rxn-d",
                    node_features=np.arange(8, dtype=float).reshape(4, 2),
                    edge_index=np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
                ),
            )
        )

        dataset = build_graph_sweep_dataset(wide_df, graph_view)
        split = TrainValTestSweepRunnerInput(
            dataset=dataset,
            sweep_size=3,
            train_idx=np.array([3, 1]),
            val_idx=np.array([0]),
            test_idx=np.array([2]),
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
                np.testing.assert_array_equal(
                    subset_sample.mlip_features,
                    expected.mlip_features,
                )
                self.assertEqual(subset_sample.target, expected.target)
                self.assertIs(subset_sample.graph, expected.graph)

    def test_build_graph_sweep_dataset_rejects_missing_graph_for_frame_row(self) -> None:
        wide_df = _Frame(
            {
                "reaction": ["rxn-a", "rxn-b"],
                "reference_ads_eng": [1.0, 2.0],
                "model_a_mlip_ads_eng_median": [1.1, 2.2],
            }
        )
        graph_view = GraphDatasetView.from_records(
            (
                GraphRecord(
                    sample_id="rxn-a",
                    node_features=np.arange(4, dtype=float).reshape(2, 2),
                    edge_index=np.array([[0], [1]], dtype=np.int64),
                ),
            )
        )

        with self.assertRaisesRegex(
            KeyError,
            r"missing graphs for reaction values: 'rxn-b'",
        ):
            build_graph_sweep_dataset(wide_df, graph_view)

    def test_build_graph_sweep_dataset_rejects_extra_graph_without_frame_row(self) -> None:
        wide_df = _Frame(
            {
                "reaction": ["rxn-a"],
                "reference_ads_eng": [1.0],
                "model_a_mlip_ads_eng_median": [1.1],
            }
        )
        graph_view = GraphDatasetView.from_records(
            (
                GraphRecord(
                    sample_id="rxn-a",
                    node_features=np.arange(4, dtype=float).reshape(2, 2),
                    edge_index=np.array([[0], [1]], dtype=np.int64),
                ),
                GraphRecord(
                    sample_id="rxn-b",
                    node_features=np.arange(6, dtype=float).reshape(3, 2),
                    edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
                ),
            )
        )

        with self.assertRaisesRegex(
            KeyError,
            r"extra sample_ids with no matching reaction: 'rxn-b'",
        ):
            build_graph_sweep_dataset(wide_df, graph_view)

    def test_build_graph_sweep_dataset_rejects_duplicate_frame_ids(self) -> None:
        wide_df = _Frame(
            {
                "reaction": ["rxn-a", "rxn-a"],
                "reference_ads_eng": [1.0, 2.0],
                "model_a_mlip_ads_eng_median": [1.1, 2.2],
            }
        )
        graph_view = GraphDatasetView.from_records(
            (
                GraphRecord(
                    sample_id="rxn-a",
                    node_features=np.arange(4, dtype=float).reshape(2, 2),
                    edge_index=np.array([[0], [1]], dtype=np.int64),
                ),
            )
        )

        with self.assertRaisesRegex(
            ValueError,
            r"duplicate reaction values: 'rxn-a'",
        ):
            build_graph_sweep_dataset(wide_df, graph_view)

    def test_build_graph_sweep_dataset_rejects_duplicate_graph_ids(self) -> None:
        wide_df = _Frame(
            {
                "reaction": ["rxn-a", "rxn-b"],
                "reference_ads_eng": [1.0, 2.0],
                "model_a_mlip_ads_eng_median": [1.1, 2.2],
            }
        )
        graph_view = GraphDatasetView.from_records(
            (
                GraphRecord(
                    sample_id="rxn-a",
                    node_features=np.arange(4, dtype=float).reshape(2, 2),
                    edge_index=np.array([[0], [1]], dtype=np.int64),
                ),
                GraphRecord(
                    sample_id="rxn-b",
                    node_features=np.arange(6, dtype=float).reshape(3, 2),
                    edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
                ),
            )
        )

        with patch.object(
            GraphDatasetView,
            "sample_ids",
            new_callable=PropertyMock,
            return_value=("rxn-a", "rxn-a"),
        ):
            with self.assertRaisesRegex(
                ValueError,
                r"duplicate sample_ids: 'rxn-a'",
            ):
                build_graph_sweep_dataset(wide_df, graph_view)


class _Frame:
    def __init__(self, columns: dict[str, list[object]]) -> None:
        self._columns = {name: np.asarray(values) for name, values in columns.items()}
        self.columns = list(columns)

    def __getitem__(self, key: str) -> np.ndarray:
        return self._columns[key]
