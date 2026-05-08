from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import PropertyMock, patch

from ase import Atoms
import numpy as np

from oasis.graphs import (
    atoms_to_graph_dataset_view,
    atoms_to_graph_record,
    build_graph_sweep_dataset,
    load_graph_dataset_view,
)
from oasis.sweep import GraphDatasetView, GraphRecord


class LoadGraphDatasetViewTests(unittest.TestCase):
    def test_load_graph_dataset_view_reads_valid_records(self) -> None:
        payload = [
            {
                "sample_id": "s0",
                "node_features": [[0.0, 1.0], [1.0, 0.0]],
                "edge_index": [[0], [1]],
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
        np.testing.assert_array_equal(
            record.edge_index,
            np.array([[0, 0, 1, 2], [1, 2, 0, 0]], dtype=np.int64),
        )
        np.testing.assert_allclose(
            record.edge_features,
            np.full((4, 1), 0.96),
        )
        self.assertIsNone(record.graph_features)

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
