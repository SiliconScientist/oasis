from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import PropertyMock, patch

import numpy as np

from oasis.graphs import build_graph_sweep_dataset, load_graph_dataset_view
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

        with self.assertRaisesRegex(ValueError, "duplicate graph record"):
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

        with self.assertRaisesRegex(KeyError, "missing graphs"):
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

        with self.assertRaisesRegex(KeyError, "extra sample_ids"):
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

        with self.assertRaisesRegex(ValueError, "duplicate reaction values"):
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
            with self.assertRaisesRegex(ValueError, "duplicate sample_ids"):
                build_graph_sweep_dataset(wide_df, graph_view)


class _Frame:
    def __init__(self, columns: dict[str, list[object]]) -> None:
        self._columns = {name: np.asarray(values) for name, values in columns.items()}
        self.columns = list(columns)

    def __getitem__(self, key: str) -> np.ndarray:
        return self._columns[key]
