from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from oasis.graphs import load_graph_dataset_view


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
