from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from oasis.sweep import GraphDatasetView, GraphRecord


def load_graph_dataset_view(path: str | Path) -> GraphDatasetView:
    """Load a graph dataset view from a JSON list of graph records."""

    resolved_path = Path(path)
    with resolved_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        raise TypeError("graph dataset JSON must be a top-level list of records.")

    records = tuple(_graph_record_from_mapping(item) for item in payload)
    return GraphDatasetView.from_records(records)


def _graph_record_from_mapping(payload: Any) -> GraphRecord:
    if not isinstance(payload, dict):
        raise TypeError("each graph record must be a JSON object.")

    missing_fields = {
        "sample_id",
        "node_features",
        "edge_index",
    }.difference(payload)
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"graph record is missing required fields: {missing}")

    return GraphRecord(
        sample_id=payload["sample_id"],
        node_features=np.asarray(payload["node_features"]),
        edge_index=np.asarray(payload["edge_index"]),
        edge_features=_optional_array(payload.get("edge_features")),
        graph_features=_optional_array(payload.get("graph_features")),
    )


def _optional_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value)
