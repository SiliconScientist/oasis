from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from oasis.exp import column_to_numpy, mlip_columns
from oasis.sweep import GraphDatasetView, GraphRecord
from oasis.sweep import SweepDataset


def load_graph_dataset_view(path: str | Path) -> GraphDatasetView:
    """Load a graph dataset view from a JSON list of graph records."""

    resolved_path = Path(path)
    with resolved_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        raise TypeError("graph dataset JSON must be a top-level list of records.")

    records = tuple(_graph_record_from_mapping(item) for item in payload)
    return GraphDatasetView.from_records(records)


def build_graph_sweep_dataset(
    wide_df: Any,
    graph_view: GraphDatasetView,
    *,
    join_key: str = "reaction",
) -> SweepDataset:
    """Build a SweepDataset by aligning wide-frame rows with structure graphs."""

    columns = getattr(wide_df, "columns", ())
    if join_key not in columns:
        raise ValueError(f"wide_df is missing required join column: {join_key}")
    if "reference_ads_eng" not in columns:
        raise ValueError("wide_df is missing required target column: reference_ads_eng")

    feature_cols = mlip_columns(wide_df)
    if not feature_cols:
        raise ValueError(
            "No MLIP prediction columns found (expected *_mlip_ads_eng_median)."
        )

    sample_ids = tuple(column_to_numpy(wide_df, join_key).tolist())
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"wide_df contains duplicate {join_key} values.")

    graph_sample_ids = graph_view.sample_ids
    if len(graph_sample_ids) != len(set(graph_sample_ids)):
        raise ValueError("graph_view contains duplicate sample_ids.")

    missing_graph_ids = tuple(sample_id for sample_id in sample_ids if sample_id not in graph_view.records_by_sample_id)
    if missing_graph_ids:
        preview = ", ".join(repr(sample_id) for sample_id in missing_graph_ids[:5])
        raise KeyError(
            f"missing graphs for {join_key} values: {preview}"
        )

    extra_graph_ids = tuple(sample_id for sample_id in graph_sample_ids if sample_id not in set(sample_ids))
    if extra_graph_ids:
        preview = ", ".join(repr(sample_id) for sample_id in extra_graph_ids[:5])
        raise KeyError(
            f"graph_view contains extra sample_ids with no matching {join_key}: {preview}"
        )

    if hasattr(wide_df, "select"):
        mlip_features = wide_df.select(feature_cols).to_numpy()
    else:
        mlip_features = np.column_stack(
            [column_to_numpy(wide_df, column_name) for column_name in feature_cols]
        )
    targets = column_to_numpy(wide_df, "reference_ads_eng")
    aligned_graphs = GraphDatasetView.from_records(
        tuple(graph_view[sample_id] for sample_id in sample_ids)
    )
    return SweepDataset(
        mlip_features=mlip_features,
        targets=targets,
        sample_ids=np.asarray(sample_ids),
        graph_view=aligned_graphs,
    )


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
