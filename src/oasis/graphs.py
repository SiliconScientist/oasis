from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from ase.neighborlist import natural_cutoffs, neighbor_list

from oasis.exp import column_to_numpy, mlip_columns
from oasis.sweep import (
    GraphDatasetView,
    GraphRecord,
    SweepDatasetInputs,
    _duplicate_sample_ids,
    _format_sample_id_list,
)
from oasis.sweep import SweepDataset

if TYPE_CHECKING:
    from ase import Atoms


@dataclass(frozen=True, slots=True)
class AtomsToGraphPolicy:
    """Canonical policy for converting ASE Atoms into GraphRecord objects.

    Policy:
    - preserve the input atom order as node order
    - store Cartesian coordinates as `node_positions`
    - use atomic numbers as the sole node feature
    - build directed edges from ASE's neighbor list with natural covalent cutoffs
    - store one edge feature per edge: the interatomic distance
    - omit graph-level features unless a caller defines a different policy later
    """

    cutoff_multiplier: float = 1.25
    include_self_interactions: bool = False

    def __post_init__(self) -> None:
        if self.cutoff_multiplier <= 0:
            raise ValueError("cutoff_multiplier must be positive.")


DEFAULT_ATOMS_TO_GRAPH_POLICY = AtomsToGraphPolicy()


def load_graph_dataset_view(path: str | Path) -> GraphDatasetView:
    """Load a graph dataset view from a JSON list of graph records."""

    resolved_path = Path(path)
    with resolved_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        raise TypeError("graph dataset JSON must be a top-level list of records.")

    records = tuple(_graph_record_from_mapping(item) for item in payload)
    return GraphDatasetView.from_records(records)


def dump_graph_dataset_view(graph_view: GraphDatasetView) -> list[dict[str, Any]]:
    """Convert a graph dataset view into JSON-serializable graph artifacts."""

    return [_graph_record_to_mapping(graph_view[sample_id]) for sample_id in graph_view.sample_ids]


def save_graph_dataset_view(
    graph_view: GraphDatasetView,
    path: str | Path,
) -> Path:
    """Write graph artifacts as a JSON list of graph records."""

    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", encoding="utf-8") as f:
        json.dump(dump_graph_dataset_view(graph_view), f, indent=2)
    return resolved_path


def atoms_to_graph_record(
    atoms: Atoms,
    *,
    sample_id: Any,
    policy: AtomsToGraphPolicy = DEFAULT_ATOMS_TO_GRAPH_POLICY,
) -> GraphRecord:
    """Convert one ASE Atoms object into a canonical GraphRecord."""

    node_features = np.asarray(atoms.numbers, dtype=float).reshape(-1, 1)
    node_positions = np.asarray(atoms.positions, dtype=float)
    cutoffs = natural_cutoffs(atoms, mult=policy.cutoff_multiplier)
    edge_sources, edge_targets, edge_distances = neighbor_list(
        "ijd",
        atoms,
        cutoffs,
        self_interaction=policy.include_self_interactions,
    )

    if len(edge_sources):
        order = np.lexsort((edge_distances, edge_targets, edge_sources))
        edge_index = np.vstack(
            (
                np.asarray(edge_sources[order], dtype=np.int64),
                np.asarray(edge_targets[order], dtype=np.int64),
            )
        )
        edge_features = np.asarray(edge_distances[order], dtype=float).reshape(-1, 1)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_features = np.empty((0, 1), dtype=float)

    return GraphRecord(
        sample_id=sample_id,
        node_features=node_features,
        edge_index=edge_index,
        node_positions=node_positions,
        edge_features=edge_features,
    )


def atoms_to_graph_dataset_view(
    sample_ids: Sequence[Any],
    atoms_list: Sequence[Atoms],
    *,
    policy: AtomsToGraphPolicy = DEFAULT_ATOMS_TO_GRAPH_POLICY,
) -> GraphDatasetView:
    """Convert aligned sample IDs and ASE Atoms objects into a GraphDatasetView."""

    if len(sample_ids) != len(atoms_list):
        raise ValueError("sample_ids and atoms_list must have the same length.")

    return GraphDatasetView.from_records(
        tuple(
            atoms_to_graph_record(atoms, sample_id=sample_id, policy=policy)
            for sample_id, atoms in zip(sample_ids, atoms_list, strict=True)
        )
    )


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
    duplicate_frame_ids = _duplicate_sample_ids(sample_ids)
    if duplicate_frame_ids:
        raise ValueError(
            f"wide_df contains duplicate {join_key} values: "
            f"{_format_sample_id_list(duplicate_frame_ids)}."
        )

    graph_sample_ids = graph_view.sample_ids
    duplicate_graph_ids = _duplicate_sample_ids(graph_sample_ids)
    if duplicate_graph_ids:
        raise ValueError(
            "graph_view contains duplicate sample_ids: "
            f"{_format_sample_id_list(duplicate_graph_ids)}."
        )

    graph_records = graph_view.records_by_sample_id
    sample_id_set = set(sample_ids)
    missing_graph_ids = tuple(
        sample_id for sample_id in sample_ids if sample_id not in graph_records
    )
    if missing_graph_ids:
        raise KeyError(
            f"missing graphs for {join_key} values: "
            f"{_format_sample_id_list(missing_graph_ids)}."
        )

    extra_graph_ids = tuple(
        sample_id for sample_id in graph_sample_ids if sample_id not in sample_id_set
    )
    if extra_graph_ids:
        raise KeyError(
            f"graph_view contains extra sample_ids with no matching {join_key}: "
            f"{_format_sample_id_list(extra_graph_ids)}."
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
    return SweepDataset.from_inputs(
        inputs=SweepDatasetInputs(
            mlip_features=mlip_features,
            graph_view=aligned_graphs,
        ),
        targets=targets,
        sample_ids=np.asarray(sample_ids),
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
        node_features=np.asarray(payload["node_features"], dtype=float),
        edge_index=np.asarray(payload["edge_index"], dtype=np.int64),
        node_positions=_optional_array(payload.get("node_positions"), dtype=float),
        edge_features=_optional_array(payload.get("edge_features"), dtype=float),
        graph_features=_optional_array(payload.get("graph_features"), dtype=float),
    )


def _graph_record_to_mapping(record: GraphRecord) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "sample_id": record.sample_id,
        "node_features": record.node_features.tolist(),
        "edge_index": record.edge_index.tolist(),
    }
    if record.node_positions is not None:
        payload["node_positions"] = record.node_positions.tolist()
    if record.edge_features is not None:
        payload["edge_features"] = record.edge_features.tolist()
    if record.graph_features is not None:
        payload["graph_features"] = record.graph_features.tolist()
    return payload


def _optional_array(value: Any, *, dtype: Any | None = None) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=dtype)
