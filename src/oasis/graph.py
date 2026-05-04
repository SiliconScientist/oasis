from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ase import Atoms
from ase.neighborlist import neighbor_list
import numpy as np
import polars as pl
import torch


@dataclass(frozen=True)
class AdsorptionGraph:
    reaction: str
    adsorbate: str | None
    z: torch.Tensor
    pos: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    batch: torch.Tensor
    y: torch.Tensor
    mlip_energies: torch.Tensor
    mlip_names: tuple[str, ...]


@dataclass(frozen=True)
class AdsorptionGraphBatch:
    z: torch.Tensor
    pos: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    batch: torch.Tensor
    ptr: torch.Tensor
    y: torch.Tensor
    mlip_energies: torch.Tensor
    reactions: list[str]
    adsorbates: list[str | None]
    mlip_names: tuple[str, ...]


def _mlip_prediction_columns(wide_df: pl.DataFrame) -> list[str]:
    return sorted(c for c in wide_df.columns if c.endswith("_mlip_ads_eng_median"))


def _build_edge_tensors(
    atoms: Atoms,
    cutoff: float,
    max_neighbors: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    senders, receivers, distances = neighbor_list("ijd", atoms, cutoff)
    if len(senders) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float32)
        return edge_index, edge_weight

    if max_neighbors is not None:
        kept_indices: list[int] = []
        counts: dict[int, int] = {}
        for idx, sender in enumerate(senders.tolist()):
            counts.setdefault(sender, 0)
            if counts[sender] >= max_neighbors:
                continue
            kept_indices.append(idx)
            counts[sender] += 1
        senders = senders[kept_indices]
        receivers = receivers[kept_indices]
        distances = distances[kept_indices]

    edge_index = torch.from_numpy(
        np.stack([senders, receivers], axis=0)
    ).to(dtype=torch.long)
    edge_weight = torch.tensor(distances, dtype=torch.float32)
    return edge_index, edge_weight


def atoms_to_graph(
    atoms: Atoms,
    *,
    reaction: str,
    adsorbate: str | None,
    reference_ads_eng: float,
    mlip_energies: torch.Tensor,
    mlip_names: Sequence[str],
    cutoff: float = 6.0,
    max_neighbors: int | None = None,
) -> AdsorptionGraph:
    z = torch.tensor(atoms.numbers, dtype=torch.long)
    pos = torch.tensor(atoms.positions, dtype=torch.float32)
    edge_index, edge_weight = _build_edge_tensors(
        atoms,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
    )
    batch = torch.zeros(len(atoms), dtype=torch.long)
    y = torch.tensor([reference_ads_eng], dtype=torch.float32)

    return AdsorptionGraph(
        reaction=reaction,
        adsorbate=adsorbate,
        z=z,
        pos=pos,
        edge_index=edge_index,
        edge_weight=edge_weight,
        batch=batch,
        y=y,
        mlip_energies=mlip_energies.to(dtype=torch.float32),
        mlip_names=tuple(mlip_names),
    )


def build_adsorption_graphs(
    wide_df: pl.DataFrame,
    atoms_list: Sequence[Atoms],
    *,
    cutoff: float = 6.0,
    max_neighbors: int | None = None,
) -> list[AdsorptionGraph]:
    if wide_df.height != len(atoms_list):
        raise ValueError(
            f"wide_df row count ({wide_df.height}) does not match atoms_list length "
            f"({len(atoms_list)})"
        )

    mlip_cols = _mlip_prediction_columns(wide_df)
    if not mlip_cols:
        raise ValueError(
            "No MLIP prediction columns found (expected *_mlip_ads_eng_median)."
        )

    graphs: list[AdsorptionGraph] = []
    rows = wide_df.iter_rows(named=True)
    for row, atoms in zip(rows, atoms_list, strict=True):
        mlip_values = torch.tensor(
            [float(row[col]) for col in mlip_cols],
            dtype=torch.float32,
        )
        graphs.append(
            atoms_to_graph(
                atoms,
                reaction=str(row["reaction"]),
                adsorbate=row.get("adsorbate"),
                reference_ads_eng=float(row["reference_ads_eng"]),
                mlip_energies=mlip_values,
                mlip_names=mlip_cols,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
            )
        )

    return graphs


def batch_adsorption_graphs(
    graphs: Sequence[AdsorptionGraph],
) -> AdsorptionGraphBatch:
    if not graphs:
        raise ValueError("Cannot batch an empty graph list")

    mlip_names = graphs[0].mlip_names
    for graph in graphs[1:]:
        if graph.mlip_names != mlip_names:
            raise ValueError("All graphs in a batch must share the same MLIP ordering")

    z_parts: list[torch.Tensor] = []
    pos_parts: list[torch.Tensor] = []
    edge_index_parts: list[torch.Tensor] = []
    edge_weight_parts: list[torch.Tensor] = []
    batch_parts: list[torch.Tensor] = []
    y_parts: list[torch.Tensor] = []
    mlip_parts: list[torch.Tensor] = []
    reactions: list[str] = []
    adsorbates: list[str | None] = []
    ptr = [0]
    node_offset = 0

    for graph_idx, graph in enumerate(graphs):
        n_nodes = graph.z.shape[0]
        z_parts.append(graph.z)
        pos_parts.append(graph.pos)
        edge_index_parts.append(graph.edge_index + node_offset)
        edge_weight_parts.append(graph.edge_weight)
        batch_parts.append(torch.full((n_nodes,), graph_idx, dtype=torch.long))
        y_parts.append(graph.y)
        mlip_parts.append(graph.mlip_energies.unsqueeze(0))
        reactions.append(graph.reaction)
        adsorbates.append(graph.adsorbate)
        node_offset += n_nodes
        ptr.append(node_offset)

    return AdsorptionGraphBatch(
        z=torch.cat(z_parts, dim=0),
        pos=torch.cat(pos_parts, dim=0),
        edge_index=torch.cat(edge_index_parts, dim=1),
        edge_weight=torch.cat(edge_weight_parts, dim=0),
        batch=torch.cat(batch_parts, dim=0),
        ptr=torch.tensor(ptr, dtype=torch.long),
        y=torch.cat(y_parts, dim=0),
        mlip_energies=torch.cat(mlip_parts, dim=0),
        reactions=reactions,
        adsorbates=adsorbates,
        mlip_names=mlip_names,
    )
