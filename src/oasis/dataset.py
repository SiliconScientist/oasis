from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import polars as pl
import torch
from torch.utils.data import Dataset

from oasis.graph import AdsorptionGraph, AdsorptionGraphBatch, batch_adsorption_graphs


@dataclass(frozen=True)
class GatingSample:
    graph: AdsorptionGraph
    mlip_energies: torch.Tensor
    target_ads_eng: torch.Tensor
    expert_labels: dict[str, str]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class GatingBatch:
    graph_batch: AdsorptionGraphBatch
    mlip_energies: torch.Tensor
    target_ads_eng: torch.Tensor
    expert_labels: list[dict[str, str]]
    metadata: list[dict[str, Any]]


def _label_columns(wide_df: pl.DataFrame) -> list[str]:
    return sorted(c for c in wide_df.columns if c.endswith("_label"))


def _sample_metadata(graph: AdsorptionGraph) -> dict[str, Any]:
    return {
        "reaction": graph.reaction,
        "adsorbate": graph.adsorbate,
        "n_atoms": int(graph.z.shape[0]),
        "mlip_names": graph.mlip_names,
    }


class GatingDataset(Dataset[GatingSample]):
    def __init__(
        self,
        graphs: Sequence[AdsorptionGraph],
        wide_df: pl.DataFrame,
    ) -> None:
        if len(graphs) != wide_df.height:
            raise ValueError(
                f"graphs length ({len(graphs)}) does not match wide_df row count "
                f"({wide_df.height})"
            )

        reactions = wide_df["reaction"].to_list()
        graph_reactions = [graph.reaction for graph in graphs]
        if reactions != graph_reactions:
            raise ValueError("graphs are not aligned with wide_df reaction order")

        self._graphs = list(graphs)
        self._label_cols = _label_columns(wide_df)
        self._rows = list(
            wide_df.select(["reaction", *self._label_cols]).iter_rows(named=True)
        )

    def __len__(self) -> int:
        return len(self._graphs)

    def __getitem__(self, index: int) -> GatingSample:
        graph = self._graphs[index]
        row = self._rows[index]
        expert_labels = {
            col.removesuffix("_label"): str(row[col])
            for col in self._label_cols
        }
        return GatingSample(
            graph=graph,
            mlip_energies=graph.mlip_energies,
            target_ads_eng=graph.y,
            expert_labels=expert_labels,
            metadata=_sample_metadata(graph),
        )


def collate_gating_samples(samples: Sequence[GatingSample]) -> GatingBatch:
    if not samples:
        raise ValueError("Cannot collate an empty sample list")

    graphs = [sample.graph for sample in samples]
    graph_batch = batch_adsorption_graphs(graphs)
    mlip_energies = torch.stack([sample.mlip_energies for sample in samples], dim=0)
    target_ads_eng = torch.cat([sample.target_ads_eng for sample in samples], dim=0)
    expert_labels = [sample.expert_labels for sample in samples]
    metadata = [sample.metadata for sample in samples]

    return GatingBatch(
        graph_batch=graph_batch,
        mlip_energies=mlip_energies,
        target_ads_eng=target_ads_eng,
        expert_labels=expert_labels,
        metadata=metadata,
    )
