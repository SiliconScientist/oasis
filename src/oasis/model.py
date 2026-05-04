from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from oasis.graph import AdsorptionGraphBatch


@dataclass(frozen=True)
class GatingOutput:
    logits: torch.Tensor
    weights: torch.Tensor
    prediction: torch.Tensor


class MLPGatingNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int | None = None,
        hidden_dims: Sequence[int] = (32, 32),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if output_dim is None:
            output_dim = input_dim
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            if hidden_dim <= 0:
                raise ValueError(
                    f"hidden dimensions must be positive, got {hidden_dim}"
                )
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, mlip_energies: torch.Tensor) -> torch.Tensor:
        if mlip_energies.ndim != 2:
            raise ValueError(
                f"mlip_energies must have shape (batch, n_experts), got {tuple(mlip_energies.shape)}"
            )
        return self.network(mlip_energies)


def weighted_expert_prediction(
    mlip_energies: torch.Tensor,
    expert_weights: torch.Tensor,
) -> torch.Tensor:
    if mlip_energies.shape != expert_weights.shape:
        raise ValueError(
            "mlip_energies and expert_weights must have the same shape, got "
            f"{tuple(mlip_energies.shape)} and {tuple(expert_weights.shape)}"
        )
    return (mlip_energies * expert_weights).sum(dim=-1)


class MixtureOfExpertsPredictionHead(nn.Module):
    def forward(
        self,
        logits: torch.Tensor,
        mlip_energies: torch.Tensor,
    ) -> GatingOutput:
        if logits.shape != mlip_energies.shape:
            raise ValueError(
                "logits and mlip_energies must have the same shape, got "
                f"{tuple(logits.shape)} and {tuple(mlip_energies.shape)}"
            )

        weights = torch.softmax(logits, dim=-1)
        prediction = weighted_expert_prediction(mlip_energies, weights)
        return GatingOutput(
            logits=logits,
            weights=weights,
            prediction=prediction,
        )


class GaussianSmearing(nn.Module):
    def __init__(
        self,
        start: float = 0.0,
        stop: float = 6.0,
        n_rbf: int = 32,
    ) -> None:
        super().__init__()
        offsets = torch.linspace(start, stop, n_rbf)
        self.register_buffer("offsets", offsets)
        delta = offsets[1] - offsets[0] if n_rbf > 1 else torch.tensor(1.0)
        self.coeff = -0.5 / float(delta * delta)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        diff = distances.unsqueeze(-1) - self.offsets
        return torch.exp(self.coeff * diff * diff)


class SchNetInteraction(nn.Module):
    def __init__(self, hidden_dim: int, n_rbf: int) -> None:
        super().__init__()
        self.filter_network = nn.Sequential(
            nn.Linear(n_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        filters = self.filter_network(edge_attr)
        messages = self.dense(x[src]) * filters
        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, dst, messages)
        return x + self.update(aggregated)


def mean_pool_by_batch(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"x must be rank-2, got shape {tuple(x.shape)}")
    if batch.ndim != 1 or batch.shape[0] != x.shape[0]:
        raise ValueError(
            f"batch must have shape ({x.shape[0]},), got {tuple(batch.shape)}"
        )
    n_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
    pooled = torch.zeros((n_graphs, x.shape[1]), dtype=x.dtype, device=x.device)
    counts = torch.zeros((n_graphs,), dtype=x.dtype, device=x.device)
    pooled.index_add_(0, batch, x)
    counts.index_add_(0, batch, torch.ones_like(batch, dtype=x.dtype))
    return pooled / counts.clamp_min(1.0).unsqueeze(-1)


class SchNetStructureEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        n_interactions: int = 3,
        n_rbf: int = 32,
        cutoff: float = 6.0,
        max_atomic_num: int = 100,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_atomic_num, hidden_dim)
        self.distance_expansion = GaussianSmearing(
            start=0.0,
            stop=cutoff,
            n_rbf=n_rbf,
        )
        self.interactions = nn.ModuleList(
            [SchNetInteraction(hidden_dim, n_rbf) for _ in range(n_interactions)]
        )

    def forward(self, graph_batch: AdsorptionGraphBatch) -> torch.Tensor:
        x = self.embedding(graph_batch.z)
        edge_attr = self.distance_expansion(graph_batch.edge_weight)
        for interaction in self.interactions:
            x = interaction(x, graph_batch.edge_index, edge_attr)
        return mean_pool_by_batch(x, graph_batch.batch)


class BaselineMLPGatedMoE(nn.Module):
    def __init__(
        self,
        n_experts: int,
        hidden_dims: Sequence[int] = (32, 32),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.gate = MLPGatingNetwork(
            input_dim=n_experts,
            output_dim=n_experts,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.head = MixtureOfExpertsPredictionHead()

    def forward(self, mlip_energies: torch.Tensor) -> GatingOutput:
        logits = self.gate(mlip_energies)
        return self.head(logits, mlip_energies)


class SchNetGatedMoE(nn.Module):
    def __init__(
        self,
        n_experts: int,
        structure_hidden_dim: int = 64,
        n_interactions: int = 3,
        n_rbf: int = 32,
        cutoff: float = 6.0,
        gate_hidden_dims: Sequence[int] = (64, 32),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = SchNetStructureEncoder(
            hidden_dim=structure_hidden_dim,
            n_interactions=n_interactions,
            n_rbf=n_rbf,
            cutoff=cutoff,
        )
        self.gate = MLPGatingNetwork(
            input_dim=structure_hidden_dim + n_experts,
            output_dim=n_experts,
            hidden_dims=gate_hidden_dims,
            dropout=dropout,
        )
        self.head = MixtureOfExpertsPredictionHead()

    def forward(
        self,
        graph_batch: AdsorptionGraphBatch,
        mlip_energies: torch.Tensor,
    ) -> GatingOutput:
        structure_embedding = self.encoder(graph_batch)
        gate_input = torch.cat([structure_embedding, mlip_energies], dim=-1)
        logits = self.gate(gate_input)
        return self.head(logits, mlip_energies)
