from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class GatingOutput:
    logits: torch.Tensor
    weights: torch.Tensor
    prediction: torch.Tensor


class MLPGatingNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (32, 32),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")

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
        layers.append(nn.Linear(prev_dim, input_dim))
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
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

    def forward(self, mlip_energies: torch.Tensor) -> GatingOutput:
        logits = self.gate(mlip_energies)
        weights = torch.softmax(logits, dim=-1)
        prediction = weighted_expert_prediction(mlip_energies, weights)
        return GatingOutput(
            logits=logits,
            weights=weights,
            prediction=prediction,
        )
