from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from oasis.dataset import GatingBatch
from oasis.model import BaselineMLPGatedMoE, SchNetGatedMoE
from oasis.train import _forward_gating_model, _move_batch_to_device


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    rmse: float


@dataclass(frozen=True)
class BaselineEvaluation:
    best_single_expert: RegressionMetrics
    best_single_expert_name: str
    mean_ensemble: RegressionMetrics


@dataclass(frozen=True)
class ModelEvaluation:
    model_name: str
    metrics: RegressionMetrics


@dataclass(frozen=True)
class EvaluationReport:
    baselines: BaselineEvaluation
    models: list[ModelEvaluation]


def regression_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> RegressionMetrics:
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target must have same shape, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    errors = prediction - target
    mae = torch.mean(torch.abs(errors)).item()
    rmse = sqrt(torch.mean(errors * errors).item())
    return RegressionMetrics(mae=mae, rmse=rmse)


def mean_ensemble_prediction(mlip_energies: torch.Tensor) -> torch.Tensor:
    if mlip_energies.ndim != 2:
        raise ValueError(
            f"mlip_energies must have shape (batch, n_experts), got {tuple(mlip_energies.shape)}"
        )
    return mlip_energies.mean(dim=-1)


def best_single_expert_metrics(
    mlip_energies: torch.Tensor,
    target: torch.Tensor,
    mlip_names: Sequence[str],
) -> tuple[str, RegressionMetrics]:
    if mlip_energies.ndim != 2:
        raise ValueError(
            f"mlip_energies must have shape (batch, n_experts), got {tuple(mlip_energies.shape)}"
        )
    if len(mlip_names) != mlip_energies.shape[1]:
        raise ValueError(
            f"Expected {mlip_energies.shape[1]} mlip names, got {len(mlip_names)}"
        )

    best_name = ""
    best_metrics: RegressionMetrics | None = None
    for idx, name in enumerate(mlip_names):
        metrics = regression_metrics(mlip_energies[:, idx], target)
        if best_metrics is None or metrics.rmse < best_metrics.rmse:
            best_name = str(name)
            best_metrics = metrics

    if best_metrics is None:
        raise ValueError("No expert metrics were computed")
    return best_name, best_metrics


def evaluate_baselines(
    batch: GatingBatch,
) -> BaselineEvaluation:
    target = batch.target_ads_eng
    mlip_energies = batch.mlip_energies
    mlip_names = batch.graph_batch.mlip_names
    best_name, best_metrics = best_single_expert_metrics(
        mlip_energies,
        target,
        mlip_names,
    )
    mean_metrics = regression_metrics(mean_ensemble_prediction(mlip_energies), target)
    return BaselineEvaluation(
        best_single_expert=best_metrics,
        best_single_expert_name=best_name,
        mean_ensemble=mean_metrics,
    )


def evaluate_gating_model(
    model: nn.Module,
    data_loader: DataLoader[GatingBatch],
    *,
    device: str = "cpu",
) -> RegressionMetrics:
    device_obj = torch.device(device)
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in data_loader:
            batch = _move_batch_to_device(batch, device_obj)
            output = _forward_gating_model(model, batch)
            predictions.append(output.prediction.cpu())
            targets.append(batch.target_ads_eng.cpu())

    if not predictions:
        raise ValueError("Data loader is empty")
    prediction = torch.cat(predictions, dim=0)
    target = torch.cat(targets, dim=0)
    return regression_metrics(prediction, target)


def evaluate_models_and_baselines(
    baseline_batch: GatingBatch,
    *,
    data_loader: DataLoader[GatingBatch],
    models: Sequence[tuple[str, nn.Module]],
    device: str = "cpu",
) -> EvaluationReport:
    baselines = evaluate_baselines(baseline_batch)
    model_evals = [
        ModelEvaluation(
            model_name=name,
            metrics=evaluate_gating_model(model, data_loader, device=device),
        )
        for name, model in models
    ]
    return EvaluationReport(
        baselines=baselines,
        models=model_evals,
    )


def default_model_baselines(
    n_experts: int,
) -> list[tuple[str, nn.Module]]:
    return [
        ("mlp_gate", BaselineMLPGatedMoE(n_experts=n_experts)),
        ("schnet_gate", SchNetGatedMoE(n_experts=n_experts)),
    ]
