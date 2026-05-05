from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from oasis.dataset import GatingBatch, GatingDataset, collate_gating_samples
from oasis.model import BaselineMLPGatedMoE, GatingOutput, SchNetGatedMoE


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    val_fraction: float = 0.2
    random_seed: int = 0
    checkpoint_dir: str | Path | None = None
    device: str = "cpu"


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float


@dataclass(frozen=True)
class TrainResult:
    history: list[EpochMetrics]
    best_epoch: int | None
    best_val_loss: float
    best_checkpoint_path: Path | None
    latest_checkpoint_path: Path | None


def split_gating_dataset(
    dataset: Dataset[Any],
    *,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[Subset[Any], Subset[Any]]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")

    n_total = len(dataset)
    if n_total < 2:
        raise ValueError("Need at least 2 samples to create train/val splits")

    n_val = max(1, int(round(n_total * val_fraction)))
    n_val = min(n_val, n_total - 1)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        dataset,
        [n_train, n_val],
        generator=generator,
    )
    return train_subset, val_subset


def build_gating_dataloaders(
    dataset: GatingDataset,
    *,
    batch_size: int = 16,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[DataLoader[GatingBatch], DataLoader[GatingBatch]]:
    train_subset, val_subset = split_gating_dataset(
        dataset,
        val_fraction=val_fraction,
        seed=seed,
    )
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_gating_samples,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_gating_samples,
    )
    return train_loader, val_loader


def build_gating_dataloaders_from_indices(
    dataset: GatingDataset,
    *,
    train_indices: Sequence[int],
    eval_indices: Sequence[int],
    batch_size: int = 16,
) -> tuple[DataLoader[GatingBatch], DataLoader[GatingBatch]]:
    if not train_indices:
        raise ValueError("train_indices must not be empty")
    if not eval_indices:
        raise ValueError("eval_indices must not be empty")

    train_subset = Subset(dataset, list(train_indices))
    eval_subset = Subset(dataset, list(eval_indices))
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_gating_samples,
    )
    eval_loader = DataLoader(
        eval_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_gating_samples,
    )
    return train_loader, eval_loader


def build_nested_gating_dataloaders(
    dataset: GatingDataset,
    *,
    train_pool_indices: Sequence[int],
    test_indices: Sequence[int],
    batch_size: int = 16,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[DataLoader[GatingBatch], DataLoader[GatingBatch], DataLoader[GatingBatch]]:
    if not train_pool_indices:
        raise ValueError("train_pool_indices must not be empty")
    if not test_indices:
        raise ValueError("test_indices must not be empty")

    outer_train_subset = Subset(dataset, list(train_pool_indices))
    inner_train_subset, inner_val_subset = split_gating_dataset(
        outer_train_subset,
        val_fraction=val_fraction,
        seed=seed,
    )
    outer_test_subset = Subset(dataset, list(test_indices))

    inner_train_loader = DataLoader(
        inner_train_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_gating_samples,
    )
    inner_val_loader = DataLoader(
        inner_val_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_gating_samples,
    )
    outer_test_loader = DataLoader(
        outer_test_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_gating_samples,
    )
    return inner_train_loader, inner_val_loader, outer_test_loader


def _move_batch_to_device(batch: GatingBatch, device: torch.device) -> GatingBatch:
    graph_batch = batch.graph_batch
    moved_graph_batch = type(graph_batch)(
        z=graph_batch.z.to(device),
        pos=graph_batch.pos.to(device),
        edge_index=graph_batch.edge_index.to(device),
        edge_weight=graph_batch.edge_weight.to(device),
        batch=graph_batch.batch.to(device),
        ptr=graph_batch.ptr.to(device),
        y=graph_batch.y.to(device),
        mlip_energies=graph_batch.mlip_energies.to(device),
        reactions=graph_batch.reactions,
        adsorbates=graph_batch.adsorbates,
        mlip_names=graph_batch.mlip_names,
    )
    return GatingBatch(
        graph_batch=moved_graph_batch,
        mlip_energies=batch.mlip_energies.to(device),
        target_ads_eng=batch.target_ads_eng.to(device),
        expert_labels=batch.expert_labels,
        metadata=batch.metadata,
    )


def _forward_gating_model(
    model: nn.Module,
    batch: GatingBatch,
) -> GatingOutput:
    if isinstance(model, SchNetGatedMoE):
        return model(batch.graph_batch, batch.mlip_energies)
    if isinstance(model, BaselineMLPGatedMoE):
        return model(batch.mlip_energies)
    raise TypeError(f"Unsupported gating model type: {type(model).__name__}")


def evaluate_gating_model(
    model: nn.Module,
    data_loader: DataLoader[GatingBatch],
    *,
    device: str = "cpu",
    loss_fn: nn.Module | None = None,
) -> float:
    if loss_fn is None:
        loss_fn = nn.MSELoss()

    device_obj = torch.device(device)
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = _move_batch_to_device(batch, device_obj)
            output = _forward_gating_model(model, batch)
            loss = loss_fn(output.prediction, batch.target_ads_eng)
            batch_size = batch.target_ads_eng.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

    if total_count == 0:
        raise ValueError("Data loader is empty")
    return total_loss / total_count


def _save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[EpochMetrics],
    best_val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "history": [metric.__dict__ for metric in history],
            "best_val_loss": best_val_loss,
        },
        path,
    )


def _clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def train_gating_model(
    model: nn.Module,
    train_loader: DataLoader[GatingBatch],
    val_loader: DataLoader[GatingBatch],
    *,
    config: TrainConfig | None = None,
) -> TrainResult:
    config = config or TrainConfig()
    device_obj = torch.device(config.device)
    model = model.to(device_obj)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    checkpoint_dir = (
        Path(config.checkpoint_dir) if config.checkpoint_dir is not None else None
    )
    best_checkpoint_path = (
        checkpoint_dir / "best.pt" if checkpoint_dir is not None else None
    )
    latest_checkpoint_path = (
        checkpoint_dir / "latest.pt" if checkpoint_dir is not None else None
    )

    history: list[EpochMetrics] = []
    best_epoch: int | None = None
    best_val_loss = float("inf")
    best_model_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_train_count = 0

        for batch in train_loader:
            batch = _move_batch_to_device(batch, device_obj)
            optimizer.zero_grad()
            output = _forward_gating_model(model, batch)
            loss = loss_fn(output.prediction, batch.target_ads_eng)
            loss.backward()
            optimizer.step()

            batch_size = batch.target_ads_eng.shape[0]
            total_train_loss += float(loss.item()) * batch_size
            total_train_count += batch_size

        if total_train_count == 0:
            raise ValueError("Training data loader is empty")

        train_loss = total_train_loss / total_train_count
        val_loss = evaluate_gating_model(
            model,
            val_loader,
            device=config.device,
            loss_fn=loss_fn,
        )
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
        )
        history.append(metrics)

        if latest_checkpoint_path is not None:
            _save_checkpoint(
                latest_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                history=history,
                best_val_loss=min(best_val_loss, val_loss),
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = _clone_state_dict(model)
            if best_checkpoint_path is not None:
                _save_checkpoint(
                    best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    history=history,
                    best_val_loss=best_val_loss,
                )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return TrainResult(
        history=history,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_checkpoint_path=best_checkpoint_path,
        latest_checkpoint_path=latest_checkpoint_path,
    )
