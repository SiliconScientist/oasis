from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from oasis.dataset import GatingBatch, GatingDataset, collate_gating_samples
from oasis.evaluate import RegressionMetrics, evaluate_gating_model
from oasis.train import TrainConfig, TrainResult, split_gating_dataset, train_gating_model


@dataclass(frozen=True)
class SweepPoint:
    fraction: float
    train_size: int
    val_size: int
    train_loss_final: float
    val_loss_final: float
    val_mae: float
    val_rmse: float


def _build_subset_loader(
    dataset: Subset,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader[GatingBatch]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_gating_samples,
    )


def _fraction_train_counts(
    n_train_total: int,
    fractions: Sequence[float],
) -> list[tuple[float, int]]:
    counts: list[tuple[float, int]] = []
    last_count = 0
    for fraction in fractions:
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fractions must be in (0, 1], got {fraction}")
        count = max(1, int(round(n_train_total * fraction)))
        count = min(count, n_train_total)
        if count <= last_count:
            continue
        counts.append((fraction, count))
        last_count = count
    if not counts:
        raise ValueError("No valid train counts were generated from fractions")
    return counts


def _nested_train_subsets(
    train_subset: Subset,
    fractions: Sequence[float],
    *,
    seed: int,
) -> list[tuple[float, Subset]]:
    indices = getattr(train_subset, "indices", None)
    if indices is None:
        raise TypeError("Expected train_subset to be a torch.utils.data.Subset")

    perm = torch.randperm(len(indices), generator=torch.Generator().manual_seed(seed))
    shuffled_indices = [indices[i] for i in perm.tolist()]
    subsets: list[tuple[float, Subset]] = []
    for fraction, count in _fraction_train_counts(len(shuffled_indices), fractions):
        subsets.append((fraction, Subset(train_subset.dataset, shuffled_indices[:count])))
    return subsets


def run_data_fraction_sweep(
    dataset: GatingDataset,
    *,
    model_factory: Callable[[], nn.Module],
    fractions: Sequence[float],
    train_config: TrainConfig,
) -> tuple[list[SweepPoint], list[TrainResult]]:
    train_subset, val_subset = split_gating_dataset(
        dataset,
        val_fraction=train_config.val_fraction,
        seed=train_config.random_seed,
    )
    nested_train_subsets = _nested_train_subsets(
        train_subset,
        fractions,
        seed=train_config.random_seed,
    )
    val_loader = _build_subset_loader(
        val_subset,
        batch_size=train_config.batch_size,
        shuffle=False,
    )

    sweep_points: list[SweepPoint] = []
    train_results: list[TrainResult] = []

    for fraction, train_fraction_subset in nested_train_subsets:
        train_loader = _build_subset_loader(
            train_fraction_subset,
            batch_size=train_config.batch_size,
            shuffle=True,
        )
        model = model_factory()
        train_result = train_gating_model(
            model,
            train_loader,
            val_loader,
            config=train_config,
        )
        train_results.append(train_result)
        val_metrics = evaluate_gating_model(
            model,
            val_loader,
            device=train_config.device,
        )
        final_epoch = train_result.history[-1]
        sweep_points.append(
            SweepPoint(
                fraction=fraction,
                train_size=len(train_fraction_subset),
                val_size=len(val_subset),
                train_loss_final=final_epoch.train_loss,
                val_loss_final=final_epoch.val_loss,
                val_mae=val_metrics.mae,
                val_rmse=val_metrics.rmse,
            )
        )

    return sweep_points, train_results


def save_sweep_points_csv(
    sweep_points: Sequence[SweepPoint],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(point) for point in sweep_points]
    pl.DataFrame(rows).write_csv(output_path)
    return output_path
