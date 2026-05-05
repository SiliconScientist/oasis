from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from oasis.dataset import GatingBatch, GatingDataset, collate_gating_samples
from oasis.evaluate import evaluate_gating_model
from oasis.methods import (
    residual_rmse,
    ridge_rmse,
)
from oasis.train import (
    TrainConfig,
    TrainResult,
    build_gating_dataloaders_from_indices,
    split_gating_dataset,
    train_gating_model,
)


@dataclass(frozen=True)
class SweepPoint:
    fraction: float
    train_size: int
    val_size: int
    train_loss_final: float
    val_loss_final: float
    val_mae: float
    val_rmse: float


@dataclass(frozen=True)
class MethodSweepRecord:
    method: str
    sweep_axis: str
    size: int
    repeat: int
    split_id: str
    rmse: float


@dataclass(frozen=True)
class SweepSplit:
    train_idx: tuple[int, ...]
    eval_idx: tuple[int, ...]
    size: int
    repeat: int
    axis: Literal["train_size"]


@dataclass(frozen=True)
class TabularMethodSpec:
    name: str
    sweep_axis: Literal["train_size"]
    evaluator: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float]
    n_repeats: int
    seed: int


@dataclass(frozen=True)
class GatingMethodSpec:
    name: str
    model_factory: Callable[[], nn.Module]
    train_config: TrainConfig
    n_repeats: int
    seed: int


def _mlip_columns(df: Any) -> list[str]:
    return [c for c in df.columns if c.endswith("_mlip_ads_eng_median")]


def _normalize_sizes(
    sizes: Sequence[int],
    *,
    max_size: int,
) -> list[int]:
    normalized = sorted({size for size in sizes if 0 < size <= max_size})
    if not normalized:
        raise ValueError("No valid sweep sizes were provided")
    return normalized


def _train_size_sweep(
    evaluator: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float],
    X: np.ndarray,
    y: np.ndarray,
    splits: Sequence[SweepSplit],
) -> list[tuple[SweepSplit, float]]:
    results: list[tuple[SweepSplit, float]] = []
    for split in splits:
        train_idx = np.asarray(split.train_idx, dtype=int)
        eval_idx = np.asarray(split.eval_idx, dtype=int)
        rmse = evaluator(X[train_idx], y[train_idx], X[eval_idx], y[eval_idx])
        results.append((split, rmse))
    return results


def _limit_splits_by_repeats(
    splits: Sequence[SweepSplit],
    *,
    axis: Literal["train_size"],
    n_repeats: int,
) -> list[SweepSplit]:
    return [
        split
        for split in splits
        if split.axis == axis and split.repeat < n_repeats
    ]


def build_train_size_splits(
    n_total: int,
    *,
    train_sizes: Sequence[int],
    n_repeats: int,
    seed: int,
) -> list[SweepSplit]:
    if n_total < 2:
        raise ValueError("Need at least 2 samples to create train/eval splits")
    rng = np.random.default_rng(seed)
    splits: list[SweepSplit] = []
    indices = np.arange(n_total)
    for n_train in _normalize_sizes(train_sizes, max_size=n_total - 1):
        for repeat in range(n_repeats):
            train_idx = rng.choice(indices, size=n_train, replace=False)
            eval_idx = np.setdiff1d(indices, train_idx, assume_unique=False)
            splits.append(
                SweepSplit(
                    train_idx=tuple(int(i) for i in train_idx.tolist()),
                    eval_idx=tuple(int(i) for i in eval_idx.tolist()),
                    size=n_train,
                    repeat=repeat,
                    axis="train_size",
                )
            )
    return splits
def run_gating_method_sweep(
    dataset: GatingDataset,
    *,
    method: GatingMethodSpec,
    train_sizes: Sequence[int],
    train_splits: Sequence[SweepSplit] | None = None,
) -> list[MethodSweepRecord]:
    n_total = len(dataset)
    if n_total < 2:
        raise ValueError("Need at least 2 samples to run a gating method sweep")

    rows: list[MethodSweepRecord] = []
    splits = (
        _limit_splits_by_repeats(
            train_splits,
            axis="train_size",
            n_repeats=method.n_repeats,
        )
        if train_splits is not None
        else build_train_size_splits(
            n_total,
            train_sizes=train_sizes,
            n_repeats=method.n_repeats,
            seed=method.seed,
        )
    )
    for split in splits:
        train_loader, val_loader = build_gating_dataloaders_from_indices(
            dataset,
            batch_size=method.train_config.batch_size,
            train_indices=split.train_idx,
            eval_indices=split.eval_idx,
        )
        model = method.model_factory()
        train_gating_model(
            model,
            train_loader,
            val_loader,
            config=TrainConfig(
                batch_size=method.train_config.batch_size,
                epochs=method.train_config.epochs,
                learning_rate=method.train_config.learning_rate,
                weight_decay=method.train_config.weight_decay,
                val_fraction=method.train_config.val_fraction,
                random_seed=method.train_config.random_seed + split.repeat,
                checkpoint_dir=None,
                device=method.train_config.device,
            ),
        )
        metrics = evaluate_gating_model(
            model,
            val_loader,
            device=method.train_config.device,
        )
        rows.append(
            MethodSweepRecord(
                method=method.name,
                sweep_axis="train_size",
                size=split.size,
                repeat=split.repeat,
                split_id=f"train_size:{split.size}:repeat:{split.repeat}",
                rmse=metrics.rmse,
            )
        )
    return rows


def default_tabular_method_specs(
    *,
    use_ridge: bool,
    use_residual: bool,
    n_repeats: int,
) -> list[TabularMethodSpec]:
    specs: list[TabularMethodSpec] = []
    if use_ridge:
        specs.append(
            TabularMethodSpec(
                name="ridge",
                sweep_axis="train_size",
                evaluator=ridge_rmse,
                n_repeats=n_repeats,
                seed=41,
            )
        )
    if use_residual:
        specs.append(
            TabularMethodSpec(
                name="residual",
                sweep_axis="train_size",
                evaluator=residual_rmse,
                n_repeats=n_repeats,
                seed=999,
            )
        )
    return specs


def run_tabular_method_sweeps(
    df: pl.DataFrame,
    *,
    methods: Sequence[TabularMethodSpec],
    train_sizes: Sequence[int],
    train_splits: Sequence[SweepSplit] | None = None,
) -> list[MethodSweepRecord]:
    feature_cols = _mlip_columns(df)
    if not feature_cols:
        raise ValueError(
            "No MLIP prediction columns found (expected *_mlip_ads_eng_median)."
        )
    if df.height <= 5:
        raise ValueError("Not enough data to evaluate (need >5 samples).")
    X = df.select(feature_cols).to_numpy()
    y = df["reference_ads_eng"].to_numpy()
    rows: list[MethodSweepRecord] = []
    for method in methods:
        splits = (
            _limit_splits_by_repeats(
                train_splits,
                axis="train_size",
                n_repeats=method.n_repeats,
            )
            if train_splits is not None
            else build_train_size_splits(
                len(X),
                train_sizes=train_sizes,
                n_repeats=method.n_repeats,
                seed=method.seed,
            )
        )
        results = _train_size_sweep(
            method.evaluator,
            X,
            y,
            splits,
        )
        rows.extend(
            MethodSweepRecord(
                method=method.name,
                sweep_axis="train_size",
                size=split.size,
                repeat=split.repeat,
                split_id=f"train_size:{split.size}:repeat:{split.repeat}",
                rmse=float(rmse),
            )
            for split, rmse in results
        )
    return rows


def run_all_method_sweeps(
    *,
    wide_df: pl.DataFrame,
    gating_dataset: GatingDataset,
    tabular_methods: Sequence[TabularMethodSpec],
    gating_methods: Sequence[GatingMethodSpec],
    tabular_train_sizes: Sequence[int],
    gating_train_sizes: Sequence[int],
    shared_train_seed: int = 0,
) -> list[MethodSweepRecord]:
    max_train_repeats = max(
        [m.n_repeats for m in tabular_methods] + [m.n_repeats for m in gating_methods],
        default=0,
    )
    shared_train_splits = (
        build_train_size_splits(
            len(gating_dataset),
            train_sizes=sorted(set(tabular_train_sizes).union(gating_train_sizes)),
            n_repeats=max_train_repeats,
            seed=shared_train_seed,
        )
        if max_train_repeats > 0
        else []
    )
    rows = run_tabular_method_sweeps(
        wide_df,
        methods=tabular_methods,
        train_sizes=tabular_train_sizes,
        train_splits=shared_train_splits,
    )
    for method in gating_methods:
        rows.extend(
            run_gating_method_sweep(
                gating_dataset,
                method=method,
                train_sizes=gating_train_sizes,
                train_splits=shared_train_splits,
            )
    )
    return rows


def run_single_split_comparison(
    *,
    wide_df: pl.DataFrame,
    gating_dataset: GatingDataset,
    tabular_methods: Sequence[TabularMethodSpec],
    gating_methods: Sequence[GatingMethodSpec],
    split: SweepSplit,
) -> list[MethodSweepRecord]:
    feature_cols = _mlip_columns(wide_df)
    if not feature_cols:
        raise ValueError(
            "No MLIP prediction columns found (expected *_mlip_ads_eng_median)."
        )

    X = wide_df.select(feature_cols).to_numpy()
    y = wide_df["reference_ads_eng"].to_numpy()
    train_idx = np.asarray(split.train_idx, dtype=int)
    eval_idx = np.asarray(split.eval_idx, dtype=int)
    rows: list[MethodSweepRecord] = []

    for method in tabular_methods:
        rmse = method.evaluator(X[train_idx], y[train_idx], X[eval_idx], y[eval_idx])
        rows.append(
            MethodSweepRecord(
                method=method.name,
                sweep_axis=split.axis,
                size=split.size,
                repeat=split.repeat,
                split_id=f"{split.axis}:{split.size}:repeat:{split.repeat}",
                rmse=float(rmse),
            )
        )

    for method in gating_methods:
        train_loader, val_loader = build_gating_dataloaders_from_indices(
            gating_dataset,
            batch_size=method.train_config.batch_size,
            train_indices=split.train_idx,
            eval_indices=split.eval_idx,
        )
        model = method.model_factory()
        train_gating_model(
            model,
            train_loader,
            val_loader,
            config=TrainConfig(
                batch_size=method.train_config.batch_size,
                epochs=method.train_config.epochs,
                learning_rate=method.train_config.learning_rate,
                weight_decay=method.train_config.weight_decay,
                val_fraction=method.train_config.val_fraction,
                random_seed=method.train_config.random_seed + split.repeat,
                checkpoint_dir=None,
                device=method.train_config.device,
            ),
        )
        metrics = evaluate_gating_model(
            model,
            val_loader,
            device=method.train_config.device,
        )
        rows.append(
            MethodSweepRecord(
                method=method.name,
                sweep_axis=split.axis,
                size=split.size,
                repeat=split.repeat,
                split_id=f"{split.axis}:{split.size}:repeat:{split.repeat}",
                rmse=metrics.rmse,
            )
        )

    return rows


def save_method_sweep_rows_csv(
    rows: Sequence[MethodSweepRecord],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([asdict(row) for row in rows]).write_csv(output_path)
    return output_path


def aggregate_method_sweep_records(
    rows: Sequence[MethodSweepRecord] | pl.DataFrame,
) -> pl.DataFrame:
    if isinstance(rows, pl.DataFrame):
        df = rows.clone()
    else:
        df = pl.DataFrame([asdict(row) for row in rows])
    if df.height == 0:
        return pl.DataFrame(
            schema={
                "method": pl.String,
                "sweep_axis": pl.String,
                "size": pl.Int64,
                "n_repeats": pl.Int64,
                "rmse_mean": pl.Float64,
                "rmse_std": pl.Float64,
            }
        )
    if "rmse" not in df.columns:
        return df.sort(["method", "size"])
    return (
        df.group_by(["method", "sweep_axis", "size"])
        .agg(
            [
                pl.len().alias("n_repeats"),
                pl.col("rmse").mean().alias("rmse_mean"),
                pl.col("rmse").std(ddof=0).fill_null(0.0).alias("rmse_std"),
            ]
        )
        .sort(["method", "size"])
    )


def build_learning_curve_sweeps(
    df: pl.DataFrame,
    *,
    min_train: int,
    max_train: int,
    n_repeats: int,
    use_ridge: bool,
    use_residual: bool,
) -> dict[str, pd.DataFrame | None]:
    specs = default_tabular_method_specs(
        use_ridge=use_ridge,
        use_residual=use_residual,
        n_repeats=n_repeats,
    )
    rows = run_tabular_method_sweeps(
        df,
        methods=specs,
        train_sizes=list(range(min_train, max_train + 1)),
    )
    aggregated = aggregate_method_sweep_records(rows)

    def _rows_to_df(method_name: str, axis_col: str) -> pd.DataFrame | None:
        method_rows = aggregated.filter(pl.col("method") == method_name).sort("size")
        if method_rows.height == 0:
            return None
        return pd.DataFrame(
            {
                axis_col: method_rows["size"].to_list(),
                "rmse_mean": method_rows["rmse_mean"].to_list(),
                "rmse_std": method_rows["rmse_std"].to_list(),
            }
        )

    return {
        "ridge_df": _rows_to_df("ridge", "n_train"),
        "resid_df": _rows_to_df("residual", "n_train"),
    }


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
