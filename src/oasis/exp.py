from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import numpy as np
import pandas as pd
import polars as pl
import torch
from sklearn.linear_model import Ridge
from torch import nn
from torch.utils.data import DataLoader, Subset

from oasis.dataset import GatingBatch, GatingDataset, collate_gating_samples
from oasis.evaluate import evaluate_gating_model
from oasis.methods import (
    model_predict,
    residual_correction_predict,
    rmse,
)
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


@dataclass(frozen=True)
class MethodSweepRow:
    method: str
    sweep_axis: str
    size: int
    n_repeats: int
    rmse_mean: float
    rmse_std: float


@dataclass(frozen=True)
class SweepSplit:
    train_idx: tuple[int, ...]
    eval_idx: tuple[int, ...]
    size: int
    repeat: int
    axis: Literal["train_size", "holdout_size"]


@dataclass(frozen=True)
class TabularMethodSpec:
    name: str
    sweep_axis: Literal["train_size", "holdout_size"]
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
) -> pd.DataFrame:
    results = []
    sizes = sorted({split.size for split in splits if split.axis == "train_size"})
    for n_train in sizes:
        rmses = []
        for split in splits:
            if split.axis != "train_size" or split.size != n_train:
                continue
            train_idx = np.asarray(split.train_idx, dtype=int)
            eval_idx = np.asarray(split.eval_idx, dtype=int)
            rmses.append(evaluator(X[train_idx], y[train_idx], X[eval_idx], y[eval_idx]))
        results.append(
            {
                "n_train": n_train,
                "rmse_mean": float(np.mean(rmses)),
                "rmse_std": float(np.std(rmses)),
            }
        )
    return pd.DataFrame(results)


def _holdout_size_sweep(
    evaluator: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float],
    X: np.ndarray,
    y: np.ndarray,
    splits: Sequence[SweepSplit],
) -> pd.DataFrame:
    results = []
    sizes = sorted({split.size for split in splits if split.axis == "holdout_size"})
    for n_hold in sizes:
        rmses = []
        for split in splits:
            if split.axis != "holdout_size" or split.size != n_hold:
                continue
            hold_idx = np.asarray(split.train_idx, dtype=int)
            eval_idx = np.asarray(split.eval_idx, dtype=int)
            rmses.append(evaluator(X[hold_idx], y[hold_idx], X[eval_idx], y[eval_idx]))
        results.append(
            {
                "n_holdout": n_hold,
                "rmse_mean": float(np.mean(rmses)),
                "rmse_std": float(np.std(rmses)),
            }
        )
    return pd.DataFrame(results)


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


def build_holdout_size_splits(
    n_total: int,
    *,
    holdout_sizes: Sequence[int],
    n_repeats: int,
    seed: int,
) -> list[SweepSplit]:
    if n_total < 2:
        raise ValueError("Need at least 2 samples to create holdout/eval splits")
    rng = np.random.default_rng(seed)
    splits: list[SweepSplit] = []
    indices = np.arange(n_total)
    for n_hold in _normalize_sizes(holdout_sizes, max_size=n_total - 1):
        for repeat in range(n_repeats):
            hold_idx = rng.choice(indices, size=n_hold, replace=False)
            eval_idx = np.setdiff1d(indices, hold_idx, assume_unique=False)
            splits.append(
                SweepSplit(
                    train_idx=tuple(int(i) for i in hold_idx.tolist()),
                    eval_idx=tuple(int(i) for i in eval_idx.tolist()),
                    size=n_hold,
                    repeat=repeat,
                    axis="holdout_size",
                )
            )
    return splits


def _build_subset_loader_from_indices(
    dataset: GatingDataset,
    indices: Sequence[int],
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader[GatingBatch]:
    return _build_subset_loader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def run_gating_method_sweep(
    dataset: GatingDataset,
    *,
    method: GatingMethodSpec,
    train_sizes: Sequence[int],
) -> list[MethodSweepRow]:
    n_total = len(dataset)
    if n_total < 2:
        raise ValueError("Need at least 2 samples to run a gating method sweep")

    rows: list[MethodSweepRow] = []
    splits = build_train_size_splits(
        n_total,
        train_sizes=train_sizes,
        n_repeats=method.n_repeats,
        seed=method.seed,
    )
    valid_sizes = sorted({split.size for split in splits})
    for train_size in valid_sizes:
        rmses: list[float] = []
        for split in splits:
            if split.size != train_size:
                continue
            train_loader = _build_subset_loader_from_indices(
                dataset,
                split.train_idx,
                batch_size=method.train_config.batch_size,
                shuffle=True,
            )
            val_loader = _build_subset_loader_from_indices(
                dataset,
                split.eval_idx,
                batch_size=method.train_config.batch_size,
                shuffle=False,
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
            rmses.append(metrics.rmse)
        rows.append(
            MethodSweepRow(
                method=method.name,
                sweep_axis="train_size",
                size=train_size,
                n_repeats=method.n_repeats,
                rmse_mean=float(np.mean(rmses)),
                rmse_std=float(np.std(rmses)),
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
                evaluator=lambda X_train, y_train, X_test, y_test: rmse(
                    y_test,
                    model_predict(
                        lambda: Ridge(alpha=0.1),
                        X_train,
                        y_train,
                        X_test,
                    ),
                ),
                n_repeats=n_repeats,
                seed=41,
            )
        )
    if use_residual:
        specs.append(
            TabularMethodSpec(
                name="residual",
                sweep_axis="holdout_size",
                evaluator=lambda X_holdout, y_holdout, X_eval, y_eval: rmse(
                    y_eval,
                    residual_correction_predict(X_holdout, y_holdout, X_eval),
                ),
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
    holdout_sizes: Sequence[int] | None = None,
) -> list[MethodSweepRow]:
    feature_cols = _mlip_columns(df)
    if not feature_cols:
        raise ValueError(
            "No MLIP prediction columns found (expected *_mlip_ads_eng_median)."
        )
    if df.height <= 5:
        raise ValueError("Not enough data to evaluate (need >5 samples).")
    X = df.select(feature_cols).to_numpy()
    y = df["reference_ads_eng"].to_numpy()
    holdout_sizes = holdout_sizes if holdout_sizes is not None else train_sizes

    rows: list[MethodSweepRow] = []
    for method in methods:
        if method.sweep_axis == "train_size":
            splits = build_train_size_splits(
                len(X),
                train_sizes=train_sizes,
                n_repeats=method.n_repeats,
                seed=method.seed,
            )
            results = _train_size_sweep(
                method.evaluator,
                X,
                y,
                splits,
            )
            rows.extend(
                MethodSweepRow(
                    method=method.name,
                    sweep_axis="train_size",
                    size=int(row["n_train"]),
                    n_repeats=method.n_repeats,
                    rmse_mean=float(row["rmse_mean"]),
                    rmse_std=float(row["rmse_std"]),
                )
                for row in results.to_dict("records")
            )
        else:
            splits = build_holdout_size_splits(
                len(X),
                holdout_sizes=holdout_sizes,
                n_repeats=method.n_repeats,
                seed=method.seed,
            )
            results = _holdout_size_sweep(
                method.evaluator,
                X,
                y,
                splits,
            )
            rows.extend(
                MethodSweepRow(
                    method=method.name,
                    sweep_axis="holdout_size",
                    size=int(row["n_holdout"]),
                    n_repeats=method.n_repeats,
                    rmse_mean=float(row["rmse_mean"]),
                    rmse_std=float(row["rmse_std"]),
                )
                for row in results.to_dict("records")
            )
    return rows


def run_all_method_sweeps(
    *,
    wide_df: pl.DataFrame,
    gating_dataset: GatingDataset,
    tabular_methods: Sequence[TabularMethodSpec],
    gating_methods: Sequence[GatingMethodSpec],
    tabular_train_sizes: Sequence[int],
    tabular_holdout_sizes: Sequence[int] | None = None,
    gating_train_sizes: Sequence[int],
) -> list[MethodSweepRow]:
    rows = run_tabular_method_sweeps(
        wide_df,
        methods=tabular_methods,
        train_sizes=tabular_train_sizes,
        holdout_sizes=tabular_holdout_sizes,
    )
    for method in gating_methods:
        rows.extend(
            run_gating_method_sweep(
                gating_dataset,
                method=method,
                train_sizes=gating_train_sizes,
            )
        )
    return rows


def save_method_sweep_rows_csv(
    rows: Sequence[MethodSweepRow],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([asdict(row) for row in rows]).write_csv(output_path)
    return output_path


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
        holdout_sizes=list(range(min_train, max_train + 1)),
    )
    results_by_method = {row.method: [] for row in rows}
    for row in rows:
        results_by_method[row.method].append(row)

    def _rows_to_df(method_name: str, axis_col: str) -> pd.DataFrame | None:
        method_rows = results_by_method.get(method_name, [])
        if not method_rows:
            return None
        return pd.DataFrame(
            {
                axis_col: [row.size for row in method_rows],
                "rmse_mean": [row.rmse_mean for row in method_rows],
                "rmse_std": [row.rmse_std for row in method_rows],
            }
        )

    return {
        "ridge_df": _rows_to_df("ridge", "n_train"),
        "resid_df": _rows_to_df("residual", "n_holdout"),
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
