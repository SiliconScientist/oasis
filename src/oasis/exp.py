from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import polars as pl
import torch
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, Subset

from oasis.dataset import GatingBatch, GatingDataset, collate_gating_samples
from oasis.evaluate import evaluate_gating_model
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


def _mlip_columns(df: Any) -> list[str]:
    return [c for c in df.columns if c.endswith("_mlip_ads_eng_median")]


def _trimmed_mean_predictions(X_corrected: np.ndarray) -> np.ndarray:
    row_means = X_corrected.mean(axis=1)
    row_stds = X_corrected.std(axis=1)
    mask = np.abs(X_corrected - row_means[:, None]) <= row_stds[:, None]
    empty_mask = mask.sum(axis=1) == 0
    if empty_mask.any():
        mask[empty_mask] = True
    return (X_corrected * mask).sum(axis=1) / mask.sum(axis=1)


def _sweep_model(
    model_factory: Callable[[], Any],
    X: np.ndarray,
    y: np.ndarray,
    min_train: int,
    max_train: int,
    n_repeats: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    results = []
    max_train = min(max_train, len(X) - 1)
    for n_train in range(min_train, max_train + 1):
        rmses = []
        for _ in range(n_repeats):
            idx = np.arange(len(X))
            train_idx = rng.choice(idx, size=n_train, replace=False)
            test_idx = np.setdiff1d(idx, train_idx, assume_unique=False)
            model = model_factory()
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[test_idx])
            rmses.append(np.sqrt(mean_squared_error(y[test_idx], preds)))
        results.append(
            {
                "n_train": n_train,
                "rmse_mean": float(np.mean(rmses)),
                "rmse_std": float(np.std(rmses)),
            }
        )
    return pd.DataFrame(results)


def _sweep_model_trimmed(
    model_factory: Callable[[], Any],
    X: np.ndarray,
    y: np.ndarray,
    min_train: int,
    max_train: int,
    n_repeats: int,
    rng: np.random.Generator,
    z_thresh: float = 1.0,
) -> pd.DataFrame:
    results = []
    max_train = min(max_train, len(X) - 1)
    for n_train in range(min_train, max_train + 1):
        rmses = []
        for _ in range(n_repeats):
            idx = np.arange(len(X))
            train_idx = rng.choice(idx, size=n_train, replace=False)
            test_idx = np.setdiff1d(idx, train_idx, assume_unique=False)
            model = model_factory()
            model.fit(X[train_idx], y[train_idx])
            X_test = X[test_idx]
            preds = model.predict(X_test)

            w = model.coef_
            contrib = X_test * w
            mu = contrib.mean()
            sigma = contrib.std() if contrib.std() > 0 else 1.0
            z = (contrib - mu) / sigma
            keep_mask = (np.abs(z) <= z_thresh).all(axis=1)
            if keep_mask.sum() == 0:
                keep_mask = np.ones(len(X_test), dtype=bool)

            rmses.append(
                np.sqrt(mean_squared_error(y[test_idx][keep_mask], preds[keep_mask]))
            )
        results.append(
            {
                "n_train": n_train,
                "rmse_mean": float(np.mean(rmses)),
                "rmse_std": float(np.std(rmses)),
            }
        )
    return pd.DataFrame(results)


def _residual_sweep(
    X: np.ndarray,
    y: np.ndarray,
    min_hold: int,
    max_hold: int,
    n_repeats: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    results = []
    max_hold = min(max_hold, len(X) - 1)
    for n_hold in range(min_hold, max_hold + 1):
        rmses = []
        for _ in range(n_repeats):
            idx = np.arange(len(X))
            hold_idx = rng.choice(idx, size=n_hold, replace=False)
            keep_idx = np.setdiff1d(idx, hold_idx, assume_unique=False)

            residuals = y[hold_idx, None] - X[hold_idx]
            mean_residuals = residuals.mean(axis=0)
            preds = (X[keep_idx] + mean_residuals).mean(axis=1)
            rmses.append(np.sqrt(mean_squared_error(y[keep_idx], preds)))

        results.append(
            {
                "n_holdout": n_hold,
                "rmse_mean": float(np.mean(rmses)),
                "rmse_std": float(np.std(rmses)),
            }
        )
    return pd.DataFrame(results)


def _residual_sweep_trimmed(
    X: np.ndarray,
    y: np.ndarray,
    min_hold: int,
    max_hold: int,
    n_repeats: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    results = []
    max_hold = min(max_hold, len(X) - 1)
    for n_hold in range(min_hold, max_hold + 1):
        rmses = []
        for _ in range(n_repeats):
            idx = np.arange(len(X))
            hold_idx = rng.choice(idx, size=n_hold, replace=False)
            keep_idx = np.setdiff1d(idx, hold_idx, assume_unique=False)

            residuals = y[hold_idx, None] - X[hold_idx]
            mean_residuals = residuals.mean(axis=0)
            preds = _trimmed_mean_predictions(X[keep_idx] + mean_residuals)
            rmses.append(np.sqrt(mean_squared_error(y[keep_idx], preds)))

        results.append(
            {
                "n_holdout": n_hold,
                "rmse_mean": float(np.mean(rmses)),
                "rmse_std": float(np.std(rmses)),
            }
        )
    return pd.DataFrame(results)


def _linearization_sweep(
    X: np.ndarray,
    y: np.ndarray,
    min_hold: int,
    max_hold: int,
    n_repeats: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    results = []
    max_hold = min(max_hold, len(X) - 1)
    for n_hold in range(min_hold, max_hold + 1):
        rmses = []
        for _ in range(n_repeats):
            idx = np.arange(len(X))
            hold_idx = rng.choice(idx, size=n_hold, replace=False)
            keep_idx = np.setdiff1d(idx, hold_idx, assume_unique=False)

            X_hold = X[hold_idx]
            mu_h = (
                X_hold.reshape(-1, 1)
                if X_hold.ndim == 1
                else X_hold.mean(axis=1, keepdims=True)
            )
            lr = LinearRegression().fit(mu_h, y[hold_idx].reshape(-1, 1))
            a = float(lr.coef_.ravel()[0])
            b = float(lr.intercept_.ravel()[0])
            preds = (a * X[keep_idx] + b).mean(axis=1)
            rmses.append(np.sqrt(mean_squared_error(y[keep_idx], preds)))

        results.append(
            {
                "n_holdout": n_hold,
                "rmse_mean": float(np.mean(rmses)),
                "rmse_std": float(np.std(rmses)),
            }
        )
    return pd.DataFrame(results)


def _linearization_sweep_trimmed(
    X: np.ndarray,
    y: np.ndarray,
    min_hold: int,
    max_hold: int,
    n_repeats: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    results = []
    max_hold = min(max_hold, len(X) - 1)
    for n_hold in range(min_hold, max_hold + 1):
        rmses = []
        for _ in range(n_repeats):
            idx = np.arange(len(X))
            hold_idx = rng.choice(idx, size=n_hold, replace=False)
            keep_idx = np.setdiff1d(idx, hold_idx, assume_unique=False)

            X_hold = X[hold_idx]
            mu_h = (
                X_hold.reshape(-1, 1)
                if X_hold.ndim == 1
                else _trimmed_mean_predictions(X_hold).reshape(-1, 1)
            )
            lr = LinearRegression().fit(mu_h, y[hold_idx].reshape(-1, 1))
            a = float(lr.coef_.ravel()[0])
            b = float(lr.intercept_.ravel()[0])
            preds = _trimmed_mean_predictions(a * X[keep_idx] + b)
            rmses.append(np.sqrt(mean_squared_error(y[keep_idx], preds)))

        results.append(
            {
                "n_holdout": n_hold,
                "rmse_mean": float(np.mean(rmses)),
                "rmse_std": float(np.std(rmses)),
            }
        )
    return pd.DataFrame(results)


def build_learning_curve_sweeps(
    df: pl.DataFrame,
    *,
    min_train: int,
    max_train: int,
    n_repeats: int,
    use_trim: bool,
    use_ridge: bool,
    use_kernel_ridge: bool,
    use_lasso: bool,
    use_elastic: bool,
    use_residual: bool,
    use_linearization: bool,
) -> dict[str, pd.DataFrame | None]:
    feature_cols = _mlip_columns(df)
    if not feature_cols:
        raise ValueError(
            "No MLIP prediction columns found (expected *_mlip_ads_eng_median)."
        )
    if df.height <= 5:
        raise ValueError("Not enough data to evaluate (need >5 samples).")

    X = df.select(feature_cols).to_numpy()
    y = df["reference_ads_eng"].to_numpy()

    return {
        "ridge_df": (
            _sweep_model(
                lambda: Ridge(alpha=0.1),
                X,
                y,
                min_train,
                max_train,
                n_repeats,
                np.random.default_rng(41),
            )
            if use_ridge
            else None
        ),
        "kernel_ridge_df": (
            _sweep_model(
                lambda: KernelRidge(alpha=1.0, kernel="rbf"),
                X,
                y,
                min_train,
                max_train,
                n_repeats,
                np.random.default_rng(2718),
            )
            if use_kernel_ridge
            else None
        ),
        "ridge_trimmed_df": (
            _sweep_model_trimmed(
                lambda: Ridge(alpha=0.1),
                X,
                y,
                min_train,
                max_train,
                n_repeats,
                np.random.default_rng(42),
                z_thresh=1.0,
            )
            if use_trim and use_ridge
            else None
        ),
        "lasso_df": (
            _sweep_model(
                lambda: Lasso(alpha=0.1, max_iter=10000),
                X,
                y,
                min_train,
                max_train,
                n_repeats,
                np.random.default_rng(123),
            )
            if use_lasso
            else None
        ),
        "lasso_trimmed_df": (
            _sweep_model_trimmed(
                lambda: Lasso(alpha=0.1, max_iter=10000),
                X,
                y,
                min_train,
                max_train,
                n_repeats,
                np.random.default_rng(124),
                z_thresh=1.0,
            )
            if use_trim and use_lasso
            else None
        ),
        "elastic_df": (
            _sweep_model(
                lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=20000),
                X,
                y,
                min_train,
                max_train,
                n_repeats,
                np.random.default_rng(321),
            )
            if use_elastic
            else None
        ),
        "elastic_trimmed_df": (
            _sweep_model_trimmed(
                lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=20000),
                X,
                y,
                min_train,
                max_train,
                n_repeats,
                np.random.default_rng(322),
                z_thresh=1.0,
            )
            if use_trim and use_elastic
            else None
        ),
        "resid_df": (
            _residual_sweep(
                X, y, min_train, max_train, n_repeats, np.random.default_rng(999)
            )
            if use_residual
            else None
        ),
        "resid_trimmed_df": (
            _residual_sweep_trimmed(
                X, y, min_train, max_train, n_repeats, np.random.default_rng(77)
            )
            if use_trim and use_residual
            else None
        ),
        "linear_df": (
            _linearization_sweep(
                X, y, min_train, max_train, n_repeats, np.random.default_rng(2024)
            )
            if use_linearization
            else None
        ),
        "linear_trimmed_df": (
            _linearization_sweep_trimmed(
                X, y, min_train, max_train, n_repeats, np.random.default_rng(2025)
            )
            if use_trim and use_linearization
            else None
        ),
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
