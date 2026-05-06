from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from oasis.exp import SweepSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


_SWEEP_RESULT_COLUMNS = ["n_train", "rmse_mean", "rmse_std"]


def trimmed_mean_predictions(X_corrected: np.ndarray) -> np.ndarray:
    """
    Compute per-sample mean after dropping MLIP predictions outside 1 std from row mean.
    """
    row_means = X_corrected.mean(axis=1)
    row_stds = X_corrected.std(axis=1)
    # Broadcast masks; keep all if std is zero or mask would be empty.
    mask = np.abs(X_corrected - row_means[:, None]) <= row_stds[:, None]
    empty_mask = mask.sum(axis=1) == 0
    if empty_mask.any():
        mask[empty_mask] = True
    return (X_corrected * mask).sum(axis=1) / mask.sum(axis=1)


def sweep_results_frame(rmses_by_size: dict[int, list[float]]) -> pd.DataFrame:
    """Build the standard sweep result frame from per-size RMSE samples."""

    rows = [
        {
            "n_train": n_train,
            "rmse_mean": float(np.mean(rmses)),
            "rmse_std": float(np.std(rmses)),
        }
        for n_train, rmses in sorted(rmses_by_size.items())
    ]
    return pd.DataFrame(rows, columns=_SWEEP_RESULT_COLUMNS)


def sweep_model(
    model_factory,
    X: np.ndarray,
    y: np.ndarray,
    splits: Sequence[SweepSplit],
) -> pd.DataFrame:
    """Evaluate a supervised model across precomputed sweep splits."""

    rmses_by_size: dict[int, list[float]] = {}
    for split in splits:
        model = model_factory()
        model.fit(X[split.train_idx], y[split.train_idx])
        X_test = X[split.test_idx]
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y[split.test_idx], preds))
        rmses_by_size.setdefault(split.sweep_size, []).append(rmse)
    return sweep_results_frame(rmses_by_size)


def sweep_model_trimmed(
    model_factory,
    X: np.ndarray,
    y: np.ndarray,
    splits: Sequence[SweepSplit],
    z_thresh: float = 1.0,
) -> pd.DataFrame:
    """
    Fit the model, drop test samples with large contribution z-scores, and refit.
    """
    rmses_by_size: dict[int, list[float]] = {}
    for split in splits:
        model = model_factory()
        model.fit(X[split.train_idx], y[split.train_idx])

        X_test = X[split.test_idx]
        preds = model.predict(X_test)

        # Compute contribution z-scores per sample
        w = model.coef_
        contrib = X_test * w
        mu = contrib.mean()
        sigma = contrib.std() if contrib.std() > 0 else 1.0
        z = (contrib - mu) / sigma
        keep_mask = (np.abs(z) <= z_thresh).all(axis=1)
        if keep_mask.sum() == 0:
            keep_mask = np.ones(len(X_test), dtype=bool)

        preds_eval = preds[keep_mask]
        y_eval = y[split.test_idx][keep_mask]
        rmse = np.sqrt(mean_squared_error(y_eval, preds_eval))
        rmses_by_size.setdefault(split.sweep_size, []).append(rmse)
    return sweep_results_frame(rmses_by_size)


def residual_sweep(
    X: np.ndarray,
    y: np.ndarray,
    splits: Sequence[SweepSplit],
) -> pd.DataFrame:
    """Evaluate residual correction across precomputed sweep splits."""

    rmses_by_size: dict[int, list[float]] = {}
    for split in splits:
        X_train = X[split.train_idx]
        y_train = y[split.train_idx]

        residuals = y_train[:, None] - X_train
        mean_residuals = residuals.mean(axis=0)

        X_corrected = X[split.test_idx] + mean_residuals
        preds = X_corrected.mean(axis=1)
        rmse = np.sqrt(mean_squared_error(y[split.test_idx], preds))
        rmses_by_size.setdefault(split.sweep_size, []).append(rmse)
    return sweep_results_frame(rmses_by_size)


def residual_sweep_trimmed(
    X: np.ndarray,
    y: np.ndarray,
    splits: Sequence[SweepSplit],
) -> pd.DataFrame:
    """
    Residual correction with per-sample outlier MLIP removal before averaging.
    """
    rmses_by_size: dict[int, list[float]] = {}
    for split in splits:
        X_train = X[split.train_idx]
        y_train = y[split.train_idx]

        residuals = y_train[:, None] - X_train
        mean_residuals = residuals.mean(axis=0)

        X_corrected = X[split.test_idx] + mean_residuals
        preds = trimmed_mean_predictions(X_corrected)
        rmse = np.sqrt(mean_squared_error(y[split.test_idx], preds))
        rmses_by_size.setdefault(split.sweep_size, []).append(rmse)
    return sweep_results_frame(rmses_by_size)


def linearization_sweep(
    X: np.ndarray,
    y: np.ndarray,
    splits: Sequence[SweepSplit],
) -> pd.DataFrame:
    """Evaluate linearization correction across precomputed sweep splits."""

    rmses_by_size: dict[int, list[float]] = {}
    for split in splits:
        X_train = X[split.train_idx]
        y_train = y[split.train_idx]

        Xh = np.asarray(X_train)
        yh = np.asarray(y_train).reshape(-1, 1)

        if Xh.ndim == 1:
            mu_h = Xh.reshape(-1, 1)
        else:
            mu_h = Xh.mean(axis=1, keepdims=True)

        lr = LinearRegression().fit(mu_h, yh)
        a = float(lr.coef_.ravel()[0])
        b = float(lr.intercept_.ravel()[0])

        X_linearized = a * X + b
        preds = X_linearized[split.test_idx].mean(axis=1)
        rmse = np.sqrt(mean_squared_error(y[split.test_idx], preds))
        rmses_by_size.setdefault(split.sweep_size, []).append(rmse)
    return sweep_results_frame(rmses_by_size)


def linearization_sweep_trimmed(
    X: np.ndarray,
    y: np.ndarray,
    splits: Sequence[SweepSplit],
) -> pd.DataFrame:
    """
    Linearize against trimmed train means, then trim ensemble averaging.
    """
    rmses_by_size: dict[int, list[float]] = {}
    for split in splits:
        X_train = X[split.train_idx]
        y_train = y[split.train_idx]

        Xh = np.asarray(X_train)
        yh = np.asarray(y_train).reshape(-1, 1)

        if Xh.ndim == 1:
            mu_h = Xh.reshape(-1, 1)
        else:
            mu_h = trimmed_mean_predictions(Xh).reshape(-1, 1)

        lr = LinearRegression().fit(mu_h, yh)
        a = float(lr.coef_.ravel()[0])
        b = float(lr.intercept_.ravel()[0])

        X_linearized = a * X + b
        preds = trimmed_mean_predictions(X_linearized[split.test_idx])
        rmse = np.sqrt(mean_squared_error(y[split.test_idx], preds))
        rmses_by_size.setdefault(split.sweep_size, []).append(rmse)
    return sweep_results_frame(rmses_by_size)
