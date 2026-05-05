from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def trimmed_mean_predictions(X: np.ndarray) -> np.ndarray:
    row_means = X.mean(axis=1)
    row_stds = X.std(axis=1)
    mask = np.abs(X - row_means[:, None]) <= row_stds[:, None]
    empty_mask = mask.sum(axis=1) == 0
    if empty_mask.any():
        mask[empty_mask] = True
    return (X * mask).sum(axis=1) / mask.sum(axis=1)


def model_predict(
    model_factory: Callable[[], Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    model = model_factory()
    model.fit(X_train, y_train)
    return np.asarray(model.predict(X_test))


def trimmed_model_predict(
    model_factory: Callable[[], Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    z_thresh: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    model = model_factory()
    model.fit(X_train, y_train)
    preds = np.asarray(model.predict(X_test))

    w = model.coef_
    contrib = X_test * w
    mu = contrib.mean()
    sigma = contrib.std() if contrib.std() > 0 else 1.0
    z = (contrib - mu) / sigma
    keep_mask = (np.abs(z) <= z_thresh).all(axis=1)
    if keep_mask.sum() == 0:
        keep_mask = np.ones(len(X_test), dtype=bool)
    return preds[keep_mask], keep_mask


def residual_correction_predict(
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    X_eval: np.ndarray,
) -> np.ndarray:
    residuals = y_holdout[:, None] - X_holdout
    mean_residuals = residuals.mean(axis=0)
    return (X_eval + mean_residuals).mean(axis=1)


def residual_correction_trimmed_predict(
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    X_eval: np.ndarray,
) -> np.ndarray:
    residuals = y_holdout[:, None] - X_holdout
    mean_residuals = residuals.mean(axis=0)
    return trimmed_mean_predictions(X_eval + mean_residuals)


def linearization_predict(
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    X_eval: np.ndarray,
) -> np.ndarray:
    mu_holdout = (
        X_holdout.reshape(-1, 1)
        if X_holdout.ndim == 1
        else X_holdout.mean(axis=1, keepdims=True)
    )
    lr = LinearRegression().fit(mu_holdout, y_holdout.reshape(-1, 1))
    a = float(lr.coef_.ravel()[0])
    b = float(lr.intercept_.ravel()[0])
    return (a * X_eval + b).mean(axis=1)


def linearization_trimmed_predict(
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    X_eval: np.ndarray,
) -> np.ndarray:
    mu_holdout = (
        X_holdout.reshape(-1, 1)
        if X_holdout.ndim == 1
        else trimmed_mean_predictions(X_holdout).reshape(-1, 1)
    )
    lr = LinearRegression().fit(mu_holdout, y_holdout.reshape(-1, 1))
    a = float(lr.coef_.ravel()[0])
    b = float(lr.intercept_.ravel()[0])
    return trimmed_mean_predictions(a * X_eval + b)
