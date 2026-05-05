from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def model_predict(
    model_factory: Callable[[], Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
) -> np.ndarray:
    model = model_factory()
    model.fit(X_train, y_train)
    return np.asarray(model.predict(X_eval))


def ridge_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
) -> np.ndarray:
    return model_predict(lambda: Ridge(alpha=0.1), X_train, y_train, X_eval)


def ridge_rmse(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> float:
    return rmse(y_eval, ridge_predict(X_train, y_train, X_eval))


def residual_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
) -> np.ndarray:
    residuals = y_train[:, None] - X_train
    mean_residuals = residuals.mean(axis=0)
    return (X_eval + mean_residuals).mean(axis=1)


def residual_rmse(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> float:
    return rmse(y_eval, residual_predict(X_train, y_train, X_eval))
