from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.metrics import mean_squared_error


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def model_predict(
    model_factory: Callable[[], Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    model = model_factory()
    model.fit(X_train, y_train)
    return np.asarray(model.predict(X_test))


def residual_correction_predict(
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    X_eval: np.ndarray,
) -> np.ndarray:
    residuals = y_holdout[:, None] - X_holdout
    mean_residuals = residuals.mean(axis=0)
    return (X_eval + mean_residuals).mean(axis=1)
