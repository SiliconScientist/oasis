from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING
import warnings

import numpy as np
import pandas as pd
from oasis.sweep import (
    SweepRunnerPayload,
    SweepRunPayload,
    TrainTestSweepRunnerInput,
    TrainValTestSweepRunnerInput,
)
from oasis.tune import SweepRunnerArtifacts, ValidationAwareEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

try:
    from sklearn.exceptions import ConvergenceWarning
except ImportError:
    ConvergenceWarning = Warning

if TYPE_CHECKING:
    from oasis.learning_curve.runners import (
        SweepExperimentRunner,
        TrainTestLearnedEstimator,
        TrainValTestLearnedEstimator,
        ValidationAwareSweepExperimentRunner,
    )


_SWEEP_RESULT_COLUMNS = ["n_train", "rmse_mean", "rmse_std"]


def _fit_model_safely(model: object, X: np.ndarray, y: np.ndarray) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(X, y)


def _as_runner_payload(
    payload: SweepRunPayload | SweepRunnerPayload,
) -> SweepRunnerPayload:
    if isinstance(payload, SweepRunnerPayload):
        return payload
    return payload.to_runner_payload()


def _payload_uses_validation(payload: SweepRunnerPayload) -> bool:
    if payload.planning_requirements.requires_validation:
        return True
    return any(
        isinstance(split, TrainValTestSweepRunnerInput) for split in payload.splits
    )


def _assert_train_test_payload(
    payload: SweepRunnerPayload,
) -> tuple[TrainTestSweepRunnerInput, ...]:
    splits = payload.splits
    if not all(isinstance(split, TrainTestSweepRunnerInput) for split in splits):
        raise TypeError("expected train/test sweep payload")
    return splits


def _assert_train_val_test_payload(
    payload: SweepRunnerPayload,
) -> tuple[TrainValTestSweepRunnerInput, ...]:
    splits = payload.splits
    if not all(isinstance(split, TrainValTestSweepRunnerInput) for split in splits):
        raise TypeError("expected train/val/test sweep payload")
    return splits


def _select_runner_call(
    runner: SweepExperimentRunner | ValidationAwareSweepExperimentRunner,
    payload: SweepRunnerPayload,
) -> Callable[..., pd.DataFrame | SweepRunnerArtifacts]:
    from oasis.learning_curve.runners import (
        SweepExperimentRunner,
        ValidationAwareSweepExperimentRunner,
    )

    uses_validation = _payload_uses_validation(payload)
    if uses_validation:
        if not isinstance(runner, ValidationAwareSweepExperimentRunner):
            raise TypeError("runner does not support validation-aware sweep payloads")
        return runner.run_with_validation
    if not isinstance(runner, SweepExperimentRunner):
        raise TypeError("runner does not support train/test sweep payloads")
    return runner.run


def _normalize_runner_output(
    output: pd.DataFrame | SweepRunnerArtifacts,
) -> SweepRunnerArtifacts:
    if isinstance(output, SweepRunnerArtifacts):
        return output
    return SweepRunnerArtifacts(metrics=output)


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
    payload: SweepRunPayload | SweepRunnerPayload,
    model_factory: Callable[[], object],
) -> pd.DataFrame:
    """Evaluate a supervised model across precomputed sweep splits."""

    payload = _as_runner_payload(payload)
    rmses_by_size: dict[int, list[float]] = {}
    for split in _assert_train_test_payload(payload):
        X = split.dataset.mlip_features
        y = split.dataset.targets
        model = model_factory()
        _fit_model_safely(model, X[split.train_idx], y[split.train_idx])
        X_test = X[split.test_idx]
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y[split.test_idx], preds))
        rmses_by_size.setdefault(split.sweep_size, []).append(rmse)
    return sweep_results_frame(rmses_by_size)


def sweep_model_with_validation(
    payload: SweepRunPayload | SweepRunnerPayload,
    model_factory: Callable[[], ValidationAwareEstimator],
) -> pd.DataFrame:
    """Evaluate a validation-aware estimator across precomputed sweep splits."""

    payload = _as_runner_payload(payload)
    splits = _assert_train_val_test_payload(payload)
    rmses_by_size: dict[int, list[float]] = {}
    for split in splits:
        X = split.dataset.mlip_features
        y = split.dataset.targets
        model = model_factory()
        model.fit(
            X[split.train_idx],
            y[split.train_idx],
            X[split.val_idx],
            y[split.val_idx],
        )
        preds = model.predict(X[split.test_idx])
        rmse = np.sqrt(mean_squared_error(y[split.test_idx], preds))
        rmses_by_size.setdefault(split.sweep_size, []).append(rmse)
    return sweep_results_frame(rmses_by_size)


def sweep_learned_model(
    payload: SweepRunPayload | SweepRunnerPayload,
    model_factory: Callable[[], TrainTestLearnedEstimator],
) -> pd.DataFrame:
    """Evaluate a learned estimator that consumes full train/test split inputs."""

    payload = _as_runner_payload(payload)
    rmses_by_size: dict[int, list[float]] = {}
    for split in _assert_train_test_payload(payload):
        X = split.dataset.mlip_features
        y = split.dataset.targets
        model = model_factory()
        model.fit(split)
        preds = model.predict(X[split.test_idx])
        rmse = np.sqrt(mean_squared_error(y[split.test_idx], preds))
        rmses_by_size.setdefault(split.sweep_size, []).append(rmse)
    return sweep_results_frame(rmses_by_size)


def sweep_learned_model_with_validation(
    payload: SweepRunPayload | SweepRunnerPayload,
    model_factory: Callable[[], TrainValTestLearnedEstimator],
) -> pd.DataFrame:
    """Evaluate a learned estimator that consumes full train/val/test split inputs."""

    payload = _as_runner_payload(payload)
    rmses_by_size: dict[int, list[float]] = {}
    for split in _assert_train_val_test_payload(payload):
        X = split.dataset.mlip_features
        y = split.dataset.targets
        model = model_factory()
        model.fit(split)
        preds = model.predict(X[split.test_idx])
        rmse = np.sqrt(mean_squared_error(y[split.test_idx], preds))
        rmses_by_size.setdefault(split.sweep_size, []).append(rmse)
    return sweep_results_frame(rmses_by_size)


def residual_sweep(
    payload: SweepRunPayload | SweepRunnerPayload,
) -> pd.DataFrame:
    """Evaluate residual correction across precomputed sweep splits."""

    payload = _as_runner_payload(payload)
    rmses_by_size: dict[int, list[float]] = {}
    for split in _assert_train_test_payload(payload):
        X = split.dataset.mlip_features
        y = split.dataset.targets
        X_train = X[split.train_idx]
        y_train = y[split.train_idx]

        residuals = y_train[:, None] - X_train
        mean_residuals = residuals.mean(axis=0)

        X_corrected = X[split.test_idx] + mean_residuals
        preds = X_corrected.mean(axis=1)
        rmse = np.sqrt(mean_squared_error(y[split.test_idx], preds))
        rmses_by_size.setdefault(split.sweep_size, []).append(rmse)
    return sweep_results_frame(rmses_by_size)


def weighted_linear_sweep(
    payload: SweepRunPayload | SweepRunnerPayload,
    *,
    fit_intercept: bool = True,
) -> pd.DataFrame:
    """Fit an unconstrained linear combiner over MLIP columns on each sweep split."""

    payload = _as_runner_payload(payload)
    rmses_by_size: dict[int, list[float]] = {}
    for split in _assert_train_test_payload(payload):
        X = split.dataset.mlip_features
        y = split.dataset.targets
        model = LinearRegression(fit_intercept=fit_intercept)
        _fit_model_safely(model, X[split.train_idx], y[split.train_idx])
        preds = model.predict(X[split.test_idx])
        rmse = np.sqrt(mean_squared_error(y[split.test_idx], preds))
        rmses_by_size.setdefault(split.sweep_size, []).append(rmse)
    return sweep_results_frame(rmses_by_size)


def _simplex_weights(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> np.ndarray:
    model = LinearRegression(fit_intercept=False, positive=True)
    _fit_model_safely(model, X_train, y_train)
    weights = np.clip(np.asarray(model.coef_, dtype=float), 0.0, None)
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        return np.full(X_train.shape[1], 1.0 / X_train.shape[1], dtype=float)
    return weights / weight_sum


def weighted_simplex_sweep(
    payload: SweepRunPayload | SweepRunnerPayload,
) -> pd.DataFrame:
    """Fit a simplex-style combiner with nonnegative weights that sum to one."""

    payload = _as_runner_payload(payload)
    rmses_by_size: dict[int, list[float]] = {}
    for split in _assert_train_test_payload(payload):
        X = split.dataset.mlip_features
        y = split.dataset.targets
        weights = _simplex_weights(
            X[split.train_idx],
            y[split.train_idx],
        )
        preds = X[split.test_idx] @ weights
        rmse = np.sqrt(mean_squared_error(y[split.test_idx], preds))
        rmses_by_size.setdefault(split.sweep_size, []).append(rmse)
    return sweep_results_frame(rmses_by_size)
