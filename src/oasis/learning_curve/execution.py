from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from statistics import NormalDist
from time import perf_counter
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
_TIMED_SWEEP_RESULT_COLUMNS = [
    "n_train",
    "rmse_mean",
    "rmse_std",
    "fit_time_mean_s",
    "fit_time_std_s",
]
_UQ_SUMMARY_COLUMNS = [
    "n_train",
    "miscalibration_area",
    "miscalibration_area_std",
    "sharpness",
    "sharpness_std",
    "dispersion",
    "dispersion_std",
    "uncertainty_kind",
]
_DEFAULT_CALIBRATION_COVERAGES = np.linspace(0.1, 0.9, 9, dtype=float)
_STANDARD_NORMAL = NormalDist()
_TRAPEZOID = getattr(np, "trapezoid", np.trapz)


@dataclass(frozen=True, slots=True)
class SplitPredictionArtifact:
    sweep_size: int
    y_true: np.ndarray
    y_pred: np.ndarray
    spread: np.ndarray

    def __post_init__(self) -> None:
        y_true = np.asarray(self.y_true, dtype=float).reshape(-1)
        y_pred = np.asarray(self.y_pred, dtype=float).reshape(-1)
        spread = np.asarray(self.spread, dtype=float).reshape(-1)
        if len(y_true) != len(y_pred) or len(y_true) != len(spread):
            raise ValueError(
                "y_true, y_pred, and spread must have the same number of rows."
            )
        if np.any(spread < 0.0):
            raise ValueError("spread must be nonnegative.")
        object.__setattr__(self, "y_true", y_true)
        object.__setattr__(self, "y_pred", y_pred)
        object.__setattr__(self, "spread", spread)


def require_min_mlip_feature_count(
    mlip_features: np.ndarray,
    *,
    min_features: int,
    method_name: str,
) -> int:
    feature_count = int(mlip_features.shape[1]) if mlip_features.ndim >= 2 else 1
    if feature_count < min_features:
        raise ValueError(
            f"{method_name} requires at least {min_features} MLIP feature columns; "
            f"got {feature_count}."
        )
    return feature_count


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


def _measure_duration_s(work: Callable[[], object]) -> float:
    t0 = perf_counter()
    work()
    return float(perf_counter() - t0)


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


def timed_sweep_results_frame(
    rmses_by_size: dict[int, list[float]],
    fit_times_by_size: dict[int, list[float]],
) -> pd.DataFrame:
    rows = []
    for n_train, rmses in sorted(rmses_by_size.items()):
        fit_times = np.asarray(fit_times_by_size.get(n_train, ()), dtype=float)
        rows.append(
            {
                "n_train": n_train,
                "rmse_mean": float(np.mean(rmses)),
                "rmse_std": float(np.std(rmses)),
                "fit_time_mean_s": (
                    0.0 if fit_times.size == 0 else float(np.mean(fit_times))
                ),
                "fit_time_std_s": (
                    0.0 if fit_times.size == 0 else float(np.std(fit_times))
                ),
            }
        )
    return pd.DataFrame(rows, columns=_TIMED_SWEEP_RESULT_COLUMNS)


def calibration_curve_frame(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    spread: np.ndarray,
    *,
    nominal_coverages: np.ndarray | None = None,
) -> pd.DataFrame:
    resolved_coverages = _resolve_nominal_coverages(nominal_coverages)
    abs_errors = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
    resolved_spread = np.asarray(spread, dtype=float).reshape(-1)
    if len(abs_errors) != len(resolved_spread):
        raise ValueError("y_true, y_pred, and spread must have the same number of rows.")
    z_scores = np.asarray(
        [
            _STANDARD_NORMAL.inv_cdf((1.0 + float(coverage)) / 2.0)
            for coverage in resolved_coverages
        ],
        dtype=float,
    )
    observed_coverages = np.asarray(
        [
            float(np.mean(abs_errors <= z_score * resolved_spread))
            for z_score in z_scores
        ],
        dtype=float,
    )
    absolute_gap = np.abs(observed_coverages - resolved_coverages)
    return pd.DataFrame(
        {
            "nominal_coverage": resolved_coverages,
            "observed_coverage": observed_coverages,
            "absolute_gap": absolute_gap,
        }
    )


def miscalibration_area(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    spread: np.ndarray,
    *,
    nominal_coverages: np.ndarray | None = None,
) -> float:
    curve = calibration_curve_frame(
        y_true,
        y_pred,
        spread,
        nominal_coverages=nominal_coverages,
    )
    if len(curve) == 1:
        return float(curve["absolute_gap"].iloc[0])
    return float(
        _TRAPEZOID(
            curve["absolute_gap"].to_numpy(),
            curve["nominal_coverage"].to_numpy(),
        )
    )


def sharpness_from_spread(spread: np.ndarray) -> float:
    resolved_spread = np.abs(np.asarray(spread, dtype=float).reshape(-1))
    if len(resolved_spread) == 0:
        return 0.0
    return float(np.mean(resolved_spread))


def dispersion_from_spread(spread: np.ndarray) -> float:
    resolved_spread = np.abs(np.asarray(spread, dtype=float).reshape(-1))
    if len(resolved_spread) == 0:
        return 0.0
    mean_spread = float(np.mean(resolved_spread))
    if mean_spread <= 0.0:
        return 0.0
    return float(np.std(resolved_spread) / mean_spread)


def aggregate_uq_summary(
    split_artifacts: list[SplitPredictionArtifact] | tuple[SplitPredictionArtifact, ...],
    *,
    uncertainty_kind: str,
    nominal_coverages: np.ndarray | None = None,
) -> pd.DataFrame:
    grouped = collect_split_prediction_artifacts(split_artifacts)
    rows: list[dict[str, float | int | str]] = []
    for sweep_size, artifacts in sorted(grouped.items()):
        miscalibration_values = np.asarray(
            [
                miscalibration_area(
                    artifact.y_true,
                    artifact.y_pred,
                    artifact.spread,
                    nominal_coverages=nominal_coverages,
                )
                for artifact in artifacts
            ],
            dtype=float,
        )
        sharpness_values = np.asarray(
            [sharpness_from_spread(artifact.spread) for artifact in artifacts],
            dtype=float,
        )
        dispersion_values = np.asarray(
            [dispersion_from_spread(artifact.spread) for artifact in artifacts],
            dtype=float,
        )
        rows.append(
            {
                "n_train": sweep_size,
                "miscalibration_area": float(np.mean(miscalibration_values)),
                "miscalibration_area_std": float(np.std(miscalibration_values)),
                "sharpness": float(np.mean(sharpness_values)),
                "sharpness_std": float(np.std(sharpness_values)),
                "dispersion": float(np.mean(dispersion_values)),
                "dispersion_std": float(np.std(dispersion_values)),
                "uncertainty_kind": uncertainty_kind,
            }
        )
    return pd.DataFrame(rows, columns=_UQ_SUMMARY_COLUMNS)


def collect_split_prediction_artifacts(
    split_artifacts: list[SplitPredictionArtifact] | tuple[SplitPredictionArtifact, ...],
) -> dict[int, list[SplitPredictionArtifact]]:
    grouped: dict[int, list[SplitPredictionArtifact]] = {}
    for artifact in split_artifacts:
        grouped.setdefault(int(artifact.sweep_size), []).append(artifact)
    return grouped


def build_split_prediction_artifact(
    *,
    sweep_size: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    spread: np.ndarray,
) -> SplitPredictionArtifact:
    return SplitPredictionArtifact(
        sweep_size=int(sweep_size),
        y_true=y_true,
        y_pred=y_pred,
        spread=spread,
    )


def _resolve_nominal_coverages(
    nominal_coverages: np.ndarray | None,
) -> np.ndarray:
    resolved_coverages = (
        _DEFAULT_CALIBRATION_COVERAGES
        if nominal_coverages is None
        else np.asarray(nominal_coverages, dtype=float).reshape(-1)
    )
    if len(resolved_coverages) == 0:
        raise ValueError("nominal_coverages must contain at least one value.")
    if np.any((resolved_coverages <= 0.0) | (resolved_coverages >= 1.0)):
        raise ValueError("nominal_coverages must lie strictly between 0 and 1.")
    return resolved_coverages


def sweep_model(
    payload: SweepRunPayload | SweepRunnerPayload,
    model_factory: Callable[[], object],
) -> pd.DataFrame:
    """Evaluate a supervised model across precomputed sweep splits."""

    payload = _as_runner_payload(payload)
    rmses_by_size: dict[int, list[float]] = {}
    fit_times_by_size: dict[int, list[float]] = {}
    for split in _assert_train_test_payload(payload):
        X = split.dataset.mlip_features
        y = split.dataset.targets
        model = model_factory()
        fit_time_s = _measure_duration_s(
            lambda: _fit_model_safely(model, X[split.train_idx], y[split.train_idx])
        )
        X_test = X[split.test_idx]
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y[split.test_idx], preds))
        rmses_by_size.setdefault(split.sweep_size, []).append(rmse)
        fit_times_by_size.setdefault(split.sweep_size, []).append(fit_time_s)
    return timed_sweep_results_frame(rmses_by_size, fit_times_by_size)


def sweep_model_with_validation(
    payload: SweepRunPayload | SweepRunnerPayload,
    model_factory: Callable[[], ValidationAwareEstimator],
) -> pd.DataFrame:
    """Evaluate a validation-aware estimator across precomputed sweep splits."""

    payload = _as_runner_payload(payload)
    splits = _assert_train_val_test_payload(payload)
    rmses_by_size: dict[int, list[float]] = {}
    fit_times_by_size: dict[int, list[float]] = {}
    for split in splits:
        X = split.dataset.mlip_features
        y = split.dataset.targets
        model = model_factory()
        fit_time_s = _measure_duration_s(
            lambda: model.fit(
                X[split.train_idx],
                y[split.train_idx],
                X[split.val_idx],
                y[split.val_idx],
            )
        )
        preds = model.predict(X[split.test_idx])
        rmse = np.sqrt(mean_squared_error(y[split.test_idx], preds))
        rmses_by_size.setdefault(split.sweep_size, []).append(rmse)
        fit_times_by_size.setdefault(split.sweep_size, []).append(fit_time_s)
    return timed_sweep_results_frame(rmses_by_size, fit_times_by_size)


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
) -> SweepRunnerArtifacts:
    """Evaluate residual correction across precomputed sweep splits."""

    payload = _as_runner_payload(payload)
    rmses_by_size: dict[int, list[float]] = {}
    fit_times_by_size: dict[int, list[float]] = {}
    split_artifacts: list[SplitPredictionArtifact] = []
    for split in _assert_train_test_payload(payload):
        X = split.dataset.mlip_features
        y = split.dataset.targets
        X_train = X[split.train_idx]
        y_train = y[split.train_idx]
        mean_residuals = None

        def fit_residual_model() -> None:
            nonlocal mean_residuals
            residuals = y_train[:, None] - X_train
            mean_residuals = residuals.mean(axis=0)

        fit_time_s = _measure_duration_s(fit_residual_model)
        assert mean_residuals is not None

        X_corrected = X[split.test_idx] + mean_residuals
        preds = X_corrected.mean(axis=1)
        spread = np.std(X_corrected, axis=1)
        y_true = y[split.test_idx]
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        rmses_by_size.setdefault(split.sweep_size, []).append(rmse)
        fit_times_by_size.setdefault(split.sweep_size, []).append(fit_time_s)
        split_artifacts.append(
            build_split_prediction_artifact(
                sweep_size=split.sweep_size,
                y_true=y_true,
                y_pred=preds,
                spread=spread,
            )
        )
    return SweepRunnerArtifacts(
        metrics=timed_sweep_results_frame(rmses_by_size, fit_times_by_size),
        uq_summary=aggregate_uq_summary(
            split_artifacts,
            uncertainty_kind="spread_only",
        ),
    )


def weighted_linear_sweep(
    payload: SweepRunPayload | SweepRunnerPayload,
    *,
    fit_intercept: bool = True,
) -> SweepRunnerArtifacts:
    """Fit an unconstrained linear combiner over MLIP columns on each sweep split."""

    payload = _as_runner_payload(payload)
    rmses_by_size: dict[int, list[float]] = {}
    fit_times_by_size: dict[int, list[float]] = {}
    split_artifacts: list[SplitPredictionArtifact] = []
    for split in _assert_train_test_payload(payload):
        X = split.dataset.mlip_features
        y = split.dataset.targets
        model = LinearRegression(fit_intercept=fit_intercept)
        fit_time_s = _measure_duration_s(
            lambda: _fit_model_safely(model, X[split.train_idx], y[split.train_idx])
        )
        X_test = X[split.test_idx]
        preds = model.predict(X_test)
        centered = X_test - preds[:, None]
        coef_weights = np.abs(np.asarray(model.coef_, dtype=float).reshape(-1))
        weight_sum = float(coef_weights.sum())
        if weight_sum <= 0.0:
            normalized_weights = np.full(X_test.shape[1], 1.0 / X_test.shape[1], dtype=float)
        else:
            normalized_weights = coef_weights / weight_sum
        spread = np.sqrt(np.sum(normalized_weights[None, :] * centered**2, axis=1))
        y_true = y[split.test_idx]
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        rmses_by_size.setdefault(split.sweep_size, []).append(rmse)
        fit_times_by_size.setdefault(split.sweep_size, []).append(fit_time_s)
        split_artifacts.append(
            build_split_prediction_artifact(
                sweep_size=split.sweep_size,
                y_true=y_true,
                y_pred=preds,
                spread=spread,
            )
        )
    return SweepRunnerArtifacts(
        metrics=timed_sweep_results_frame(rmses_by_size, fit_times_by_size),
        uq_summary=aggregate_uq_summary(
            split_artifacts,
            uncertainty_kind="spread_only",
        ),
    )


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
) -> SweepRunnerArtifacts:
    """Fit a simplex-style combiner with nonnegative weights that sum to one."""

    payload = _as_runner_payload(payload)
    rmses_by_size: dict[int, list[float]] = {}
    fit_times_by_size: dict[int, list[float]] = {}
    split_artifacts: list[SplitPredictionArtifact] = []
    for split in _assert_train_test_payload(payload):
        X = split.dataset.mlip_features
        y = split.dataset.targets
        require_min_mlip_feature_count(
            X,
            min_features=2,
            method_name="weighted_simplex",
        )
        weights = None

        def fit_simplex_model() -> None:
            nonlocal weights
            weights = _simplex_weights(
                X[split.train_idx],
                y[split.train_idx],
            )

        fit_time_s = _measure_duration_s(fit_simplex_model)
        assert weights is not None
        X_test = X[split.test_idx]
        preds = X_test @ weights
        centered = X_test - preds[:, None]
        spread = np.sqrt(np.sum(weights[None, :] * centered**2, axis=1))
        y_true = y[split.test_idx]
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        rmses_by_size.setdefault(split.sweep_size, []).append(rmse)
        fit_times_by_size.setdefault(split.sweep_size, []).append(fit_time_s)
        split_artifacts.append(
            build_split_prediction_artifact(
                sweep_size=split.sweep_size,
                y_true=y_true,
                y_pred=preds,
                spread=spread,
            )
        )
    return SweepRunnerArtifacts(
        metrics=timed_sweep_results_frame(rmses_by_size, fit_times_by_size),
        uq_summary=aggregate_uq_summary(
            split_artifacts,
            uncertainty_kind="spread_only",
        ),
    )
