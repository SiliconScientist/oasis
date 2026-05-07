from __future__ import annotations

from collections.abc import Collection
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd
from oasis.exp import LearningCurveResults, SweepRunPayload
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


_SWEEP_RESULT_COLUMNS = ["n_train", "rmse_mean", "rmse_std"]


@dataclass(frozen=True, slots=True)
class SklearnSweepModelSpec:
    result_field: str
    trimmed_result_field: str | None
    model_factory: Callable[[], object]
    trim_z_thresh: float = 1.0


@dataclass(frozen=True, slots=True)
class LearningCurveModelRegistration:
    name: str
    config_attr: str
    family_factory: Callable[[], SweepModelFamily]


class SweepModelFamily(Protocol):
    """Contract for a model family runnable over a shared split sweep."""

    def run(self, payload: SweepRunPayload) -> LearningCurveResults: ...


class SweepExperimentRunner(Protocol):
    """Common runner interface for sweep methods with optional trimmed output."""

    def run(self, payload: SweepRunPayload) -> pd.DataFrame: ...

    def run_trimmed(
        self,
        payload: SweepRunPayload,
        *,
        z_thresh: float = 1.0,
    ) -> pd.DataFrame: ...


@dataclass(frozen=True, slots=True)
class SweepFamilySpec:
    result_field: str
    trimmed_result_field: str | None
    runner: SweepExperimentRunner
    trim_z_thresh: float = 1.0


@dataclass(frozen=True, slots=True)
class SupervisedModelSweepRunner:
    """Reusable adapter for supervised estimators over the shared sweep payload."""

    model_factory: Callable[[], object]

    def run(self, payload: SweepRunPayload) -> pd.DataFrame:
        return sweep_model(payload, self.model_factory)

    def run_trimmed(
        self,
        payload: SweepRunPayload,
        *,
        z_thresh: float = 1.0,
    ) -> pd.DataFrame:
        return sweep_model_trimmed(
            payload,
            self.model_factory,
            z_thresh=z_thresh,
        )


@dataclass(frozen=True, slots=True)
class FunctionalSweepRunner:
    base_runner: Callable[[SweepRunPayload], pd.DataFrame]
    trimmed_runner: Callable[[SweepRunPayload], pd.DataFrame]

    def run(self, payload: SweepRunPayload) -> pd.DataFrame:
        return self.base_runner(payload)

    def run_trimmed(
        self,
        payload: SweepRunPayload,
        *,
        z_thresh: float = 1.0,
    ) -> pd.DataFrame:
        del z_thresh
        return self.trimmed_runner(payload)


@dataclass(frozen=True, slots=True)
class ConfiguredSweepModelFamily:
    spec: SweepFamilySpec

    def run(self, payload: SweepRunPayload) -> LearningCurveResults:
        results: dict[str, pd.DataFrame | None] = {
            self.spec.result_field: self.spec.runner.run(payload),
        }
        if payload.use_trim and self.spec.trimmed_result_field is not None:
            results[self.spec.trimmed_result_field] = self.spec.runner.run_trimmed(
                payload,
                z_thresh=self.spec.trim_z_thresh,
            )
        return LearningCurveResults.from_mapping(results)


def default_sweep_model_families(
    enabled_model_names: Collection[str] | None = None,
) -> tuple[SweepModelFamily, ...]:
    registrations = learning_curve_model_registry()
    enabled_names = (
        set(registration.name for registration in registrations)
        if enabled_model_names is None
        else set(enabled_model_names)
    )
    return tuple(
        registration.family_factory()
        for registration in registrations
        if registration.name in enabled_names
    )


def enabled_learning_curve_model_names_from_config(
    plot_cfg: Any | None,
) -> tuple[str, ...]:
    registrations = learning_curve_model_registry()
    if plot_cfg is None:
        return tuple(registration.name for registration in registrations)
    return tuple(
        registration.name
        for registration in registrations
        if getattr(plot_cfg, registration.config_attr)
    )


def learning_curve_model_registry() -> tuple[LearningCurveModelRegistration, ...]:
    sklearn_registrations = tuple(
        LearningCurveModelRegistration(
            name=name,
            config_attr=config_attr,
            family_factory=lambda spec=spec: ConfiguredSweepModelFamily(
                SweepFamilySpec(
                    result_field=spec.result_field,
                    trimmed_result_field=spec.trimmed_result_field,
                    runner=SupervisedModelSweepRunner(spec.model_factory),
                    trim_z_thresh=spec.trim_z_thresh,
                )
            ),
        )
        for name, config_attr, spec in sklearn_sweep_model_specs()
    )
    return sklearn_registrations + (
        LearningCurveModelRegistration(
            name="residual",
            config_attr="use_residual",
            family_factory=lambda: ConfiguredSweepModelFamily(
                SweepFamilySpec(
                    result_field="resid_df",
                    trimmed_result_field="resid_trimmed_df",
                    runner=FunctionalSweepRunner(
                        base_runner=residual_sweep,
                        trimmed_runner=residual_sweep_trimmed,
                    ),
                )
            ),
        ),
        LearningCurveModelRegistration(
            name="linearization",
            config_attr="use_linearization",
            family_factory=lambda: ConfiguredSweepModelFamily(
                SweepFamilySpec(
                    result_field="linear_df",
                    trimmed_result_field="linear_trimmed_df",
                    runner=FunctionalSweepRunner(
                        base_runner=linearization_sweep,
                        trimmed_runner=linearization_sweep_trimmed,
                    ),
                )
            ),
        ),
    )


def sklearn_sweep_model_specs(
) -> tuple[tuple[str, str, SklearnSweepModelSpec], ...]:
    return (
        (
            "ridge",
            "use_ridge",
            SklearnSweepModelSpec(
                result_field="ridge_df",
                trimmed_result_field="ridge_trimmed_df",
                model_factory=lambda: Ridge(alpha=0.1),
            ),
        ),
        (
            "kernel_ridge",
            "use_kernel_ridge",
            SklearnSweepModelSpec(
                result_field="kernel_ridge_df",
                trimmed_result_field=None,
                model_factory=lambda: KernelRidge(alpha=1.0, kernel="rbf"),
            ),
        ),
        (
            "lasso",
            "use_lasso",
            SklearnSweepModelSpec(
                result_field="lasso_df",
                trimmed_result_field="lasso_trimmed_df",
                model_factory=lambda: Lasso(alpha=0.1, max_iter=10000),
            ),
        ),
        (
            "elastic",
            "use_elastic_net",
            SklearnSweepModelSpec(
                result_field="elastic_df",
                trimmed_result_field="elastic_trimmed_df",
                model_factory=lambda: ElasticNet(
                    alpha=0.1,
                    l1_ratio=0.5,
                    max_iter=20000,
                ),
            ),
        ),
    )


def sklearn_model_families(
    specs: tuple[tuple[str, str, SklearnSweepModelSpec], ...],
) -> tuple[SweepModelFamily, ...]:
    return tuple(
        ConfiguredSweepModelFamily(
            SweepFamilySpec(
                result_field=spec.result_field,
                trimmed_result_field=spec.trimmed_result_field,
                runner=SupervisedModelSweepRunner(spec.model_factory),
                trim_z_thresh=spec.trim_z_thresh,
            )
        )
        for _, _, spec in specs
    )


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
    payload: SweepRunPayload,
    model_factory,
) -> pd.DataFrame:
    """Evaluate a supervised model across precomputed sweep splits."""

    X = payload.dataset.X
    y = payload.dataset.y
    rmses_by_size: dict[int, list[float]] = {}
    for split in payload.split_collection.splits:
        model = model_factory()
        model.fit(X[split.train_idx], y[split.train_idx])
        X_test = X[split.test_idx]
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y[split.test_idx], preds))
        rmses_by_size.setdefault(split.sweep_size, []).append(rmse)
    return sweep_results_frame(rmses_by_size)


def sweep_model_trimmed(
    payload: SweepRunPayload,
    model_factory,
    z_thresh: float = 1.0,
) -> pd.DataFrame:
    """
    Fit the model, drop test samples with large contribution z-scores, and refit.
    """
    X = payload.dataset.X
    y = payload.dataset.y
    rmses_by_size: dict[int, list[float]] = {}
    for split in payload.split_collection.splits:
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
    payload: SweepRunPayload,
) -> pd.DataFrame:
    """Evaluate residual correction across precomputed sweep splits."""

    X = payload.dataset.X
    y = payload.dataset.y
    rmses_by_size: dict[int, list[float]] = {}
    for split in payload.split_collection.splits:
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
    payload: SweepRunPayload,
) -> pd.DataFrame:
    """
    Residual correction with per-sample outlier MLIP removal before averaging.
    """
    X = payload.dataset.X
    y = payload.dataset.y
    rmses_by_size: dict[int, list[float]] = {}
    for split in payload.split_collection.splits:
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
    payload: SweepRunPayload,
) -> pd.DataFrame:
    """Evaluate linearization correction across precomputed sweep splits."""

    X = payload.dataset.X
    y = payload.dataset.y
    rmses_by_size: dict[int, list[float]] = {}
    for split in payload.split_collection.splits:
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
    payload: SweepRunPayload,
) -> pd.DataFrame:
    """
    Linearize against trimmed train means, then trim ensemble averaging.
    """
    X = payload.dataset.X
    y = payload.dataset.y
    rmses_by_size: dict[int, list[float]] = {}
    for split in payload.split_collection.splits:
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
