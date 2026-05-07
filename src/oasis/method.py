from __future__ import annotations

from collections.abc import Collection
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Protocol

import numpy as np
import pandas as pd
from oasis.exp import (
    LearningCurveResults,
    SweepFamilyRequirements,
    SweepRunPayload,
)
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

    def requirements(self) -> SweepFamilyRequirements: ...

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
    requirements: SweepFamilyRequirements = field(
        default_factory=SweepFamilyRequirements
    )


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
class WeightedCombinerSweepRunner:
    fit_intercept: bool = True

    def run(self, payload: SweepRunPayload) -> pd.DataFrame:
        return weighted_combiner_sweep(
            payload,
            fit_intercept=self.fit_intercept,
        )

    def run_trimmed(
        self,
        payload: SweepRunPayload,
        *,
        z_thresh: float = 1.0,
    ) -> pd.DataFrame:
        del z_thresh
        return self.run(payload)


@dataclass(frozen=True, slots=True)
class NormalizedWeightedCombinerSweepRunner:
    def run(self, payload: SweepRunPayload) -> pd.DataFrame:
        return normalized_weighted_combiner_sweep(payload)

    def run_trimmed(
        self,
        payload: SweepRunPayload,
        *,
        z_thresh: float = 1.0,
    ) -> pd.DataFrame:
        del z_thresh
        return self.run(payload)


@dataclass(frozen=True, slots=True)
class ConfiguredSweepModelFamily:
    spec: SweepFamilySpec

    def requirements(self) -> SweepFamilyRequirements:
        return self.spec.requirements

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
            name="weighted_combiner",
            config_attr="use_weighted_combiner",
            family_factory=lambda: ConfiguredSweepModelFamily(
                SweepFamilySpec(
                    result_field="weighted_combiner_df",
                    trimmed_result_field=None,
                    runner=WeightedCombinerSweepRunner(fit_intercept=True),
                )
            ),
        ),
        LearningCurveModelRegistration(
            name="normalized_weighted_combiner",
            config_attr="use_normalized_weighted_combiner",
            family_factory=lambda: ConfiguredSweepModelFamily(
                SweepFamilySpec(
                    result_field="normalized_weighted_combiner_df",
                    trimmed_result_field=None,
                    runner=NormalizedWeightedCombinerSweepRunner(),
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


def weighted_combiner_sweep(
    payload: SweepRunPayload,
    *,
    fit_intercept: bool = True,
) -> pd.DataFrame:
    """Fit unconstrained linear weights over MLIP columns for each sweep split."""

    X = payload.dataset.X
    y = payload.dataset.y
    rmses_by_size: dict[int, list[float]] = {}
    for split in payload.split_collection.splits:
        model = LinearRegression(fit_intercept=fit_intercept)
        model.fit(X[split.train_idx], y[split.train_idx])
        preds = model.predict(X[split.test_idx])
        rmse = np.sqrt(mean_squared_error(y[split.test_idx], preds))
        rmses_by_size.setdefault(split.sweep_size, []).append(rmse)
    return sweep_results_frame(rmses_by_size)


def _normalized_nonnegative_weights(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> np.ndarray:
    model = LinearRegression(fit_intercept=False, positive=True)
    model.fit(X_train, y_train)
    weights = np.clip(np.asarray(model.coef_, dtype=float), 0.0, None)
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        return np.full(X_train.shape[1], 1.0 / X_train.shape[1], dtype=float)
    return weights / weight_sum


def normalized_weighted_combiner_sweep(
    payload: SweepRunPayload,
) -> pd.DataFrame:
    """
    Fit nonnegative weights and renormalize them to sum to one on each sweep split.
    """

    X = payload.dataset.X
    y = payload.dataset.y
    rmses_by_size: dict[int, list[float]] = {}
    for split in payload.split_collection.splits:
        weights = _normalized_nonnegative_weights(
            X[split.train_idx],
            y[split.train_idx],
        )
        preds = X[split.test_idx] @ weights
        rmse = np.sqrt(mean_squared_error(y[split.test_idx], preds))
        rmses_by_size.setdefault(split.sweep_size, []).append(rmse)
    return sweep_results_frame(rmses_by_size)
