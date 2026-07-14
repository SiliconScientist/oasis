from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from oasis.sweep import TrainValTestSweepRunnerInput
from oasis.tune import (
    GridHyperparameterSpec,
    HyperparameterSpec,
    SelectionRefitPolicy,
    TrialTuningSpec,
)

try:
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.linear_model import ElasticNet, Lasso, Ridge
    from sklearn.metrics.pairwise import pairwise_kernels
except ImportError:
    KernelRidge = ElasticNet = Lasso = Ridge = pairwise_kernels = None  # type: ignore[assignment,misc]


@dataclass(frozen=True, slots=True)
class SklearnSweepModelSpec:
    """Registered supervised model family configuration."""

    result_field: str
    model_factory: Callable[[], object]
    hyperparameter_spec: HyperparameterSpec | None = None
    trial_tuning_spec: TrialTuningSpec | None = None
    optuna_n_trials: int | None = None
    optuna_timeout_s: int | None = None
    optuna_study_factory: Callable[[TrainValTestSweepRunnerInput], Any] | None = None
    selection_metadata_field: str | None = None
    uq_summary_field: str | None = None
    selection_refit_policy: SelectionRefitPolicy = "train_plus_val"
    requires_calibration: bool = False


def _linear_model_spread(model: object, X: np.ndarray) -> np.ndarray:
    X_arr = np.asarray(X, dtype=float)
    preds = np.asarray(model.predict(X_arr), dtype=float).reshape(-1)
    coefs = np.ravel(np.asarray(getattr(model, "coef_", np.ones(X_arr.shape[1])), dtype=float))
    if len(coefs) != X_arr.shape[1]:
        coefs = np.ones(X_arr.shape[1], dtype=float)
    weights = np.abs(coefs)
    weight_sum = float(weights.sum())
    normalized = (
        np.full(X_arr.shape[1], 1.0 / X_arr.shape[1], dtype=float)
        if weight_sum <= 0.0
        else weights / weight_sum
    )
    centered = X_arr - preds[:, None]
    return np.sqrt(np.sum(normalized[None, :] * centered**2, axis=1))


def _kernel_ridge_spread(model: object, X: np.ndarray) -> np.ndarray:
    X_arr = np.asarray(X, dtype=float)
    X_fit = np.asarray(getattr(model, "X_fit_"), dtype=float)
    alpha = np.asarray(getattr(model, "alpha", 1.0), dtype=float)
    if alpha.ndim == 0:
        regularizer = np.eye(X_fit.shape[0], dtype=float) * float(alpha)
    elif alpha.shape == (X_fit.shape[0],):
        regularizer = np.diag(alpha.astype(float))
    else:
        raise ValueError("kernel ridge uncertainty proxy requires scalar or per-sample alpha")

    kernel = getattr(model, "kernel", "linear")
    kernel_params = {
        "gamma": getattr(model, "gamma", None),
        "degree": getattr(model, "degree", None),
        "coef0": getattr(model, "coef0", None),
    }
    kernel_params = {
        key: value for key, value in kernel_params.items() if value is not None
    }
    train_kernel = pairwise_kernels(
        X_fit,
        X_fit,
        metric=kernel,
        filter_params=True,
        **kernel_params,
    )
    cross_kernel = pairwise_kernels(
        X_arr,
        X_fit,
        metric=kernel,
        filter_params=True,
        **kernel_params,
    )
    self_kernel = np.diag(
        pairwise_kernels(
            X_arr,
            X_arr,
            metric=kernel,
            filter_params=True,
            **kernel_params,
        )
    )
    solved = np.linalg.solve(train_kernel + regularizer, cross_kernel.T)
    variance = self_kernel - np.sum(cross_kernel * solved.T, axis=1)
    return np.sqrt(np.clip(variance, 0.0, None))


def sklearn_sweep_model_specs() -> tuple[tuple[str, str, SklearnSweepModelSpec], ...]:
    return (
        (
            "ridge",
            "use_ridge",
            SklearnSweepModelSpec(
                result_field="ridge_df",
                model_factory=lambda: Ridge(alpha=0.1),
                hyperparameter_spec=GridHyperparameterSpec(
                    estimator_factory=Ridge,
                    grid={"alpha": (0.01, 0.1, 1.0, 10.0)},
                    predictive_spread_extractor=_linear_model_spread,
                    uncertainty_kind="spread_only",
                    uncertainty_note="spread-only; not probabilistically interpretable",
                ),
                selection_metadata_field="ridge_selection_df",
                uq_summary_field="ridge_uq_df",
                requires_calibration=True,
            ),
        ),
        (
            "kernel_ridge",
            "use_kernel_ridge",
            SklearnSweepModelSpec(
                result_field="kernel_ridge_df",
                model_factory=lambda: KernelRidge(
                    alpha=1.0,
                    kernel="rbf",
                ),
                hyperparameter_spec=GridHyperparameterSpec(
                    estimator_factory=KernelRidge,
                    grid={
                        "alpha": (0.1, 1.0, 10.0),
                        "gamma": (0.1, 1.0),
                    },
                    fixed_params={"kernel": "rbf"},
                    metadata_keys=("alpha", "gamma", "kernel"),
                    predictive_spread_extractor=_kernel_ridge_spread,
                    uncertainty_kind="spread_only",
                    uncertainty_note="spread-only; not probabilistically interpretable",
                ),
                selection_metadata_field="kernel_ridge_selection_df",
                uq_summary_field="kernel_ridge_uq_df",
                requires_calibration=True,
            ),
        ),
        (
            "lasso",
            "use_lasso",
            SklearnSweepModelSpec(
                result_field="lasso_df",
                model_factory=lambda: Lasso(alpha=0.1, max_iter=10000),
                hyperparameter_spec=GridHyperparameterSpec(
                    estimator_factory=Lasso,
                    grid={"alpha": (0.001, 0.01, 0.1, 1.0)},
                    fixed_params={"max_iter": 10000},
                ),
                selection_metadata_field="lasso_selection_df",
            ),
        ),
        (
            "elastic",
            "use_elastic_net",
            SklearnSweepModelSpec(
                result_field="elastic_df",
                model_factory=lambda: ElasticNet(
                    alpha=0.1,
                    l1_ratio=0.5,
                    max_iter=20000,
                ),
                hyperparameter_spec=GridHyperparameterSpec(
                    estimator_factory=ElasticNet,
                    grid={
                        "alpha": (0.001, 0.01, 0.1),
                        "l1_ratio": (0.2, 0.5, 0.8),
                    },
                    fixed_params={"max_iter": 20000},
                    metadata_keys=("alpha", "l1_ratio"),
                ),
                selection_metadata_field="elastic_selection_df",
            ),
        ),
    )
