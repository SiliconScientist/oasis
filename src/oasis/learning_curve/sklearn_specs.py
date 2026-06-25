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
except ImportError:
    KernelRidge = ElasticNet = Lasso = Ridge = None  # type: ignore[assignment,misc]


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
                    uncertainty_kind="calibrated",
                    uncertainty_note="post-hoc scalar calibrated spread",
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
                ),
                selection_metadata_field="kernel_ridge_selection_df",
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
