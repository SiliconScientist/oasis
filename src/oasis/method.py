from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from oasis.learning_curve.execution import (
    _normalize_runner_output,
    _select_runner_call,
    residual_sweep,
    sweep_learned_model,
    sweep_learned_model_with_validation,
    sweep_model,
    sweep_model_with_validation,
    sweep_results_frame,
    weighted_linear_sweep,
    weighted_simplex_sweep,
)
from oasis.learning_curve.families.graph_mean import (
    _graph_feature_means,
    _graph_feature_means_from_batches,
    _graph_mean_study_factory,
    _GRAPH_MEAN_BATCH_ADAPTER,
    _FixedScaleStudy,
    _FixedScaleTrial,
    _targets_from_batches,
    GraphMeanConstantModel,
    GraphMeanLearnedTrialTuningSpec,
)
from oasis.learning_curve.registry import (
    _configured_trial_tuned_family_for_learned_family_spec,
    _config_flag_enabled,
    _family_factory_for_learned_family_spec,
    _is_enabled_for_learned_family_spec,
    _moe_enabled,
    _sklearn_capabilities_for_spec,
    _sklearn_runner_for_spec,
    _sklearn_selection_metadata_field_for_spec,
    default_sweep_model_families,
    enabled_learning_curve_model_names_from_config,
    learned_family_registration,
    learning_curve_model_registry,
    sklearn_model_families,
    LearningCurveModelRegistration,
)
from oasis.learning_curve.runners import (
    ConfiguredSweepModelFamily,
    FunctionalSweepRunner,
    LearnedModelSweepRunner,
    PlaceholderLearnedSweepModelFamily,
    SupervisedModelSweepRunner,
    SweepFamilySpec,
    SweepModelFamily,
    TrainTestLearnedEstimator,
    TrainValTestLearnedEstimator,
    ValidationAwareLearnedModelSweepRunner,
    ValidationAwareSupervisedModelSweepRunner,
    WeightedLinearSweepRunner,
    WeightedSimplexSweepRunner,
)
from oasis.sweep import SweepModelCapabilities, TrainValTestSweepRunnerInput
from oasis.sweep import SweepDatasetBatchLoaderAdapter, TrainEvalLoaderPolicy
from oasis.tune import (
    GridHyperparameterSpec,
    HyperparameterSpec,
    LearnedOptunaModelSelectionSweepRunner,
    LearnedTrialTuningSpec,
    SelectionRefitPolicy,
    TrialTuningSpec,
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_squared_error


@dataclass(frozen=True, slots=True)
class SklearnSweepModelSpec:
    """Registered supervised model family configuration.

    When `hyperparameter_spec` is present, the family runs through the generic
    model-selection runner on train/val/test splits and therefore requires inner
    validation. `selection_refit_policy` controls whether the chosen candidate is
    refit on `train_idx` only or on `train_idx + val_idx` before the single outer
    test evaluation.
    """

    result_field: str
    model_factory: Callable[[], object]
    hyperparameter_spec: HyperparameterSpec | None = None
    trial_tuning_spec: TrialTuningSpec | None = None
    optuna_n_trials: int | None = None
    optuna_timeout_s: int | None = None
    optuna_study_factory: Callable[[TrainValTestSweepRunnerInput], Any] | None = None
    selection_metadata_field: str | None = None
    selection_refit_policy: SelectionRefitPolicy = "train_plus_val"


def _ridge_optuna_study_factory(split: TrainValTestSweepRunnerInput) -> Any:
    del split
    import optuna

    return optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.GridSampler(
            search_space={"alpha": [0.01, 0.1, 1.0, 10.0]},
            seed=0,
        ),
    )


@dataclass(frozen=True, slots=True)
class RidgeOptunaTrialTuningSpec:
    def build_trial_objective(
        self,
        split: TrainValTestSweepRunnerInput,
    ) -> Callable[[Any], float]:
        X = split.dataset.mlip_features
        y = split.dataset.targets

        def objective(trial: Any) -> float:
            alpha = float(trial.suggest_categorical("alpha", [0.01, 0.1, 1.0, 10.0]))
            model = Ridge(alpha=alpha)
            model.fit(X[split.train_idx], y[split.train_idx])
            val_preds = model.predict(X[split.val_idx])
            return float(np.sqrt(mean_squared_error(y[split.val_idx], val_preds)))

        return objective

    def fit_selected_model(
        self,
        split: TrainValTestSweepRunnerInput,
        best_trial: Any,
        *,
        refit_policy: SelectionRefitPolicy,
    ) -> object:
        alpha = float(best_trial.params["alpha"])
        model = Ridge(alpha=alpha)
        X = split.dataset.mlip_features
        y = split.dataset.targets
        if refit_policy == "train_only":
            fit_idx = split.train_idx
        else:
            fit_idx = np.concatenate([split.train_idx, split.val_idx])
        model.fit(X[fit_idx], y[fit_idx])
        return model

    def trial_metadata(
        self,
        best_trial: Any,
        model: object,
    ) -> dict[str, Any]:
        del model
        return {"alpha": float(best_trial.params["alpha"])}


@dataclass(frozen=True, slots=True)
class LearnedFamilyRegistrationSpec:
    """Registry spec for non-sklearn learning-curve families.

    Families can register either as:
    - a concrete `family_factory` that returns a `SweepModelFamily`, or
    - a trial-tuned family declaration that provides result fields plus a
      `TrialTuningSpec`, which the registry wires into the shared
      `OptunaModelSelectionSweepRunner`.
    """

    name: str
    capabilities: SweepModelCapabilities
    family_factory: Callable[[], SweepModelFamily] | None = None
    result_field: str | None = None
    trial_tuning_spec: TrialTuningSpec | None = None
    learned_trial_tuning_spec: LearnedTrialTuningSpec | None = None
    optuna_n_trials: int | None = None
    optuna_timeout_s: int | None = None
    optuna_study_factory: Callable[[TrainValTestSweepRunnerInput], Any] | None = None
    selection_metadata_field: str | None = None
    config_key: str | None = None
    is_enabled: Callable[[Any], bool] | None = None
    default_enabled: bool = True
def learned_family_registration_specs() -> tuple[LearnedFamilyRegistrationSpec, ...]:
    return (
        LearnedFamilyRegistrationSpec(
            name="residual",
            config_key="use_residual",
            capabilities=SweepModelCapabilities(),
            family_factory=lambda: ConfiguredSweepModelFamily(
                SweepFamilySpec(
                    result_field="resid_df",
                    runner=FunctionalSweepRunner(base_runner=residual_sweep),
                )
            ),
        ),
        LearnedFamilyRegistrationSpec(
            name="weighted_linear",
            config_key="use_weighted_linear",
            capabilities=SweepModelCapabilities(),
            family_factory=lambda: ConfiguredSweepModelFamily(
                SweepFamilySpec(
                    result_field="weighted_linear_df",
                    runner=WeightedLinearSweepRunner(fit_intercept=True),
                )
            ),
        ),
        LearnedFamilyRegistrationSpec(
            name="weighted_simplex",
            config_key="use_weighted_simplex",
            capabilities=SweepModelCapabilities(),
            family_factory=lambda: ConfiguredSweepModelFamily(
                SweepFamilySpec(
                    result_field="weighted_simplex_df",
                    runner=WeightedSimplexSweepRunner(),
                )
            ),
        ),
        LearnedFamilyRegistrationSpec(
            name="graph_mean",
            config_key="use_graph_mean",
            capabilities=SweepModelCapabilities(requires_validation=True),
            result_field="graph_mean_df",
            learned_trial_tuning_spec=GraphMeanLearnedTrialTuningSpec(),
            optuna_n_trials=3,
            optuna_study_factory=_graph_mean_study_factory,
            selection_metadata_field="graph_mean_selection_df",
            default_enabled=False,
        ),
        LearnedFamilyRegistrationSpec(
            name="moe",
            is_enabled=_moe_enabled,
            capabilities=SweepModelCapabilities(requires_validation=True),
            # A real MoE implementation can keep this registry entry shape but
            # swap the placeholder family_factory for `trial_tuning_spec=...`
            # plus Optuna study settings to run through the shared trial-based
            # selection runner.
            family_factory=lambda: PlaceholderLearnedSweepModelFamily(
                name="moe",
                declared_capabilities=SweepModelCapabilities(requires_validation=True),
            ),
            default_enabled=False,
        ),
    )


def sklearn_sweep_model_specs() -> tuple[tuple[str, str, SklearnSweepModelSpec], ...]:
    return (
        (
            "ridge",
            "use_ridge",
            SklearnSweepModelSpec(
                result_field="ridge_df",
                model_factory=lambda: Ridge(alpha=0.1),
                trial_tuning_spec=RidgeOptunaTrialTuningSpec(),
                optuna_n_trials=4,
                optuna_study_factory=_ridge_optuna_study_factory,
                selection_metadata_field="ridge_selection_df",
            ),
        ),
        (
            "kernel_ridge",
            "use_kernel_ridge",
            SklearnSweepModelSpec(
                result_field="kernel_ridge_df",
                model_factory=lambda: KernelRidge(alpha=1.0, kernel="rbf"),
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
