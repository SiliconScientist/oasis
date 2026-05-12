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
from oasis.sweep import (
    LoaderAdapterInput,
    SweepDataset,
    SweepDatasetBatchLoaderAdapter,
    LearningCurveResults,
    SweepFamilyRequirements,
    SweepModelCapabilities,
    SweepRunnerPayload,
    SweepRunPayload,
    SweepBatch,
    TrainEvalLoaderPolicy,
    TrainTestSweepRunnerInput,
    TrainValTestSweepRunnerInput,
)
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


def _graph_feature_means(dataset: SweepDataset) -> np.ndarray:
    if not dataset.has_graphs:
        raise ValueError(
            "graph_mean learned family requires graph_view on the dataset."
        )
    return np.asarray(
        [
            float(np.mean(dataset.graphs[sample_id].node_features))
            for sample_id in dataset.sample_ids.tolist()
        ],
        dtype=float,
    )


_GRAPH_MEAN_BATCH_ADAPTER = SweepDatasetBatchLoaderAdapter(
    policy=TrainEvalLoaderPolicy(
        batch_size=None,
        eval_batch_size=None,
        train_shuffle=False,
        eval_shuffle=False,
    )
)


def _graph_feature_means_from_batches(batches: tuple[SweepBatch, ...]) -> np.ndarray:
    return np.asarray(
        [
            float(np.mean(graph.node_features))
            for batch in batches
            for graph in batch.graphs
            if graph is not None
        ],
        dtype=float,
    )


def _targets_from_batches(batches: tuple[SweepBatch, ...]) -> np.ndarray:
    return np.concatenate([batch.targets for batch in batches])


@dataclass(frozen=True, slots=True)
class GraphMeanConstantModel:
    scale: float
    offset: float


@dataclass(frozen=True, slots=True)
class GraphMeanLearnedTrialTuningSpec:
    def build_trial_objective(
        self,
        split: TrainValTestSweepRunnerInput,
    ) -> Callable[[Any], float]:
        loaders = split.loaders(_GRAPH_MEAN_BATCH_ADAPTER)
        graph_means = _graph_feature_means_from_batches(loaders.val)
        targets = _targets_from_batches(loaders.val)

        def objective(trial: Any) -> float:
            scale = float(trial.params["scale"])
            preds = graph_means * scale
            return float(np.sqrt(mean_squared_error(targets, preds)))

        return objective

    def fit_selected_model(
        self,
        split: TrainValTestSweepRunnerInput,
        best_trial: Any,
        *,
        refit_policy: SelectionRefitPolicy,
    ) -> GraphMeanConstantModel:
        loaders = split.loaders(_GRAPH_MEAN_BATCH_ADAPTER)
        scale = float(best_trial.params["scale"])
        if refit_policy == "train_only":
            fit_graph_means = _graph_feature_means_from_batches(loaders.train)
            fit_targets = _targets_from_batches(loaders.train)
        else:
            fit_graph_means = np.concatenate(
                [
                    _graph_feature_means_from_batches(loaders.train),
                    _graph_feature_means_from_batches(loaders.val),
                ]
            )
            fit_targets = np.concatenate(
                [
                    _targets_from_batches(loaders.train),
                    _targets_from_batches(loaders.val),
                ]
            )
        offset = float(np.mean(fit_targets - (fit_graph_means * scale)))
        return GraphMeanConstantModel(scale=scale, offset=offset)

    def predict(
        self,
        model: GraphMeanConstantModel,
        dataset: SweepDataset,
    ) -> np.ndarray:
        batches = _GRAPH_MEAN_BATCH_ADAPTER.build_loader(
            LoaderAdapterInput(
                dataset=dataset,
                split_name="test",
                batching=_GRAPH_MEAN_BATCH_ADAPTER.batching_for_split(
                    split_name="test"
                ),
            )
        )
        return (_graph_feature_means_from_batches(batches) * model.scale) + model.offset

    def trial_metadata(
        self,
        best_trial: Any,
        model: GraphMeanConstantModel,
    ) -> dict[str, Any]:
        return {
            "scale": float(best_trial.params["scale"]),
            "offset": float(model.offset),
        }


@dataclass(frozen=True, slots=True)
class _FixedScaleTrial:
    scale: float
    value: float | None = None

    @property
    def params(self) -> dict[str, float]:
        return {"scale": self.scale}


@dataclass(frozen=True, slots=True)
class _FixedScaleStudy:
    trials: tuple[_FixedScaleTrial, ...]
    best_trial: _FixedScaleTrial | None = None

    def optimize(
        self,
        objective: Callable[[Any], float],
        *,
        n_trials: int,
        timeout: int | None,
    ) -> None:
        del timeout
        best_value = np.inf
        best_trial = None
        for trial in self.trials[:n_trials]:
            objective_value = objective(trial)
            object.__setattr__(trial, "value", objective_value)
            if objective_value < best_value:
                best_value = objective_value
                best_trial = trial
        object.__setattr__(self, "best_trial", best_trial)


def _graph_mean_study_factory(split: TrainValTestSweepRunnerInput) -> Any:
    del split
    return _FixedScaleStudy(
        trials=(
            _FixedScaleTrial(scale=0.5),
            _FixedScaleTrial(scale=1.0),
            _FixedScaleTrial(scale=1.5),
        )
    )


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
