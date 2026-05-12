from __future__ import annotations

from collections.abc import Collection
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
from oasis.learning_curve.runners import (
    ConfiguredSweepModelFamily,
    FunctionalSweepRunner,
    LearnedModelSweepRunner,
    PlaceholderLearnedSweepModelFamily,
    SupervisedModelSweepRunner,
    SweepExperimentRunner,
    SweepFamilySpec,
    SweepModelFamily,
    TrainTestLearnedEstimator,
    TrainValTestLearnedEstimator,
    ValidationAwareLearnedModelSweepRunner,
    ValidationAwareSupervisedModelSweepRunner,
    ValidationAwareSweepExperimentRunner,
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
    OptunaModelSelectionSweepRunner,
    SelectionRefitPolicy,
    SupervisedModelSelectionSweepRunner,
    SweepRunnerArtifacts,
    TrialTuningSpec,
    ValidationAwareEstimator,
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
class LearningCurveModelRegistration:
    name: str
    is_enabled: Callable[[Any], bool]
    family_factory: Callable[[], SweepModelFamily]
    default_enabled: bool = True


def _config_flag_enabled(config_key: str) -> Callable[[Any], bool]:
    return lambda config_section: bool(getattr(config_section, config_key, False))


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


def _moe_enabled(config_section: Any) -> bool:
    return bool(getattr(getattr(config_section, "moe", None), "enabled", False))


def _is_enabled_for_learned_family_spec(
    spec: LearnedFamilyRegistrationSpec,
) -> Callable[[Any], bool]:
    if spec.is_enabled is not None:
        return spec.is_enabled
    if spec.config_key is not None:
        return _config_flag_enabled(spec.config_key)
    raise ValueError(
        f"learned family registration '{spec.name}' must define config_key or is_enabled"
    )


def _family_factory_for_learned_family_spec(
    spec: LearnedFamilyRegistrationSpec,
) -> Callable[[], SweepModelFamily]:
    def build_family() -> SweepModelFamily:
        if spec.family_factory is not None:
            family = spec.family_factory()
        else:
            family = _configured_trial_tuned_family_for_learned_family_spec(spec)
        if family.capabilities() != spec.capabilities:
            raise ValueError(
                f"learned family registration '{spec.name}' declared capabilities "
                f"{spec.capabilities!r} but factory produced {family.capabilities()!r}"
            )
        return family

    return build_family


def _configured_trial_tuned_family_for_learned_family_spec(
    spec: LearnedFamilyRegistrationSpec,
) -> SweepModelFamily:
    if (
        spec.trial_tuning_spec is not None
        and spec.learned_trial_tuning_spec is not None
    ):
        raise ValueError(
            f"learned family registration '{spec.name}' cannot define both "
            "trial_tuning_spec and learned_trial_tuning_spec"
        )
    if spec.trial_tuning_spec is None and spec.learned_trial_tuning_spec is None:
        raise ValueError(
            f"learned family registration '{spec.name}' must define family_factory "
            "or a trial tuning spec"
        )
    if spec.result_field is None:
        raise ValueError(
            f"learned family registration '{spec.name}' must define result_field "
            "for trial-tuned registration"
        )
    if spec.optuna_n_trials is None:
        raise ValueError(
            f"learned family registration '{spec.name}' must define optuna_n_trials "
            "for trial-tuned registration"
        )
    runner_kwargs = {
        "n_trials": spec.optuna_n_trials,
        "timeout_s": spec.optuna_timeout_s,
    }
    if spec.optuna_study_factory is not None:
        runner_kwargs["study_factory"] = spec.optuna_study_factory
    runner = (
        OptunaModelSelectionSweepRunner(
            spec.trial_tuning_spec,
            **runner_kwargs,
        )
        if spec.trial_tuning_spec is not None
        else LearnedOptunaModelSelectionSweepRunner(
            spec.learned_trial_tuning_spec,
            **runner_kwargs,
        )
    )
    return ConfiguredSweepModelFamily(
        SweepFamilySpec(
            result_field=spec.result_field,
            runner=runner,
            selection_metadata_field=spec.selection_metadata_field,
            capabilities=spec.capabilities,
        )
    )


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
                batching=_GRAPH_MEAN_BATCH_ADAPTER.batching_for_split(split_name="test"),
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


def learned_family_registration(
    spec: LearnedFamilyRegistrationSpec,
) -> LearningCurveModelRegistration:
    return LearningCurveModelRegistration(
        name=spec.name,
        is_enabled=_is_enabled_for_learned_family_spec(spec),
        family_factory=_family_factory_for_learned_family_spec(spec),
        default_enabled=spec.default_enabled,
    )



def default_sweep_model_families(
    enabled_model_names: Collection[str] | None = None,
) -> tuple[SweepModelFamily, ...]:
    registrations = learning_curve_model_registry()
    enabled_names = (
        set(
            registration.name
            for registration in registrations
            if registration.default_enabled
        )
        if enabled_model_names is None
        else set(enabled_model_names)
    )
    return tuple(
        registration.family_factory()
        for registration in registrations
        if registration.name in enabled_names
    )


def enabled_learning_curve_model_names_from_config(
    model_cfg: Any | None,
) -> tuple[str, ...]:
    registrations = learning_curve_model_registry()
    if model_cfg is None:
        return tuple(
            registration.name
            for registration in registrations
            if registration.default_enabled
        )
    return tuple(
        registration.name
        for registration in registrations
        if registration.is_enabled(model_cfg)
    )


def _sklearn_runner_for_spec(
    spec: SklearnSweepModelSpec,
) -> SweepExperimentRunner | ValidationAwareSweepExperimentRunner:
    if spec.trial_tuning_spec is not None:
        if spec.optuna_n_trials is None:
            raise ValueError("optuna-backed sklearn spec must declare optuna_n_trials")
        runner_kwargs = {
            "n_trials": spec.optuna_n_trials,
            "timeout_s": spec.optuna_timeout_s,
            "refit_policy": spec.selection_refit_policy,
        }
        if spec.optuna_study_factory is not None:
            runner_kwargs["study_factory"] = spec.optuna_study_factory
        return OptunaModelSelectionSweepRunner(
            spec.trial_tuning_spec,
            **runner_kwargs,
        )
    if spec.hyperparameter_spec is not None:
        return SupervisedModelSelectionSweepRunner(
            spec.hyperparameter_spec,
            refit_policy=spec.selection_refit_policy,
        )
    return SupervisedModelSweepRunner(spec.model_factory)


def _sklearn_capabilities_for_spec(
    spec: SklearnSweepModelSpec,
) -> SweepModelCapabilities:
    if spec.hyperparameter_spec is None and spec.trial_tuning_spec is None:
        return SweepModelCapabilities()
    return SweepModelCapabilities(requires_validation=True)


def _sklearn_selection_metadata_field_for_spec(
    spec: SklearnSweepModelSpec,
) -> str | None:
    return spec.selection_metadata_field


def learning_curve_model_registry() -> tuple[LearningCurveModelRegistration, ...]:
    sklearn_registrations = tuple(
        LearningCurveModelRegistration(
            name=name,
            is_enabled=_config_flag_enabled(config_key),
            family_factory=lambda spec=spec: ConfiguredSweepModelFamily(
                SweepFamilySpec(
                    result_field=spec.result_field,
                    selection_metadata_field=_sklearn_selection_metadata_field_for_spec(
                        spec
                    ),
                    runner=_sklearn_runner_for_spec(spec),
                    capabilities=_sklearn_capabilities_for_spec(spec),
                )
            ),
        )
        for name, config_key, spec in sklearn_sweep_model_specs()
    )
    return sklearn_registrations + tuple(
        learned_family_registration(spec)
        for spec in learned_family_registration_specs()
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


def sklearn_model_families(
    specs: tuple[tuple[str, str, SklearnSweepModelSpec], ...],
) -> tuple[SweepModelFamily, ...]:
    return tuple(
        ConfiguredSweepModelFamily(
            SweepFamilySpec(
                result_field=spec.result_field,
                selection_metadata_field=_sklearn_selection_metadata_field_for_spec(
                    spec
                ),
                runner=_sklearn_runner_for_spec(spec),
                capabilities=_sklearn_capabilities_for_spec(spec),
            )
        )
        for _, _, spec in specs
    )
