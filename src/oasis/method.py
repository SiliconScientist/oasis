from __future__ import annotations

from collections.abc import Collection
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from itertools import product
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from oasis.sweep import (
    LearningCurveResults,
    SweepFamilyRequirements,
    SweepModelCapabilities,
    SweepRunnerPayload,
    SweepRunPayload,
    TrainTestSweepRunnerInput,
    TrainValTestSweepRunnerInput,
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


_SWEEP_RESULT_COLUMNS = ["n_train", "rmse_mean", "rmse_std"]
SelectionRefitPolicy = Literal["train_only", "train_plus_val"]


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
    trimmed_result_field: str | None
    model_factory: Callable[[], object]
    hyperparameter_spec: HyperparameterSpec | None = None
    selection_metadata_field: str | None = None
    selection_refit_policy: SelectionRefitPolicy = "train_plus_val"
    trim_z_thresh: float = 1.0


@runtime_checkable
class HyperparameterSpec(Protocol):
    """Contract for candidate estimators used in validation-based selection.

    Candidate ranking must be based on `val_idx` only. `test_idx` is reserved
    for one final outer evaluation after selection and any optional refit.
    """

    def candidate_factories(self) -> tuple[Callable[[], object], ...]: ...

    def selection_metadata(self, model: object) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class FactoryListHyperparameterSpec:
    factories: tuple[Callable[[], object], ...]
    metadata_extractor: Callable[[object], Mapping[str, Any]] | None = None

    def candidate_factories(self) -> tuple[Callable[[], object], ...]:
        if not self.factories:
            raise ValueError("hyperparameter spec must declare at least one candidate")
        return self.factories

    def selection_metadata(self, model: object) -> Mapping[str, Any]:
        if self.metadata_extractor is None:
            return {}
        return self.metadata_extractor(model)


@dataclass(frozen=True, slots=True)
class GridHyperparameterSpec:
    estimator_factory: Callable[..., object]
    grid: Mapping[str, tuple[Any, ...]]
    fixed_params: Mapping[str, Any] = field(default_factory=dict)
    metadata_keys: tuple[str, ...] | None = None

    def candidate_factories(self) -> tuple[Callable[[], object], ...]:
        items = tuple(self.grid.items())
        if not items:
            return (lambda: self.estimator_factory(**self.fixed_params),)
        candidate_factories = []
        keys = tuple(key for key, _ in items)
        value_sets = tuple(values for _, values in items)
        for values in product(*value_sets):
            params = dict(self.fixed_params)
            params.update(zip(keys, values, strict=True))
            candidate_factories.append(_estimator_factory_with_params(self.estimator_factory, params))
        return tuple(candidate_factories)

    def selection_metadata(self, model: object) -> Mapping[str, Any]:
        if hasattr(model, "get_params"):
            params = model.get_params(deep=False)
        else:
            params = {
                key: getattr(model, key)
                for key in self._selected_metadata_keys()
                if hasattr(model, key)
            }
        return {
            key: params[key]
            for key in self._selected_metadata_keys()
            if key in params
        }

    def _selected_metadata_keys(self) -> tuple[str, ...]:
        if self.metadata_keys is not None:
            return self.metadata_keys
        return tuple(self.grid.keys())


@runtime_checkable
class ValidationAwareEstimator(Protocol):
    """Estimator that selects on validation data before outer-test prediction."""

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


def _as_runner_payload(
    payload: SweepRunPayload | SweepRunnerPayload,
) -> SweepRunnerPayload:
    if isinstance(payload, SweepRunnerPayload):
        return payload
    return payload.to_runner_payload()


def _estimator_factory_with_params(
    estimator_factory: Callable[..., object],
    params: Mapping[str, Any],
) -> Callable[[], object]:
    return lambda: estimator_factory(**params)


def _validate_selection_refit_policy(policy: SelectionRefitPolicy) -> None:
    if policy not in ("train_only", "train_plus_val"):
        raise ValueError(f"unsupported selection refit policy: {policy}")


def _payload_uses_validation(payload: SweepRunnerPayload) -> bool:
    if payload.planning_requirements.requires_validation:
        return True
    return any(isinstance(split, TrainValTestSweepRunnerInput) for split in payload.splits)


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
    *,
    trimmed: bool,
) -> Callable[..., pd.DataFrame]:
    uses_validation = _payload_uses_validation(payload)
    if uses_validation:
        if not isinstance(runner, ValidationAwareSweepExperimentRunner):
            raise TypeError("runner does not support validation-aware sweep payloads")
        return (
            runner.run_trimmed_with_validation
            if trimmed
            else runner.run_with_validation
        )
    if not isinstance(runner, SweepExperimentRunner):
        raise TypeError("runner does not support train/test sweep payloads")
    return runner.run_trimmed if trimmed else runner.run


def _normalize_runner_output(
    output: pd.DataFrame | SweepRunnerArtifacts,
) -> SweepRunnerArtifacts:
    if isinstance(output, SweepRunnerArtifacts):
        return output
    return SweepRunnerArtifacts(metrics=output)


@dataclass(frozen=True, slots=True)
class LearningCurveModelRegistration:
    name: str
    config_attr: str
    family_factory: Callable[[], SweepModelFamily]


class SweepModelFamily(Protocol):
    """Contract for a model family runnable over a shared split sweep."""

    def capabilities(self) -> SweepModelCapabilities: ...

    def requirements(self) -> SweepFamilyRequirements: ...

    def run(self, payload: SweepRunPayload) -> LearningCurveResults: ...


@runtime_checkable
class SweepExperimentRunner(Protocol):
    """Common runner interface for sweep methods with optional trimmed output."""

    def run(self, payload: SweepRunnerPayload) -> pd.DataFrame: ...

    def run_trimmed(
        self,
        payload: SweepRunnerPayload,
        *,
        z_thresh: float = 1.0,
    ) -> pd.DataFrame: ...


@runtime_checkable
class ValidationAwareSweepExperimentRunner(Protocol):
    """Runner interface for methods that select on val and evaluate on outer test."""

    def run_with_validation(self, payload: SweepRunnerPayload) -> pd.DataFrame: ...

    def run_trimmed_with_validation(
        self,
        payload: SweepRunnerPayload,
        *,
        z_thresh: float = 1.0,
    ) -> pd.DataFrame: ...


@dataclass(frozen=True, slots=True)
class SweepFamilySpec:
    result_field: str
    trimmed_result_field: str | None
    runner: SweepExperimentRunner | ValidationAwareSweepExperimentRunner
    selection_metadata_field: str | None = None
    trim_z_thresh: float = 1.0
    capabilities: SweepModelCapabilities = field(
        default_factory=SweepModelCapabilities
    )


@dataclass(frozen=True, slots=True)
class SupervisedModelSweepRunner:
    """Reusable adapter for supervised estimators over the shared sweep payload."""

    model_factory: Callable[[], object]

    def run(self, payload: SweepRunnerPayload) -> pd.DataFrame:
        return sweep_model(payload, self.model_factory)

    def run_trimmed(
        self,
        payload: SweepRunnerPayload,
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
    base_runner: Callable[[SweepRunnerPayload], pd.DataFrame]
    trimmed_runner: Callable[[SweepRunnerPayload], pd.DataFrame]

    def run(self, payload: SweepRunnerPayload) -> pd.DataFrame:
        return self.base_runner(payload)

    def run_trimmed(
        self,
        payload: SweepRunnerPayload,
        *,
        z_thresh: float = 1.0,
    ) -> pd.DataFrame:
        del z_thresh
        return self.trimmed_runner(payload)


@dataclass(frozen=True, slots=True)
class ValidationAwareSupervisedModelSweepRunner:
    """Adapter for estimators that tune on val before outer-test evaluation."""

    model_factory: Callable[[], object]

    def run_with_validation(self, payload: SweepRunnerPayload) -> pd.DataFrame:
        return sweep_model_with_validation(payload, self.model_factory)

    def run_trimmed_with_validation(
        self,
        payload: SweepRunnerPayload,
        *,
        z_thresh: float = 1.0,
    ) -> pd.DataFrame:
        return sweep_model_with_validation_trimmed(
            payload,
            self.model_factory,
            z_thresh=z_thresh,
        )


@dataclass(frozen=True, slots=True)
class SupervisedModelSelectionSweepRunner:
    """Reusable runner for train/val/test supervised model selection.

    The policy is:
    - fit each candidate on `train_idx`
    - score each candidate on `val_idx`
    - pick the best validation candidate deterministically by candidate order
      when scores tie
    - optionally refit the chosen candidate on `train_idx + val_idx`, or keep
      the train-only fit, depending on `refit_policy`
    - evaluate once on the outer `test_idx`
    """

    hyperparameter_spec: HyperparameterSpec
    refit_policy: SelectionRefitPolicy = "train_plus_val"

    def run_with_validation(self, payload: SweepRunnerPayload) -> pd.DataFrame:
        return sweep_supervised_model_selection(
            payload,
            self.hyperparameter_spec,
            refit_policy=self.refit_policy,
        )

    def run_trimmed_with_validation(
        self,
        payload: SweepRunnerPayload,
        *,
        z_thresh: float = 1.0,
    ) -> pd.DataFrame:
        return sweep_supervised_model_selection_trimmed(
            payload,
            self.hyperparameter_spec,
            refit_policy=self.refit_policy,
            z_thresh=z_thresh,
        )


HyperparameterSelectionSweepRunner = SupervisedModelSelectionSweepRunner


@dataclass(frozen=True, slots=True)
class WeightedLinearSweepRunner:
    fit_intercept: bool = True

    def run(self, payload: SweepRunnerPayload) -> pd.DataFrame:
        return weighted_linear_sweep(
            payload,
            fit_intercept=self.fit_intercept,
        )

    def run_trimmed(
        self,
        payload: SweepRunnerPayload,
        *,
        z_thresh: float = 1.0,
    ) -> pd.DataFrame:
        del z_thresh
        return self.run(payload)


@dataclass(frozen=True, slots=True)
class WeightedSimplexSweepRunner:
    def run(self, payload: SweepRunnerPayload) -> pd.DataFrame:
        return weighted_simplex_sweep(payload)

    def run_trimmed(
        self,
        payload: SweepRunnerPayload,
        *,
        z_thresh: float = 1.0,
    ) -> pd.DataFrame:
        del z_thresh
        return self.run(payload)


@dataclass(frozen=True, slots=True)
class SweepRunnerArtifacts:
    metrics: pd.DataFrame
    selection_metadata: pd.DataFrame | None = None


@dataclass(frozen=True, slots=True)
class ConfiguredSweepModelFamily:
    spec: SweepFamilySpec

    def capabilities(self) -> SweepModelCapabilities:
        return self.spec.capabilities

    def requirements(self) -> SweepFamilyRequirements:
        return self.spec.capabilities.to_requirements()

    def run(self, payload: SweepRunPayload) -> LearningCurveResults:
        runner_payload = payload.to_runner_payload()
        run = _select_runner_call(self.spec.runner, runner_payload, trimmed=False)
        base_output = _normalize_runner_output(run(runner_payload))
        results: dict[str, pd.DataFrame | None] = {
            self.spec.result_field: base_output.metrics,
        }
        if self.spec.selection_metadata_field is not None:
            results[self.spec.selection_metadata_field] = base_output.selection_metadata
        if payload.use_trim and self.spec.trimmed_result_field is not None:
            run_trimmed = _select_runner_call(
                self.spec.runner,
                runner_payload,
                trimmed=True,
            )
            trimmed_output = _normalize_runner_output(
                run_trimmed(
                    runner_payload,
                    z_thresh=self.spec.trim_z_thresh,
                )
            )
            results[self.spec.trimmed_result_field] = trimmed_output.metrics
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


def _sklearn_runner_for_spec(
    spec: SklearnSweepModelSpec,
) -> SweepExperimentRunner | ValidationAwareSweepExperimentRunner:
    if spec.hyperparameter_spec is not None:
        return SupervisedModelSelectionSweepRunner(
            spec.hyperparameter_spec,
            refit_policy=spec.selection_refit_policy,
        )
    return SupervisedModelSweepRunner(spec.model_factory)


def _sklearn_capabilities_for_spec(
    spec: SklearnSweepModelSpec,
) -> SweepModelCapabilities:
    if spec.hyperparameter_spec is None:
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
            config_attr=config_attr,
            family_factory=lambda spec=spec: ConfiguredSweepModelFamily(
                SweepFamilySpec(
                    result_field=spec.result_field,
                    trimmed_result_field=spec.trimmed_result_field,
                    selection_metadata_field=_sklearn_selection_metadata_field_for_spec(spec),
                    runner=_sklearn_runner_for_spec(spec),
                    trim_z_thresh=spec.trim_z_thresh,
                    capabilities=_sklearn_capabilities_for_spec(spec),
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
            name="weighted_linear",
            config_attr="use_weighted_linear",
            family_factory=lambda: ConfiguredSweepModelFamily(
                SweepFamilySpec(
                    result_field="weighted_linear_df",
                    trimmed_result_field=None,
                    runner=WeightedLinearSweepRunner(fit_intercept=True),
                )
            ),
        ),
        LearningCurveModelRegistration(
            name="weighted_simplex",
            config_attr="use_weighted_simplex",
            family_factory=lambda: ConfiguredSweepModelFamily(
                SweepFamilySpec(
                    result_field="weighted_simplex_df",
                    trimmed_result_field=None,
                    runner=WeightedSimplexSweepRunner(),
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
                hyperparameter_spec=GridHyperparameterSpec(
                    estimator_factory=Ridge,
                    grid={"alpha": (0.01, 0.1, 1.0, 10.0)},
                ),
                selection_metadata_field="ridge_selection_df",
            ),
        ),
        (
            "kernel_ridge",
            "use_kernel_ridge",
            SklearnSweepModelSpec(
                result_field="kernel_ridge_df",
                trimmed_result_field=None,
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
                trimmed_result_field="lasso_trimmed_df",
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
                trimmed_result_field="elastic_trimmed_df",
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
                trimmed_result_field=spec.trimmed_result_field,
                selection_metadata_field=_sklearn_selection_metadata_field_for_spec(spec),
                runner=_sklearn_runner_for_spec(spec),
                trim_z_thresh=spec.trim_z_thresh,
                capabilities=_sklearn_capabilities_for_spec(spec),
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
    payload: SweepRunPayload | SweepRunnerPayload,
    model_factory,
) -> pd.DataFrame:
    """Evaluate a supervised model across precomputed sweep splits."""

    payload = _as_runner_payload(payload)
    rmses_by_size: dict[int, list[float]] = {}
    for split in _assert_train_test_payload(payload):
        X = split.dataset.X
        y = split.dataset.y
        model = model_factory()
        model.fit(X[split.train_idx], y[split.train_idx])
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
        X = split.dataset.X
        y = split.dataset.y
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


def _select_candidate_factory_by_validation(
    split: TrainValTestSweepRunnerInput,
    hyperparameter_spec: HyperparameterSpec,
) -> Callable[[], object]:
    X = split.dataset.X
    y = split.dataset.y
    best_candidate_factory = None
    best_rmse = np.inf
    for candidate_factory in hyperparameter_spec.candidate_factories():
        model = candidate_factory()
        model.fit(X[split.train_idx], y[split.train_idx])
        val_preds = model.predict(X[split.val_idx])
        val_rmse = np.sqrt(mean_squared_error(y[split.val_idx], val_preds))
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_candidate_factory = candidate_factory
    if best_candidate_factory is None:
        raise ValueError("hyperparameter spec did not yield any candidate estimators")
    return best_candidate_factory


def _fit_selected_supervised_model(
    split: TrainValTestSweepRunnerInput,
    hyperparameter_spec: HyperparameterSpec,
    *,
    refit_policy: SelectionRefitPolicy,
) -> tuple[object, Mapping[str, Any]]:
    """Return the selected model after applying the configured refit policy."""

    _validate_selection_refit_policy(refit_policy)
    candidate_factory = _select_candidate_factory_by_validation(
        split,
        hyperparameter_spec,
    )
    model = candidate_factory()
    X = split.dataset.X
    y = split.dataset.y
    if refit_policy == "train_only":
        fit_idx = split.train_idx
    else:
        fit_idx = np.concatenate([split.train_idx, split.val_idx])
    model.fit(X[fit_idx], y[fit_idx])
    return model, hyperparameter_spec.selection_metadata(model)


def _selection_metadata_frame(
    metadata_by_size: dict[int, list[Mapping[str, Any]]],
) -> pd.DataFrame:
    rows = []
    for n_train, records in sorted(metadata_by_size.items()):
        row: dict[str, Any] = {"n_train": n_train}
        keys = tuple(
            sorted(
                {
                    key
                    for record in records
                    for key in record
                }
            )
        )
        for key in keys:
            values = [record[key] for record in records if key in record]
            unique_values = []
            for value in values:
                if value not in unique_values:
                    unique_values.append(value)
            row[key] = unique_values[0] if len(unique_values) == 1 else tuple(unique_values)
        rows.append(row)
    return pd.DataFrame(rows)


def sweep_supervised_model_selection(
    payload: SweepRunPayload | SweepRunnerPayload,
    hyperparameter_spec: HyperparameterSpec,
    *,
    refit_policy: SelectionRefitPolicy = "train_plus_val",
) -> SweepRunnerArtifacts:
    """Select on validation, optionally refit, then evaluate once on outer test.

    This is the generic supervised model-selection path. It never uses
    `test_idx` during candidate ranking. Final fitting follows `refit_policy`:
    `train_only` keeps the fit performed on `train_idx`, while
    `train_plus_val` refits the chosen candidate on the full outer training
    budget `train_idx + val_idx` before outer-test evaluation.
    """

    payload = _as_runner_payload(payload)
    splits = _assert_train_val_test_payload(payload)
    rmses_by_size: dict[int, list[float]] = {}
    metadata_by_size: dict[int, list[Mapping[str, Any]]] = {}
    for split in splits:
        y = split.dataset.y
        model, metadata = _fit_selected_supervised_model(
            split,
            hyperparameter_spec,
            refit_policy=refit_policy,
        )
        preds = model.predict(split.dataset.X[split.test_idx])
        rmse = np.sqrt(mean_squared_error(y[split.test_idx], preds))
        rmses_by_size.setdefault(split.sweep_size, []).append(rmse)
        metadata_by_size.setdefault(split.sweep_size, []).append(metadata)
    return SweepRunnerArtifacts(
        metrics=sweep_results_frame(rmses_by_size),
        selection_metadata=_selection_metadata_frame(metadata_by_size),
    )


def sweep_model_trimmed(
    payload: SweepRunPayload | SweepRunnerPayload,
    model_factory,
    z_thresh: float = 1.0,
) -> pd.DataFrame:
    """
    Fit the model, drop test samples with large contribution z-scores, and refit.
    """
    payload = _as_runner_payload(payload)
    rmses_by_size: dict[int, list[float]] = {}
    for split in _assert_train_test_payload(payload):
        X = split.dataset.X
        y = split.dataset.y
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


def sweep_model_with_validation_trimmed(
    payload: SweepRunPayload | SweepRunnerPayload,
    model_factory: Callable[[], ValidationAwareEstimator],
    z_thresh: float = 1.0,
) -> pd.DataFrame:
    """
    Validation-aware evaluation with test-time trimming by contribution z-score.
    """

    payload = _as_runner_payload(payload)
    splits = _assert_train_val_test_payload(payload)
    rmses_by_size: dict[int, list[float]] = {}
    for split in splits:
        X = split.dataset.X
        y = split.dataset.y
        model = model_factory()
        model.fit(
            X[split.train_idx],
            y[split.train_idx],
            X[split.val_idx],
            y[split.val_idx],
        )

        X_test = X[split.test_idx]
        preds = model.predict(X_test)

        w = getattr(model, "coef_", None)
        if w is None:
            raise AttributeError(
                "validation-aware trimmed sweep requires estimator.coef_"
            )
        contrib = X_test * np.asarray(w)
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


def sweep_supervised_model_selection_trimmed(
    payload: SweepRunPayload | SweepRunnerPayload,
    hyperparameter_spec: HyperparameterSpec,
    *,
    refit_policy: SelectionRefitPolicy = "train_plus_val",
    z_thresh: float = 1.0,
) -> SweepRunnerArtifacts:
    """
    Validation-based model selection with test-time trimming by contribution z-score.
    """

    payload = _as_runner_payload(payload)
    splits = _assert_train_val_test_payload(payload)
    rmses_by_size: dict[int, list[float]] = {}
    metadata_by_size: dict[int, list[Mapping[str, Any]]] = {}
    for split in splits:
        X = split.dataset.X
        y = split.dataset.y
        model, metadata = _fit_selected_supervised_model(
            split,
            hyperparameter_spec,
            refit_policy=refit_policy,
        )

        X_test = X[split.test_idx]
        preds = model.predict(X_test)

        w = getattr(model, "coef_", None)
        if w is None:
            raise AttributeError(
                "trimmed hyperparameter selection requires estimator.coef_"
            )
        contrib = X_test * np.asarray(w)
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
        metadata_by_size.setdefault(split.sweep_size, []).append(metadata)
    return SweepRunnerArtifacts(
        metrics=sweep_results_frame(rmses_by_size),
        selection_metadata=_selection_metadata_frame(metadata_by_size),
    )


def sweep_model_with_hyperparameter_selection(
    payload: SweepRunPayload | SweepRunnerPayload,
    hyperparameter_spec: HyperparameterSpec,
    *,
    refit_policy: SelectionRefitPolicy = "train_plus_val",
) -> pd.DataFrame:
    return sweep_supervised_model_selection(
        payload,
        hyperparameter_spec,
        refit_policy=refit_policy,
    ).metrics


def sweep_model_with_hyperparameter_selection_trimmed(
    payload: SweepRunPayload | SweepRunnerPayload,
    hyperparameter_spec: HyperparameterSpec,
    *,
    refit_policy: SelectionRefitPolicy = "train_plus_val",
    z_thresh: float = 1.0,
) -> pd.DataFrame:
    return sweep_supervised_model_selection_trimmed(
        payload,
        hyperparameter_spec,
        refit_policy=refit_policy,
        z_thresh=z_thresh,
    ).metrics


def residual_sweep(
    payload: SweepRunPayload | SweepRunnerPayload,
) -> pd.DataFrame:
    """Evaluate residual correction across precomputed sweep splits."""

    payload = _as_runner_payload(payload)
    rmses_by_size: dict[int, list[float]] = {}
    for split in _assert_train_test_payload(payload):
        X = split.dataset.X
        y = split.dataset.y
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
    payload: SweepRunPayload | SweepRunnerPayload,
) -> pd.DataFrame:
    """
    Residual correction with per-sample outlier MLIP removal before averaging.
    """
    payload = _as_runner_payload(payload)
    rmses_by_size: dict[int, list[float]] = {}
    for split in _assert_train_test_payload(payload):
        X = split.dataset.X
        y = split.dataset.y
        X_train = X[split.train_idx]
        y_train = y[split.train_idx]

        residuals = y_train[:, None] - X_train
        mean_residuals = residuals.mean(axis=0)

        X_corrected = X[split.test_idx] + mean_residuals
        preds = trimmed_mean_predictions(X_corrected)
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
        X = split.dataset.X
        y = split.dataset.y
        model = LinearRegression(fit_intercept=fit_intercept)
        model.fit(X[split.train_idx], y[split.train_idx])
        preds = model.predict(X[split.test_idx])
        rmse = np.sqrt(mean_squared_error(y[split.test_idx], preds))
        rmses_by_size.setdefault(split.sweep_size, []).append(rmse)
    return sweep_results_frame(rmses_by_size)


def _simplex_weights(
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


def weighted_simplex_sweep(
    payload: SweepRunPayload | SweepRunnerPayload,
) -> pd.DataFrame:
    """
    Fit a simplex-style combiner with nonnegative weights that sum to one.
    """

    payload = _as_runner_payload(payload)
    rmses_by_size: dict[int, list[float]] = {}
    for split in _assert_train_test_payload(payload):
        X = split.dataset.X
        y = split.dataset.y
        weights = _simplex_weights(
            X[split.train_idx],
            y[split.train_idx],
        )
        preds = X[split.test_idx] @ weights
        rmse = np.sqrt(mean_squared_error(y[split.test_idx], preds))
        rmses_by_size.setdefault(split.sweep_size, []).append(rmse)
    return sweep_results_frame(rmses_by_size)
