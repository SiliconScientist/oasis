from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from itertools import product
from time import perf_counter
from typing import Any, Literal, Protocol, runtime_checkable
import warnings

import numpy as np
import pandas as pd
from oasis.calibration import ScalarSpreadCalibrator
from oasis.sweep import (
    SweepDataset,
    SweepRunnerPayload,
    SweepRunPayload,
    TrainValCalTestSweepRunnerInput,
    TrainValTestSweepRunnerInput,
)

try:
    from pydantic import BaseModel
except ModuleNotFoundError:
    class BaseModel:
        def __init__(self, **data: Any) -> None:
            annotations = getattr(type(self), "__annotations__", {})
            unexpected = set(data) - set(annotations)
            if unexpected:
                unexpected_names = ", ".join(sorted(unexpected))
                raise TypeError(f"unexpected config fields: {unexpected_names}")
            for field_name in annotations:
                if field_name in data:
                    value = data[field_name]
                else:
                    value = getattr(type(self), field_name)
                setattr(self, field_name, value)

try:
    from sklearn.exceptions import ConvergenceWarning
except ImportError:
    ConvergenceWarning = Warning

SelectionRefitPolicy = Literal["train_only", "train_plus_val"]


class OptunaTuningConfig(BaseModel):
    n_trials: int = 50
    sampler: str | None = None
    pruner: str | None = None
    timeout_s: int | None = None
    seed: int | None = None


DEFAULT_TRAINING_EPOCH_CHOICES: tuple[int, ...] = (25, 50, 100)


def resolved_training_epochs(training_cfg: Any, trial: Any | None = None) -> int:
    configured_epochs = getattr(training_cfg, "epochs", None)
    if configured_epochs is not None:
        return int(configured_epochs)
    if trial is None:
        raise ValueError("trial is required when training epochs are not fixed.")
    return int(
        trial.suggest_categorical("epochs", list(DEFAULT_TRAINING_EPOCH_CHOICES))
    )


def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return float(np.mean((y_true_arr - y_pred_arr) ** 2))


def _fit_model_safely(model: object, X: np.ndarray, y: np.ndarray) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(X, y)


def _measure_duration_s(work: Callable[[], object]) -> float:
    t0 = perf_counter()
    work()
    return float(perf_counter() - t0)


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
    predictive_spread_extractor: Callable[[object, np.ndarray], np.ndarray] | None = None
    uncertainty_kind: str | None = None
    uncertainty_note: str | None = None

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
            candidate_factories.append(
                _estimator_factory_with_params(self.estimator_factory, params)
            )
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


@runtime_checkable
class TrialTuningSpec(Protocol):
    """Contract for validation-based trial tuning over one sweep split.

    Implementations define how one split is converted into a trial objective,
    how the chosen trial is refit for the final outer-test evaluation, and
    which metadata should be reported for the selected trial.
    """

    def build_trial_objective(
        self,
        split: TrainValTestSweepRunnerInput,
    ) -> Callable[[Any], float]: ...

    def fit_selected_model(
        self,
        split: TrainValTestSweepRunnerInput,
        best_trial: Any,
        *,
        refit_policy: SelectionRefitPolicy,
    ) -> object: ...

    def trial_metadata(
        self,
        best_trial: Any,
        model: object,
    ) -> Mapping[str, Any]: ...


@runtime_checkable
class LearnedTrialTuningSpec(Protocol):
    """Contract for learned-model trial tuning over split-aware dataset subsets.

    Implementations receive the full `TrainValTestSweepRunnerInput`, so they may
    derive framework-specific loaders from `split.dataset_subsets()`,
    `split.loader_inputs(...)`, or `split.loaders(...)` instead of slicing raw
    numpy arrays directly.

    Selection contract:
    - candidate ranking may use only train/val data
    - `fit_selected_model(...)` may refit on train only or train+val according
      to `refit_policy`
    - `predict(...)` is called exactly once on the held-out outer test subset
      after selection/refit
    - outer test targets must never influence candidate ranking
    """

    def build_trial_objective(
        self,
        split: TrainValTestSweepRunnerInput,
    ) -> Callable[[Any], float]: ...

    def fit_selected_model(
        self,
        split: TrainValTestSweepRunnerInput,
        best_trial: Any,
        *,
        refit_policy: SelectionRefitPolicy,
    ) -> object: ...

    def predict(
        self,
        model: object,
        dataset: SweepDataset,
    ) -> np.ndarray: ...

    def trial_metadata(
        self,
        best_trial: Any,
        model: object,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class SweepRunnerArtifacts:
    metrics: pd.DataFrame
    selection_metadata: pd.DataFrame | None = None
    uq_summary: pd.DataFrame | None = None


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


HyperparameterSelectionSweepRunner = SupervisedModelSelectionSweepRunner


@dataclass(frozen=True, slots=True)
class TrialModelSelectionSweepRunner:
    """Reusable runner for train/val/test trial-based model selection."""

    tuning_spec: TrialTuningSpec
    trial_factory: Callable[[TrainValTestSweepRunnerInput], Iterable[Any]]
    refit_policy: SelectionRefitPolicy = "train_plus_val"

    def run_with_validation(self, payload: SweepRunnerPayload) -> pd.DataFrame:
        return sweep_model_with_trial_tuning(
            payload,
            self.tuning_spec,
            self.trial_factory,
            refit_policy=self.refit_policy,
        )

    def run_artifacts_with_validation(
        self,
        payload: SweepRunnerPayload,
    ) -> SweepRunnerArtifacts:
        return sweep_trial_model_selection(
            payload,
            self.tuning_spec,
            self.trial_factory,
            refit_policy=self.refit_policy,
        )


@dataclass(frozen=True, slots=True)
class LearnedTrialModelSelectionSweepRunner:
    """Reusable runner for learned-model trial selection on split-aware datasets."""

    tuning_spec: LearnedTrialTuningSpec
    trial_factory: Callable[[TrainValTestSweepRunnerInput], Iterable[Any]]
    refit_policy: SelectionRefitPolicy = "train_plus_val"

    def run_with_validation(self, payload: SweepRunnerPayload) -> pd.DataFrame:
        return sweep_learned_model_with_trial_tuning(
            payload,
            self.tuning_spec,
            self.trial_factory,
            refit_policy=self.refit_policy,
        )

    def run_artifacts_with_validation(
        self,
        payload: SweepRunnerPayload,
    ) -> SweepRunnerArtifacts:
        return sweep_learned_trial_model_selection(
            payload,
            self.tuning_spec,
            self.trial_factory,
            refit_policy=self.refit_policy,
        )


def _default_optuna_study_factory(split: TrainValTestSweepRunnerInput) -> Any:
    del split
    import optuna

    return optuna.create_study(direction="minimize")


def study_factory_from_optuna_cfg(
    cfg: OptunaTuningConfig,
) -> Callable[[TrainValTestSweepRunnerInput], Any]:
    def factory(split: TrainValTestSweepRunnerInput) -> Any:
        del split
        import optuna

        sampler_name = cfg.sampler
        seed = cfg.seed
        if sampler_name is None or sampler_name == "tpe":
            sampler = optuna.samplers.TPESampler(seed=seed)
        elif sampler_name == "random":
            sampler = optuna.samplers.RandomSampler(seed=seed)
        elif sampler_name == "cmaes":
            sampler = optuna.samplers.CmaEsSampler(seed=seed)
        else:
            raise ValueError(f"Unknown Optuna sampler: {sampler_name!r}")

        pruner_name = cfg.pruner
        if pruner_name is None:
            return optuna.create_study(direction="minimize", sampler=sampler)
        if pruner_name == "median":
            pruner = optuna.pruners.MedianPruner()
        else:
            raise ValueError(f"Unknown Optuna pruner: {pruner_name!r}")
        return optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    return factory


@dataclass(frozen=True, slots=True)
class OptunaModelSelectionSweepRunner:
    """Reusable runner for Optuna-backed train/val/test model selection."""

    tuning_spec: TrialTuningSpec
    n_trials: int
    timeout_s: int | None = None
    study_factory: Callable[[TrainValTestSweepRunnerInput], Any] = (
        _default_optuna_study_factory
    )
    refit_policy: SelectionRefitPolicy = "train_plus_val"

    def run_with_validation(self, payload: SweepRunnerPayload) -> SweepRunnerArtifacts:
        return sweep_optuna_model_selection(
            payload,
            self.tuning_spec,
            n_trials=self.n_trials,
            timeout_s=self.timeout_s,
            study_factory=self.study_factory,
            refit_policy=self.refit_policy,
        )

    def run_artifacts_with_validation(
        self,
        payload: SweepRunnerPayload,
    ) -> SweepRunnerArtifacts:
        return sweep_optuna_model_selection(
            payload,
            self.tuning_spec,
            n_trials=self.n_trials,
            timeout_s=self.timeout_s,
            study_factory=self.study_factory,
            refit_policy=self.refit_policy,
        )


@dataclass(frozen=True, slots=True)
class LearnedOptunaModelSelectionSweepRunner:
    """Reusable runner for Optuna-backed learned-model selection."""

    tuning_spec: LearnedTrialTuningSpec
    n_trials: int
    timeout_s: int | None = None
    study_factory: Callable[[TrainValTestSweepRunnerInput], Any] = (
        _default_optuna_study_factory
    )
    refit_policy: SelectionRefitPolicy = "train_plus_val"

    def run_with_validation(self, payload: SweepRunnerPayload) -> SweepRunnerArtifacts:
        return sweep_learned_optuna_model_selection(
            payload,
            self.tuning_spec,
            n_trials=self.n_trials,
            timeout_s=self.timeout_s,
            study_factory=self.study_factory,
            refit_policy=self.refit_policy,
        )

    def run_artifacts_with_validation(
        self,
        payload: SweepRunnerPayload,
    ) -> SweepRunnerArtifacts:
        return sweep_learned_optuna_model_selection(
            payload,
            self.tuning_spec,
            n_trials=self.n_trials,
            timeout_s=self.timeout_s,
            study_factory=self.study_factory,
            refit_policy=self.refit_policy,
        )


def _as_runner_payload(
    payload: SweepRunPayload | SweepRunnerPayload,
) -> SweepRunnerPayload:
    if isinstance(payload, SweepRunnerPayload):
        return payload
    return payload.to_runner_payload()


def _assert_train_val_test_payload(
    payload: SweepRunnerPayload,
) -> tuple[TrainValTestSweepRunnerInput, ...]:
    splits = payload.splits
    if not all(isinstance(split, TrainValTestSweepRunnerInput) for split in splits):
        raise TypeError("expected train/val/test sweep payload")
    return splits


def _assert_train_val_cal_test_payload(
    payload: SweepRunnerPayload,
) -> tuple[TrainValCalTestSweepRunnerInput, ...]:
    splits = payload.splits
    if not all(isinstance(split, TrainValCalTestSweepRunnerInput) for split in splits):
        raise TypeError("expected train/val/cal/test sweep payload")
    return splits


def _effective_uncertainty_metadata(
    *,
    uncertainty_kind: str | None,
    uncertainty_note: str | None,
    requires_calibration: bool,
) -> tuple[str | None, str | None]:
    if requires_calibration and uncertainty_kind is not None:
        return "calibrated", "post-hoc scalar calibrated spread"
    return uncertainty_kind, uncertainty_note


def _estimator_factory_with_params(
    estimator_factory: Callable[..., object],
    params: Mapping[str, Any],
) -> Callable[[], object]:
    return lambda: estimator_factory(**params)


def _validate_selection_refit_policy(policy: SelectionRefitPolicy) -> None:
    if policy not in ("train_only", "train_plus_val"):
        raise ValueError(f"unsupported selection refit policy: {policy}")


def _select_candidate_factory_by_validation(
    split: TrainValTestSweepRunnerInput,
    hyperparameter_spec: HyperparameterSpec,
) -> Callable[[], object]:
    X = split.dataset.mlip_features
    y = split.dataset.targets
    best_candidate_factory = None
    best_rmse = np.inf
    for candidate_factory in hyperparameter_spec.candidate_factories():
        model = candidate_factory()
        _fit_model_safely(model, X[split.train_idx], y[split.train_idx])
        val_preds = model.predict(X[split.val_idx])
        val_rmse = np.sqrt(_mean_squared_error(y[split.val_idx], val_preds))
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
) -> tuple[object, Mapping[str, Any], float]:
    """Return the selected model after applying the configured refit policy."""

    _validate_selection_refit_policy(refit_policy)
    X = split.dataset.mlip_features
    y = split.dataset.targets
    candidate_factory = None
    model = None

    def select_and_fit() -> None:
        nonlocal candidate_factory, model
        candidate_factory = _select_candidate_factory_by_validation(
            split,
            hyperparameter_spec,
        )
        model = candidate_factory()
        if refit_policy == "train_only":
            fit_idx = split.train_idx
        else:
            fit_idx = np.concatenate([split.train_idx, split.val_idx])
        _fit_model_safely(model, X[fit_idx], y[fit_idx])

    fit_time_s = _measure_duration_s(select_and_fit)
    assert model is not None
    return model, hyperparameter_spec.selection_metadata(model), fit_time_s


def _selection_metadata_frame(
    metadata_by_size: dict[int, list[Mapping[str, Any]]],
) -> pd.DataFrame:
    rows = []
    for n_train, records in sorted(metadata_by_size.items()):
        row: dict[str, Any] = {"n_train": n_train}
        keys = tuple(sorted({key for record in records for key in record}))
        for key in keys:
            values = [record[key] for record in records if key in record]
            unique_values = []
            for value in values:
                if value not in unique_values:
                    unique_values.append(value)
            row[key] = unique_values[0] if len(unique_values) == 1 else tuple(unique_values)
        rows.append(row)
    return pd.DataFrame(rows)


def _timed_sweep_results_frame(
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
    return pd.DataFrame(
        rows,
        columns=[
            "n_train",
            "rmse_mean",
            "rmse_std",
            "fit_time_mean_s",
            "fit_time_std_s",
        ],
    )


def _uq_summary_frame(
    split_artifacts: list[dict[str, Any]],
    *,
    uncertainty_kind: str,
    uncertainty_note: str | None = None,
) -> pd.DataFrame:
    from oasis.learning_curve.execution import (
        aggregate_uq_summary,
        build_split_prediction_artifact,
    )

    summary = aggregate_uq_summary(
        [
            build_split_prediction_artifact(
                sweep_size=artifact["sweep_size"],
                y_true=artifact["y_true"],
                y_pred=artifact["y_pred"],
                spread=artifact["spread"],
            )
            for artifact in split_artifacts
        ],
        uncertainty_kind=uncertainty_kind,
    )
    if uncertainty_note is not None and not summary.empty:
        summary["uncertainty_note"] = uncertainty_note
    return summary


def _hyperparameter_spec_predictive_spread(
    hyperparameter_spec: HyperparameterSpec,
    model: object,
    X_test: np.ndarray,
) -> np.ndarray | None:
    extractor = getattr(hyperparameter_spec, "predictive_spread_extractor", None)
    if extractor is None:
        return None
    return np.asarray(extractor(model, X_test), dtype=float)


def _learned_tuning_spec_predictive_spread(
    tuning_spec: LearnedTrialTuningSpec,
    model: object,
    dataset: SweepDataset,
) -> np.ndarray | None:
    extractor = getattr(tuning_spec, "predictive_spread", None)
    if extractor is None:
        return None
    return np.asarray(extractor(model, dataset), dtype=float)


def _select_best_trial_by_validation(
    split: TrainValTestSweepRunnerInput,
    tuning_spec: TrialTuningSpec,
    trials: Iterable[Any],
) -> Any:
    objective = tuning_spec.build_trial_objective(split)
    best_trial = None
    best_value = np.inf
    for trial in trials:
        objective_value = objective(trial)
        if objective_value < best_value:
            best_value = objective_value
            best_trial = trial
    if best_trial is None:
        raise ValueError("trial tuning did not yield any candidate trials")
    return best_trial


def _fit_trial_selected_model(
    split: TrainValTestSweepRunnerInput,
    tuning_spec: TrialTuningSpec,
    trials: Iterable[Any],
    *,
    refit_policy: SelectionRefitPolicy,
) -> tuple[object, Mapping[str, Any]]:
    _validate_selection_refit_policy(refit_policy)
    best_trial = _select_best_trial_by_validation(split, tuning_spec, trials)
    model = tuning_spec.fit_selected_model(
        split,
        best_trial,
        refit_policy=refit_policy,
    )
    return model, tuning_spec.trial_metadata(best_trial, model)


def _fit_learned_trial_selected_model(
    split: TrainValTestSweepRunnerInput,
    tuning_spec: LearnedTrialTuningSpec,
    trials: Iterable[Any],
    *,
    refit_policy: SelectionRefitPolicy,
) -> tuple[object, Mapping[str, Any], float]:
    _validate_selection_refit_policy(refit_policy)
    best_trial = None
    model = None

    def select_and_fit() -> None:
        nonlocal best_trial, model
        best_trial = _select_best_trial_by_validation(split, tuning_spec, trials)
        model = tuning_spec.fit_selected_model(
            split,
            best_trial,
            refit_policy=refit_policy,
        )

    fit_time_s = _measure_duration_s(select_and_fit)
    assert best_trial is not None
    assert model is not None
    return model, tuning_spec.trial_metadata(best_trial, model), fit_time_s


def _optimize_study_best_trial(
    split: TrainValTestSweepRunnerInput,
    tuning_spec: TrialTuningSpec,
    *,
    n_trials: int,
    timeout_s: int | None,
    study_factory: Callable[[TrainValTestSweepRunnerInput], Any],
) -> tuple[Any, Any]:
    study = study_factory(split)
    objective = tuning_spec.build_trial_objective(split)
    study.optimize(objective, n_trials=n_trials, timeout=timeout_s)
    best_trial = getattr(study, "best_trial", None)
    if best_trial is None:
        raise ValueError("optuna study did not produce a best trial")
    return study, best_trial


def _fit_optuna_selected_model(
    split: TrainValTestSweepRunnerInput,
    tuning_spec: TrialTuningSpec,
    *,
    n_trials: int,
    timeout_s: int | None,
    study_factory: Callable[[TrainValTestSweepRunnerInput], Any],
    refit_policy: SelectionRefitPolicy,
) -> tuple[object, Mapping[str, Any]]:
    _validate_selection_refit_policy(refit_policy)
    study, best_trial = _optimize_study_best_trial(
        split,
        tuning_spec,
        n_trials=n_trials,
        timeout_s=timeout_s,
        study_factory=study_factory,
    )
    model = tuning_spec.fit_selected_model(
        split,
        best_trial,
        refit_policy=refit_policy,
    )
    return model, _optuna_selection_metadata(
        study,
        best_trial,
        model,
        tuning_spec,
    )


def _fit_learned_optuna_selected_model(
    split: TrainValTestSweepRunnerInput,
    tuning_spec: LearnedTrialTuningSpec,
    *,
    n_trials: int,
    timeout_s: int | None,
    study_factory: Callable[[TrainValTestSweepRunnerInput], Any],
    refit_policy: SelectionRefitPolicy,
) -> tuple[object, Mapping[str, Any], float]:
    _validate_selection_refit_policy(refit_policy)
    study = None
    best_trial = None
    model = None

    def select_and_fit() -> None:
        nonlocal study, best_trial, model
        study, best_trial = _optimize_study_best_trial(
            split,
            tuning_spec,
            n_trials=n_trials,
            timeout_s=timeout_s,
            study_factory=study_factory,
        )
        model = tuning_spec.fit_selected_model(
            split,
            best_trial,
            refit_policy=refit_policy,
        )

    fit_time_s = _measure_duration_s(select_and_fit)
    assert study is not None
    assert best_trial is not None
    assert model is not None
    return model, _optuna_selection_metadata(
        study,
        best_trial,
        model,
        tuning_spec,
    ), fit_time_s


def _optuna_selection_metadata(
    study: Any,
    best_trial: Any,
    model: object,
    tuning_spec: TrialTuningSpec,
) -> Mapping[str, Any]:
    metadata = dict(tuning_spec.trial_metadata(best_trial, model))
    params = getattr(best_trial, "params", None)
    if isinstance(params, Mapping):
        metadata.update(params)
    best_value = getattr(best_trial, "value", None)
    if best_value is not None:
        metadata.setdefault("best_validation_score", float(best_value))
    trials = getattr(study, "trials", None)
    if trials is not None:
        metadata.setdefault("trial_count", len(trials))
    sampler = getattr(study, "sampler", None)
    if sampler is not None:
        metadata.setdefault("sampler", type(sampler).__name__)
    pruner = getattr(study, "pruner", None)
    if pruner is not None:
        metadata.setdefault("pruner", type(pruner).__name__)
    return metadata


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
    if payload.planning_requirements.requires_calibration:
        splits = _assert_train_val_cal_test_payload(payload)
    else:
        splits = _assert_train_val_test_payload(payload)
    rmses_by_size: dict[int, list[float]] = {}
    fit_times_by_size: dict[int, list[float]] = {}
    metadata_by_size: dict[int, list[Mapping[str, Any]]] = {}
    uq_artifacts: list[dict[str, Any]] = []
    uncertainty_kind, uncertainty_note = _effective_uncertainty_metadata(
        uncertainty_kind=getattr(hyperparameter_spec, "uncertainty_kind", None),
        uncertainty_note=getattr(hyperparameter_spec, "uncertainty_note", None),
        requires_calibration=payload.planning_requirements.requires_calibration,
    )
    for split in splits:
        y = split.dataset.targets
        model, metadata, fit_time_s = _fit_selected_supervised_model(
            split,
            hyperparameter_spec,
            refit_policy=refit_policy,
        )
        X_test = split.dataset.mlip_features[split.test_idx]
        y_test = y[split.test_idx]
        preds = model.predict(X_test)
        rmse = np.sqrt(_mean_squared_error(y_test, preds))
        rmses_by_size.setdefault(split.sweep_size, []).append(rmse)
        fit_times_by_size.setdefault(split.sweep_size, []).append(fit_time_s)
        metadata_by_size.setdefault(split.sweep_size, []).append(metadata)
        spread = _hyperparameter_spec_predictive_spread(
            hyperparameter_spec,
            model,
            X_test,
        )
        if (
            spread is not None
            and uncertainty_kind is not None
            and isinstance(split, TrainValCalTestSweepRunnerInput)
        ):
            X_cal = split.dataset.mlip_features[split.cal_idx]
            y_cal = y[split.cal_idx]
            cal_preds = np.asarray(model.predict(X_cal), dtype=float)
            cal_spread = _hyperparameter_spec_predictive_spread(
                hyperparameter_spec,
                model,
                X_cal,
            )
            if cal_spread is None:
                raise ValueError("calibration-aware UQ requires predictive spread on cal split.")
            calibrator = ScalarSpreadCalibrator.fit(
                y_true=y_cal,
                y_pred=cal_preds,
                spread=cal_spread,
            )
            spread = calibrator.apply(spread)
        if spread is not None and uncertainty_kind is not None:
            uq_artifacts.append(
                {
                    "sweep_size": split.sweep_size,
                    "y_true": y_test,
                    "y_pred": preds,
                    "spread": spread,
                }
            )
    uq_summary = None
    if uq_artifacts and uncertainty_kind is not None:
        uq_summary = _uq_summary_frame(
            uq_artifacts,
            uncertainty_kind=uncertainty_kind,
            uncertainty_note=uncertainty_note,
        )
    return SweepRunnerArtifacts(
        metrics=_timed_sweep_results_frame(rmses_by_size, fit_times_by_size),
        selection_metadata=_selection_metadata_frame(metadata_by_size),
        uq_summary=uq_summary,
    )


def sweep_trial_model_selection(
    payload: SweepRunPayload | SweepRunnerPayload,
    tuning_spec: TrialTuningSpec,
    trial_factory: Callable[[TrainValTestSweepRunnerInput], Iterable[Any]],
    *,
    refit_policy: SelectionRefitPolicy = "train_plus_val",
) -> SweepRunnerArtifacts:
    """Select the best trial on validation, refit, then evaluate once on test."""

    payload = _as_runner_payload(payload)
    splits = _assert_train_val_test_payload(payload)
    rmses_by_size: dict[int, list[float]] = {}
    metadata_by_size: dict[int, list[Mapping[str, Any]]] = {}
    for split in splits:
        y = split.dataset.targets
        model, metadata = _fit_trial_selected_model(
            split,
            tuning_spec,
            trial_factory(split),
            refit_policy=refit_policy,
        )
        preds = model.predict(split.dataset.mlip_features[split.test_idx])
        rmse = np.sqrt(_mean_squared_error(y[split.test_idx], preds))
        rmses_by_size.setdefault(split.sweep_size, []).append(rmse)
        metadata_by_size.setdefault(split.sweep_size, []).append(metadata)
    return SweepRunnerArtifacts(
        metrics=_sweep_results_frame(rmses_by_size),
        selection_metadata=_selection_metadata_frame(metadata_by_size),
    )


def sweep_learned_trial_model_selection(
    payload: SweepRunPayload | SweepRunnerPayload,
    tuning_spec: LearnedTrialTuningSpec,
    trial_factory: Callable[[TrainValTestSweepRunnerInput], Iterable[Any]],
    *,
    refit_policy: SelectionRefitPolicy = "train_plus_val",
) -> SweepRunnerArtifacts:
    """Select learned-model trials on validation subsets, then evaluate on test."""

    payload = _as_runner_payload(payload)
    if payload.planning_requirements.requires_calibration:
        splits = _assert_train_val_cal_test_payload(payload)
    else:
        splits = _assert_train_val_test_payload(payload)
    rmses_by_size: dict[int, list[float]] = {}
    fit_times_by_size: dict[int, list[float]] = {}
    metadata_by_size: dict[int, list[Mapping[str, Any]]] = {}
    uq_artifacts: list[dict[str, Any]] = []
    uncertainty_kind, uncertainty_note = _effective_uncertainty_metadata(
        uncertainty_kind=getattr(tuning_spec, "uncertainty_kind", None),
        uncertainty_note=getattr(tuning_spec, "uncertainty_note", None),
        requires_calibration=payload.planning_requirements.requires_calibration,
    )
    for split in splits:
        test_dataset = split.dataset_subsets().test
        y_test = test_dataset.targets
        model, metadata, fit_time_s = _fit_learned_trial_selected_model(
            split,
            tuning_spec,
            trial_factory(split),
            refit_policy=refit_policy,
        )
        preds = np.asarray(tuning_spec.predict(model, test_dataset), dtype=float)
        rmse = np.sqrt(_mean_squared_error(y_test, preds))
        rmses_by_size.setdefault(split.sweep_size, []).append(rmse)
        fit_times_by_size.setdefault(split.sweep_size, []).append(fit_time_s)
        metadata_by_size.setdefault(split.sweep_size, []).append(metadata)
        spread = _learned_tuning_spec_predictive_spread(
            tuning_spec,
            model,
            test_dataset,
        )
        if (
            spread is not None
            and uncertainty_kind is not None
            and isinstance(split, TrainValCalTestSweepRunnerInput)
        ):
            cal_dataset = split.dataset_subsets().cal
            cal_preds = np.asarray(tuning_spec.predict(model, cal_dataset), dtype=float)
            cal_spread = _learned_tuning_spec_predictive_spread(
                tuning_spec,
                model,
                cal_dataset,
            )
            if cal_spread is None:
                raise ValueError("calibration-aware UQ requires predictive spread on cal split.")
            calibrator = ScalarSpreadCalibrator.fit(
                y_true=cal_dataset.targets,
                y_pred=cal_preds,
                spread=cal_spread,
            )
            spread = calibrator.apply(spread)
        if spread is not None and uncertainty_kind is not None:
            uq_artifacts.append(
                {
                    "sweep_size": split.sweep_size,
                    "y_true": y_test,
                    "y_pred": preds,
                    "spread": spread,
                }
            )
    uq_summary = None
    if uq_artifacts and uncertainty_kind is not None:
        uq_summary = _uq_summary_frame(
            uq_artifacts,
            uncertainty_kind=uncertainty_kind,
            uncertainty_note=uncertainty_note,
        )
    return SweepRunnerArtifacts(
        metrics=_timed_sweep_results_frame(rmses_by_size, fit_times_by_size),
        selection_metadata=_selection_metadata_frame(metadata_by_size),
        uq_summary=uq_summary,
    )


def sweep_optuna_model_selection(
    payload: SweepRunPayload | SweepRunnerPayload,
    tuning_spec: TrialTuningSpec,
    *,
    n_trials: int,
    timeout_s: int | None = None,
    study_factory: Callable[[TrainValTestSweepRunnerInput], Any] = (
        _default_optuna_study_factory
    ),
    refit_policy: SelectionRefitPolicy = "train_plus_val",
) -> SweepRunnerArtifacts:
    """Optimize trials on train/val, refit once, then evaluate once on outer test."""

    payload = _as_runner_payload(payload)
    splits = _assert_train_val_test_payload(payload)
    rmses_by_size: dict[int, list[float]] = {}
    metadata_by_size: dict[int, list[Mapping[str, Any]]] = {}
    for split in splits:
        y = split.dataset.targets
        model, metadata = _fit_optuna_selected_model(
            split,
            tuning_spec,
            n_trials=n_trials,
            timeout_s=timeout_s,
            study_factory=study_factory,
            refit_policy=refit_policy,
        )
        preds = model.predict(split.dataset.mlip_features[split.test_idx])
        rmse = np.sqrt(_mean_squared_error(y[split.test_idx], preds))
        rmses_by_size.setdefault(split.sweep_size, []).append(rmse)
        metadata_by_size.setdefault(split.sweep_size, []).append(metadata)
    return SweepRunnerArtifacts(
        metrics=_sweep_results_frame(rmses_by_size),
        selection_metadata=_selection_metadata_frame(metadata_by_size),
    )


def sweep_learned_optuna_model_selection(
    payload: SweepRunPayload | SweepRunnerPayload,
    tuning_spec: LearnedTrialTuningSpec,
    *,
    n_trials: int,
    timeout_s: int | None = None,
    study_factory: Callable[[TrainValTestSweepRunnerInput], Any] = (
        _default_optuna_study_factory
    ),
    refit_policy: SelectionRefitPolicy = "train_plus_val",
) -> SweepRunnerArtifacts:
    """Optimize learned-model trials on train/val subsets, then test once."""

    payload = _as_runner_payload(payload)
    if payload.planning_requirements.requires_calibration:
        splits = _assert_train_val_cal_test_payload(payload)
    else:
        splits = _assert_train_val_test_payload(payload)
    rmses_by_size: dict[int, list[float]] = {}
    fit_times_by_size: dict[int, list[float]] = {}
    metadata_by_size: dict[int, list[Mapping[str, Any]]] = {}
    uq_artifacts: list[dict[str, Any]] = []
    uncertainty_kind, uncertainty_note = _effective_uncertainty_metadata(
        uncertainty_kind=getattr(tuning_spec, "uncertainty_kind", None),
        uncertainty_note=getattr(tuning_spec, "uncertainty_note", None),
        requires_calibration=payload.planning_requirements.requires_calibration,
    )
    for split in splits:
        test_dataset = split.dataset_subsets().test
        y_test = test_dataset.targets
        model, metadata, fit_time_s = _fit_learned_optuna_selected_model(
            split,
            tuning_spec,
            n_trials=n_trials,
            timeout_s=timeout_s,
            study_factory=study_factory,
            refit_policy=refit_policy,
        )
        preds = np.asarray(tuning_spec.predict(model, test_dataset), dtype=float)
        rmse = np.sqrt(_mean_squared_error(y_test, preds))
        rmses_by_size.setdefault(split.sweep_size, []).append(rmse)
        fit_times_by_size.setdefault(split.sweep_size, []).append(fit_time_s)
        metadata_by_size.setdefault(split.sweep_size, []).append(metadata)
        spread = _learned_tuning_spec_predictive_spread(
            tuning_spec,
            model,
            test_dataset,
        )
        if (
            spread is not None
            and uncertainty_kind is not None
            and isinstance(split, TrainValCalTestSweepRunnerInput)
        ):
            cal_dataset = split.dataset_subsets().cal
            cal_preds = np.asarray(tuning_spec.predict(model, cal_dataset), dtype=float)
            cal_spread = _learned_tuning_spec_predictive_spread(
                tuning_spec,
                model,
                cal_dataset,
            )
            if cal_spread is None:
                raise ValueError("calibration-aware UQ requires predictive spread on cal split.")
            calibrator = ScalarSpreadCalibrator.fit(
                y_true=cal_dataset.targets,
                y_pred=cal_preds,
                spread=cal_spread,
            )
            spread = calibrator.apply(spread)
        if spread is not None and uncertainty_kind is not None:
            uq_artifacts.append(
                {
                    "sweep_size": split.sweep_size,
                    "y_true": y_test,
                    "y_pred": preds,
                    "spread": spread,
                }
            )
    uq_summary = None
    if uq_artifacts and uncertainty_kind is not None:
        uq_summary = _uq_summary_frame(
            uq_artifacts,
            uncertainty_kind=uncertainty_kind,
            uncertainty_note=uncertainty_note,
        )
    return SweepRunnerArtifacts(
        metrics=_timed_sweep_results_frame(rmses_by_size, fit_times_by_size),
        selection_metadata=_selection_metadata_frame(metadata_by_size),
        uq_summary=uq_summary,
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


def sweep_model_with_trial_tuning(
    payload: SweepRunPayload | SweepRunnerPayload,
    tuning_spec: TrialTuningSpec,
    trial_factory: Callable[[TrainValTestSweepRunnerInput], Iterable[Any]],
    *,
    refit_policy: SelectionRefitPolicy = "train_plus_val",
) -> pd.DataFrame:
    return sweep_trial_model_selection(
        payload,
        tuning_spec,
        trial_factory,
        refit_policy=refit_policy,
    ).metrics


def sweep_learned_model_with_trial_tuning(
    payload: SweepRunPayload | SweepRunnerPayload,
    tuning_spec: LearnedTrialTuningSpec,
    trial_factory: Callable[[TrainValTestSweepRunnerInput], Iterable[Any]],
    *,
    refit_policy: SelectionRefitPolicy = "train_plus_val",
) -> pd.DataFrame:
    return sweep_learned_trial_model_selection(
        payload,
        tuning_spec,
        trial_factory,
        refit_policy=refit_policy,
    ).metrics


def sweep_model_with_optuna_selection(
    payload: SweepRunPayload | SweepRunnerPayload,
    tuning_spec: TrialTuningSpec,
    *,
    n_trials: int,
    timeout_s: int | None = None,
    study_factory: Callable[[TrainValTestSweepRunnerInput], Any] = (
        _default_optuna_study_factory
    ),
    refit_policy: SelectionRefitPolicy = "train_plus_val",
) -> pd.DataFrame:
    return sweep_optuna_model_selection(
        payload,
        tuning_spec,
        n_trials=n_trials,
        timeout_s=timeout_s,
        study_factory=study_factory,
        refit_policy=refit_policy,
    ).metrics


def sweep_learned_model_with_optuna_selection(
    payload: SweepRunPayload | SweepRunnerPayload,
    tuning_spec: LearnedTrialTuningSpec,
    *,
    n_trials: int,
    timeout_s: int | None = None,
    study_factory: Callable[[TrainValTestSweepRunnerInput], Any] = (
        _default_optuna_study_factory
    ),
    refit_policy: SelectionRefitPolicy = "train_plus_val",
) -> pd.DataFrame:
    return sweep_learned_optuna_model_selection(
        payload,
        tuning_spec,
        n_trials=n_trials,
        timeout_s=timeout_s,
        study_factory=study_factory,
        refit_policy=refit_policy,
    ).metrics


def _sweep_results_frame(rmses_by_size: dict[int, list[float]]) -> pd.DataFrame:
    rows = [
        {
            "n_train": n_train,
            "rmse_mean": float(np.mean(rmses)),
            "rmse_std": float(np.std(rmses)),
        }
        for n_train, rmses in sorted(rmses_by_size.items())
    ]
    return pd.DataFrame(rows, columns=["n_train", "rmse_mean", "rmse_std"])
