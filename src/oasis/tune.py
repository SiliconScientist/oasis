from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from itertools import product
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from oasis.sweep import (
    SweepDataset,
    SweepRunnerPayload,
    SweepRunPayload,
    TrainValTestSweepRunnerInput,
)
from pydantic import BaseModel
from sklearn.metrics import mean_squared_error

SelectionRefitPolicy = Literal["train_only", "train_plus_val"]


class OptunaTuningConfig(BaseModel):
    n_trials: int = 50
    sampler: str | None = None
    pruner: str | None = None
    timeout_s: int | None = None
    seed: int | None = None


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
    """Contract for learned-model trial tuning over split-aware dataset subsets."""

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
    X = split.dataset.mlip_features
    y = split.dataset.targets
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
) -> tuple[object, Mapping[str, Any]]:
    _validate_selection_refit_policy(refit_policy)
    best_trial = _select_best_trial_by_validation(split, tuning_spec, trials)
    model = tuning_spec.fit_selected_model(
        split,
        best_trial,
        refit_policy=refit_policy,
    )
    return model, tuning_spec.trial_metadata(best_trial, model)


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
    splits = _assert_train_val_test_payload(payload)
    rmses_by_size: dict[int, list[float]] = {}
    metadata_by_size: dict[int, list[Mapping[str, Any]]] = {}
    for split in splits:
        y = split.dataset.targets
        model, metadata = _fit_selected_supervised_model(
            split,
            hyperparameter_spec,
            refit_policy=refit_policy,
        )
        preds = model.predict(split.dataset.mlip_features[split.test_idx])
        rmse = np.sqrt(mean_squared_error(y[split.test_idx], preds))
        rmses_by_size.setdefault(split.sweep_size, []).append(rmse)
        metadata_by_size.setdefault(split.sweep_size, []).append(metadata)
    return SweepRunnerArtifacts(
        metrics=_sweep_results_frame(rmses_by_size),
        selection_metadata=_selection_metadata_frame(metadata_by_size),
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
        rmse = np.sqrt(mean_squared_error(y[split.test_idx], preds))
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
    splits = _assert_train_val_test_payload(payload)
    rmses_by_size: dict[int, list[float]] = {}
    metadata_by_size: dict[int, list[Mapping[str, Any]]] = {}
    for split in splits:
        test_dataset = split.dataset_subsets().test
        y_test = test_dataset.targets
        model, metadata = _fit_learned_trial_selected_model(
            split,
            tuning_spec,
            trial_factory(split),
            refit_policy=refit_policy,
        )
        preds = np.asarray(tuning_spec.predict(model, test_dataset), dtype=float)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        rmses_by_size.setdefault(split.sweep_size, []).append(rmse)
        metadata_by_size.setdefault(split.sweep_size, []).append(metadata)
    return SweepRunnerArtifacts(
        metrics=_sweep_results_frame(rmses_by_size),
        selection_metadata=_selection_metadata_frame(metadata_by_size),
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
        rmse = np.sqrt(mean_squared_error(y[split.test_idx], preds))
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
    splits = _assert_train_val_test_payload(payload)
    rmses_by_size: dict[int, list[float]] = {}
    metadata_by_size: dict[int, list[Mapping[str, Any]]] = {}
    for split in splits:
        test_dataset = split.dataset_subsets().test
        y_test = test_dataset.targets
        model, metadata = _fit_learned_optuna_selected_model(
            split,
            tuning_spec,
            n_trials=n_trials,
            timeout_s=timeout_s,
            study_factory=study_factory,
            refit_policy=refit_policy,
        )
        preds = np.asarray(tuning_spec.predict(model, test_dataset), dtype=float)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        rmses_by_size.setdefault(split.sweep_size, []).append(rmse)
        metadata_by_size.setdefault(split.sweep_size, []).append(metadata)
    return SweepRunnerArtifacts(
        metrics=_sweep_results_frame(rmses_by_size),
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
