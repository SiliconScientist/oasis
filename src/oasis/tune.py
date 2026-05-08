from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from itertools import product
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from oasis.sweep import SweepRunnerPayload, SweepRunPayload, TrainValTestSweepRunnerInput
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


def sweep_supervised_model_selection_trimmed(
    payload: SweepRunPayload | SweepRunnerPayload,
    hyperparameter_spec: HyperparameterSpec,
    *,
    refit_policy: SelectionRefitPolicy = "train_plus_val",
    z_thresh: float = 1.0,
) -> SweepRunnerArtifacts:
    """Validation-based model selection with test-time trimming by contribution z-score."""

    payload = _as_runner_payload(payload)
    splits = _assert_train_val_test_payload(payload)
    rmses_by_size: dict[int, list[float]] = {}
    metadata_by_size: dict[int, list[Mapping[str, Any]]] = {}
    for split in splits:
        X = split.dataset.mlip_features
        y = split.dataset.targets
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
