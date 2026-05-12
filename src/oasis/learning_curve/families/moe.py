from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from oasis.sweep import SweepDataset, TrainValTestSweepRunnerInput
from oasis.tune import SelectionRefitPolicy


@dataclass(frozen=True, slots=True)
class MoEModel:
    weights: np.ndarray
    bias: float = 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


def _weights_from_trial(params: dict[str, Any], n_experts: int) -> np.ndarray:
    return _softmax(np.array([params[f"logit_{i}"] for i in range(n_experts)]))


@dataclass(frozen=True, slots=True)
class MoELearnedTrialTuningSpec:
    def build_trial_objective(
        self,
        split: TrainValTestSweepRunnerInput,
    ) -> Callable[[Any], float]:
        subsets = split.dataset_subsets()
        X_val = subsets.val.mlip_features
        y_val = subsets.val.targets
        n_experts = X_val.shape[1]

        def objective(trial: Any) -> float:
            logits = np.array(
                [trial.suggest_float(f"logit_{i}", -5.0, 5.0) for i in range(n_experts)]
            )
            residuals = y_val - (X_val @ _softmax(logits))
            return float(np.sqrt(np.mean(residuals**2)))

        return objective

    def fit_selected_model(
        self,
        split: TrainValTestSweepRunnerInput,
        best_trial: Any,
        *,
        refit_policy: SelectionRefitPolicy,
    ) -> MoEModel:
        subsets = split.dataset_subsets()
        n_experts = split.dataset.mlip_features.shape[1]
        weights = _weights_from_trial(best_trial.params, n_experts)
        if refit_policy == "train_only":
            X = subsets.train.mlip_features
            y = subsets.train.targets
        else:
            X = np.concatenate([subsets.train.mlip_features, subsets.val.mlip_features])
            y = np.concatenate([subsets.train.targets, subsets.val.targets])
        bias = float(np.mean(y - (X @ weights)))
        return MoEModel(weights=weights, bias=bias)

    def predict(
        self,
        model: MoEModel,
        dataset: SweepDataset,
    ) -> np.ndarray:
        return model.predict(dataset.mlip_features)

    def trial_metadata(
        self,
        best_trial: Any,
        model: MoEModel,
    ) -> dict[str, Any]:
        return {f"weight_{i}": float(w) for i, w in enumerate(model.weights)}
