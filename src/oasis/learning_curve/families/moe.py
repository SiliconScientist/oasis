from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from oasis.learning_curve.families.gating_policy import DenseGatingPolicy, GatingPolicy
from oasis.learning_curve.execution import require_min_mlip_feature_count
from oasis.sweep import SweepDataset, TrainValTestSweepRunnerInput
from oasis.tune import SelectionRefitPolicy


@dataclass(frozen=True, slots=True)
class MoEModel:
    weights: np.ndarray
    bias: float = 0.0
    policy: GatingPolicy = field(default_factory=DenseGatingPolicy)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias


@dataclass(frozen=True, slots=True)
class MlipBaselineGateTuningSpec:
    policy: GatingPolicy = field(default_factory=DenseGatingPolicy)
    uncertainty_kind: str = "spread_only"
    uncertainty_note: str | None = None

    def build_trial_objective(
        self,
        split: TrainValTestSweepRunnerInput,
    ) -> Callable[[Any], float]:
        subsets = split.dataset_subsets()
        X_val = subsets.val.mlip_features
        y_val = subsets.val.targets
        n_experts = require_min_mlip_feature_count(
            split.dataset.mlip_features,
            min_features=2,
            method_name="moe",
        )
        policy = self.policy

        def objective(trial: Any) -> float:
            logits = np.array(
                [trial.suggest_float(f"logit_{i}", -5.0, 5.0) for i in range(n_experts)]
            )
            residuals = y_val - (X_val @ policy.apply(logits))
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
        n_experts = require_min_mlip_feature_count(
            split.dataset.mlip_features,
            min_features=2,
            method_name="moe",
        )
        logits = np.array([best_trial.params[f"logit_{i}"] for i in range(n_experts)])
        weights = self.policy.apply(logits)
        if refit_policy == "train_only":
            X = subsets.train.mlip_features
            y = subsets.train.targets
        else:
            X = np.concatenate([subsets.train.mlip_features, subsets.val.mlip_features])
            y = np.concatenate([subsets.train.targets, subsets.val.targets])
        bias = float(np.mean(y - (X @ weights)))
        return MoEModel(weights=weights, bias=bias, policy=self.policy)

    def predict(
        self,
        model: MoEModel,
        dataset: SweepDataset,
    ) -> np.ndarray:
        return model.predict(dataset.mlip_features)

    def predictive_spread(
        self,
        model: MoEModel,
        dataset: SweepDataset,
    ) -> np.ndarray:
        X = np.asarray(dataset.mlip_features, dtype=float)
        preds = model.predict(X)
        centered = X - preds[:, None]
        return np.sqrt(np.sum(model.weights[None, :] * centered**2, axis=1))

    def trial_metadata(
        self,
        best_trial: Any,
        model: MoEModel,
    ) -> dict[str, Any]:
        return {f"weight_{i}": float(w) for i, w in enumerate(model.weights)}
