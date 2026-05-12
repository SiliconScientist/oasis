from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from oasis.sweep import (
    LoaderAdapterInput,
    SweepBatch,
    SweepDataset,
    SweepDatasetBatchLoaderAdapter,
    TrainEvalLoaderPolicy,
    TrainValTestSweepRunnerInput,
)
from oasis.tune import SelectionRefitPolicy
from sklearn.metrics import mean_squared_error


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
