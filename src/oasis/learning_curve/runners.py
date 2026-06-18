from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from oasis.learning_curve.execution import (
    _normalize_runner_output,
    _select_runner_call,
    sweep_learned_model,
    sweep_learned_model_with_validation,
    sweep_model,
    sweep_model_with_validation,
    weighted_linear_sweep,
    weighted_simplex_sweep,
)
from oasis.sweep import (
    LearningCurveResults,
    SweepFamilyRequirements,
    SweepModelCapabilities,
    SweepRunPayload,
    SweepRunnerPayload,
    TrainTestSweepRunnerInput,
    TrainValTestSweepRunnerInput,
)
from oasis.tune import SweepRunnerArtifacts, ValidationAwareEstimator


class SweepModelFamily(Protocol):
    """Contract for a model family runnable over a shared split sweep."""

    def capabilities(self) -> SweepModelCapabilities: ...

    def requirements(self) -> SweepFamilyRequirements: ...

    def run(self, payload: SweepRunPayload) -> LearningCurveResults: ...


@runtime_checkable
class SweepExperimentRunner(Protocol):
    """Common runner interface for train/test sweep methods."""

    def run(self, payload: SweepRunnerPayload) -> pd.DataFrame | SweepRunnerArtifacts: ...


@runtime_checkable
class ValidationAwareSweepExperimentRunner(Protocol):
    """Runner interface for methods that select on val and evaluate on outer test."""

    def run_with_validation(
        self,
        payload: SweepRunnerPayload,
    ) -> pd.DataFrame | SweepRunnerArtifacts: ...


@runtime_checkable
class TrainTestLearnedEstimator(Protocol):
    """Estimator that trains from the full train/test split input."""

    def fit(self, split: TrainTestSweepRunnerInput) -> None: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class TrainValTestLearnedEstimator(Protocol):
    """Estimator that trains from the full train/val/test split input."""

    def fit(self, split: TrainValTestSweepRunnerInput) -> None: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class SweepFamilySpec:
    result_field: str
    runner: SweepExperimentRunner | ValidationAwareSweepExperimentRunner
    selection_metadata_field: str | None = None
    uq_summary_field: str | None = None
    capabilities: SweepModelCapabilities = field(default_factory=SweepModelCapabilities)


@dataclass(frozen=True, slots=True)
class SupervisedModelSweepRunner:
    """Reusable adapter for supervised estimators over the shared sweep payload."""

    model_factory: Callable[[], object]

    def run(self, payload: SweepRunnerPayload) -> pd.DataFrame:
        return sweep_model(payload, self.model_factory)


@dataclass(frozen=True, slots=True)
class FunctionalSweepRunner:
    base_runner: Callable[[SweepRunnerPayload], pd.DataFrame | SweepRunnerArtifacts]

    def run(self, payload: SweepRunnerPayload) -> pd.DataFrame | SweepRunnerArtifacts:
        return self.base_runner(payload)


@dataclass(frozen=True, slots=True)
class ValidationAwareSupervisedModelSweepRunner:
    """Adapter for estimators that tune on val before outer-test evaluation."""

    model_factory: Callable[[], ValidationAwareEstimator]

    def run_with_validation(self, payload: SweepRunnerPayload) -> pd.DataFrame:
        return sweep_model_with_validation(payload, self.model_factory)


@dataclass(frozen=True, slots=True)
class LearnedModelSweepRunner:
    """Adapter for learned estimators that need the full train/test split input."""

    model_factory: Callable[[], TrainTestLearnedEstimator]

    def run(self, payload: SweepRunnerPayload) -> pd.DataFrame:
        return sweep_learned_model(payload, self.model_factory)


@dataclass(frozen=True, slots=True)
class ValidationAwareLearnedModelSweepRunner:
    """Adapter for learned estimators that need the full train/val/test split input."""

    model_factory: Callable[[], TrainValTestLearnedEstimator]

    def run_with_validation(self, payload: SweepRunnerPayload) -> pd.DataFrame:
        return sweep_learned_model_with_validation(payload, self.model_factory)


@dataclass(frozen=True, slots=True)
class WeightedLinearSweepRunner:
    fit_intercept: bool = True

    def run(self, payload: SweepRunnerPayload) -> pd.DataFrame | SweepRunnerArtifacts:
        return weighted_linear_sweep(
            payload,
            fit_intercept=self.fit_intercept,
        )


@dataclass(frozen=True, slots=True)
class WeightedSimplexSweepRunner:
    def run(self, payload: SweepRunnerPayload) -> pd.DataFrame | SweepRunnerArtifacts:
        return weighted_simplex_sweep(payload)


@dataclass(frozen=True, slots=True)
class ConfiguredSweepModelFamily:
    spec: SweepFamilySpec

    def capabilities(self) -> SweepModelCapabilities:
        return self.spec.capabilities

    def requirements(self) -> SweepFamilyRequirements:
        return self.spec.capabilities.to_requirements()

    def run(self, payload: SweepRunPayload) -> LearningCurveResults:
        runner_payload = payload.to_runner_payload()
        run = _select_runner_call(self.spec.runner, runner_payload)
        base_output = _normalize_runner_output(run(runner_payload))
        results: dict[str, pd.DataFrame | None] = {
            self.spec.result_field: base_output.metrics,
        }
        if self.spec.selection_metadata_field is not None:
            results[self.spec.selection_metadata_field] = base_output.selection_metadata
        if self.spec.uq_summary_field is not None:
            results[self.spec.uq_summary_field] = base_output.uq_summary
        return LearningCurveResults.from_mapping(results)


@dataclass(frozen=True, slots=True)
class PlaceholderLearnedSweepModelFamily:
    name: str
    declared_capabilities: SweepModelCapabilities = field(
        default_factory=SweepModelCapabilities
    )

    def capabilities(self) -> SweepModelCapabilities:
        return self.declared_capabilities

    def requirements(self) -> SweepFamilyRequirements:
        return self.declared_capabilities.to_requirements()

    def run(self, payload: SweepRunPayload) -> LearningCurveResults:
        del payload
        raise NotImplementedError(
            f"learning-curve family '{self.name}' is registered but not implemented"
        )
