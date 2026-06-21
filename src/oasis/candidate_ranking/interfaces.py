from __future__ import annotations

from typing import Protocol, runtime_checkable

from oasis.candidate_ranking.types import (
    AdslabCandidate,
    ParentCandidate,
    RankingContext,
    RankingResult,
)


@runtime_checkable
class CandidateGenerator(Protocol):
    """Extension point for method-specific adslab candidate generation.

    Implementations may use MLIP ensembles, learned regressors, or
    registry-backed predictors to emit canonical per-adslab candidates.
    """

    def generate(
        self,
        context: RankingContext,
    ) -> list[AdslabCandidate]: ...


@runtime_checkable
class ParentReducer(Protocol):
    """Extension point for parent-level candidate reduction policies."""

    def reduce(
        self,
        candidates: list[AdslabCandidate],
        context: RankingContext,
    ) -> list[ParentCandidate]: ...


@runtime_checkable
class CandidateScorer(Protocol):
    """Extension point for ranking policies over reduced parent candidates.

    Scorers may combine target proximity, uncertainty, acquisition value, or
    method-specific signals.
    """

    def score(
        self,
        candidates: list[ParentCandidate],
        context: RankingContext,
    ) -> list[ParentCandidate]: ...


@runtime_checkable
class MethodSignalProvider(Protocol):
    """Optional extension point for registry-backed auxiliary method signals."""

    def build_signals(self, context: RankingContext) -> dict[str, object]: ...


@runtime_checkable
class CandidatePredictor(Protocol):
    """Extension point for fitted n-shot predictors over validated references.

    Predictors such as `residual`, `weighted_simplex`, and `ridge` should plug
    in here by registration rather than by branching orchestration code.
    """

    name: str
    min_validated_references: int

    def is_feasible(self, validated_reference_count: int) -> bool: ...


@runtime_checkable
class RankingStrategy(Protocol):
    """Extension point for zero-shot, few-shot, and n-shot ranking workflows."""

    name: str

    def rank(self, context: RankingContext) -> RankingResult: ...
