from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class AdslabCandidate:
    """Canonical per-adslab record shared by all ranking methods."""

    parent_slab_id: str
    adslab_id: str
    predicted_binding_energy: float | None = None
    uncertainty: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ParentCandidate:
    """Canonical per-parent candidate emitted after reduction."""

    parent_slab_id: str
    selected_adslab_id: str
    predicted_binding_energy: float | None = None
    uncertainty: float | None = None
    score: float | None = None
    adslab_metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RankingContext:
    """Method-agnostic inputs for zero-shot, few-shot, and iterative ranking."""

    shot_count: int = 0
    target_binding_energy: float | None = None
    dataset_metadata: dict[str, Any] = field(default_factory=dict)
    candidate_records: tuple[dict[str, Any], ...] = ()
    prior_observations: tuple[dict[str, Any], ...] = ()
    method_config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RankingResult:
    """Complete output from one ranking strategy execution."""

    strategy_name: str
    adslab_candidates: tuple[AdslabCandidate, ...]
    parent_candidates: tuple[ParentCandidate, ...]
    ranked_candidates: tuple[ParentCandidate, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
