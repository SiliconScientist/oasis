from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class MlipAnomalyMetadata:
    """Normalized anomaly payload for one MLIP prediction."""

    label: str
    labels: tuple[str, ...] = ()
    details: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MlipModelPrediction:
    """One model-specific prediction carried into candidate screening."""

    model_name: str
    result_path: str
    predicted_binding_energy: float | None = None
    single_point_binding_energy: float | None = None
    anomaly: MlipAnomalyMetadata | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScreeningInputRecord:
    """Method-agnostic per-adslab input shared by ranking strategies."""

    reaction: str
    parent_slab_id: str
    adslab_id: str
    adsorbate: str | None = None
    reference_binding_energy: float | None = None
    site_metadata: dict[str, Any] = field(default_factory=dict)
    slab_metadata: dict[str, Any] = field(default_factory=dict)
    adslab_metadata: dict[str, Any] = field(default_factory=dict)
    shared_metadata: dict[str, Any] = field(default_factory=dict)
    model_predictions: tuple[MlipModelPrediction, ...] = ()


@dataclass(frozen=True, slots=True)
class MethodProvenance:
    """Method-level provenance for one candidate or derived signal."""

    method_name: str
    stage: str
    shot_count: int = 0
    source_methods: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class UncertaintyEstimate:
    """Canonical uncertainty payload with explicit provenance."""

    value: float
    metric: str
    provenance: MethodProvenance
    is_calibrated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SupportingSignal:
    """Optional auxiliary signal used by current or future ranking methods."""

    name: str
    value: float
    provenance: MethodProvenance
    objective: str = "minimize"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AdslabCandidate:
    """Canonical per-adslab record shared by all ranking methods.

    The shape is intentionally broad enough for zero-shot heuristics, few-shot
    learned scores, and iterative registry-backed methods without forcing later
    schema changes.
    """

    parent_slab_id: str
    adslab_id: str
    predicted_binding_energy: float | None = None
    method_provenance: MethodProvenance | None = None
    uncertainty: UncertaintyEstimate | None = None
    supporting_signals: tuple[SupportingSignal, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ParentCandidate:
    """Canonical per-parent candidate emitted after reduction.

    This record preserves the selected child identity plus method, uncertainty,
    and auxiliary signal hooks so later ranking methods can add learned or
    acquisition-style scores without changing the public contract.
    """

    parent_slab_id: str
    selected_adslab_id: str
    predicted_binding_energy: float | None = None
    method_provenance: MethodProvenance | None = None
    uncertainty: UncertaintyEstimate | None = None
    score: float | None = None
    supporting_signals: tuple[SupportingSignal, ...] = ()
    selected_adslab_ids: tuple[str, ...] = ()
    adslab_metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RankingContext:
    """Method-agnostic inputs for zero-shot, few-shot, and iterative ranking."""

    shot_count: int = 0
    target_binding_energy: float | None = None
    dataset_metadata: dict[str, Any] = field(default_factory=dict)
    candidate_records: tuple[ScreeningInputRecord, ...] = ()
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
