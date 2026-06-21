from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterable

from oasis.candidate_ranking.interfaces import CandidatePredictor

_PREDICTOR_REGISTRY: dict[str, CandidatePredictor] = {}


@dataclass(frozen=True, slots=True)
class PredictorSpec:
    """Lightweight registry record for one candidate-ranking predictor."""

    name: str
    min_validated_references: int
    metadata: dict[str, object] = field(default_factory=dict)

    def is_feasible(self, validated_reference_count: int) -> bool:
        return validated_reference_count >= self.min_validated_references


def register_predictor(predictor: CandidatePredictor) -> CandidatePredictor:
    """Register a predictor by name.

    This corrects the abstraction boundary: extension happens by adding fitted
    predictors such as `residual`, `weighted_simplex`, or `ridge`, while
    zero-shot remains the unfitted fallback when no validated references exist.
    """

    name = getattr(predictor, "name", None)
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Candidate predictors must define a non-empty string name.")
    if name in _PREDICTOR_REGISTRY:
        raise ValueError(f"Candidate predictor {name!r} is already registered.")
    _PREDICTOR_REGISTRY[name] = predictor
    return predictor


def register_builtin_predictors() -> None:
    """Register built-in predictor specs.

    New few-shot predictors should plug in by registration here or in a
    caller-owned bootstrap path, rather than by branching on shot regime.
    """
    builtins = (
        PredictorSpec(
            name="residual",
            min_validated_references=1,
            metadata={"family": "few_shot", "hyperparameter_free": True},
        ),
        PredictorSpec(
            name="weighted_simplex",
            min_validated_references=1,
            metadata={"family": "few_shot", "hyperparameter_free": True},
        ),
        PredictorSpec(
            name="ridge",
            min_validated_references=2,
            metadata={"family": "few_shot", "hyperparameter_free": False},
        ),
    )
    for predictor in builtins:
        if predictor.name not in _PREDICTOR_REGISTRY:
            register_predictor(predictor)


def ensure_predictor(name: str) -> CandidatePredictor:
    """Resolve one predictor, lazily registering built-ins on first use."""

    if name not in _PREDICTOR_REGISTRY:
        register_builtin_predictors()
    return get_predictor(name)


def get_predictor(name: str) -> CandidatePredictor:
    """Return one registered predictor by name."""

    try:
        return _PREDICTOR_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(_PREDICTOR_REGISTRY)) or "<none>"
        raise KeyError(
            f"Unknown candidate predictor {name!r}. Registered predictors: {known}"
        ) from exc


def registered_predictor_names() -> tuple[str, ...]:
    """Return registered predictor names in sorted order."""

    return tuple(sorted(_PREDICTOR_REGISTRY))


def clear_registered_predictors() -> None:
    """Test helper for resetting the predictor registry."""

    _PREDICTOR_REGISTRY.clear()


def register_predictors(predictors: Iterable[CandidatePredictor]) -> None:
    """Register several predictors."""

    for predictor in predictors:
        register_predictor(predictor)
