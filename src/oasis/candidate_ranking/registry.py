from __future__ import annotations

from collections.abc import Iterable

from oasis.candidate_ranking.interfaces import RankingStrategy

_STRATEGY_REGISTRY: dict[str, RankingStrategy] = {}


def register_strategy(strategy: RankingStrategy) -> RankingStrategy:
    """Register a ranking strategy by name.

    New ranking methods should register here rather than branching orchestration
    code elsewhere in the pipeline.
    """

    name = getattr(strategy, "name", None)
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Ranking strategies must define a non-empty string name.")
    if name in _STRATEGY_REGISTRY:
        raise ValueError(f"Ranking strategy {name!r} is already registered.")
    _STRATEGY_REGISTRY[name] = strategy
    return strategy


def get_strategy(name: str) -> RankingStrategy:
    """Return one registered strategy by name."""

    try:
        return _STRATEGY_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(_STRATEGY_REGISTRY)) or "<none>"
        raise KeyError(
            f"Unknown ranking strategy {name!r}. Registered strategies: {known}"
        ) from exc


def registered_strategy_names() -> tuple[str, ...]:
    """Return registered strategy names in sorted order."""

    return tuple(sorted(_STRATEGY_REGISTRY))


def clear_registered_strategies() -> None:
    """Test helper for resetting the ranking strategy registry."""

    _STRATEGY_REGISTRY.clear()


def register_strategies(strategies: Iterable[RankingStrategy]) -> None:
    """Register several ranking strategies."""

    for strategy in strategies:
        register_strategy(strategy)
