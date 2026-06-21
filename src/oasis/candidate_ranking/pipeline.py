from __future__ import annotations

from pathlib import Path
from typing import Any

from oasis.candidate_ranking.loaders import load_screening_input_records
from oasis.candidate_ranking.registry import ensure_strategy
from oasis.candidate_ranking.types import (
    RankingContext,
    RankingResult,
    ScreeningInputRecord,
)
from oasis.mlip.artifacts import find_result_files


def build_ranking_context(
    *,
    candidate_records: tuple[ScreeningInputRecord, ...],
    shot_count: int = 0,
    target_binding_energy: float | None = None,
    dataset_metadata: dict[str, Any] | None = None,
    prior_observations: tuple[dict[str, Any], ...] = (),
    method_config: dict[str, Any] | None = None,
) -> RankingContext:
    """Build a method-agnostic ranking context for any registered strategy."""

    return RankingContext(
        shot_count=shot_count,
        target_binding_energy=target_binding_energy,
        dataset_metadata=dict(dataset_metadata or {}),
        candidate_records=candidate_records,
        prior_observations=prior_observations,
        method_config=dict(method_config or {}),
    )


def rank_candidates(
    *,
    candidate_records: tuple[ScreeningInputRecord, ...],
    method_name: str = "zero_shot",
    shot_count: int = 0,
    target_binding_energy: float | None = None,
    dataset_metadata: dict[str, Any] | None = None,
    prior_observations: tuple[dict[str, Any], ...] = (),
    method_config: dict[str, Any] | None = None,
) -> RankingResult:
    """Resolve one registered strategy and rank normalized candidate records."""

    strategy = ensure_strategy(method_name)
    context = build_ranking_context(
        candidate_records=candidate_records,
        shot_count=shot_count,
        target_binding_energy=target_binding_energy,
        dataset_metadata=dataset_metadata,
        prior_observations=prior_observations,
        method_config=method_config,
    )
    return strategy.rank(context)


def rank_candidates_from_result_files(
    result_files: list[Path],
    *,
    method_name: str = "zero_shot",
    shot_count: int = 0,
    target_binding_energy: float | None = None,
    dataset_metadata: dict[str, Any] | None = None,
    prior_observations: tuple[dict[str, Any], ...] = (),
    method_config: dict[str, Any] | None = None,
) -> RankingResult:
    """Load MLIP bundles into normalized records and run one ranking strategy."""

    candidate_records = load_screening_input_records(result_files)
    return rank_candidates(
        candidate_records=candidate_records,
        method_name=method_name,
        shot_count=shot_count,
        target_binding_energy=target_binding_energy,
        dataset_metadata=dataset_metadata,
        prior_observations=prior_observations,
        method_config=method_config,
    )


def rank_candidates_from_results_dir(
    results_dir: str | Path,
    *,
    pattern: str = "*/*_result.json",
    exclude_processed: bool = True,
    method_name: str = "zero_shot",
    shot_count: int = 0,
    target_binding_energy: float | None = None,
    dataset_metadata: dict[str, Any] | None = None,
    prior_observations: tuple[dict[str, Any], ...] = (),
    method_config: dict[str, Any] | None = None,
) -> RankingResult:
    """Discover MLIP result files under one directory and run ranking."""

    result_files = find_result_files(
        Path(results_dir),
        pattern=pattern,
        exclude_processed=exclude_processed,
    )
    return rank_candidates_from_result_files(
        result_files,
        method_name=method_name,
        shot_count=shot_count,
        target_binding_energy=target_binding_energy,
        dataset_metadata=dataset_metadata,
        prior_observations=prior_observations,
        method_config=method_config,
    )
