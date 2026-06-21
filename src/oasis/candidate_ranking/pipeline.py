from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from oasis.candidate_ranking.loaders import load_screening_input_records
from oasis.candidate_ranking.methods import ZeroShotCandidateRanker
from oasis.candidate_ranking.registry import ensure_predictor
from oasis.candidate_ranking.types import (
    RankingContext,
    RankingResult,
    ScreeningInputRecord,
    ValidatedReference,
)
from oasis.mlip.artifacts import find_result_files


def normalize_validated_references(
    validated_references: Iterable[ValidatedReference | dict[str, Any]],
) -> tuple[ValidatedReference, ...]:
    """Normalize validated reference inputs into canonical records."""

    normalized: list[ValidatedReference] = []
    for reference in validated_references:
        if isinstance(reference, ValidatedReference):
            normalized.append(reference)
            continue
        normalized.append(ValidatedReference(**dict(reference)))
    return tuple(normalized)


def build_ranking_context(
    *,
    candidate_records: tuple[ScreeningInputRecord, ...],
    shot_count: int = 0,
    target_binding_energy: float | None = None,
    dataset_metadata: dict[str, Any] | None = None,
    validated_references: tuple[ValidatedReference, ...] | tuple[dict[str, Any], ...] = (),
    prior_observations: tuple[dict[str, Any], ...] = (),
    method_config: dict[str, Any] | None = None,
) -> RankingContext:
    """Build a method-agnostic ranking context for any registered strategy.

    Shot count is inferred from `validated_references` when they are supplied.
    The explicit `shot_count` argument remains as a temporary compatibility
    fallback during the refactor away from user-selected shot modes.
    """

    return RankingContext(
        shot_count=shot_count,
        target_binding_energy=target_binding_energy,
        dataset_metadata=dict(dataset_metadata or {}),
        candidate_records=candidate_records,
        validated_references=normalize_validated_references(validated_references),
        prior_observations=prior_observations,
        method_config=dict(method_config or {}),
    )


def rank_candidates(
    *,
    candidate_records: tuple[ScreeningInputRecord, ...],
    predictor_name: str | None = None,
    method_name: str | None = None,
    shot_count: int = 0,
    target_binding_energy: float | None = None,
    dataset_metadata: dict[str, Any] | None = None,
    validated_references: tuple[ValidatedReference, ...] = (),
    prior_observations: tuple[dict[str, Any], ...] = (),
    method_config: dict[str, Any] | None = None,
) -> RankingResult:
    """Rank normalized candidate records via zero-shot fallback or predictor path.

    When there are no validated references, zero-shot ranking is the unfitted
    fallback and no predictor registry lookup occurs. When validated references
    exist, the caller must select a registered predictor.
    """

    context = build_ranking_context(
        candidate_records=candidate_records,
        shot_count=shot_count,
        target_binding_energy=target_binding_energy,
        dataset_metadata=dataset_metadata,
        validated_references=validated_references,
        prior_observations=prior_observations,
        method_config=method_config,
    )
    if context.inferred_shot_count == 0:
        return ZeroShotCandidateRanker().rank(context)

    resolved_predictor_name = predictor_name or method_name
    if resolved_predictor_name is None:
        raise ValueError(
            "predictor_name must be provided when validated references are available."
        )
    predictor = ensure_predictor(resolved_predictor_name)
    if not predictor.is_feasible(context.inferred_shot_count):
        raise ValueError(
            f"Predictor {predictor.name!r} requires at least "
            f"{predictor.min_validated_references} validated references; "
            f"got {context.inferred_shot_count}."
        )
    raise NotImplementedError(
        f"Predictor-backed n-shot ranking is not implemented yet for "
        f"{predictor.name!r}."
    )


def rank_candidates_from_result_files(
    result_files: list[Path],
    *,
    predictor_name: str | None = None,
    method_name: str | None = None,
    shot_count: int = 0,
    target_binding_energy: float | None = None,
    dataset_metadata: dict[str, Any] | None = None,
    validated_references: tuple[ValidatedReference, ...] = (),
    prior_observations: tuple[dict[str, Any], ...] = (),
    method_config: dict[str, Any] | None = None,
) -> RankingResult:
    """Load MLIP bundles into normalized records and run one ranking strategy."""

    candidate_records = load_screening_input_records(result_files)
    return rank_candidates(
        candidate_records=candidate_records,
        predictor_name=predictor_name,
        method_name=method_name,
        shot_count=shot_count,
        target_binding_energy=target_binding_energy,
        dataset_metadata=dataset_metadata,
        validated_references=validated_references,
        prior_observations=prior_observations,
        method_config=method_config,
    )


def rank_candidates_from_results_dir(
    results_dir: str | Path,
    *,
    pattern: str = "*/*_result.json",
    exclude_processed: bool = True,
    predictor_name: str | None = None,
    method_name: str | None = None,
    shot_count: int = 0,
    target_binding_energy: float | None = None,
    dataset_metadata: dict[str, Any] | None = None,
    validated_references: tuple[ValidatedReference, ...] = (),
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
        predictor_name=predictor_name,
        method_name=method_name,
        shot_count=shot_count,
        target_binding_energy=target_binding_energy,
        dataset_metadata=dataset_metadata,
        validated_references=validated_references,
        prior_observations=prior_observations,
        method_config=method_config,
    )
