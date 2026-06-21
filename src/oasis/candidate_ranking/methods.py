from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import fmean, pstdev
import numpy as np

from oasis.candidate_ranking.interfaces import (
    CandidateGenerator,
    CandidateScorer,
    ParentReducer,
    RankingStrategy,
)
from oasis.candidate_ranking.types import (
    AdslabCandidate,
    MethodProvenance,
    ParentCandidate,
    RankingContext,
    RankingResult,
    ScreeningInputRecord,
    SupportingSignal,
    UncertaintyEstimate,
)
from oasis.mlip.artifacts import INFERENCE_DETAIL_COLUMNS


@dataclass(frozen=True, slots=True)
class UnfittedEnsembleBaselineConfig:
    """Configuration for anomaly-aware unfitted ensemble candidate generation."""

    exclude_anomalous: bool = False
    label_allowlist: tuple[str, ...] = ("normal",)
    strict_inference_anomaly: bool = False
    min_valid_mlips: int = 2

    @classmethod
    def from_context(
        cls,
        context: RankingContext,
    ) -> UnfittedEnsembleBaselineConfig:
        method_cfg = dict(context.method_config)
        allowlist = tuple(method_cfg.get("label_allowlist", ("normal",)))
        return cls(
            exclude_anomalous=bool(method_cfg.get("exclude_anomalous", False)),
            label_allowlist=allowlist or ("normal",),
            strict_inference_anomaly=bool(
                method_cfg.get("strict_inference_anomaly", False)
            ),
            min_valid_mlips=int(method_cfg.get("min_valid_mlips", 2)),
        )


@dataclass(frozen=True, slots=True)
class TargetAwareScoringConfig:
    """Configurable scoring policy over reduced parent candidates.

    This stage is intentionally separate from candidate generation so future
    methods can reuse the same reducer output while swapping in acquisition,
    few-shot, or registry-backed scoring policies.
    """

    target_distance_weight: float = 1.0
    uncertainty_weight: float = 1.0
    score_function: str = "target_uncertainty_cost"
    target_uncertainty_alpha: float = 0.75
    supporting_signal_weights: dict[str, float] | None = None

    @classmethod
    def from_context(cls, context: RankingContext) -> TargetAwareScoringConfig:
        method_cfg = dict(context.method_config)
        signal_weights = method_cfg.get("supporting_signal_weights", {})
        return cls(
            target_distance_weight=float(method_cfg.get("target_distance_weight", 1.0)),
            uncertainty_weight=float(method_cfg.get("uncertainty_weight", 1.0)),
            score_function=str(
                method_cfg.get("score_function", "target_uncertainty_cost")
            ),
            target_uncertainty_alpha=float(
                method_cfg.get("target_uncertainty_alpha", 0.75)
            ),
            supporting_signal_weights={
                str(name): float(weight)
                for name, weight in dict(signal_weights).items()
            },
        )


def _prediction_passes_strict_details(record: ScreeningInputRecord, model_name: str) -> bool:
    for prediction in record.model_predictions:
        if prediction.model_name != model_name:
            continue
        anomaly = prediction.anomaly
        if anomaly is None:
            return False
        return all(anomaly.details.get(detail_name, 0) == 0 for detail_name in INFERENCE_DETAIL_COLUMNS)
    return False


def _globally_allowed_model_names(
    records: tuple[ScreeningInputRecord, ...],
    cfg: UnfittedEnsembleBaselineConfig,
) -> tuple[str, ...]:
    model_names = sorted(
        {
            prediction.model_name
            for record in records
            for prediction in record.model_predictions
        }
    )
    if not cfg.exclude_anomalous:
        return tuple(model_names)
    if cfg.strict_inference_anomaly:
        return tuple(model_names)

    allowlist = set(cfg.label_allowlist)
    kept: list[str] = []
    for model_name in model_names:
        labels = [
            prediction.anomaly.label
            for record in records
            for prediction in record.model_predictions
            if prediction.model_name == model_name and prediction.anomaly is not None
        ]
        if labels and all(label in allowlist for label in labels):
            kept.append(model_name)
    if not kept:
        raise ValueError("No MLIP prediction columns remain after anomaly-aware selection.")
    return tuple(kept)


def _valid_predictions_for_record(
    record: ScreeningInputRecord,
    *,
    cfg: UnfittedEnsembleBaselineConfig,
    globally_allowed_model_names: tuple[str, ...],
) -> tuple:
    allowlist = set(cfg.label_allowlist)
    valid_predictions = []
    for prediction in record.model_predictions:
        if prediction.model_name not in globally_allowed_model_names:
            continue
        if prediction.predicted_binding_energy is None:
            continue
        if not cfg.exclude_anomalous:
            valid_predictions.append(prediction)
            continue
        if cfg.strict_inference_anomaly:
            if _prediction_passes_strict_details(record, prediction.model_name):
                valid_predictions.append(prediction)
            continue
        if prediction.anomaly is None:
            continue
        if prediction.anomaly.label in allowlist:
            valid_predictions.append(prediction)
    return tuple(valid_predictions)


class UnfittedEnsembleBaselineGenerator(CandidateGenerator):
    """Generate per-adslab baseline candidates from normalized MLIP inputs."""

    method_name = "unfitted_ensemble_baseline"

    def generate(
        self,
        context: RankingContext,
    ) -> list[AdslabCandidate]:
        cfg = UnfittedEnsembleBaselineConfig.from_context(context)
        globally_allowed_model_names = _globally_allowed_model_names(
            context.candidate_records,
            cfg,
        )
        candidates: list[AdslabCandidate] = []
        for record in context.candidate_records:
            valid_predictions = _valid_predictions_for_record(
                record,
                cfg=cfg,
                globally_allowed_model_names=globally_allowed_model_names,
            )
            if cfg.exclude_anomalous and len(valid_predictions) < cfg.min_valid_mlips:
                continue
            if not valid_predictions:
                continue

            energies = [float(pred.predicted_binding_energy) for pred in valid_predictions]
            provenance = MethodProvenance(
                method_name=self.method_name,
                stage="candidate_generation",
                shot_count=context.inferred_shot_count,
                source_methods=tuple(pred.model_name for pred in valid_predictions),
                metadata={
                    "globally_allowed_models": list(globally_allowed_model_names),
                    "excluded_models": [
                        pred.model_name
                        for pred in record.model_predictions
                        if pred.model_name not in {p.model_name for p in valid_predictions}
                    ],
                    "filtering_mode": (
                        "details"
                        if cfg.strict_inference_anomaly and cfg.exclude_anomalous
                        else "labels"
                    ),
                },
            )
            uncertainty = UncertaintyEstimate(
                value=float(pstdev(energies)) if len(energies) > 1 else 0.0,
                metric="ensemble_std",
                provenance=provenance,
                is_calibrated=False,
                metadata={"valid_mlip_count": len(valid_predictions)},
            )
            candidates.append(
                AdslabCandidate(
                    parent_slab_id=record.parent_slab_id,
                    adslab_id=record.adslab_id,
                    predicted_binding_energy=float(fmean(energies)),
                    method_provenance=provenance,
                    uncertainty=uncertainty,
                    supporting_signals=(
                        SupportingSignal(
                            name="valid_mlip_count",
                            value=float(len(valid_predictions)),
                            provenance=provenance,
                            objective="maximize",
                        ),
                    ),
                    metadata={
                        "reaction": record.reaction,
                        "adsorbate": record.adsorbate,
                        "site_metadata": dict(record.site_metadata),
                        "available_model_names": [
                            prediction.model_name for prediction in record.model_predictions
                        ],
                    },
                    provenance={
                        "record_model_predictions": [
                            {
                                "model_name": prediction.model_name,
                                "result_path": prediction.result_path,
                                "label": prediction.anomaly.label if prediction.anomaly else None,
                            }
                            for prediction in valid_predictions
                        ]
                    },
                )
            )
        return candidates


class LowestEnergyParentReducer(ParentReducer):
    """Reduce adslab candidates to one lowest-energy child per parent slab.

    This reducer is intentionally standalone so future ranking methods can swap
    it out or augment it with acquisition logic without changing candidate
    generation.
    """

    def reduce(
        self,
        candidates: list[AdslabCandidate],
        context: RankingContext,
    ) -> list[ParentCandidate]:
        del context
        grouped_candidates: dict[str, list[AdslabCandidate]] = {}
        for candidate in candidates:
            grouped_candidates.setdefault(candidate.parent_slab_id, []).append(candidate)

        reduced: list[ParentCandidate] = []
        for parent_slab_id in sorted(grouped_candidates):
            grouped = grouped_candidates[parent_slab_id]
            selected = min(
                grouped,
                key=lambda candidate: (
                    float("inf")
                    if candidate.predicted_binding_energy is None
                    else float(candidate.predicted_binding_energy),
                    float("inf")
                    if candidate.uncertainty is None
                    else float(candidate.uncertainty.value),
                    candidate.adslab_id,
                ),
            )
            reduced.append(
                ParentCandidate(
                    parent_slab_id=selected.parent_slab_id,
                    selected_adslab_id=selected.adslab_id,
                    predicted_binding_energy=selected.predicted_binding_energy,
                    method_provenance=selected.method_provenance,
                    uncertainty=selected.uncertainty,
                    supporting_signals=selected.supporting_signals,
                    selected_adslab_ids=tuple(
                        candidate.adslab_id
                        for candidate in sorted(grouped, key=lambda item: item.adslab_id)
                    ),
                    adslab_metadata=dict(selected.metadata),
                    provenance={
                        **dict(selected.provenance),
                        "reduction_policy": "lowest_energy_child_per_parent",
                        "candidate_count_for_parent": len(grouped),
                    },
                )
            )
        return reduced


def _target_distance(
    candidate: ParentCandidate,
    context: RankingContext,
) -> float:
    if (
        candidate.predicted_binding_energy is None
        or context.target_binding_energy is None
    ):
        return 0.0
    return abs(
        float(candidate.predicted_binding_energy) - float(context.target_binding_energy)
    )


def _uncertainty_value(candidate: ParentCandidate) -> float:
    if candidate.uncertainty is None:
        return 0.0
    return float(candidate.uncertainty.value)


def _score_source_metadata(candidate: ParentCandidate) -> dict[str, object]:
    provenance = candidate.method_provenance
    if provenance is None:
        return {
            "candidate_source_kind": "unknown",
            "candidate_source_method": None,
            "candidate_source_stage": None,
            "candidate_source_shot_count": None,
        }
    return {
        "candidate_source_kind": (
            "baseline"
            if provenance.method_name == "unfitted_ensemble_baseline"
            else "predictor"
        ),
        "candidate_source_method": provenance.method_name,
        "candidate_source_stage": provenance.stage,
        "candidate_source_shot_count": provenance.shot_count,
    }


def _supporting_signal_penalty(
    candidate: ParentCandidate,
    *,
    signal_weights: dict[str, float],
) -> tuple[float, dict[str, float]]:
    by_name = {signal.name: signal for signal in candidate.supporting_signals}
    total = 0.0
    components: dict[str, float] = {}
    for signal_name, weight in sorted(signal_weights.items()):
        signal = by_name.get(signal_name)
        if signal is None:
            continue
        contribution = float(signal.value) * float(weight)
        if signal.objective == "maximize":
            contribution *= -1.0
        total += contribution
        components[signal_name] = contribution
    return total, components


def target_uncertainty_cost(
    mean: np.ndarray,
    std: np.ndarray,
    target: float,
    alpha: float = 0.75,
) -> np.ndarray:
    """Normalized target/uncertainty tradeoff from transfer-shot.

    Lower cost is better. The accuracy term uses absolute distance to target and
    the uncertainty term uses the predicted spread, each normalized by their
    respective range across the candidate set.
    """

    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    range_mu = np.ptp(mean)
    range_sigma = np.ptp(std)
    epsilon = 1e-8
    range_mu = range_mu if range_mu > 0 else epsilon
    range_sigma = range_sigma if range_sigma > 0 else epsilon
    accuracy_term = np.abs(mean - float(target)) / range_mu
    uncertainty_term = std / range_sigma
    return alpha * accuracy_term + (1.0 - alpha) * uncertainty_term


class TargetAwareCandidateScorer(CandidateScorer):
    """Rank reduced candidates using target proximity, uncertainty, and signals.

    Future policies can replace this scorer entirely or extend it with new
    supporting signals without modifying the generator or reducer stages.
    """

    def score(
        self,
        candidates: list[ParentCandidate],
        context: RankingContext,
    ) -> list[ParentCandidate]:
        cfg = TargetAwareScoringConfig.from_context(context)
        if cfg.score_function == "target_uncertainty_cost":
            return self._score_with_target_uncertainty_cost(candidates, context, cfg)
        if cfg.score_function == "weighted_sum":
            return self._score_with_weighted_sum(candidates, context, cfg)
        raise ValueError(
            f"Unknown score_function {cfg.score_function!r}. "
            "Expected 'target_uncertainty_cost' or 'weighted_sum'."
        )

    def _score_with_target_uncertainty_cost(
        self,
        candidates: list[ParentCandidate],
        context: RankingContext,
        cfg: TargetAwareScoringConfig,
    ) -> list[ParentCandidate]:
        if context.target_binding_energy is None:
            raise ValueError(
                "target_binding_energy must be provided for target_uncertainty_cost scoring."
            )
        means = np.asarray(
            [
                float(candidate.predicted_binding_energy or 0.0)
                for candidate in candidates
            ],
            dtype=float,
        )
        stds = np.asarray(
            [_uncertainty_value(candidate) for candidate in candidates],
            dtype=float,
        )
        costs = target_uncertainty_cost(
            mean=means,
            std=stds,
            target=float(context.target_binding_energy),
            alpha=cfg.target_uncertainty_alpha,
        )
        scored: list[ParentCandidate] = []
        for candidate, cost in zip(candidates, costs, strict=False):
            target_distance = _target_distance(candidate, context)
            uncertainty_penalty = _uncertainty_value(candidate)
            scored.append(
                replace(
                    candidate,
                    score=float(cost),
                    provenance={
                        **dict(candidate.provenance),
                        "scoring_policy": "target_uncertainty_cost",
                        "score_source": _score_source_metadata(candidate),
                        "score_components": {
                            "target_distance": target_distance,
                            "uncertainty": uncertainty_penalty,
                            "target_uncertainty_alpha": cfg.target_uncertainty_alpha,
                        },
                    },
                )
            )
        return sorted(
            scored,
            key=lambda candidate: (
                float("inf") if candidate.score is None else float(candidate.score),
                candidate.selected_adslab_id,
            ),
        )

    def _score_with_weighted_sum(
        self,
        candidates: list[ParentCandidate],
        context: RankingContext,
        cfg: TargetAwareScoringConfig,
    ) -> list[ParentCandidate]:
        scored = []
        for candidate in candidates:
            target_distance = _target_distance(candidate, context)
            uncertainty_penalty = _uncertainty_value(candidate)
            signal_penalty, signal_components = _supporting_signal_penalty(
                candidate,
                signal_weights=cfg.supporting_signal_weights or {},
            )
            score = (
                cfg.target_distance_weight * target_distance
                + cfg.uncertainty_weight * uncertainty_penalty
                + signal_penalty
            )
            provenance = {
                **dict(candidate.provenance),
                "scoring_policy": "target_aware_weighted_sum",
                "score_source": _score_source_metadata(candidate),
                "score_components": {
                    "target_distance": cfg.target_distance_weight * target_distance,
                    "uncertainty": cfg.uncertainty_weight * uncertainty_penalty,
                    "supporting_signals": signal_components,
                },
            }
            scored.append(
                replace(
                    candidate,
                    score=float(score),
                    provenance=provenance,
                )
            )
        return sorted(
            scored,
            key=lambda candidate: (
                float("inf") if candidate.score is None else float(candidate.score),
                candidate.selected_adslab_id,
            ),
        )


class UnfittedEnsembleBaselineRanker(RankingStrategy):
    """Rank candidates with the unfitted MLIP-ensemble baseline path."""

    name = "unfitted_ensemble_baseline"

    def __init__(
        self,
        *,
        generator: CandidateGenerator | None = None,
        reducer: ParentReducer | None = None,
        scorer: CandidateScorer | None = None,
    ) -> None:
        self.generator = generator or UnfittedEnsembleBaselineGenerator()
        self.reducer = reducer or LowestEnergyParentReducer()
        self.scorer = scorer or TargetAwareCandidateScorer()

    def rank(self, context: RankingContext) -> RankingResult:
        adslab_candidates = tuple(self.generator.generate(context))
        parent_candidates = tuple(self.reducer.reduce(list(adslab_candidates), context))
        ranked_candidates = tuple(self.scorer.score(list(parent_candidates), context))
        return RankingResult(
            strategy_name=self.name,
            adslab_candidates=adslab_candidates,
            parent_candidates=parent_candidates,
            ranked_candidates=ranked_candidates,
            metadata={"shot_count": context.inferred_shot_count},
        )
