from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, pstdev

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
class ZeroShotRankingConfig:
    """Configuration for anomaly-aware zero-shot candidate generation."""

    exclude_anomalous: bool = False
    label_allowlist: tuple[str, ...] = ("normal",)
    strict_inference_anomaly: bool = False
    min_valid_mlips: int = 2

    @classmethod
    def from_context(cls, context: RankingContext) -> ZeroShotRankingConfig:
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
    cfg: ZeroShotRankingConfig,
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
    cfg: ZeroShotRankingConfig,
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


class ZeroShotCandidateGenerator(CandidateGenerator):
    """Generate per-adslab zero-shot candidates from normalized MLIP inputs."""

    method_name = "zero_shot"

    def generate(
        self,
        context: RankingContext,
    ) -> list[AdslabCandidate]:
        cfg = ZeroShotRankingConfig.from_context(context)
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
                shot_count=context.shot_count,
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


class ZeroShotIdentityReducer(ParentReducer):
    """Temporary reducer that preserves one adslab candidate per parent output."""

    def reduce(
        self,
        candidates: list[AdslabCandidate],
        context: RankingContext,
    ) -> list[ParentCandidate]:
        del context
        return [
            ParentCandidate(
                parent_slab_id=candidate.parent_slab_id,
                selected_adslab_id=candidate.adslab_id,
                predicted_binding_energy=candidate.predicted_binding_energy,
                method_provenance=candidate.method_provenance,
                uncertainty=candidate.uncertainty,
                supporting_signals=candidate.supporting_signals,
                selected_adslab_ids=(candidate.adslab_id,),
                adslab_metadata=dict(candidate.metadata),
                provenance=dict(candidate.provenance),
            )
            for candidate in candidates
        ]


class ZeroShotIdentityScorer(CandidateScorer):
    """Temporary scorer that preserves generation order until ranking policy lands."""

    def score(
        self,
        candidates: list[ParentCandidate],
        context: RankingContext,
    ) -> list[ParentCandidate]:
        del context
        return candidates


class ZeroShotCandidateRanker(RankingStrategy):
    """Registry-compatible zero-shot ranking strategy."""

    name = "zero_shot"

    def __init__(
        self,
        *,
        generator: CandidateGenerator | None = None,
        reducer: ParentReducer | None = None,
        scorer: CandidateScorer | None = None,
    ) -> None:
        self.generator = generator or ZeroShotCandidateGenerator()
        self.reducer = reducer or ZeroShotIdentityReducer()
        self.scorer = scorer or ZeroShotIdentityScorer()

    def rank(self, context: RankingContext) -> RankingResult:
        adslab_candidates = tuple(self.generator.generate(context))
        parent_candidates = tuple(self.reducer.reduce(list(adslab_candidates), context))
        ranked_candidates = tuple(self.scorer.score(list(parent_candidates), context))
        return RankingResult(
            strategy_name=self.name,
            adslab_candidates=adslab_candidates,
            parent_candidates=parent_candidates,
            ranked_candidates=ranked_candidates,
            metadata={"shot_count": context.shot_count},
        )
