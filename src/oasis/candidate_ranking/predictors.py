from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from oasis.candidate_ranking.methods import (
    LowestEnergyParentReducer,
    TargetAwareCandidateScorer,
    UnfittedEnsembleBaselineConfig,
    _globally_allowed_model_names,
    _valid_predictions_for_record,
)
from oasis.candidate_ranking.types import (
    AdslabCandidate,
    MethodProvenance,
    RankingContext,
    RankingResult,
    ScreeningInputRecord,
    SupportingSignal,
    UncertaintyEstimate,
    ValidatedReference,
)
from oasis.learning_curve.execution import (
    _simplex_weights,
    require_min_mlip_feature_count,
)

try:
    from sklearn.linear_model import Ridge
except ImportError:
    Ridge = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class PreparedPredictorInputs:
    predictor_name: str
    model_names: tuple[str, ...]
    candidate_records: tuple[ScreeningInputRecord, ...]
    candidate_matrix: np.ndarray
    training_records: tuple[ScreeningInputRecord, ...]
    training_matrix: np.ndarray
    training_targets: np.ndarray
    training_references: tuple[ValidatedReference, ...]


def _record_identity_keys(record: ScreeningInputRecord) -> tuple[str, ...]:
    return (
        f"adslab:{record.adslab_id}",
        f"reaction:{record.reaction}",
    )


def _resolve_reference_key(reference: ValidatedReference) -> str:
    return reference.identity


def _prepare_predictor_inputs(
    context: RankingContext,
    *,
    predictor_name: str,
) -> PreparedPredictorInputs:
    cfg = UnfittedEnsembleBaselineConfig.from_context(context)
    model_names = _globally_allowed_model_names(context.candidate_records, cfg)
    matrix_rows: list[np.ndarray] = []
    matrix_records: list[ScreeningInputRecord] = []
    for record in context.candidate_records:
        valid_predictions = _valid_predictions_for_record(
            record,
            cfg=cfg,
            globally_allowed_model_names=model_names,
        )
        by_name = {
            prediction.model_name: float(prediction.predicted_binding_energy)
            for prediction in valid_predictions
            if prediction.predicted_binding_energy is not None
        }
        if len(by_name) < len(model_names):
            continue
        matrix_rows.append(
            np.asarray([by_name[model_name] for model_name in model_names], dtype=float)
        )
        matrix_records.append(record)

    if not matrix_rows:
        raise ValueError(
            f"Predictor {predictor_name!r} could not find any complete candidate rows "
            "after anomaly-aware MLIP filtering."
        )

    candidate_matrix = np.vstack(matrix_rows)
    record_by_identity = {
        identity: record
        for record in matrix_records
        for identity in _record_identity_keys(record)
    }
    feature_by_identity = {
        identity: features
        for record, features in zip(matrix_records, candidate_matrix, strict=True)
        for identity in _record_identity_keys(record)
    }

    training_records: list[ScreeningInputRecord] = []
    training_rows: list[np.ndarray] = []
    training_targets: list[float] = []
    matched_references: list[ValidatedReference] = []
    for reference in context.validated_references:
        record = record_by_identity.get(_resolve_reference_key(reference))
        features = feature_by_identity.get(_resolve_reference_key(reference))
        if record is None or features is None:
            continue
        training_records.append(record)
        training_rows.append(features)
        training_targets.append(float(reference.adsorption_energy))
        matched_references.append(reference)

    return PreparedPredictorInputs(
        predictor_name=predictor_name,
        model_names=model_names,
        candidate_records=tuple(matrix_records),
        candidate_matrix=candidate_matrix,
        training_records=tuple(training_records),
        training_matrix=np.vstack(training_rows) if training_rows else np.empty((0, len(model_names))),
        training_targets=np.asarray(training_targets, dtype=float),
        training_references=tuple(matched_references),
    )


def _build_candidate_outputs(
    *,
    context: RankingContext,
    predictor_name: str,
    prepared: PreparedPredictorInputs,
    predicted_binding_energies: np.ndarray,
    spread: np.ndarray,
    extra_metadata: dict[str, object] | None = None,
) -> list[AdslabCandidate]:
    candidates: list[AdslabCandidate] = []
    for record, predicted_energy, uncertainty_value, features in zip(
        prepared.candidate_records,
        predicted_binding_energies,
        spread,
        prepared.candidate_matrix,
        strict=True,
    ):
        provenance = MethodProvenance(
            method_name=predictor_name,
            stage="candidate_generation",
            shot_count=context.inferred_shot_count,
            source_methods=prepared.model_names,
            metadata={
                "training_reference_count": len(prepared.training_references),
                "training_reference_ids": [
                    reference.identity for reference in prepared.training_references
                ],
                "model_names": list(prepared.model_names),
                **dict(extra_metadata or {}),
            },
        )
        uncertainty = UncertaintyEstimate(
            value=float(uncertainty_value),
            metric="spread_only",
            provenance=provenance,
            is_calibrated=False,
            metadata={"feature_count": len(prepared.model_names)},
        )
        candidates.append(
            AdslabCandidate(
                parent_slab_id=record.parent_slab_id,
                adslab_id=record.adslab_id,
                predicted_binding_energy=float(predicted_energy),
                method_provenance=provenance,
                uncertainty=uncertainty,
                supporting_signals=(
                    SupportingSignal(
                        name="training_reference_count",
                        value=float(len(prepared.training_references)),
                        provenance=provenance,
                        objective="maximize",
                    ),
                    SupportingSignal(
                        name="valid_mlip_count",
                        value=float(len(prepared.model_names)),
                        provenance=provenance,
                        objective="maximize",
                    ),
                ),
                metadata={
                    "reaction": record.reaction,
                    "adsorbate": record.adsorbate,
                    "site_metadata": dict(record.site_metadata),
                    "available_model_names": list(prepared.model_names),
                },
                provenance={
                    "feature_vector": {
                        model_name: float(value)
                        for model_name, value in zip(
                            prepared.model_names,
                            features,
                            strict=True,
                        )
                    }
                },
            )
        )
    return candidates


def generate_residual_candidates(context: RankingContext) -> list[AdslabCandidate]:
    prepared = _prepare_predictor_inputs(context, predictor_name="residual")
    if len(prepared.training_targets) < 1:
        raise ValueError("Predictor 'residual' requires at least 1 usable validated reference.")

    residuals = prepared.training_targets[:, None] - prepared.training_matrix
    mean_residuals = residuals.mean(axis=0)
    corrected = prepared.candidate_matrix + mean_residuals
    predicted = corrected.mean(axis=1)
    spread = np.std(corrected, axis=1)
    return _build_candidate_outputs(
        context=context,
        predictor_name="residual",
        prepared=prepared,
        predicted_binding_energies=predicted,
        spread=spread,
        extra_metadata={"residual_mean_by_model": mean_residuals.tolist()},
    )


def generate_weighted_simplex_candidates(context: RankingContext) -> list[AdslabCandidate]:
    prepared = _prepare_predictor_inputs(context, predictor_name="weighted_simplex")
    if len(prepared.training_targets) < 1:
        raise ValueError(
            "Predictor 'weighted_simplex' requires at least 1 usable validated reference."
        )
    require_min_mlip_feature_count(
        prepared.training_matrix,
        min_features=2,
        method_name="weighted_simplex",
    )
    weights = _simplex_weights(prepared.training_matrix, prepared.training_targets)
    predicted = prepared.candidate_matrix @ weights
    centered = prepared.candidate_matrix - predicted[:, None]
    spread = np.sqrt(np.sum(weights[None, :] * centered**2, axis=1))
    return _build_candidate_outputs(
        context=context,
        predictor_name="weighted_simplex",
        prepared=prepared,
        predicted_binding_energies=predicted,
        spread=spread,
        extra_metadata={"simplex_weights": weights.tolist()},
    )


def _linear_spread_from_coefficients(
    *,
    features: np.ndarray,
    predicted: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    weights = np.abs(np.asarray(coefficients, dtype=float).reshape(-1))
    weight_sum = float(weights.sum())
    normalized = (
        np.full(features.shape[1], 1.0 / features.shape[1], dtype=float)
        if weight_sum <= 0.0
        else weights / weight_sum
    )
    centered = features - predicted[:, None]
    return np.sqrt(np.sum(normalized[None, :] * centered**2, axis=1))


def generate_ridge_candidates(context: RankingContext) -> list[AdslabCandidate]:
    if Ridge is None:
        raise ImportError("scikit-learn is required for predictor 'ridge'.")
    prepared = _prepare_predictor_inputs(context, predictor_name="ridge")
    if len(prepared.training_targets) < 2:
        raise ValueError("Predictor 'ridge' requires at least 2 usable validated references.")
    alpha = float(context.method_config.get("alpha", 0.1))
    model = Ridge(alpha=alpha)
    model.fit(prepared.training_matrix, prepared.training_targets)
    predicted = np.asarray(model.predict(prepared.candidate_matrix), dtype=float)
    spread = _linear_spread_from_coefficients(
        features=prepared.candidate_matrix,
        predicted=predicted,
        coefficients=np.asarray(getattr(model, "coef_", np.ones(prepared.candidate_matrix.shape[1]))),
    )
    return _build_candidate_outputs(
        context=context,
        predictor_name="ridge",
        prepared=prepared,
        predicted_binding_energies=predicted,
        spread=spread,
        extra_metadata={
            "alpha": alpha,
            "coefficients": np.asarray(model.coef_, dtype=float).tolist(),
            "intercept": float(model.intercept_),
        },
    )


@dataclass(frozen=True, slots=True)
class PredictorBackedRanker:
    predictor_name: str
    generate_candidates: Callable[[RankingContext], list[AdslabCandidate]]

    def rank(self, context: RankingContext) -> RankingResult:
        adslab_candidates = tuple(self.generate_candidates(context))
        parent_candidates = tuple(
            LowestEnergyParentReducer().reduce(list(adslab_candidates), context)
        )
        ranked_candidates = tuple(
            TargetAwareCandidateScorer().score(list(parent_candidates), context)
        )
        return RankingResult(
            strategy_name=self.predictor_name,
            adslab_candidates=adslab_candidates,
            parent_candidates=parent_candidates,
            ranked_candidates=ranked_candidates,
            metadata={
                "shot_count": context.inferred_shot_count,
                "predictor_name": self.predictor_name,
            },
        )
