from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from oasis.calibration import fit_scalar_spread_calibrator_with_identity_fallback
from oasis.experiment import calibration_size_if_sweep_feasible
from oasis.calibration_metrics import (
    dispersion_from_spread,
    miscalibration_area,
    sharpness_from_spread,
)

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
    UncertaintyCalibration,
    UncertaintyEstimate,
    ValidatedReference,
)
from oasis.experiment import build_sweep_split_collection
from oasis.sweep import SweepFamilyRequirements
from oasis.learning_curve.execution import _simplex_weights, require_min_mlip_feature_count

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


@dataclass(frozen=True, slots=True)
class PredictorFitResult:
    predicted_binding_energies: np.ndarray
    spread: np.ndarray
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class PredictorDiagnosticsResult:
    metrics_by_predictor: dict[str, dict[str, object]]


@dataclass(frozen=True, slots=True)
class CalibratedPredictorFitResult:
    fit: PredictorFitResult
    raw_spread: np.ndarray
    calibrator: ScalarSpreadCalibrator | None = None


_CALIBRATABLE_PREDICTORS = frozenset({"residual", "weighted_simplex"})


def _record_identity_keys(record: ScreeningInputRecord) -> tuple[str, ...]:
    return (
        f"adslab:{record.adslab_id}",
        f"reaction:{record.reaction}",
    )


def _resolve_reference_key(reference: ValidatedReference) -> str:
    return reference.identity


def _fit_predictor_arrays(
    predictor_name: str,
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    method_config: dict[str, object],
) -> PredictorFitResult:
    if predictor_name == "residual":
        residuals = y_train[:, None] - X_train
        mean_residuals = residuals.mean(axis=0)
        corrected = X_test + mean_residuals
        return PredictorFitResult(
            predicted_binding_energies=corrected.mean(axis=1),
            spread=np.std(corrected, axis=1),
            metadata={"residual_mean_by_model": mean_residuals.tolist()},
        )
    if predictor_name == "weighted_simplex":
        require_min_mlip_feature_count(
            X_train,
            min_features=2,
            method_name="weighted_simplex",
        )
        weights = _simplex_weights(X_train, y_train)
        predicted = X_test @ weights
        centered = X_test - predicted[:, None]
        return PredictorFitResult(
            predicted_binding_energies=predicted,
            spread=np.sqrt(np.sum(weights[None, :] * centered**2, axis=1)),
            metadata={"simplex_weights": weights.tolist()},
        )
    if predictor_name == "ridge":
        if Ridge is None:
            raise ImportError("scikit-learn is required for predictor 'ridge'.")
        alpha = float(method_config.get("alpha", 0.1))
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        predicted = np.asarray(model.predict(X_test), dtype=float)
        return PredictorFitResult(
            predicted_binding_energies=predicted,
            spread=_linear_spread_from_coefficients(
                features=X_test,
                predicted=predicted,
                coefficients=np.asarray(
                    getattr(model, "coef_", np.ones(X_test.shape[1])),
                    dtype=float,
                ),
            ),
            metadata={
                "alpha": alpha,
                "coefficients": np.asarray(model.coef_, dtype=float).tolist(),
                "intercept": float(model.intercept_),
            },
        )
    raise KeyError(f"Unknown candidate predictor {predictor_name!r}.")


def _maybe_partition_calibration_indices(
    base_idx: np.ndarray,
    *,
    method_config: dict[str, object],
    seed: int,
    min_inner_train_size: int = 1,
) -> tuple[np.ndarray, np.ndarray | None]:
    n_cal = calibration_size_if_sweep_feasible(
        len(base_idx),
        calibration_fraction=float(method_config.get("calibration_fraction", 0.2)),
        min_cal_size=int(method_config.get("min_cal_size", 1)),
        min_post_calibration_size=min_inner_train_size,
    )
    if n_cal is None:
        return np.asarray(base_idx, dtype=int), None
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(np.asarray(base_idx, dtype=int))
    return shuffled[n_cal:], shuffled[:n_cal]


def _fit_predictor_with_optional_calibration(
    predictor_name: str,
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    method_config: dict[str, object],
    X_cal: np.ndarray | None = None,
    y_cal: np.ndarray | None = None,
) -> CalibratedPredictorFitResult:
    fit = _fit_predictor_arrays(
        predictor_name,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        method_config=method_config,
    )
    raw_spread = np.asarray(fit.spread, dtype=float)
    if predictor_name not in _CALIBRATABLE_PREDICTORS or X_cal is None or y_cal is None:
        return CalibratedPredictorFitResult(
            fit=fit,
            raw_spread=raw_spread,
        )

    calibration_fit = _fit_predictor_arrays(
        predictor_name,
        X_train=X_train,
        y_train=y_train,
        X_test=X_cal,
        method_config=method_config,
    )
    calibrator = fit_scalar_spread_calibrator_with_identity_fallback(
        y_true=y_cal,
        y_pred=calibration_fit.predicted_binding_energies,
        spread=calibration_fit.spread,
    )
    return CalibratedPredictorFitResult(
        fit=PredictorFitResult(
            predicted_binding_energies=fit.predicted_binding_energies,
            spread=calibrator.apply(raw_spread),
            metadata={
                **fit.metadata,
                "calibration_method": calibrator.method,
                "calibration_scale": float(calibrator.scale),
            },
        ),
        raw_spread=raw_spread,
        calibrator=calibrator,
    )


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


def evaluate_predictors(
    predictor_names: tuple[str, ...],
    *,
    context: RankingContext,
) -> PredictorDiagnosticsResult | None:
    prepared = _prepare_predictor_inputs(context, predictor_name="selection")
    if len(prepared.training_targets) == 0:
        return None

    split_collection = build_sweep_split_collection(
        len(prepared.training_targets),
        min_train=1,
        max_train=1,
        step=1,
        n_repeats=int(context.method_config.get("selection_n_repeats", 1)),
        seed=int(context.dataset_metadata.get("seed", 0) or 0),
        requested_sweep_sizes=(len(prepared.training_targets),),
        requirements=SweepFamilyRequirements(),
        budget_mode="screening_fraction",
        screen_fraction=float(context.method_config.get("selection_screen_fraction", 0.2)),
        min_screen_size=int(context.method_config.get("selection_min_screen_size", 1)),
        validation_fraction=float(context.method_config.get("validation_fraction", 0.2)),
        min_val_size=int(context.method_config.get("min_val_size", 1)),
        min_tuning_val_size=int(context.method_config.get("min_tuning_val_size", 1)),
        min_inner_train_size=int(context.method_config.get("min_inner_train_size", 1)),
        min_test_size=int(context.method_config.get("selection_min_test_size", 1)),
    )
    splits = tuple(split_collection.splits)

    metrics_by_predictor: dict[str, dict[str, object]] = {}
    for predictor_name in predictor_names:
        rmse_values: list[float] = []
        miscalibration_values: list[float] = []
        sharpness_values: list[float] = []
        dispersion_values: list[float] = []
        all_splits_calibrated = True
        evaluation_mode = "screening_cv"
        if splits:
            for split_idx, split in enumerate(splits):
                train_idx, cal_idx = _maybe_partition_calibration_indices(
                    split.train_idx,
                    method_config=context.method_config,
                    seed=int(context.dataset_metadata.get("seed", 0) or 0) + split_idx,
                    min_inner_train_size=int(
                        context.method_config.get("min_inner_train_size", 1)
                    ),
                )
                calibrated_fit = _fit_predictor_with_optional_calibration(
                    predictor_name,
                    X_train=prepared.training_matrix[train_idx],
                    y_train=prepared.training_targets[train_idx],
                    X_test=prepared.training_matrix[split.test_idx],
                    method_config=context.method_config,
                    X_cal=(
                        None
                        if cal_idx is None
                        else prepared.training_matrix[cal_idx]
                    ),
                    y_cal=(
                        None
                        if cal_idx is None
                        else prepared.training_targets[cal_idx]
                    ),
                )
                fit = calibrated_fit.fit
                residual = (
                    prepared.training_targets[split.test_idx]
                    - fit.predicted_binding_energies
                )
                spread = np.asarray(fit.spread, dtype=float)
                if (
                    predictor_name in _CALIBRATABLE_PREDICTORS
                    and calibrated_fit.calibrator is None
                ):
                    all_splits_calibrated = False
                rmse_values.append(float(np.sqrt(np.mean(residual**2))))
                miscalibration_values.append(
                    float(
                        miscalibration_area(
                            prepared.training_targets[split.test_idx],
                            fit.predicted_binding_energies,
                            spread,
                        )
                    )
                )
                sharpness_values.append(float(sharpness_from_spread(spread)))
                dispersion_values.append(float(dispersion_from_spread(spread)))
        else:
            evaluation_mode = "full_fit_train_rmse"
            train_idx, cal_idx = _maybe_partition_calibration_indices(
                np.arange(len(prepared.training_targets), dtype=int),
                method_config=context.method_config,
                seed=int(context.dataset_metadata.get("seed", 0) or 0),
                min_inner_train_size=int(
                    context.method_config.get("min_inner_train_size", 1)
                ),
            )
            calibrated_fit = _fit_predictor_with_optional_calibration(
                predictor_name,
                X_train=prepared.training_matrix[train_idx],
                y_train=prepared.training_targets[train_idx],
                X_test=prepared.training_matrix,
                method_config=context.method_config,
                X_cal=(
                    None
                    if cal_idx is None
                    else prepared.training_matrix[cal_idx]
                ),
                y_cal=(
                    None
                    if cal_idx is None
                    else prepared.training_targets[cal_idx]
                ),
            )
            fit = calibrated_fit.fit
            residual = prepared.training_targets - fit.predicted_binding_energies
            spread = np.asarray(fit.spread, dtype=float)
            if (
                predictor_name in _CALIBRATABLE_PREDICTORS
                and calibrated_fit.calibrator is None
            ):
                all_splits_calibrated = False
            rmse_values.append(float(np.sqrt(np.mean(residual**2))))
            miscalibration_values.append(
                float(
                    miscalibration_area(
                        prepared.training_targets,
                        fit.predicted_binding_energies,
                        spread,
                    )
                )
            )
            sharpness_values.append(float(sharpness_from_spread(spread)))
            dispersion_values.append(float(dispersion_from_spread(spread)))

        metrics_by_predictor[predictor_name] = {
            "n_train": len(prepared.training_targets),
            "cv_rmse_mean": float(np.mean(rmse_values)),
            "cv_rmse_std": float(np.std(rmse_values)),
            "miscalibration_area": float(np.mean(miscalibration_values)),
            "miscalibration_area_std": float(np.std(miscalibration_values)),
            "sharpness": float(np.mean(sharpness_values)),
            "sharpness_std": float(np.std(sharpness_values)),
            "dispersion": float(np.mean(dispersion_values)),
            "dispersion_std": float(np.std(dispersion_values)),
            "uncertainty_kind": (
                "calibrated"
                if predictor_name in _CALIBRATABLE_PREDICTORS and all_splits_calibrated
                else "spread_only"
            ),
            "evaluation_mode": evaluation_mode,
            "split_count": len(splits) if splits else 1,
        }
    if not metrics_by_predictor:
        return None
    return PredictorDiagnosticsResult(
        metrics_by_predictor=metrics_by_predictor,
    )


def predictor_diagnostics_frame(
    diagnostics: PredictorDiagnosticsResult,
    *,
    predictor_name: str,
) -> pd.DataFrame:
    metrics = diagnostics.metrics_by_predictor[predictor_name]
    return pd.DataFrame([metrics])


def _build_candidate_outputs(
    *,
    context: RankingContext,
    predictor_name: str,
    prepared: PreparedPredictorInputs,
    predicted_binding_energies: np.ndarray,
    spread: np.ndarray,
    raw_spread: np.ndarray | None = None,
    calibrator: ScalarSpreadCalibrator | None = None,
    extra_metadata: dict[str, object] | None = None,
) -> list[AdslabCandidate]:
    candidates: list[AdslabCandidate] = []
    for idx, (record, predicted_energy, uncertainty_value, features) in enumerate(
        zip(
        prepared.candidate_records,
        predicted_binding_energies,
        spread,
        prepared.candidate_matrix,
        strict=True,
        )
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
        raw_provenance = MethodProvenance(
            method_name=predictor_name,
            stage="raw_uncertainty",
            shot_count=context.inferred_shot_count,
            source_methods=prepared.model_names,
        )
        uncertainty = UncertaintyEstimate(
            value=float(uncertainty_value),
            metric="calibrated_spread" if calibrator is not None else "spread_only",
            provenance=(
                MethodProvenance(
                    method_name=predictor_name,
                    stage="uncertainty_calibration",
                    shot_count=context.inferred_shot_count,
                    source_methods=(predictor_name,),
                    metadata={
                        "calibration_method": calibrator.method,
                        "calibration_scale": float(calibrator.scale),
                    },
                )
                if calibrator is not None
                else provenance
            ),
            is_calibrated=calibrator is not None,
            calibration=(
                UncertaintyCalibration(
                    method=calibrator.method,
                    raw_value=float(raw_spread[idx]),
                    raw_metric="spread_only",
                    raw_provenance=raw_provenance,
                    metadata={
                        "scale": float(calibrator.scale),
                        "reference_coverage": float(calibrator.reference_coverage),
                    },
                )
                if calibrator is not None and raw_spread is not None
                else None
            ),
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

    train_idx, cal_idx = _maybe_partition_calibration_indices(
        np.arange(len(prepared.training_targets), dtype=int),
        method_config=context.method_config,
        seed=int(context.dataset_metadata.get("seed", 0) or 0),
        min_inner_train_size=1,
    )
    calibrated_fit = _fit_predictor_with_optional_calibration(
        "residual",
        X_train=prepared.training_matrix[train_idx],
        y_train=prepared.training_targets[train_idx],
        X_test=prepared.candidate_matrix,
        method_config=context.method_config,
        X_cal=(
            None if cal_idx is None else prepared.training_matrix[cal_idx]
        ),
        y_cal=(
            None if cal_idx is None else prepared.training_targets[cal_idx]
        ),
    )
    return _build_candidate_outputs(
        context=context,
        predictor_name="residual",
        prepared=prepared,
        predicted_binding_energies=calibrated_fit.fit.predicted_binding_energies,
        spread=calibrated_fit.fit.spread,
        raw_spread=calibrated_fit.raw_spread,
        calibrator=calibrated_fit.calibrator,
        extra_metadata=calibrated_fit.fit.metadata,
    )


def generate_weighted_simplex_candidates(context: RankingContext) -> list[AdslabCandidate]:
    prepared = _prepare_predictor_inputs(context, predictor_name="weighted_simplex")
    if len(prepared.training_targets) < 1:
        raise ValueError(
            "Predictor 'weighted_simplex' requires at least 1 usable validated reference."
        )
    train_idx, cal_idx = _maybe_partition_calibration_indices(
        np.arange(len(prepared.training_targets), dtype=int),
        method_config=context.method_config,
        seed=int(context.dataset_metadata.get("seed", 0) or 0),
        min_inner_train_size=1,
    )
    calibrated_fit = _fit_predictor_with_optional_calibration(
        "weighted_simplex",
        X_train=prepared.training_matrix[train_idx],
        y_train=prepared.training_targets[train_idx],
        X_test=prepared.candidate_matrix,
        method_config=context.method_config,
        X_cal=(
            None if cal_idx is None else prepared.training_matrix[cal_idx]
        ),
        y_cal=(
            None if cal_idx is None else prepared.training_targets[cal_idx]
        ),
    )
    return _build_candidate_outputs(
        context=context,
        predictor_name="weighted_simplex",
        prepared=prepared,
        predicted_binding_energies=calibrated_fit.fit.predicted_binding_energies,
        spread=calibrated_fit.fit.spread,
        raw_spread=calibrated_fit.raw_spread,
        calibrator=calibrated_fit.calibrator,
        extra_metadata=calibrated_fit.fit.metadata,
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
    prepared = _prepare_predictor_inputs(context, predictor_name="ridge")
    if len(prepared.training_targets) < 2:
        raise ValueError("Predictor 'ridge' requires at least 2 usable validated references.")
    fit = _fit_predictor_arrays(
        "ridge",
        X_train=prepared.training_matrix,
        y_train=prepared.training_targets,
        X_test=prepared.candidate_matrix,
        method_config=context.method_config,
    )
    return _build_candidate_outputs(
        context=context,
        predictor_name="ridge",
        prepared=prepared,
        predicted_binding_energies=fit.predicted_binding_energies,
        spread=fit.spread,
        extra_metadata=fit.metadata,
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
