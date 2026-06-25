from __future__ import annotations

import unittest
from dataclasses import asdict
from dataclasses import replace

from oasis.candidate_ranking import (
    AdslabCandidate,
    CandidateGenerator,
    CandidatePredictor,
    CandidateScorer,
    MethodProvenance,
    ParentCandidate,
    ParentReducer,
    PredictorSpec,
    RankingContext,
    RankingResult,
    SupportingSignal,
    UncertaintyCalibration,
    UncertaintyEstimate,
    ValidatedReference,
    clear_registered_predictors,
    ensure_predictor,
    get_predictor,
    register_builtin_predictors,
    register_predictors,
    register_predictor,
    registered_predictor_names,
)
from oasis.learning_curve.registry import minimum_training_size_for_learning_curve_model


class _FakeGenerator:
    def generate(self, context: RankingContext) -> list[AdslabCandidate]:
        del context
        provenance = MethodProvenance(
            method_name="unfitted_ensemble_baseline",
            stage="candidate_generation",
            source_methods=("ensemble_mean",),
        )
        return [
            AdslabCandidate(
                parent_slab_id="slab-1",
                adslab_id="adslab-1",
                predicted_binding_energy=0.4,
                method_provenance=provenance,
                uncertainty=UncertaintyEstimate(
                    value=0.2,
                    metric="ensemble_std",
                    provenance=provenance,
                ),
                supporting_signals=(
                    SupportingSignal(
                        name="target_distance_proxy",
                        value=0.4,
                        provenance=provenance,
                    ),
                ),
            )
        ]


class _FakeReducer:
    def reduce(
        self,
        candidates: list[AdslabCandidate],
        context: RankingContext,
    ) -> list[ParentCandidate]:
        del context
        candidate = candidates[0]
        return [
            ParentCandidate(
                parent_slab_id=candidate.parent_slab_id,
                selected_adslab_id=candidate.adslab_id,
                predicted_binding_energy=candidate.predicted_binding_energy,
                method_provenance=candidate.method_provenance,
                uncertainty=candidate.uncertainty,
                supporting_signals=candidate.supporting_signals,
                selected_adslab_ids=(candidate.adslab_id,),
            )
        ]


class _FakeScorer:
    def score(
        self,
        candidates: list[ParentCandidate],
        context: RankingContext,
    ) -> list[ParentCandidate]:
        target = context.target_binding_energy or 0.0
        return [
            replace(
                candidate,
                score=abs((candidate.predicted_binding_energy or 0.0) - target),
            )
            for candidate in candidates
        ]


class _FakePredictor:
    name = "residual"
    min_validated_references = 1

    def is_feasible(self, validated_reference_count: int) -> bool:
        return validated_reference_count >= self.min_validated_references

    def generate(self, context: RankingContext) -> list[AdslabCandidate]:
        return _FakeGenerator().generate(context)


class CandidateRankingInterfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_registered_predictors()

    def tearDown(self) -> None:
        clear_registered_predictors()

    def test_fake_components_satisfy_protocols(self) -> None:
        self.assertIsInstance(_FakeGenerator(), CandidateGenerator)
        self.assertIsInstance(_FakeReducer(), ParentReducer)
        self.assertIsInstance(_FakeScorer(), CandidateScorer)
        self.assertIsInstance(_FakePredictor(), CandidatePredictor)

    def test_register_and_resolve_predictor_by_name(self) -> None:
        predictor = _FakePredictor()

        register_predictor(predictor)

        self.assertEqual(registered_predictor_names(), ("residual",))
        self.assertIs(get_predictor("residual"), predictor)

    def test_duplicate_registration_fails_cleanly(self) -> None:
        register_predictor(_FakePredictor())

        with self.assertRaisesRegex(ValueError, "already registered"):
            register_predictor(_FakePredictor())

    def test_register_builtin_predictors_seeds_predictor_names(self) -> None:
        register_builtin_predictors()

        self.assertEqual(
            registered_predictor_names(),
            ("residual", "ridge", "weighted_simplex"),
        )
        self.assertEqual(
            get_predictor("residual").min_validated_references,
            minimum_training_size_for_learning_curve_model("residual"),
        )
        self.assertEqual(
            get_predictor("weighted_simplex").min_validated_references,
            minimum_training_size_for_learning_curve_model("weighted_simplex"),
        )
        self.assertEqual(
            get_predictor("ridge").min_validated_references,
            minimum_training_size_for_learning_curve_model("ridge"),
        )

    def test_ensure_predictor_lazily_registers_builtins(self) -> None:
        predictor = ensure_predictor("weighted_simplex")

        self.assertEqual(predictor.name, "weighted_simplex")
        self.assertEqual(
            registered_predictor_names(),
            ("residual", "ridge", "weighted_simplex"),
        )

    def test_future_predictors_extend_registry_by_registration(self) -> None:
        register_builtin_predictors()
        register_predictors(
            (
                PredictorSpec(name="lasso", min_validated_references=2),
                PredictorSpec(name="kernel_ridge", min_validated_references=2),
            )
        )

        self.assertEqual(
            registered_predictor_names(),
            ("kernel_ridge", "lasso", "residual", "ridge", "weighted_simplex"),
        )

    def test_strategy_composes_generator_reducer_and_scorer(self) -> None:
        adslab_candidates = _FakeGenerator().generate(RankingContext(target_binding_energy=0.0))
        parent_candidates = _FakeReducer().reduce(adslab_candidates, RankingContext())
        ranked_candidates = _FakeScorer().score(parent_candidates, RankingContext(target_binding_energy=0.0))

        self.assertEqual(len(adslab_candidates), 1)
        self.assertEqual(len(parent_candidates), 1)
        self.assertEqual(len(ranked_candidates), 1)
        self.assertEqual(ranked_candidates[0].selected_adslab_id, "adslab-1")
        self.assertAlmostEqual(ranked_candidates[0].score or -1.0, 0.4)
        self.assertEqual(
            ranked_candidates[0].method_provenance.method_name,
            "unfitted_ensemble_baseline",
        )
        self.assertEqual(
            ranked_candidates[0].uncertainty.metric,
            "ensemble_std",
        )
        self.assertEqual(
            ranked_candidates[0].supporting_signals[0].name,
            "target_distance_proxy",
        )

    def test_candidate_records_expose_future_method_hooks(self) -> None:
        provenance = MethodProvenance(
            method_name="two_shot",
            stage="ranking",
            shot_count=2,
            source_methods=("unfitted_ensemble_baseline", "gp_refit"),
            metadata={"fit_id": "trial-7"},
        )
        uncertainty = UncertaintyEstimate(
            value=0.18,
            metric="posterior_std",
            provenance=provenance,
            is_calibrated=True,
            calibration=UncertaintyCalibration(
                method="scalar_scale",
                raw_value=0.12,
                raw_metric="spread_only",
                raw_provenance=provenance,
                metadata={"reference_coverage": 0.68},
            ),
            metadata={"calibration_split": "val"},
        )
        signal = SupportingSignal(
            name="expected_improvement",
            value=0.07,
            provenance=provenance,
            objective="maximize",
            metadata={"target_binding_energy": -0.5},
        )

        candidate = AdslabCandidate(
            parent_slab_id="slab-2",
            adslab_id="adslab-4",
            predicted_binding_energy=-0.4,
            method_provenance=provenance,
            uncertainty=uncertainty,
            supporting_signals=(signal,),
            metadata={"site_label": "bridge"},
        )
        parent = ParentCandidate(
            parent_slab_id="slab-2",
            selected_adslab_id="adslab-4",
            predicted_binding_energy=-0.4,
            method_provenance=provenance,
            uncertainty=uncertainty,
            score=0.11,
            supporting_signals=(signal,),
            selected_adslab_ids=("adslab-1", "adslab-4"),
            adslab_metadata={"site_label": "bridge"},
        )

        self.assertEqual(candidate.method_provenance.shot_count, 2)
        self.assertTrue(candidate.uncertainty.is_calibrated)
        self.assertEqual(candidate.uncertainty.calibration.raw_value, 0.12)
        self.assertEqual(candidate.supporting_signals[0].objective, "maximize")
        self.assertEqual(parent.selected_adslab_ids, ("adslab-1", "adslab-4"))
        self.assertEqual(parent.adslab_metadata["site_label"], "bridge")

    def test_uncertainty_calibration_serializes_raw_and_calibrated_payloads(self) -> None:
        raw_provenance = MethodProvenance(
            method_name="residual",
            stage="candidate_prediction",
            metadata={"fit_id": "split-3"},
        )
        uncertainty = UncertaintyEstimate(
            value=0.22,
            metric="calibrated_spread",
            provenance=MethodProvenance(
                method_name="residual",
                stage="uncertainty_calibration",
                source_methods=("residual",),
            ),
            is_calibrated=True,
            calibration=UncertaintyCalibration(
                method="scalar_scale",
                raw_value=0.11,
                raw_metric="spread_only",
                raw_provenance=raw_provenance,
                metadata={"scale": 2.0},
            ),
        )

        payload = asdict(uncertainty)

        self.assertTrue(payload["is_calibrated"])
        self.assertEqual(payload["value"], 0.22)
        self.assertEqual(payload["calibration"]["raw_value"], 0.11)
        self.assertEqual(payload["calibration"]["raw_metric"], "spread_only")
        self.assertEqual(
            payload["calibration"]["raw_provenance"]["stage"],
            "candidate_prediction",
        )

    def test_ranking_context_prefers_validated_reference_count_for_shot_inference(self) -> None:
        context = RankingContext(
            shot_count=9,
            validated_references=(
                ValidatedReference(adslab_id="adslab-1", adsorption_energy=-0.2),
                ValidatedReference(adslab_id="adslab-2", adsorption_energy=-0.4),
            ),
        )

        self.assertEqual(context.inferred_shot_count, 2)
        self.assertEqual(context.validated_reference_count, 2)
        self.assertEqual(
            context.validated_reference_identities,
            ("adslab:adslab-1", "adslab:adslab-2"),
        )

    def test_validated_reference_requires_one_stable_identifier(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires at least one stable identifier"):
            ValidatedReference(adsorption_energy=-0.2).identity
