from __future__ import annotations

import unittest
from dataclasses import replace

from oasis.candidate_ranking import (
    AdslabCandidate,
    CandidateGenerator,
    CandidateScorer,
    MethodProvenance,
    ParentCandidate,
    ParentReducer,
    RankingContext,
    RankingResult,
    SupportingSignal,
    UncertaintyEstimate,
    ValidatedReference,
    clear_registered_strategies,
    ensure_strategy,
    get_strategy,
    register_builtin_strategies,
    register_strategies,
    register_strategy,
    registered_strategy_names,
)


class _FakeGenerator:
    def generate(self, context: RankingContext) -> list[AdslabCandidate]:
        del context
        provenance = MethodProvenance(
            method_name="zero_shot",
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


class _FakeStrategy:
    name = "zero_shot"

    def __init__(self) -> None:
        self.generator = _FakeGenerator()
        self.reducer = _FakeReducer()
        self.scorer = _FakeScorer()

    def rank(self, context: RankingContext) -> RankingResult:
        adslab_candidates = self.generator.generate(context)
        parent_candidates = self.reducer.reduce(adslab_candidates, context)
        ranked_candidates = self.scorer.score(parent_candidates, context)
        return RankingResult(
            strategy_name=self.name,
            adslab_candidates=tuple(adslab_candidates),
            parent_candidates=tuple(parent_candidates),
            ranked_candidates=tuple(ranked_candidates),
        )


class CandidateRankingInterfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_registered_strategies()

    def tearDown(self) -> None:
        clear_registered_strategies()

    def test_fake_components_satisfy_protocols(self) -> None:
        self.assertIsInstance(_FakeGenerator(), CandidateGenerator)
        self.assertIsInstance(_FakeReducer(), ParentReducer)
        self.assertIsInstance(_FakeScorer(), CandidateScorer)

    def test_register_and_resolve_strategy_by_name(self) -> None:
        strategy = _FakeStrategy()

        register_strategy(strategy)

        self.assertEqual(registered_strategy_names(), ("zero_shot",))
        self.assertIs(get_strategy("zero_shot"), strategy)

    def test_duplicate_registration_fails_cleanly(self) -> None:
        register_strategy(_FakeStrategy())

        with self.assertRaisesRegex(ValueError, "already registered"):
            register_strategy(_FakeStrategy())

    def test_register_builtin_strategies_seeds_zero_shot(self) -> None:
        register_builtin_strategies()

        self.assertEqual(registered_strategy_names(), ("zero_shot",))
        self.assertEqual(get_strategy("zero_shot").name, "zero_shot")

    def test_ensure_strategy_lazily_registers_builtins(self) -> None:
        strategy = ensure_strategy("zero_shot")

        self.assertEqual(strategy.name, "zero_shot")
        self.assertEqual(registered_strategy_names(), ("zero_shot",))

    def test_future_methods_extend_registry_by_registration(self) -> None:
        class _TwoShotStrategy:
            name = "two_shot"

            def rank(self, context: RankingContext) -> RankingResult:
                del context
                return RankingResult(
                    strategy_name=self.name,
                    adslab_candidates=(),
                    parent_candidates=(),
                    ranked_candidates=(),
                    metadata={"note": "registered extension seam"},
                )

        class _NShotStrategy:
            name = "n_shot"

            def rank(self, context: RankingContext) -> RankingResult:
                del context
                return RankingResult(
                    strategy_name=self.name,
                    adslab_candidates=(),
                    parent_candidates=(),
                    ranked_candidates=(),
                )

        register_builtin_strategies()
        register_strategies([_TwoShotStrategy(), _NShotStrategy()])

        self.assertEqual(
            registered_strategy_names(),
            ("n_shot", "two_shot", "zero_shot"),
        )

    def test_strategy_composes_generator_reducer_and_scorer(self) -> None:
        strategy = _FakeStrategy()

        result = strategy.rank(RankingContext(target_binding_energy=0.0))

        self.assertEqual(result.strategy_name, "zero_shot")
        self.assertEqual(len(result.adslab_candidates), 1)
        self.assertEqual(len(result.parent_candidates), 1)
        self.assertEqual(len(result.ranked_candidates), 1)
        self.assertEqual(result.ranked_candidates[0].selected_adslab_id, "adslab-1")
        self.assertAlmostEqual(result.ranked_candidates[0].score or -1.0, 0.4)
        self.assertEqual(
            result.ranked_candidates[0].method_provenance.method_name,
            "zero_shot",
        )
        self.assertEqual(
            result.ranked_candidates[0].uncertainty.metric,
            "ensemble_std",
        )
        self.assertEqual(
            result.ranked_candidates[0].supporting_signals[0].name,
            "target_distance_proxy",
        )

    def test_candidate_records_expose_future_method_hooks(self) -> None:
        provenance = MethodProvenance(
            method_name="two_shot",
            stage="ranking",
            shot_count=2,
            source_methods=("zero_shot", "gp_refit"),
            metadata={"fit_id": "trial-7"},
        )
        uncertainty = UncertaintyEstimate(
            value=0.18,
            metric="posterior_std",
            provenance=provenance,
            is_calibrated=True,
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
        self.assertEqual(candidate.supporting_signals[0].objective, "maximize")
        self.assertEqual(parent.selected_adslab_ids, ("adslab-1", "adslab-4"))
        self.assertEqual(parent.adslab_metadata["site_label"], "bridge")

    def test_ranking_context_prefers_validated_reference_count_for_shot_inference(self) -> None:
        context = RankingContext(
            shot_count=9,
            validated_references=(
                ValidatedReference(adslab_id="adslab-1", adsorption_energy=-0.2),
                ValidatedReference(adslab_id="adslab-2", adsorption_energy=-0.4),
            ),
        )

        self.assertEqual(context.inferred_shot_count, 2)
