from __future__ import annotations

import unittest
from dataclasses import replace

from oasis.candidate_ranking import (
    AdslabCandidate,
    CandidateGenerator,
    CandidateScorer,
    ParentCandidate,
    ParentReducer,
    RankingContext,
    RankingResult,
    clear_registered_strategies,
    get_strategy,
    register_strategy,
    registered_strategy_names,
)


class _FakeGenerator:
    def generate(self, context: RankingContext) -> list[AdslabCandidate]:
        del context
        return [
            AdslabCandidate(
                parent_slab_id="slab-1",
                adslab_id="adslab-1",
                predicted_binding_energy=0.4,
                uncertainty=0.2,
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
                uncertainty=candidate.uncertainty,
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

    def test_strategy_composes_generator_reducer_and_scorer(self) -> None:
        strategy = _FakeStrategy()

        result = strategy.rank(RankingContext(target_binding_energy=0.0))

        self.assertEqual(result.strategy_name, "zero_shot")
        self.assertEqual(len(result.adslab_candidates), 1)
        self.assertEqual(len(result.parent_candidates), 1)
        self.assertEqual(len(result.ranked_candidates), 1)
        self.assertEqual(result.ranked_candidates[0].selected_adslab_id, "adslab-1")
        self.assertAlmostEqual(result.ranked_candidates[0].score or -1.0, 0.4)
