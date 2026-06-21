from __future__ import annotations

import unittest

from oasis.candidate_ranking import (
    MlipAnomalyMetadata,
    MlipModelPrediction,
    RankingContext,
    ScreeningInputRecord,
    ZeroShotCandidateGenerator,
    ZeroShotCandidateRanker,
)
from oasis.candidate_ranking.registry import (
    clear_registered_strategies,
    get_strategy,
    register_strategy,
)


def _record(
    *,
    reaction: str,
    parent_slab_id: str,
    adslab_id: str,
    predictions: tuple[MlipModelPrediction, ...],
) -> ScreeningInputRecord:
    return ScreeningInputRecord(
        reaction=reaction,
        parent_slab_id=parent_slab_id,
        adslab_id=adslab_id,
        adsorbate="N",
        site_metadata={"initial_site_label": "bridge"},
        model_predictions=predictions,
    )


def _prediction(
    model_name: str,
    energy: float,
    *,
    label: str = "normal",
    details: dict[str, int] | None = None,
) -> MlipModelPrediction:
    return MlipModelPrediction(
        model_name=model_name,
        result_path=f"/tmp/{model_name}_result.json",
        predicted_binding_energy=energy,
        anomaly=MlipAnomalyMetadata(
            label=label,
            labels=(() if label == "normal" else (label,)),
            details=dict(details or {}),
        ),
    )


class ZeroShotCandidateRankingTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_registered_strategies()

    def tearDown(self) -> None:
        clear_registered_strategies()

    def test_zero_shot_generator_averages_valid_mlips_and_tracks_provenance(self) -> None:
        generator = ZeroShotCandidateGenerator()
        context = RankingContext(
            candidate_records=(
                _record(
                    reaction="rxn-1->N*",
                    parent_slab_id="slab-1",
                    adslab_id="adslab-1",
                    predictions=(
                        _prediction("mace", 1.0),
                        _prediction("orb", 3.0),
                    ),
                ),
            ),
        )

        candidates = generator.generate(context)

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertAlmostEqual(candidate.predicted_binding_energy or -1.0, 2.0)
        self.assertAlmostEqual(candidate.uncertainty.value, 1.0)
        self.assertEqual(candidate.method_provenance.source_methods, ("mace", "orb"))
        self.assertEqual(
            candidate.provenance["record_model_predictions"][0]["model_name"],
            "mace",
        )

    def test_zero_shot_generator_reuses_label_based_anomaly_filtering(self) -> None:
        generator = ZeroShotCandidateGenerator()
        context = RankingContext(
            candidate_records=(
                _record(
                    reaction="rxn-1->N*",
                    parent_slab_id="slab-1",
                    adslab_id="adslab-1",
                    predictions=(
                        _prediction("mace", 1.0),
                        _prediction("orb", 2.0, label="energy_anomaly"),
                    ),
                ),
                _record(
                    reaction="rxn-2->N*",
                    parent_slab_id="slab-2",
                    adslab_id="adslab-2",
                    predictions=(
                        _prediction("mace", 3.0),
                        _prediction("orb", 4.0),
                    ),
                ),
            ),
            method_config={
                "exclude_anomalous": True,
                "label_allowlist": ["normal"],
                "min_valid_mlips": 1,
            },
        )

        candidates = generator.generate(context)

        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0].method_provenance.source_methods, ("mace",))
        self.assertEqual(
            candidates[0].method_provenance.metadata["globally_allowed_models"],
            ["mace"],
        )

    def test_zero_shot_generator_can_use_strict_detail_filtering(self) -> None:
        generator = ZeroShotCandidateGenerator()
        context = RankingContext(
            candidate_records=(
                _record(
                    reaction="rxn-1->N*",
                    parent_slab_id="slab-1",
                    adslab_id="adslab-1",
                    predictions=(
                        _prediction("mace", 1.0, details={"ads_move": 0}),
                        _prediction("orb", 5.0, label="normal", details={"ads_move": 1}),
                    ),
                ),
            ),
            method_config={
                "exclude_anomalous": True,
                "strict_inference_anomaly": True,
                "min_valid_mlips": 1,
            },
        )

        candidates = generator.generate(context)

        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0].predicted_binding_energy or -1.0, 1.0)
        self.assertEqual(candidates[0].method_provenance.source_methods, ("mace",))

    def test_zero_shot_ranker_is_registry_compatible(self) -> None:
        strategy = ZeroShotCandidateRanker()
        register_strategy(strategy)

        resolved = get_strategy("zero_shot")
        result = resolved.rank(
            RankingContext(
                candidate_records=(
                    _record(
                        reaction="rxn-1->N*",
                        parent_slab_id="slab-1",
                        adslab_id="adslab-1",
                        predictions=(
                            _prediction("mace", 1.0),
                            _prediction("orb", 3.0),
                        ),
                    ),
                )
            )
        )

        self.assertIs(resolved, strategy)
        self.assertEqual(result.strategy_name, "zero_shot")
        self.assertEqual(len(result.adslab_candidates), 1)
        self.assertEqual(len(result.parent_candidates), 1)
        self.assertEqual(result.parent_candidates[0].selected_adslab_id, "adslab-1")
