from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from oasis.candidate_ranking import (
    MlipAnomalyMetadata,
    MlipModelPrediction,
    RankingResult,
    ScreeningInputRecord,
    clear_registered_strategies,
    greedy_cost,
    rank_candidates,
    rank_candidates_from_result_files,
)
from oasis.candidate_ranking.registry import register_strategy


def _prediction(model_name: str, energy: float) -> MlipModelPrediction:
    return MlipModelPrediction(
        model_name=model_name,
        result_path=f"/tmp/{model_name}_result.json",
        predicted_binding_energy=energy,
        anomaly=MlipAnomalyMetadata(label="normal"),
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
        model_predictions=predictions,
    )


def _write_result_file(path: Path, entries: list[dict[str, object]]) -> None:
    payload: dict[str, object] = {
        "calculation_settings": {
            "chemical_bond_cutoff": 1.25,
            "n_crit_relax": 200,
        }
    }
    for entry in entries:
        payload[str(entry["reaction"])] = {
            "reference": {"ads_eng": 1.0},
            "single_calculation": {"ads_eng": entry["single"]},
            "final": {
                "median_num": 0,
                "ads_eng_median": entry["median"],
                "ads_seed_range": 0.0,
                "ads_eng_seed_range": 0.0,
            },
            "0": {
                "adslab_steps": 50,
                "substrate_displacement": 0.1,
                "max_bond_change": 5.0,
            },
            "metadata": {
                "reference": {
                    "parent_slab_id": entry["parent_slab_id"],
                    "adslab_id": entry["adslab_id"],
                    "adsorbate": "N",
                },
                "structures": {
                    "slab": {
                        "parent_slab_id": entry["parent_slab_id"],
                        "adslab_id": entry["adslab_id"],
                    },
                    "adslab": {
                        "parent_slab_id": entry["parent_slab_id"],
                        "adslab_id": entry["adslab_id"],
                        "adsorbate": "N",
                        "initial_site_label": entry.get("site_label", "bridge"),
                    },
                },
            },
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class CandidateRankingPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_registered_strategies()

    def tearDown(self) -> None:
        clear_registered_strategies()

    def test_greedy_cost_matches_transfer_shot_formula(self) -> None:
        costs = greedy_cost(
            mean=[0.0, 1.0, 2.0],
            std=[0.2, 0.4, 0.8],
            target=1.0,
            alpha=0.75,
        )

        self.assertEqual(costs.shape, (3,))
        self.assertAlmostEqual(float(costs[1]), 1.0 / 6.0)
        self.assertLess(float(costs[1]), float(costs[0]))
        self.assertLess(float(costs[1]), float(costs[2]))

    def test_rank_candidates_runs_zero_shot_through_registry(self) -> None:
        result = rank_candidates(
            candidate_records=(
                _record(
                    reaction="rxn-1->N*",
                    parent_slab_id="slab-1",
                    adslab_id="adslab-1",
                    predictions=(
                        _prediction("mace", 0.8),
                        _prediction("orb", 1.2),
                    ),
                ),
                _record(
                    reaction="rxn-2->N*",
                    parent_slab_id="slab-2",
                    adslab_id="adslab-2",
                    predictions=(
                        _prediction("mace", 0.0),
                        _prediction("orb", 2.0),
                    ),
                ),
            ),
            method_name="zero_shot",
            shot_count=0,
            target_binding_energy=1.0,
        )

        self.assertEqual(result.strategy_name, "zero_shot")
        self.assertEqual(result.metadata["shot_count"], 0)
        self.assertEqual(result.ranked_candidates[0].selected_adslab_id, "adslab-1")
        self.assertEqual(result.ranked_candidates[0].provenance["scoring_policy"], "greedy_cost")

    def test_rank_candidates_supports_future_registered_methods_without_branching(self) -> None:
        class _TwoShotStrategy:
            name = "two_shot"

            def rank(self, context) -> RankingResult:
                return RankingResult(
                    strategy_name=self.name,
                    adslab_candidates=(),
                    parent_candidates=(),
                    ranked_candidates=(),
                    metadata={"shot_count": context.shot_count},
                )

        register_strategy(_TwoShotStrategy())
        result = rank_candidates(
            candidate_records=(),
            method_name="two_shot",
            shot_count=2,
            target_binding_energy=1.0,
        )

        self.assertEqual(result.strategy_name, "two_shot")
        self.assertEqual(result.metadata["shot_count"], 2)

    def test_rank_candidates_from_result_files_runs_end_to_end(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            mace_path = base_dir / "mace" / "mace_result.json"
            orb_path = base_dir / "orb" / "orb_result.json"
            entries = [
                {
                    "reaction": "rxn-1->N*",
                    "parent_slab_id": "slab-1",
                    "adslab_id": "adslab-1",
                    "median": 0.8,
                    "single": 0.9,
                },
                {
                    "reaction": "rxn-2->N*",
                    "parent_slab_id": "slab-2",
                    "adslab_id": "adslab-2",
                    "median": 0.0,
                    "single": 0.1,
                },
            ]
            _write_result_file(mace_path, entries)
            _write_result_file(
                orb_path,
                [
                    {**entries[0], "median": 1.2, "single": 1.1},
                    {**entries[1], "median": 2.0, "single": 2.1},
                ],
            )

            result = rank_candidates_from_result_files(
                [mace_path, orb_path],
                method_name="zero_shot",
                target_binding_energy=1.0,
            )

        self.assertEqual(result.strategy_name, "zero_shot")
        self.assertEqual(len(result.adslab_candidates), 2)
        self.assertEqual(result.ranked_candidates[0].selected_adslab_id, "adslab-1")
