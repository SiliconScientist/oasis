from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from oasis.candidate_ranking import (
    MlipAnomalyMetadata,
    MlipModelPrediction,
    PredictorSpec,
    RankingResult,
    ScreeningInputRecord,
    ValidatedReference,
    clear_registered_predictors,
    build_ranking_context,
    normalize_validated_references,
    rank_candidates,
    rank_candidates_from_result_files,
    select_predictor_name,
    target_uncertainty_cost,
)
from oasis.candidate_ranking.registry import register_predictor


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
        clear_registered_predictors()

    def tearDown(self) -> None:
        clear_registered_predictors()

    def test_target_uncertainty_cost_matches_transfer_shot_formula(self) -> None:
        costs = target_uncertainty_cost(
            mean=[0.0, 1.0, 2.0],
            std=[0.2, 0.4, 0.8],
            target=1.0,
            alpha=0.75,
        )

        self.assertEqual(costs.shape, (3,))
        self.assertAlmostEqual(float(costs[1]), 1.0 / 6.0)
        self.assertLess(float(costs[1]), float(costs[0]))
        self.assertLess(float(costs[1]), float(costs[2]))

    def test_rank_candidates_runs_unfitted_ensemble_baseline_for_zero_references(self) -> None:
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
            target_binding_energy=1.0,
        )

        self.assertEqual(result.strategy_name, "unfitted_ensemble_baseline")
        self.assertEqual(result.metadata["shot_count"], 0)
        self.assertEqual(result.ranked_candidates[0].selected_adslab_id, "adslab-1")
        self.assertEqual(
            result.ranked_candidates[0].provenance["scoring_policy"],
            "target_uncertainty_cost",
        )

    def test_rank_candidates_uses_baseline_as_no_reference_fallback(self) -> None:
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
            ),
            target_binding_energy=1.0,
            predictor_name="ridge",
        )

        self.assertEqual(result.strategy_name, "unfitted_ensemble_baseline")
        self.assertEqual(result.metadata["shot_count"], 0)

    def test_rank_candidates_requires_registered_predictor_for_n_shot_path(self) -> None:
        register_predictor(PredictorSpec(name="ridge", min_validated_references=2))

        with self.assertRaisesRegex(NotImplementedError, "Predictor-backed n-shot ranking"):
            rank_candidates(
                candidate_records=(),
                predictor_name="ridge",
                target_binding_energy=1.0,
                validated_references=(
                    ValidatedReference(adslab_id="adslab-1", adsorption_energy=-0.2),
                    ValidatedReference(adslab_id="adslab-2", adsorption_energy=-0.4),
                ),
            )

    def test_select_predictor_name_prefers_most_data_hungry_feasible_predictor(self) -> None:
        register_predictor(PredictorSpec(name="residual", min_validated_references=1))
        register_predictor(
            PredictorSpec(name="weighted_simplex", min_validated_references=1)
        )
        register_predictor(PredictorSpec(name="ridge", min_validated_references=2))

        self.assertEqual(
            select_predictor_name(
                ("residual", "weighted_simplex", "ridge"),
                validated_reference_count=1,
            ),
            "weighted_simplex",
        )
        self.assertEqual(
            select_predictor_name(
                ("residual", "weighted_simplex", "ridge"),
                validated_reference_count=2,
            ),
            "ridge",
        )

    def test_rank_candidates_can_resolve_predictor_from_predictor_list(self) -> None:
        register_predictor(PredictorSpec(name="residual", min_validated_references=1))
        register_predictor(PredictorSpec(name="ridge", min_validated_references=2))

        with self.assertRaisesRegex(NotImplementedError, "Predictor-backed n-shot ranking"):
            rank_candidates(
                candidate_records=(),
                predictor_names=("residual", "ridge"),
                target_binding_energy=1.0,
                validated_references=(
                    ValidatedReference(adslab_id="adslab-1", adsorption_energy=-0.2),
                    ValidatedReference(adslab_id="adslab-2", adsorption_energy=-0.4),
                ),
            )

    def test_rank_candidates_checks_predictor_feasibility_by_validated_reference_count(self) -> None:
        register_predictor(PredictorSpec(name="ridge", min_validated_references=2))

        with self.assertRaisesRegex(ValueError, "requires at least 2 validated references"):
            rank_candidates(
                candidate_records=(),
                predictor_name="ridge",
                target_binding_energy=1.0,
                validated_references=(
                    ValidatedReference(adslab_id="adslab-1", adsorption_energy=-0.2),
                ),
            )

    def test_rank_candidates_requires_feasible_predictor_from_predictor_list(self) -> None:
        register_predictor(PredictorSpec(name="ridge", min_validated_references=2))

        with self.assertRaisesRegex(ValueError, "feasible predictor must be provided"):
            rank_candidates(
                candidate_records=(),
                predictor_names=("ridge",),
                target_binding_energy=1.0,
                validated_references=(
                    ValidatedReference(adslab_id="adslab-1", adsorption_energy=-0.2),
                ),
            )

    def test_build_ranking_context_infers_shot_count_from_validated_references(self) -> None:
        context = build_ranking_context(
            candidate_records=(),
            shot_count=7,
            validated_references=(
                ValidatedReference(adslab_id="adslab-1", adsorption_energy=-0.2),
            ),
        )

        self.assertEqual(context.inferred_shot_count, 1)

    def test_normalize_validated_references_accepts_mapping_inputs(self) -> None:
        references = normalize_validated_references(
            (
                {
                    "adslab_id": "adslab-1",
                    "adsorption_energy": -0.2,
                    "metadata": {"source_run": "dft-1"},
                },
                {
                    "reaction": "rxn-2->N*",
                    "adsorption_energy": -0.4,
                },
            )
        )

        self.assertEqual(len(references), 2)
        self.assertEqual(references[0].identity, "adslab:adslab-1")
        self.assertEqual(references[0].metadata["source_run"], "dft-1")
        self.assertEqual(references[1].identity, "reaction:rxn-2->N*")

    def test_validated_references_remain_separate_from_candidate_records(self) -> None:
        context = build_ranking_context(
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
            ),
            validated_references=(
                ValidatedReference(
                    adslab_id="adslab-9",
                    adsorption_energy=-0.2,
                    reference_source="dft",
                ),
            ),
        )

        self.assertEqual(len(context.candidate_records), 1)
        self.assertEqual(len(context.validated_references), 1)
        self.assertEqual(context.candidate_records[0].adslab_id, "adslab-1")
        self.assertEqual(context.validated_references[0].adslab_id, "adslab-9")

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
                target_binding_energy=1.0,
            )

        self.assertEqual(result.strategy_name, "unfitted_ensemble_baseline")
        self.assertEqual(len(result.adslab_candidates), 2)
        self.assertEqual(result.ranked_candidates[0].selected_adslab_id, "adslab-1")
