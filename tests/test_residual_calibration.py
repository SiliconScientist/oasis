from __future__ import annotations

import unittest

import numpy as np

from oasis.candidate_ranking.predictors import evaluate_predictors, generate_residual_candidates
from oasis.candidate_ranking.types import (
    MlipAnomalyMetadata,
    MlipModelPrediction,
    RankingContext,
    ScreeningInputRecord,
    ValidatedReference,
)
from oasis.experiment import build_sweep_split_collection
from oasis.learning_curve.execution import residual_sweep
from oasis.sweep import SweepDataset, SweepRunPayload, SweepFamilyRequirements


def _prediction(model_name: str, energy: float) -> MlipModelPrediction:
    return MlipModelPrediction(
        model_name=model_name,
        result_path=f"/tmp/{model_name}.json",
        predicted_binding_energy=energy,
        anomaly=MlipAnomalyMetadata(label="normal"),
    )


def _record(index: int, a: float, b: float) -> ScreeningInputRecord:
    return ScreeningInputRecord(
        reaction=f"rxn-{index}",
        parent_slab_id=f"slab-{index}",
        adslab_id=f"adslab-{index}",
        model_predictions=(
            _prediction("mace", a),
            _prediction("orb", b),
        ),
    )


class ResidualCalibrationTests(unittest.TestCase):
    def test_residual_sweep_marks_calibrated_uq_when_calibration_split_is_present(self) -> None:
        X = np.array(
            [
                [0.0, 2.0],
                [1.0, 3.0],
                [2.0, 4.0],
                [3.0, 5.0],
                [4.0, 6.0],
                [5.0, 7.0],
            ],
            dtype=float,
        )
        y = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], dtype=float)
        payload = SweepRunPayload(
            dataset=SweepDataset(mlip_features=X, targets=y),
            split_collection=build_sweep_split_collection(
                n_samples=len(y),
                min_train=4,
                max_train=4,
                n_repeats=2,
                seed=4,
                requirements=SweepFamilyRequirements(requires_calibration=True),
                calibration_fraction=0.25,
                min_cal_size=1,
                min_inner_train_size=2,
                min_test_size=2,
            ),
        )

        result = residual_sweep(payload)

        self.assertEqual(result.uq_summary["uncertainty_kind"].tolist(), ["calibrated"])

    def test_generate_residual_candidates_emits_calibrated_uncertainty_when_feasible(self) -> None:
        records = tuple(
            _record(i, float(i), float(i + 2))
            for i in range(6)
        )
        context = RankingContext(
            candidate_records=records,
            validated_references=tuple(
                ValidatedReference(adslab_id=f"adslab-{i}", adsorption_energy=float(i) + 0.5)
                for i in range(6)
            ),
            dataset_metadata={"seed": 9},
            method_config={"calibration_fraction": 0.25, "min_cal_size": 1},
        )

        candidates = generate_residual_candidates(context)

        self.assertTrue(all(candidate.uncertainty is not None for candidate in candidates))
        self.assertTrue(all(candidate.uncertainty.is_calibrated for candidate in candidates))
        self.assertTrue(
            all(candidate.uncertainty.calibration is not None for candidate in candidates)
        )
        self.assertEqual(candidates[0].uncertainty.metric, "calibrated_spread")

    def test_evaluate_predictors_marks_residual_calibrated_when_inner_calibration_is_feasible(
        self,
    ) -> None:
        records = tuple(
            _record(i, float(i), float(i + 2))
            for i in range(6)
        )
        context = RankingContext(
            candidate_records=records,
            validated_references=tuple(
                ValidatedReference(adslab_id=f"adslab-{i}", adsorption_energy=float(i) + 0.5)
                for i in range(6)
            ),
            dataset_metadata={"seed": 3},
            method_config={
                "selection_n_repeats": 1,
                "selection_screen_fraction": 0.25,
                "selection_min_screen_size": 1,
                "selection_min_test_size": 1,
                "calibration_fraction": 0.25,
                "min_cal_size": 1,
                "min_inner_train_size": 2,
            },
        )

        diagnostics = evaluate_predictors(("residual",), context=context)

        assert diagnostics is not None
        self.assertEqual(
            diagnostics.metrics_by_predictor["residual"]["uncertainty_kind"],
            "calibrated",
        )


if __name__ == "__main__":
    unittest.main()
