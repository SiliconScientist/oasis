from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from oasis.experiment.policy_diagnostic import (
    PolicySelectionDiagnosticArtifact,
    PolicySelectionDiagnosticResults,
    load_policy_selection_diagnostic_artifact,
    normalize_policy_detail_frame,
    normalize_policy_summary_frame,
    save_policy_selection_diagnostic_artifact,
)
from oasis.learning_curve.results_io import LearningCurveSweepMetadata


class PolicySelectionDiagnosticTests(unittest.TestCase):
    @staticmethod
    def _metadata() -> LearningCurveSweepMetadata:
        return LearningCurveSweepMetadata(
            seed=17,
            min_train=4,
            max_train=12,
            step=2,
            n_repeats=3,
            enabled_models=("ridge", "weighted_linear"),
            dataset_tag="example_oh",
            dataset_size=24,
        )

    def test_normalize_policy_detail_frame_orders_and_coerces_types(self) -> None:
        frame = pd.DataFrame(
            {
                "budget": [8, 4],
                "repeat": [1, 0],
                "oracle_method": ["weighted_linear", "ridge"],
                "screening_selected_method": ["ridge", "ridge"],
                "oracle_outer_rmse": [0.21, 0.31],
                "screening_selected_outer_rmse": [0.24, 0.31],
                "regret": [0.03, 0.0],
                "screening_cv_rmse": [0.19, 0.29],
                "agreement": [False, True],
                "ignored": [1, 2],
            }
        )

        normalized = normalize_policy_detail_frame(frame)

        self.assertEqual(
            normalized.columns.tolist(),
            [
                "budget",
                "repeat",
                "oracle_method",
                "screening_selected_method",
                "oracle_outer_rmse",
                "screening_selected_outer_rmse",
                "regret",
                "screening_cv_rmse",
                "agreement",
            ],
        )
        self.assertEqual(normalized["budget"].tolist(), [4, 8])
        self.assertEqual(str(normalized["agreement"].dtype), "boolean")

    def test_normalize_policy_summary_frame_requires_all_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            normalize_policy_summary_frame(pd.DataFrame({"budget": [4]}))

    def test_policy_diagnostic_artifact_round_trip(self) -> None:
        artifact = PolicySelectionDiagnosticArtifact(
            metadata=self._metadata(),
            results=PolicySelectionDiagnosticResults(
                detail_df=pd.DataFrame(
                    {
                        "budget": [4, 4, 8],
                        "repeat": [0, 1, 0],
                        "oracle_method": ["ridge", "weighted_linear", "ridge"],
                        "screening_selected_method": ["ridge", "ridge", "ridge"],
                        "oracle_outer_rmse": [0.31, 0.28, 0.25],
                        "screening_selected_outer_rmse": [0.31, 0.33, 0.25],
                        "regret": [0.0, 0.05, 0.0],
                        "screening_cv_rmse": [0.29, 0.27, 0.23],
                        "agreement": [True, False, True],
                    }
                ),
                summary_df=pd.DataFrame(
                    {
                        "budget": [8, 4],
                        "mean_regret": [0.0, 0.025],
                        "std_regret": [0.0, 0.025],
                        "se_regret": [0.0, 0.0176776695],
                        "ci95_low": [0.0, -0.009648],
                        "ci95_high": [0.0, 0.059648],
                        "agreement_rate": [1.0, 0.5],
                        "oracle_outer_rmse_mean": [0.25, 0.295],
                        "screening_selected_outer_rmse_mean": [0.25, 0.32],
                    }
                ),
            ),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "policy_diagnostic.json"
            saved_path = save_policy_selection_diagnostic_artifact(artifact, path)
            restored = load_policy_selection_diagnostic_artifact(
                saved_path,
                expected_metadata=self._metadata(),
            )

        self.assertEqual(saved_path, path)
        self.assertEqual(restored.metadata, artifact.metadata)
        pd.testing.assert_frame_equal(restored.results.detail_df, artifact.results.detail_df)
        pd.testing.assert_frame_equal(restored.results.summary_df, artifact.results.summary_df)
