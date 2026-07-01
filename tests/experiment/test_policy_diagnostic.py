from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from oasis.experiment.policy_diagnostic import (
    PolicySelectionDiagnosticArtifact,
    PolicySelectionDiagnosticResults,
    build_policy_selection_diagnostic_results,
    build_policy_selection_detail_frame,
    derive_family_split_collection_from_shared_outer_splits,
    generate_shared_outer_splits,
    load_policy_selection_diagnostic_artifact,
    normalize_policy_detail_frame,
    normalize_policy_summary_frame,
    save_policy_selection_diagnostic_artifact,
    summarize_policy_detail_frame,
)
from oasis.learning_curve.results_io import (
    LearningCurveSweepMetadata,
    learning_curve_result_field_for_method_name,
)
from oasis.learning_curve.runners import SweepFamilyRunArtifacts
from oasis.sweep import LearningCurveResults, SweepDataset, SweepFamilyRequirements


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

    def test_summarize_policy_detail_frame_aggregates_by_budget(self) -> None:
        summary = summarize_policy_detail_frame(
            pd.DataFrame(
                {
                    "budget": [4, 4, 8],
                    "repeat": [0, 1, 0],
                    "oracle_method": ["ridge", "weighted_linear", "ridge"],
                    "screening_selected_method": ["ridge", "ridge", "ridge"],
                    "oracle_outer_rmse": [0.2, 0.25, 0.3],
                    "screening_selected_outer_rmse": [0.2, 0.35, 0.3],
                    "regret": [0.0, 0.1, 0.0],
                    "screening_cv_rmse": [0.1, 0.12, 0.15],
                    "agreement": [True, False, True],
                }
            )
        )

        self.assertEqual(summary["budget"].tolist(), [4, 8])
        self.assertEqual(summary["mean_regret"].tolist(), [0.05, 0.0])
        self.assertEqual(summary["agreement_rate"].tolist(), [0.5, 1.0])

    def test_derived_family_splits_preserve_shared_outer_test_sets(self) -> None:
        shared_splits = generate_shared_outer_splits(
            12,
            min_train=4,
            max_train=4,
            step=1,
            n_repeats=2,
            seed=17,
        )

        class PlainFamily:
            method_name = "ridge"

            def requirements(self):
                return SweepFamilyRequirements()

        class ValidationFamily:
            method_name = "weighted_linear"

            def requirements(self):
                return SweepFamilyRequirements(requires_inner_validation=True)

        plain = derive_family_split_collection_from_shared_outer_splits(
            shared_splits,
            family=PlainFamily(),
            seed=17,
            validation_fraction=0.25,
            min_val_size=1,
            min_tuning_val_size=1,
            calibration_enabled=False,
            calibration_fraction=0.2,
            min_cal_size=1,
            min_inner_train_size=1,
        )
        validation = derive_family_split_collection_from_shared_outer_splits(
            shared_splits,
            family=ValidationFamily(),
            seed=17,
            validation_fraction=0.25,
            min_val_size=1,
            min_tuning_val_size=1,
            calibration_enabled=False,
            calibration_fraction=0.2,
            min_cal_size=1,
            min_inner_train_size=1,
        )

        self.assertEqual(len(plain.splits), len(validation.splits))
        for plain_split, validation_split in zip(plain.splits, validation.splits, strict=True):
            np.testing.assert_array_equal(plain_split.test_idx, validation_split.test_idx)
            self.assertEqual(plain_split.sweep_size, validation_split.sweep_size)

    def test_build_policy_selection_detail_frame_uses_shared_outer_splits(self) -> None:
        dataset = SweepDataset(
            mlip_features=np.arange(12, dtype=float).reshape(-1, 1),
            targets=np.linspace(0.0, 1.1, 12),
            sample_ids=np.arange(12, dtype=int),
        )
        shared_splits = generate_shared_outer_splits(
            dataset.n_samples,
            min_train=4,
            max_train=4,
            step=1,
            n_repeats=2,
            seed=11,
        )
        train_signature_by_repeat = {
            split.repeat: tuple(sorted(int(value) for value in split.train_idx.tolist()))
            for split in shared_splits
        }
        test_signature_by_repeat = {
            split.repeat: (
                split.budget,
                tuple(sorted(int(value) for value in split.test_idx.tolist())),
            )
            for split in shared_splits
        }

        class StubFamily:
            def __init__(
                self,
                *,
                method_name: str,
                requirements: SweepFamilyRequirements,
                outer_scores_by_signature: dict[tuple[int, tuple[int, ...]], float],
                screening_scores_by_signature: dict[tuple[int, ...], float],
            ) -> None:
                self.method_name = method_name
                self._requirements = requirements
                self._outer_scores_by_signature = outer_scores_by_signature
                self._screening_scores_by_signature = screening_scores_by_signature

            def requirements(self):
                return self._requirements

            def run(self, payload):
                split = payload.split_collection.splits[0]
                result_field = learning_curve_result_field_for_method_name(self.method_name)
                assert result_field is not None
                screening_key = tuple(sorted(int(value) for value in payload.dataset.sample_ids.tolist()))
                score = self._screening_scores_by_signature[screening_key]
                return LearningCurveResults.from_mapping(
                    {
                        result_field: pd.DataFrame(
                            {
                                "n_train": [split.sweep_size],
                                "rmse_mean": [score],
                                "rmse_std": [0.0],
                            }
                        )
                    }
                )

            def run_with_artifacts(self, payload):
                result_field = learning_curve_result_field_for_method_name(self.method_name)
                assert result_field is not None
                repeats_by_budget: dict[int, int] = {}
                repeat_rows = []
                rmse_values = []
                for split in payload.split_collection.splits:
                    repeat = repeats_by_budget.get(split.sweep_size, 0)
                    repeats_by_budget[split.sweep_size] = repeat + 1
                    signature = (
                        int(split.sweep_size),
                        tuple(sorted(int(value) for value in split.test_idx.tolist())),
                    )
                    rmse = self._outer_scores_by_signature[signature]
                    repeat_rows.append(
                        {
                            "n_train": int(split.sweep_size),
                            "repeat": repeat,
                            "outer_test_rmse": rmse,
                        }
                    )
                    rmse_values.append(rmse)
                return SweepFamilyRunArtifacts(
                    results=LearningCurveResults.from_mapping(
                        {
                            result_field: pd.DataFrame(
                                {
                                    "n_train": [payload.split_collection.splits[0].sweep_size],
                                    "rmse_mean": [float(np.mean(rmse_values))],
                                    "rmse_std": [float(np.std(rmse_values))],
                                }
                            )
                        }
                    ),
                    repeat_metrics=pd.DataFrame(repeat_rows),
                )

        ridge = StubFamily(
            method_name="ridge",
            requirements=SweepFamilyRequirements(),
            outer_scores_by_signature={
                test_signature_by_repeat[0]: 0.20,
                test_signature_by_repeat[1]: 0.35,
            },
            screening_scores_by_signature={
                train_signature_by_repeat[0]: 0.10,
                train_signature_by_repeat[1]: 0.12,
            },
        )
        weighted_linear = StubFamily(
            method_name="weighted_linear",
            requirements=SweepFamilyRequirements(requires_inner_validation=True),
            outer_scores_by_signature={
                test_signature_by_repeat[0]: 0.30,
                test_signature_by_repeat[1]: 0.25,
            },
            screening_scores_by_signature={
                train_signature_by_repeat[0]: 0.15,
                train_signature_by_repeat[1]: 0.11,
            },
        )

        detail = build_policy_selection_detail_frame(
            dataset,
            min_train=4,
            max_train=4,
            step=1,
            n_repeats=2,
            seed=11,
            model_families=[ridge, weighted_linear],
            outer_validation_fraction=0.25,
            outer_min_val_size=1,
            outer_min_tuning_val_size=1,
            outer_calibration_enabled=False,
            outer_calibration_fraction=0.2,
            outer_min_cal_size=1,
            outer_min_inner_train_size=1,
            min_test_size=1,
            screening_fraction=0.25,
            min_screen_size=1,
            screening_validation_fraction=0.25,
            screening_min_val_size=1,
            screening_min_tuning_val_size=1,
            screening_calibration_enabled=False,
            screening_calibration_fraction=0.2,
            screening_min_cal_size=1,
            screening_min_inner_train_size=1,
        )

        pd.testing.assert_frame_equal(
            detail,
            pd.DataFrame(
                {
                    "budget": pd.Series([4, 4], dtype="Int64"),
                    "repeat": pd.Series([0, 1], dtype="Int64"),
                    "oracle_method": pd.Series(
                        ["ridge", "weighted_linear"], dtype="string"
                    ),
                    "screening_selected_method": pd.Series(
                        ["ridge", "weighted_linear"], dtype="string"
                    ),
                    "oracle_outer_rmse": pd.Series([0.2, 0.25], dtype="Float64"),
                    "screening_selected_outer_rmse": pd.Series([0.2, 0.25], dtype="Float64"),
                    "regret": pd.Series([0.0, 0.0], dtype="Float64"),
                    "screening_cv_rmse": pd.Series([0.1, 0.11], dtype="Float64"),
                    "agreement": pd.Series([True, True], dtype="boolean"),
                }
            ),
        )
        results = build_policy_selection_diagnostic_results(
            dataset,
            min_train=4,
            max_train=4,
            step=1,
            n_repeats=2,
            seed=11,
            model_families=[ridge, weighted_linear],
            outer_validation_fraction=0.25,
            outer_min_val_size=1,
            outer_min_tuning_val_size=1,
            outer_calibration_enabled=False,
            outer_calibration_fraction=0.2,
            outer_min_cal_size=1,
            outer_min_inner_train_size=1,
            min_test_size=1,
            screening_fraction=0.25,
            min_screen_size=1,
            screening_validation_fraction=0.25,
            screening_min_val_size=1,
            screening_min_tuning_val_size=1,
            screening_calibration_enabled=False,
            screening_calibration_fraction=0.2,
            screening_min_cal_size=1,
            screening_min_inner_train_size=1,
        )
        self.assertEqual(results.summary_df["budget"].tolist(), [4])
        self.assertEqual(results.summary_df["agreement_rate"].tolist(), [1.0])
