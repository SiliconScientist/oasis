from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from oasis.experiment.policy_diagnostic import (
    PolicySelectionDiagnosticArtifact,
    PolicySelectionDiagnosticResults,
    ScreeningDiagnosticRowsArtifact,
    build_policy_selection_diagnostic_results,
    build_policy_selection_detail_frame,
    derive_family_split_collection_from_shared_outer_splits,
    generate_shared_outer_splits,
    load_policy_selection_diagnostic_artifact,
    load_screening_diagnostic_rows_artifact,
    normalize_policy_detail_frame,
    normalize_policy_summary_frame,
    normalize_screening_diagnostic_rows_frame,
    save_policy_selection_diagnostic_artifact,
    save_screening_diagnostic_rows_artifact,
    summarize_policy_detail_frame,
)
from oasis.learning_curve.results_io import (
    LearningCurveSweepMetadata,
    learning_curve_result_field_for_method_name,
    learning_curve_uq_field_for_method_name,
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

    @staticmethod
    def _cache_signature() -> dict[str, object]:
        return {
            "learning_curve": {
                "min_train": 4,
                "max_train": 12,
                "step": 2,
                "n_repeats": 3,
                "enabled_model_names": ["ridge", "weighted_linear"],
            },
            "screening": {
                "screen_fraction": 0.25,
                "policy_names": ["min_screening_rmse"],
                "combined_miscalibration_lambda": 1.0,
            },
        }

    def test_normalize_policy_detail_frame_orders_and_coerces_types(self) -> None:
        frame = pd.DataFrame(
            {
                "policy_name": ["min_screening_rmse", "min_screening_rmse"],
                "budget": [8, 4],
                "repeat": [1, 0],
                "oracle_method": ["weighted_linear", "ridge"],
                "screening_selected_method": ["ridge", "ridge"],
                "oracle_outer_rmse": [0.21, 0.31],
                "screening_selected_outer_rmse": [0.24, 0.31],
                "regret": [0.03, 0.0],
                "screening_cv_rmse": [0.19, 0.29],
                "screening_miscalibration_area": [0.08, 0.11],
                "agreement": [False, True],
                "ignored": [1, 2],
            }
        )

        normalized = normalize_policy_detail_frame(frame)

        self.assertEqual(
            normalized.columns.tolist(),
            [
                "policy_name",
                "budget",
                "repeat",
                "oracle_method",
                "screening_selected_method",
                "oracle_outer_rmse",
                "screening_selected_outer_rmse",
                "regret",
                "screening_cv_rmse",
                "screening_miscalibration_area",
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
                        "policy_name": [
                            "min_screening_rmse",
                            "min_screening_rmse",
                            "min_screening_rmse",
                        ],
                        "budget": [4, 4, 8],
                        "repeat": [0, 1, 0],
                        "oracle_method": ["ridge", "weighted_linear", "ridge"],
                        "screening_selected_method": ["ridge", "ridge", "ridge"],
                        "oracle_outer_rmse": [0.31, 0.28, 0.25],
                        "screening_selected_outer_rmse": [0.31, 0.33, 0.25],
                        "regret": [0.0, 0.05, 0.0],
                        "screening_cv_rmse": [0.29, 0.27, 0.23],
                        "screening_miscalibration_area": [0.11, 0.12, 0.09],
                        "agreement": [True, False, True],
                    }
                ),
                summary_df=pd.DataFrame(
                    {
                        "policy_name": ["min_screening_rmse", "min_screening_rmse"],
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
            cache_signature=self._cache_signature(),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "policy_diagnostic.json"
            saved_path = save_policy_selection_diagnostic_artifact(artifact, path)
            restored = load_policy_selection_diagnostic_artifact(
                saved_path,
                expected_metadata=self._metadata(),
                expected_cache_signature=self._cache_signature(),
            )

        self.assertEqual(saved_path, path)
        self.assertEqual(restored.metadata, artifact.metadata)
        self.assertEqual(restored.cache_signature, artifact.cache_signature)
        pd.testing.assert_frame_equal(restored.results.detail_df, artifact.results.detail_df)
        pd.testing.assert_frame_equal(restored.results.summary_df, artifact.results.summary_df)

    def test_policy_diagnostic_artifact_rejects_incompatible_cache_signature(self) -> None:
        artifact = PolicySelectionDiagnosticArtifact(
            metadata=self._metadata(),
            results=PolicySelectionDiagnosticResults(
                detail_df=pd.DataFrame(
                    {
                        "policy_name": ["min_screening_rmse"],
                        "budget": [4],
                        "repeat": [0],
                        "oracle_method": ["ridge"],
                        "screening_selected_method": ["ridge"],
                        "oracle_outer_rmse": [0.31],
                        "screening_selected_outer_rmse": [0.31],
                        "regret": [0.0],
                        "screening_cv_rmse": [0.29],
                        "screening_miscalibration_area": [0.11],
                        "agreement": [True],
                    }
                ),
                summary_df=pd.DataFrame(
                    {
                        "policy_name": ["min_screening_rmse"],
                        "budget": [4],
                        "mean_regret": [0.0],
                        "std_regret": [0.0],
                        "se_regret": [0.0],
                        "ci95_low": [0.0],
                        "ci95_high": [0.0],
                        "agreement_rate": [1.0],
                        "oracle_outer_rmse_mean": [0.31],
                        "screening_selected_outer_rmse_mean": [0.31],
                    }
                ),
            ),
            cache_signature=self._cache_signature(),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "policy_diagnostic.json"
            save_policy_selection_diagnostic_artifact(artifact, path)
            with self.assertRaisesRegex(ValueError, "cache signature is incompatible"):
                load_policy_selection_diagnostic_artifact(
                    path,
                    expected_metadata=self._metadata(),
                    expected_cache_signature={
                        **self._cache_signature(),
                        "screening": {
                            **self._cache_signature()["screening"],
                            "combined_miscalibration_lambda": 2.0,
                        },
                    },
                )

    def test_summarize_policy_detail_frame_aggregates_by_budget(self) -> None:
        summary = summarize_policy_detail_frame(
            pd.DataFrame(
                {
                    "policy_name": [
                        "min_screening_rmse",
                        "min_screening_rmse",
                        "min_screening_rmse",
                    ],
                    "budget": [4, 4, 8],
                    "repeat": [0, 1, 0],
                    "oracle_method": ["ridge", "weighted_linear", "ridge"],
                    "screening_selected_method": ["ridge", "ridge", "ridge"],
                    "oracle_outer_rmse": [0.2, 0.25, 0.3],
                    "screening_selected_outer_rmse": [0.2, 0.35, 0.3],
                    "regret": [0.0, 0.1, 0.0],
                    "screening_cv_rmse": [0.1, 0.12, 0.15],
                    "screening_miscalibration_area": [0.2, 0.3, 0.1],
                    "agreement": [True, False, True],
                }
            )
        )

        self.assertEqual(summary["budget"].tolist(), [4, 8])
        self.assertEqual(summary["mean_regret"].tolist(), [0.05, 0.0])
        self.assertEqual(summary["agreement_rate"].tolist(), [0.5, 1.0])

    def test_summarize_policy_detail_frame_groups_by_policy_name(self) -> None:
        summary = summarize_policy_detail_frame(
            pd.DataFrame(
                {
                    "policy_name": [
                        "combined_screening_rmse_miscalibration",
                        "min_screening_rmse",
                    ],
                    "budget": [4, 4],
                    "repeat": [0, 0],
                    "oracle_method": ["ridge", "ridge"],
                    "screening_selected_method": ["weighted_linear", "ridge"],
                    "oracle_outer_rmse": [0.2, 0.2],
                    "screening_selected_outer_rmse": [0.3, 0.2],
                    "regret": [0.1, 0.0],
                    "screening_cv_rmse": [0.11, 0.1],
                    "screening_miscalibration_area": [0.01, 0.05],
                    "agreement": [False, True],
                }
            )
        )

        self.assertEqual(
            summary["policy_name"].tolist(),
            [
                "combined_screening_rmse_miscalibration",
                "min_screening_rmse",
            ],
        )
        self.assertEqual(summary["mean_regret"].tolist(), [0.1, 0.0])

    def test_normalize_screening_diagnostic_rows_frame_orders_and_coerces_types(self) -> None:
        frame = pd.DataFrame(
            {
                "method": ["weighted_linear", "ridge"],
                "budget": [8, 4],
                "repeat": [1, 0],
                "screening_cv_rmse": [0.19, 0.29],
                "screening_miscalibration_area": [0.08, 0.11],
                "ignored": [1, 2],
            }
        )

        normalized = normalize_screening_diagnostic_rows_frame(frame)

        self.assertEqual(
            normalized.columns.tolist(),
            [
                "method",
                "budget",
                "repeat",
                "screening_cv_rmse",
                "screening_miscalibration_area",
            ],
        )
        self.assertEqual(normalized["method"].tolist(), ["ridge", "weighted_linear"])
        self.assertEqual(str(normalized["budget"].dtype), "Int64")

    def test_screening_diagnostic_rows_artifact_round_trip(self) -> None:
        artifact = ScreeningDiagnosticRowsArtifact(
            metadata=self._metadata(),
            screening_rows_df=pd.DataFrame(
                {
                    "method": ["weighted_linear", "ridge"],
                    "budget": [8, 4],
                    "repeat": [1, 0],
                    "screening_cv_rmse": [0.19, 0.29],
                    "screening_miscalibration_area": [0.08, 0.11],
                }
            ),
            cache_signature=self._cache_signature(),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "screening_rows.json"
            saved_path = save_screening_diagnostic_rows_artifact(artifact, path)
            restored = load_screening_diagnostic_rows_artifact(
                saved_path,
                expected_metadata=self._metadata(),
                expected_cache_signature=self._cache_signature(),
            )

        self.assertEqual(saved_path, path)
        self.assertEqual(restored.metadata, artifact.metadata)
        self.assertEqual(restored.cache_signature, artifact.cache_signature)
        pd.testing.assert_frame_equal(
            restored.screening_rows_df,
            normalize_screening_diagnostic_rows_frame(artifact.screening_rows_df),
        )

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

    def test_derived_family_splits_disable_calibration_requirement_when_config_disabled(
        self,
    ) -> None:
        shared_splits = generate_shared_outer_splits(
            12,
            min_train=4,
            max_train=4,
            step=1,
            n_repeats=1,
            seed=17,
        )

        class CalibratedValidationFamily:
            method_name = "ridge"

            def requirements(self):
                return SweepFamilyRequirements(
                    requires_inner_validation=True,
                    requires_calibration=True,
                )

        split_collection = derive_family_split_collection_from_shared_outer_splits(
            shared_splits,
            family=CalibratedValidationFamily(),
            seed=17,
            validation_fraction=0.25,
            min_val_size=1,
            min_tuning_val_size=1,
            calibration_enabled=False,
            calibration_fraction=0.2,
            min_cal_size=1,
            min_inner_train_size=1,
        )

        self.assertFalse(split_collection.planning_requirements.requires_calibration)
        self.assertTrue(split_collection.planning_requirements.requires_inner_validation)
        self.assertIsNone(split_collection.splits[0].cal_idx)
        self.assertIsNotNone(split_collection.splits[0].val_idx)

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
                screening_miscalibration_by_signature: dict[tuple[int, ...], float] | None = None,
            ) -> None:
                self.method_name = method_name
                self._requirements = requirements
                self._outer_scores_by_signature = outer_scores_by_signature
                self._screening_scores_by_signature = screening_scores_by_signature
                self._screening_miscalibration_by_signature = (
                    {}
                    if screening_miscalibration_by_signature is None
                    else screening_miscalibration_by_signature
                )

            def requirements(self):
                return self._requirements

            def run(self, payload):
                split = payload.split_collection.splits[0]
                result_field = learning_curve_result_field_for_method_name(self.method_name)
                assert result_field is not None
                uq_field = learning_curve_uq_field_for_method_name(self.method_name)
                screening_key = tuple(sorted(int(value) for value in payload.dataset.sample_ids.tolist()))
                score = self._screening_scores_by_signature[screening_key]
                frames = {
                    result_field: pd.DataFrame(
                        {
                            "n_train": [split.sweep_size],
                            "rmse_mean": [score],
                            "rmse_std": [0.0],
                        }
                    )
                }
                if uq_field is not None and screening_key in self._screening_miscalibration_by_signature:
                    frames[uq_field] = pd.DataFrame(
                        {
                            "n_train": [split.sweep_size],
                            "miscalibration_area": [
                                self._screening_miscalibration_by_signature[screening_key]
                            ],
                            "miscalibration_area_std": [0.0],
                            "sharpness": [0.0],
                            "sharpness_std": [0.0],
                            "dispersion": [0.0],
                            "dispersion_std": [0.0],
                            "uncertainty_kind": ["spread_only"],
                        }
                    )
                return LearningCurveResults.from_mapping(frames)

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
            screening_miscalibration_by_signature={
                train_signature_by_repeat[0]: 0.05,
                train_signature_by_repeat[1]: 0.07,
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
            screening_miscalibration_by_signature={
                train_signature_by_repeat[0]: 0.09,
                train_signature_by_repeat[1]: 0.04,
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
                    "policy_name": pd.Series(
                        ["min_screening_rmse", "min_screening_rmse"], dtype="string"
                    ),
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
                    "screening_miscalibration_area": pd.Series(
                        [0.05, 0.04], dtype="Float64"
                    ),
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

    def test_policy_variants_can_select_different_methods(self) -> None:
        dataset = SweepDataset(
            mlip_features=np.arange(8, dtype=float).reshape(-1, 1),
            targets=np.linspace(0.0, 0.7, 8),
            sample_ids=np.arange(8, dtype=int),
        )
        shared_splits = generate_shared_outer_splits(
            dataset.n_samples,
            min_train=4,
            max_train=4,
            step=1,
            n_repeats=1,
            seed=19,
        )
        train_signature = tuple(sorted(int(value) for value in shared_splits[0].train_idx.tolist()))
        test_signature = (
            shared_splits[0].budget,
            tuple(sorted(int(value) for value in shared_splits[0].test_idx.tolist())),
        )

        class StubFamily:
            def __init__(
                self,
                *,
                method_name: str,
                outer_rmse: float,
                screening_rmse: float,
                screening_miscalibration: float,
            ) -> None:
                self.method_name = method_name
                self.outer_rmse = outer_rmse
                self.screening_rmse = screening_rmse
                self.screening_miscalibration = screening_miscalibration

            def requirements(self):
                return SweepFamilyRequirements()

            def run(self, payload):
                result_field = learning_curve_result_field_for_method_name(self.method_name)
                uq_field = learning_curve_uq_field_for_method_name(self.method_name)
                assert result_field is not None
                frames = {
                    result_field: pd.DataFrame(
                        {
                            "n_train": [payload.split_collection.splits[0].sweep_size],
                            "rmse_mean": [self.screening_rmse],
                            "rmse_std": [0.0],
                        }
                    )
                }
                if uq_field is not None:
                    frames[uq_field] = pd.DataFrame(
                        {
                            "n_train": [payload.split_collection.splits[0].sweep_size],
                            "miscalibration_area": [self.screening_miscalibration],
                            "miscalibration_area_std": [0.0],
                            "sharpness": [0.0],
                            "sharpness_std": [0.0],
                            "dispersion": [0.0],
                            "dispersion_std": [0.0],
                            "uncertainty_kind": ["spread_only"],
                        }
                    )
                return LearningCurveResults.from_mapping(frames)

            def run_with_artifacts(self, payload):
                result_field = learning_curve_result_field_for_method_name(self.method_name)
                assert result_field is not None
                return SweepFamilyRunArtifacts(
                    results=LearningCurveResults.from_mapping(
                        {
                            result_field: pd.DataFrame(
                                {
                                    "n_train": [payload.split_collection.splits[0].sweep_size],
                                    "rmse_mean": [self.outer_rmse],
                                    "rmse_std": [0.0],
                                }
                            )
                        }
                    ),
                    repeat_metrics=pd.DataFrame(
                        [
                            {
                                "n_train": test_signature[0],
                                "repeat": 0,
                                "outer_test_rmse": self.outer_rmse,
                            }
                        ]
                    ),
                )

        detail = build_policy_selection_detail_frame(
            dataset,
            min_train=4,
            max_train=4,
            step=1,
            n_repeats=1,
            seed=19,
            model_families=[
                StubFamily(
                    method_name="ridge",
                    outer_rmse=0.2,
                    screening_rmse=0.1,
                    screening_miscalibration=0.3,
                ),
                StubFamily(
                    method_name="weighted_linear",
                    outer_rmse=0.25,
                    screening_rmse=0.2,
                    screening_miscalibration=0.05,
                ),
            ],
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
            policy_names=[
                "min_screening_rmse",
                "min_screening_miscalibration_area",
                "combined_screening_rmse_miscalibration",
            ],
            combined_miscalibration_lambda=2.0,
        )

        selected_by_policy = dict(
            zip(
                detail["policy_name"].tolist(),
                detail["screening_selected_method"].tolist(),
                strict=True,
            )
        )
        self.assertEqual(selected_by_policy["min_screening_rmse"], "ridge")
        self.assertEqual(
            selected_by_policy["min_screening_miscalibration_area"],
            "weighted_linear",
        )
        self.assertEqual(
            selected_by_policy["combined_screening_rmse_miscalibration"],
            "weighted_linear",
        )

    def test_build_policy_selection_diagnostic_results_emits_multi_policy_summary(self) -> None:
        dataset = SweepDataset(
            mlip_features=np.arange(8, dtype=float).reshape(-1, 1),
            targets=np.linspace(0.0, 0.7, 8),
            sample_ids=np.arange(8, dtype=int),
        )

        class StubFamily:
            def __init__(
                self,
                *,
                method_name: str,
                outer_rmse: float,
                screening_rmse: float,
                screening_miscalibration: float,
            ) -> None:
                self.method_name = method_name
                self.outer_rmse = outer_rmse
                self.screening_rmse = screening_rmse
                self.screening_miscalibration = screening_miscalibration

            def requirements(self):
                return SweepFamilyRequirements()

            def run(self, payload):
                result_field = learning_curve_result_field_for_method_name(self.method_name)
                uq_field = learning_curve_uq_field_for_method_name(self.method_name)
                assert result_field is not None
                frames = {
                    result_field: pd.DataFrame(
                        {
                            "n_train": [payload.split_collection.splits[0].sweep_size],
                            "rmse_mean": [self.screening_rmse],
                            "rmse_std": [0.0],
                        }
                    )
                }
                if uq_field is not None:
                    frames[uq_field] = pd.DataFrame(
                        {
                            "n_train": [payload.split_collection.splits[0].sweep_size],
                            "miscalibration_area": [self.screening_miscalibration],
                            "miscalibration_area_std": [0.0],
                            "sharpness": [0.0],
                            "sharpness_std": [0.0],
                            "dispersion": [0.0],
                            "dispersion_std": [0.0],
                            "uncertainty_kind": ["spread_only"],
                        }
                    )
                return LearningCurveResults.from_mapping(frames)

            def run_with_artifacts(self, payload):
                result_field = learning_curve_result_field_for_method_name(self.method_name)
                assert result_field is not None
                return SweepFamilyRunArtifacts(
                    results=LearningCurveResults.from_mapping(
                        {
                            result_field: pd.DataFrame(
                                {
                                    "n_train": [payload.split_collection.splits[0].sweep_size],
                                    "rmse_mean": [self.outer_rmse],
                                    "rmse_std": [0.0],
                                }
                            )
                        }
                    ),
                    repeat_metrics=pd.DataFrame(
                        [{"n_train": 4, "repeat": 0, "outer_test_rmse": self.outer_rmse}]
                    ),
                )

        results = build_policy_selection_diagnostic_results(
            dataset,
            min_train=4,
            max_train=4,
            step=1,
            n_repeats=1,
            seed=31,
            model_families=[
                StubFamily(
                    method_name="ridge",
                    outer_rmse=0.2,
                    screening_rmse=0.1,
                    screening_miscalibration=0.3,
                ),
                StubFamily(
                    method_name="weighted_linear",
                    outer_rmse=0.24,
                    screening_rmse=0.2,
                    screening_miscalibration=0.05,
                ),
            ],
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
            policy_names=[
                "min_screening_rmse",
                "min_screening_miscalibration_area",
            ],
        )

        self.assertEqual(
            results.summary_df["policy_name"].tolist(),
            [
                "min_screening_miscalibration_area",
                "min_screening_rmse",
            ],
        )
        np.testing.assert_allclose(
            results.summary_df["mean_regret"].to_numpy(),
            [0.04, 0.0],
            atol=1e-12,
        )

    def test_build_policy_selection_detail_frame_reuses_cached_outer_repeat_metrics(
        self,
    ) -> None:
        dataset = SweepDataset(
            mlip_features=np.arange(8, dtype=float).reshape(-1, 1),
            targets=np.linspace(0.0, 0.7, 8),
            sample_ids=np.arange(8, dtype=int),
        )

        class StubFamily:
            def __init__(self, *, method_name: str, screening_rmse: float) -> None:
                self.method_name = method_name
                self.screening_rmse = screening_rmse

            def requirements(self):
                return SweepFamilyRequirements()

            def run(self, payload):
                result_field = learning_curve_result_field_for_method_name(self.method_name)
                assert result_field is not None
                return LearningCurveResults.from_mapping(
                    {
                        result_field: pd.DataFrame(
                            {
                                "n_train": [payload.split_collection.splits[0].sweep_size],
                                "rmse_mean": [self.screening_rmse],
                                "rmse_std": [0.0],
                            }
                        )
                    }
                )

            def run_with_artifacts(self, payload):
                del payload
                raise AssertionError("outer metrics should be loaded from the repeat-metrics cache")

        detail = build_policy_selection_detail_frame(
            dataset,
            min_train=4,
            max_train=4,
            step=1,
            n_repeats=1,
            seed=19,
            model_families=[
                StubFamily(method_name="ridge", screening_rmse=0.1),
                StubFamily(method_name="weighted_linear", screening_rmse=0.2),
            ],
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
            cached_outer_repeat_metrics_df=pd.DataFrame(
                {
                    "method": ["ridge", "weighted_linear"],
                    "budget": [4, 4],
                    "repeat": [0, 0],
                    "outer_test_rmse": [0.2, 0.25],
                }
            ),
        )

        self.assertEqual(detail["oracle_method"].tolist(), ["ridge"])
        self.assertEqual(detail["screening_selected_method"].tolist(), ["ridge"])
        self.assertEqual(detail["oracle_outer_rmse"].tolist(), [0.2])

    def test_policy_regret_uses_paired_outer_test_rmse_not_screening_metrics(self) -> None:
        detail = normalize_policy_detail_frame(
            pd.DataFrame(
                {
                    "policy_name": ["min_screening_miscalibration_area"],
                    "budget": [4],
                    "repeat": [0],
                    "oracle_method": ["ridge"],
                    "screening_selected_method": ["weighted_linear"],
                    "oracle_outer_rmse": [0.2],
                    "screening_selected_outer_rmse": [0.35],
                    "regret": [0.15],
                    "screening_cv_rmse": [0.5],
                    "screening_miscalibration_area": [0.01],
                    "agreement": [False],
                }
            )
        )

        self.assertEqual(detail["regret"].tolist(), [0.15])

    def test_miscalibration_policy_fails_loudly_when_metric_missing(self) -> None:
        dataset = SweepDataset(
            mlip_features=np.arange(8, dtype=float).reshape(-1, 1),
            targets=np.linspace(0.0, 0.7, 8),
            sample_ids=np.arange(8, dtype=int),
        )

        class MissingMiscalibrationFamily:
            def __init__(self, method_name: str, outer_rmse: float, screening_rmse: float) -> None:
                self.method_name = method_name
                self.outer_rmse = outer_rmse
                self.screening_rmse = screening_rmse

            def requirements(self):
                return SweepFamilyRequirements()

            def run(self, payload):
                result_field = learning_curve_result_field_for_method_name(self.method_name)
                assert result_field is not None
                return LearningCurveResults.from_mapping(
                    {
                        result_field: pd.DataFrame(
                            {
                                "n_train": [payload.split_collection.splits[0].sweep_size],
                                "rmse_mean": [self.screening_rmse],
                                "rmse_std": [0.0],
                            }
                        )
                    }
                )

            def run_with_artifacts(self, payload):
                result_field = learning_curve_result_field_for_method_name(self.method_name)
                assert result_field is not None
                return SweepFamilyRunArtifacts(
                    results=LearningCurveResults.from_mapping(
                        {
                            result_field: pd.DataFrame(
                                {
                                    "n_train": [payload.split_collection.splits[0].sweep_size],
                                    "rmse_mean": [self.outer_rmse],
                                    "rmse_std": [0.0],
                                }
                            )
                        }
                    ),
                    repeat_metrics=pd.DataFrame(
                        [{"n_train": 4, "repeat": 0, "outer_test_rmse": self.outer_rmse}]
                    ),
                )

        with self.assertRaisesRegex(
            ValueError,
            "screening miscalibration area is required",
        ):
            build_policy_selection_detail_frame(
                dataset,
                min_train=4,
                max_train=4,
                step=1,
                n_repeats=1,
                seed=23,
                model_families=[
                    MissingMiscalibrationFamily("ridge", outer_rmse=0.2, screening_rmse=0.1),
                    MissingMiscalibrationFamily(
                        "weighted_linear",
                        outer_rmse=0.25,
                        screening_rmse=0.11,
                    ),
                ],
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
                policy_names=["min_screening_miscalibration_area"],
            )
