from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from oasis.exp import (
    build_sweep_split_collection,
    generate_inner_validation_sweep_splits,
    generate_sweep_splits,
    generate_sweep_splits_with_validation,
    inner_validation_size_for_sweep,
    prepare_parity_plot_data,
    run_learning_curve_experiments,
    run_learning_curve_experiments_from_config,
    run_learning_curve_experiments_from_frame,
)
from oasis.sweep import (
    LearningCurveResults,
    SweepDataset,
    SweepFamilyRequirements,
    SweepModelCapabilities,
    SweepRunnerPayload,
    SweepSplit,
    SweepSplitCollection,
    SweepRunPayload,
    TrainTestSweepRunnerInput,
    TrainValTestSweepRunnerInput,
)
from tests.support import regression_dataset

try:
    from oasis.method import (
        ConfiguredSweepModelFamily,
        SupervisedModelSweepRunner,
        SweepFamilySpec,
        sweep_results_frame,
    )

    HAS_SKLEARN = True
except ModuleNotFoundError:
    HAS_SKLEARN = False


class GenerateSweepSplitsTests(unittest.TestCase):
    def test_generate_sweep_splits_yields_disjoint_full_partitions(self) -> None:
        rng = np.random.default_rng(123)

        splits = list(
            generate_sweep_splits(
                n_samples=6,
                min_train=2,
                max_train=3,
                n_repeats=2,
                rng=rng,
            )
        )

        self.assertEqual(len(splits), 4)
        self.assertTrue(all(isinstance(split, SweepSplit) for split in splits))

        expected_sizes = [2, 2, 3, 3]
        observed_sizes = [split.sweep_size for split in splits]
        self.assertEqual(observed_sizes, expected_sizes)

        full_idx = np.arange(6)
        for split in splits:
            self.assertEqual(len(split.train_idx), split.sweep_size)
            self.assertEqual(len(split.test_idx), 6 - split.sweep_size)
            self.assertIsNone(split.val_idx)
            self.assertEqual(
                len(np.intersect1d(split.train_idx, split.test_idx)),
                0,
            )
            np.testing.assert_array_equal(
                np.sort(np.concatenate([split.train_idx, split.test_idx])),
                full_idx,
            )

    def test_same_seed_gives_same_splits(self) -> None:
        splits_a = list(
            generate_sweep_splits(
                n_samples=8,
                min_train=2,
                max_train=4,
                n_repeats=3,
                rng=np.random.default_rng(42),
            )
        )
        splits_b = list(
            generate_sweep_splits(
                n_samples=8,
                min_train=2,
                max_train=4,
                n_repeats=3,
                rng=np.random.default_rng(42),
            )
        )

        self.assertEqual(len(splits_a), len(splits_b))
        for split_a, split_b in zip(splits_a, splits_b, strict=True):
            self.assertEqual(split_a.sweep_size, split_b.sweep_size)
            np.testing.assert_array_equal(split_a.train_idx, split_b.train_idx)
            np.testing.assert_array_equal(split_a.test_idx, split_b.test_idx)
            self.assertIsNone(split_a.val_idx)
            self.assertIsNone(split_b.val_idx)

    def test_sweep_split_accepts_optional_validation_indices(self) -> None:
        split = SweepSplit(
            sweep_size=3,
            train_idx=np.array([0, 1, 2]),
            test_idx=np.array([5, 6]),
            val_idx=np.array([3, 4]),
        )

        np.testing.assert_array_equal(split.train_idx, np.array([0, 1, 2]))
        np.testing.assert_array_equal(split.val_idx, np.array([3, 4]))
        np.testing.assert_array_equal(split.test_idx, np.array([5, 6]))


class GenerateSweepSplitsWithValidationTests(unittest.TestCase):
    def test_inner_validation_size_for_sweep_uses_fraction_policy(self) -> None:
        self.assertEqual(inner_validation_size_for_sweep(1), 1)
        self.assertEqual(inner_validation_size_for_sweep(4), 1)
        self.assertEqual(inner_validation_size_for_sweep(5), 1)
        self.assertEqual(inner_validation_size_for_sweep(10), 2)

    def test_inner_validation_size_for_sweep_rejects_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "sweep_size must be positive"):
            inner_validation_size_for_sweep(0)

        with self.assertRaisesRegex(ValueError, "frac must be positive"):
            inner_validation_size_for_sweep(5, frac=0.0)

    def test_generate_sweep_splits_with_validation_yields_disjoint_full_partitions(
        self,
    ) -> None:
        rng = np.random.default_rng(123)

        splits = list(
            generate_sweep_splits_with_validation(
                n_samples=8,
                min_train=4,
                max_train=5,
                n_val=2,
                n_repeats=2,
                rng=rng,
            )
        )

        self.assertEqual(len(splits), 4)
        self.assertTrue(all(isinstance(split, SweepSplit) for split in splits))
        self.assertEqual([split.sweep_size for split in splits], [4, 4, 5, 5])

        full_idx = np.arange(8)
        for split in splits:
            self.assertEqual(len(split.train_idx), split.sweep_size - 2)
            self.assertIsNotNone(split.val_idx)
            self.assertEqual(len(split.val_idx), 2)
            self.assertEqual(len(split.test_idx), 8 - split.sweep_size)
            self.assertEqual(
                len(np.intersect1d(split.train_idx, split.val_idx)),
                0,
            )
            self.assertEqual(
                len(np.intersect1d(split.train_idx, split.test_idx)),
                0,
            )
            self.assertEqual(
                len(np.intersect1d(split.val_idx, split.test_idx)),
                0,
            )
            np.testing.assert_array_equal(
                np.sort(
                    np.concatenate(
                        [split.train_idx, split.val_idx, split.test_idx]
                    )
                ),
                full_idx,
            )
            self.assertEqual(
                len(split.train_idx) + len(split.val_idx),
                split.sweep_size,
            )

    def test_same_seed_gives_same_validation_splits(self) -> None:
        splits_a = list(
            generate_sweep_splits_with_validation(
                n_samples=9,
                min_train=2,
                max_train=4,
                n_val=2,
                n_repeats=3,
                rng=np.random.default_rng(42),
            )
        )
        splits_b = list(
            generate_sweep_splits_with_validation(
                n_samples=9,
                min_train=2,
                max_train=4,
                n_val=2,
                n_repeats=3,
                rng=np.random.default_rng(42),
            )
        )

        self.assertEqual(len(splits_a), len(splits_b))
        for split_a, split_b in zip(splits_a, splits_b, strict=True):
            self.assertEqual(split_a.sweep_size, split_b.sweep_size)
            np.testing.assert_array_equal(split_a.train_idx, split_b.train_idx)
            np.testing.assert_array_equal(split_a.val_idx, split_b.val_idx)
            np.testing.assert_array_equal(split_a.test_idx, split_b.test_idx)

    def test_generate_sweep_splits_with_validation_clamps_max_train(self) -> None:
        splits = list(
            generate_sweep_splits_with_validation(
                n_samples=7,
                min_train=4,
                max_train=6,
                n_val=2,
                n_repeats=2,
                rng=np.random.default_rng(7),
            )
        )

        self.assertEqual(len(splits), 6)
        self.assertEqual([split.sweep_size for split in splits], [4, 4, 5, 5, 6, 6])
        self.assertEqual([len(split.test_idx) for split in splits], [3, 3, 2, 2, 1, 1])

    def test_generate_sweep_splits_with_validation_returns_no_splits_when_min_train_exceeds_capacity(
        self,
    ) -> None:
        splits = list(
            generate_sweep_splits_with_validation(
                n_samples=7,
                min_train=7,
                max_train=6,
                n_val=2,
                n_repeats=2,
                rng=np.random.default_rng(7),
            )
        )

        self.assertEqual(splits, [])

    def test_generate_sweep_splits_with_validation_requires_outer_train_to_fit_validation(
        self,
    ) -> None:
        splits = list(
            generate_sweep_splits_with_validation(
                n_samples=7,
                min_train=1,
                max_train=3,
                n_val=2,
                n_repeats=1,
                rng=np.random.default_rng(7),
            )
        )

        self.assertEqual([split.sweep_size for split in splits], [3])
        self.assertTrue(all(len(split.train_idx) == 1 for split in splits))

    def test_generate_sweep_splits_with_validation_rejects_invalid_validation_size(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "n_val must be positive"):
            list(
                generate_sweep_splits_with_validation(
                    n_samples=6,
                    min_train=2,
                    max_train=3,
                    n_val=0,
                    n_repeats=1,
                    rng=np.random.default_rng(1),
                )
            )

    def test_generate_inner_validation_sweep_splits_uses_fraction_policy(self) -> None:
        splits = list(
            generate_inner_validation_sweep_splits(
                n_samples=12,
                min_train=4,
                max_train=10,
                n_repeats=1,
                rng=np.random.default_rng(7),
            )
        )

        self.assertEqual([split.sweep_size for split in splits], [4, 5, 6, 7, 8, 9, 10])
        self.assertEqual(
            [len(split.val_idx) for split in splits],
            [1, 1, 1, 1, 1, 1, 2],
        )
        self.assertEqual(
            [len(split.train_idx) for split in splits],
            [3, 4, 5, 6, 7, 8, 8],
        )

        with self.assertRaisesRegex(
            ValueError,
            "n_val must be smaller than n_samples",
        ):
            list(
                generate_sweep_splits_with_validation(
                    n_samples=6,
                    min_train=2,
                    max_train=3,
                    n_val=6,
                    n_repeats=1,
                    rng=np.random.default_rng(1),
                )
            )



class ExpIntegrationTests(unittest.TestCase):
    def test_run_learning_curve_experiments_from_frame_passes_reaction_ids_into_dataset(
        self,
    ) -> None:
        df = pd.DataFrame(
            {
                "reaction": [
                    "rxn-0",
                    "rxn-1",
                    "rxn-2",
                    "rxn-3",
                    "rxn-4",
                    "rxn-5",
                ],
                "reference_ads_eng": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "ridge_mlip_ads_eng_median": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
                "lasso_mlip_ads_eng_median": [0.9, 1.9, 2.9, 3.9, 4.9, 5.9],
            }
        )
        result_df = pd.DataFrame(
            {
                "n_train": [2, 3, 4],
                "rmse_mean": [0.4, 0.3, 0.2],
                "rmse_std": [0.05, 0.04, 0.03],
            }
        )

        with patch(
            "oasis.exp.run_learning_curve_experiments",
            autospec=True,
        ) as run_mock:
            run_mock.return_value = LearningCurveResults.from_mapping(
                {"ridge_df": result_df}
            )

            results = run_learning_curve_experiments_from_frame(
                df,
                min_train=2,
                max_train=4,
                n_repeats=1,
                seed=17,
                use_trim=False,
            )

        self.assertIsNotNone(results.ridge_df)
        dataset = run_mock.call_args.args[0]
        self.assertIsInstance(dataset, SweepDataset)
        np.testing.assert_array_equal(
            dataset.sample_ids,
            np.array(["rxn-0", "rxn-1", "rxn-2", "rxn-3", "rxn-4", "rxn-5"]),
        )
        self.assertEqual(dataset.auxiliary_views, {})
        self.assertEqual(dataset.mlip_features.shape, (6, 2))
        np.testing.assert_array_equal(
            dataset.targets,
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        )

    def test_run_learning_curve_experiments_plans_splits_from_dataset_sample_count(
        self,
    ) -> None:
        dataset = SweepDataset(
            mlip_features=np.array(
                [{"row": 0}, {"row": 1}, {"row": 2}, {"row": 3}, {"row": 4}],
                dtype=object,
            ),
            targets=np.arange(5, dtype=float),
        )
        result_df = pd.DataFrame(
            {
                "n_train": [2, 3, 4],
                "rmse_mean": [0.3, 0.2, 0.1],
                "rmse_std": [0.03, 0.02, 0.01],
            }
        )

        class RecordingFamily:
            def requirements(self) -> SweepFamilyRequirements:
                return SweepFamilyRequirements()

            def run(self, payload):
                self.last_payload = payload
                return LearningCurveResults.from_mapping({"ridge_df": result_df})

        family = RecordingFamily()

        results = run_learning_curve_experiments(
            dataset,
            min_train=2,
            max_train=4,
            n_repeats=1,
            seed=5,
            use_trim=False,
            model_families=[family],
        )

        self.assertIs(results.ridge_df, result_df)
        self.assertEqual(dataset.n_samples, 5)
        self.assertEqual(
            [split.sweep_size for split in family.last_payload.split_collection.splits],
            [2, 3, 4],
        )
        self.assertTrue(
            all(split.val_idx is None for split in family.last_payload.split_collection.splits)
        )

    def test_run_learning_curve_experiments_combines_capabilities_first(self) -> None:
        X = np.arange(21, dtype=float).reshape(7, 3)
        y = np.arange(7, dtype=float)
        result_df = pd.DataFrame(
            {
                "n_train": [4, 5],
                "rmse_mean": [0.4, 0.3],
                "rmse_std": [0.05, 0.04],
            }
        )

        class CapabilityAwareStubFamily:
            def capabilities(self) -> SweepModelCapabilities:
                return SweepModelCapabilities(
                    min_train_size=4,
                    requires_validation=True,
                )

            def run(self, payload):
                self.last_payload = payload
                return LearningCurveResults.from_mapping({"ridge_df": result_df})

        family = CapabilityAwareStubFamily()

        results = run_learning_curve_experiments(
            SweepDataset(mlip_features=X, targets=y),
            min_train=2,
            max_train=5,
            n_repeats=1,
            seed=3,
            use_trim=False,
            model_families=[family],
        )

        self.assertIs(results.ridge_df, result_df)
        self.assertEqual(
            family.last_payload.split_collection.planning_requirements,
            SweepFamilyRequirements(
                min_train_size=4,
                requires_inner_validation=True,
            ),
        )

    def test_run_learning_curve_experiments_accepts_injected_model_families(
        self,
    ) -> None:
        X = np.array(
            [
                [1.0, 1.1],
                [2.0, 2.1],
                [3.0, 3.1],
                [4.0, 4.1],
                [5.0, 5.1],
                [6.0, 6.1],
            ]
        )
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result_df = pd.DataFrame(
            {
                "n_train": [2, 3, 4],
                "rmse_mean": [0.4, 0.3, 0.2],
                "rmse_std": [0.05, 0.04, 0.03],
            }
        )

        class StubFamily:
            def __init__(self, field_name: str) -> None:
                self.field_name = field_name
                self.calls = 0

            def requirements(self) -> SweepFamilyRequirements:
                return SweepFamilyRequirements()

            def run(self, payload):
                self.calls += 1
                self.last_payload = payload
                return LearningCurveResults.from_mapping(
                    {self.field_name: result_df}
                )

        ridge_family = StubFamily("ridge_df")
        weighted_family = StubFamily("weighted_linear_df")

        results = run_learning_curve_experiments(
            SweepDataset(mlip_features=X, targets=y),
            min_train=2,
            max_train=4,
            n_repeats=1,
            seed=9,
            use_trim=False,
            model_families=[ridge_family, weighted_family],
        )

        self.assertIs(results.ridge_df, result_df)
        self.assertIs(results.weighted_linear_df, result_df)
        self.assertIsNone(results.kernel_ridge_df)
        self.assertIsNone(results.ridge_trimmed_df)
        self.assertEqual(ridge_family.calls, 1)
        self.assertEqual(weighted_family.calls, 1)
        self.assertIsInstance(ridge_family.last_payload, SweepRunPayload)
        self.assertFalse(ridge_family.last_payload.use_trim)
        self.assertEqual(
            [split.sweep_size for split in ridge_family.last_payload.split_collection.splits],
            [2, 3, 4],
        )

    def test_run_learning_curve_experiments_honors_family_split_requirements(
        self,
    ) -> None:
        X = np.array(
            [
                [1.0, 1.1],
                [2.0, 2.1],
                [3.0, 3.1],
                [4.0, 4.1],
                [5.0, 5.1],
                [6.0, 6.1],
                [7.0, 7.1],
            ]
        )
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        result_df = pd.DataFrame(
            {
                "n_train": [4, 5],
                "rmse_mean": [0.4, 0.3],
                "rmse_std": [0.05, 0.04],
            }
        )

        class ValidationAwareStubFamily:
            def requirements(self) -> SweepFamilyRequirements:
                return SweepFamilyRequirements(
                    min_train_size=4,
                    requires_inner_validation=True,
                )

            def run(self, payload):
                self.last_payload = payload
                return LearningCurveResults.from_mapping({"ridge_df": result_df})

        family = ValidationAwareStubFamily()

        results = run_learning_curve_experiments(
            SweepDataset(mlip_features=X, targets=y),
            min_train=2,
            max_train=5,
            n_repeats=1,
            seed=3,
            use_trim=False,
            model_families=[family],
        )

        self.assertIs(results.ridge_df, result_df)
        self.assertEqual(
            [split.sweep_size for split in family.last_payload.split_collection.splits],
            [4, 5],
        )
        self.assertTrue(
            all(split.val_idx is not None for split in family.last_payload.split_collection.splits)
        )
        self.assertEqual(
            [len(split.train_idx) for split in family.last_payload.split_collection.splits],
            [3, 4],
        )
        self.assertEqual(
            [len(split.val_idx) for split in family.last_payload.split_collection.splits],
            [1, 1],
        )
        self.assertEqual(
            family.last_payload.split_collection.planning_requirements,
            SweepFamilyRequirements(
                min_train_size=4,
                requires_inner_validation=True,
            ),
        )

    def test_build_sweep_split_collection_skips_invalid_small_validation_sweeps(
        self,
    ) -> None:
        split_collection = build_sweep_split_collection(
            n_samples=7,
            min_train=2,
            max_train=5,
            n_repeats=1,
            seed=3,
            requirements=SweepFamilyRequirements(
                min_train_size=4,
                requires_inner_validation=True,
            ),
        )

        self.assertEqual(
            [split.sweep_size for split in split_collection.splits],
            [4, 5],
        )
        self.assertTrue(all(split.val_idx is not None for split in split_collection.splits))

    def test_validation_aware_family_does_not_constrain_baseline_sweep_sizes(
        self,
    ) -> None:
        X = np.arange(21, dtype=float).reshape(7, 3)
        y = np.arange(7, dtype=float)
        baseline_result = pd.DataFrame(
            {
                "n_train": [2, 3, 4, 5],
                "rmse_mean": [0.5, 0.4, 0.3, 0.2],
                "rmse_std": [0.05, 0.04, 0.03, 0.02],
            }
        )
        validation_result = pd.DataFrame(
            {
                "n_train": [4, 5],
                "rmse_mean": [0.35, 0.25],
                "rmse_std": [0.03, 0.02],
            }
        )

        class BaselineStubFamily:
            def capabilities(self) -> SweepModelCapabilities:
                return SweepModelCapabilities()

            def run(self, payload):
                self.last_payload = payload
                return LearningCurveResults.from_mapping({"ridge_df": baseline_result})

        class ValidationStubFamily:
            def capabilities(self) -> SweepModelCapabilities:
                return SweepModelCapabilities(
                    min_train_size=4,
                    requires_validation=True,
                )

            def run(self, payload):
                self.last_payload = payload
                return LearningCurveResults.from_mapping(
                    {"weighted_linear_df": validation_result}
                )

        baseline_family = BaselineStubFamily()
        validation_family = ValidationStubFamily()

        results = run_learning_curve_experiments(
            SweepDataset(mlip_features=X, targets=y),
            min_train=2,
            max_train=5,
            n_repeats=1,
            seed=3,
            use_trim=False,
            model_families=[baseline_family, validation_family],
        )

        self.assertIs(results.ridge_df, baseline_result)
        self.assertIs(results.weighted_linear_df, validation_result)
        self.assertEqual(
            [split.sweep_size for split in baseline_family.last_payload.split_collection.splits],
            [2, 3, 4, 5],
        )
        self.assertTrue(
            all(split.val_idx is None for split in baseline_family.last_payload.split_collection.splits)
        )
        self.assertEqual(
            [split.sweep_size for split in validation_family.last_payload.split_collection.splits],
            [4, 5],
        )
        self.assertTrue(
            all(split.val_idx is not None for split in validation_family.last_payload.split_collection.splits)
        )

    def test_run_learning_curve_experiments_supports_mixed_runner_input_types(
        self,
    ) -> None:
        class RecordingRunner:
            def __init__(self, expected_split_type: type) -> None:
                self.expected_split_type = expected_split_type
                self.payloads: list[SweepRunnerPayload] = []

            def run(self, payload: SweepRunnerPayload) -> pd.DataFrame:
                self.payloads.append(payload)
                self._assert_payload(payload)
                rmses_by_size = {
                    split.sweep_size: [float(split.sweep_size) / 10.0]
                    for split in payload.splits
                }
                return sweep_results_frame(rmses_by_size)

            def run_with_validation(
                self,
                payload: SweepRunnerPayload,
            ) -> pd.DataFrame:
                return self.run(payload)

            def run_trimmed(
                self,
                payload: SweepRunnerPayload,
                *,
                z_thresh: float = 1.0,
            ) -> pd.DataFrame:
                del z_thresh
                return self.run(payload)

            def run_trimmed_with_validation(
                self,
                payload: SweepRunnerPayload,
                *,
                z_thresh: float = 1.0,
            ) -> pd.DataFrame:
                del z_thresh
                return self.run(payload)

            def _assert_payload(self, payload: SweepRunnerPayload) -> None:
                if not payload.splits:
                    raise AssertionError("expected non-empty payload.splits")
                for split in payload.splits:
                    if not isinstance(split, self.expected_split_type):
                        raise AssertionError(
                            f"expected {self.expected_split_type}, got {type(split)}"
                        )

        X = np.arange(21, dtype=float).reshape(7, 3)
        y = np.arange(7, dtype=float)
        baseline_runner = RecordingRunner(TrainTestSweepRunnerInput)
        validation_runner = RecordingRunner(TrainValTestSweepRunnerInput)
        baseline_family = ConfiguredSweepModelFamily(
            spec=SweepFamilySpec(
                result_field="ridge_df",
                trimmed_result_field=None,
                runner=baseline_runner,
            )
        )
        validation_family = ConfiguredSweepModelFamily(
            spec=SweepFamilySpec(
                result_field="weighted_linear_df",
                trimmed_result_field=None,
                runner=validation_runner,
                capabilities=SweepModelCapabilities(
                    min_train_size=4,
                    requires_validation=True,
                ),
            )
        )

        results = run_learning_curve_experiments(
            SweepDataset(mlip_features=X, targets=y),
            min_train=2,
            max_train=5,
            n_repeats=1,
            seed=3,
            use_trim=False,
            model_families=[baseline_family, validation_family],
        )

        self.assertEqual(results.ridge_df.columns.tolist(), ["n_train", "rmse_mean", "rmse_std"])
        self.assertEqual(
            results.ridge_df["n_train"].tolist(),
            [2, 3, 4, 5],
        )
        self.assertEqual(
            results.weighted_linear_df["n_train"].tolist(),
            [4, 5],
        )
        self.assertEqual(len(baseline_runner.payloads), 1)
        self.assertEqual(len(validation_runner.payloads), 1)

    def test_configured_family_rejects_validation_splits_for_train_test_runner(
        self,
    ) -> None:
        family = ConfiguredSweepModelFamily(
            spec=SweepFamilySpec(
                result_field="ridge_df",
                trimmed_result_field=None,
                runner=SupervisedModelSweepRunner(lambda: None),
                capabilities=SweepModelCapabilities(
                    min_train_size=4,
                    requires_validation=True,
                ),
            )
        )
        payload = SweepRunPayload(
            dataset=SweepDataset(
                mlip_features=np.arange(21, dtype=float).reshape(7, 3),
                targets=np.arange(7, dtype=float),
            ),
            split_collection=SweepSplitCollection(
                splits=tuple(
                    generate_inner_validation_sweep_splits(
                        n_samples=7,
                        min_train=4,
                        max_train=4,
                        n_repeats=1,
                        rng=np.random.default_rng(5),
                    )
                ),
                planning_requirements=family.requirements(),
            ),
            use_trim=False,
        )

        with self.assertRaisesRegex(
            TypeError,
            "runner does not support validation-aware sweep payloads",
        ):
            family.run(payload)

    def test_run_learning_curve_experiments_handles_skipped_validation_family_predictably(
        self,
    ) -> None:
        class BaselineRunner:
            def __init__(self) -> None:
                self.payloads: list[SweepRunnerPayload] = []

            def run(self, payload: SweepRunnerPayload) -> pd.DataFrame:
                self.payloads.append(payload)
                if not payload.splits:
                    raise AssertionError("expected baseline family to receive splits")
                for split in payload.splits:
                    if not isinstance(split, TrainTestSweepRunnerInput):
                        raise AssertionError(
                            f"expected {TrainTestSweepRunnerInput}, got {type(split)}"
                        )
                return sweep_results_frame(
                    {
                        split.sweep_size: [float(split.sweep_size) / 10.0]
                        for split in payload.splits
                    }
                )

            def run_trimmed(
                self,
                payload: SweepRunnerPayload,
                *,
                z_thresh: float = 1.0,
            ) -> pd.DataFrame:
                del z_thresh
                return self.run(payload)

        class EmptyAwareValidationRunner:
            def __init__(self) -> None:
                self.payloads: list[SweepRunnerPayload] = []

            def run_with_validation(
                self,
                payload: SweepRunnerPayload,
            ) -> pd.DataFrame:
                self.payloads.append(payload)
                if payload.splits != ():
                    raise AssertionError("expected skipped validation family to receive no splits")
                return sweep_results_frame({})

            def run_trimmed_with_validation(
                self,
                payload: SweepRunnerPayload,
                *,
                z_thresh: float = 1.0,
            ) -> pd.DataFrame:
                del z_thresh
                return self.run_with_validation(payload)

        X = np.arange(18, dtype=float).reshape(6, 3)
        y = np.arange(6, dtype=float)
        baseline_runner = BaselineRunner()
        baseline_family = ConfiguredSweepModelFamily(
            spec=SweepFamilySpec(
                result_field="ridge_df",
                trimmed_result_field=None,
                runner=baseline_runner,
            )
        )
        validation_runner = EmptyAwareValidationRunner()
        skipped_family = ConfiguredSweepModelFamily(
            spec=SweepFamilySpec(
                result_field="weighted_linear_df",
                trimmed_result_field=None,
                runner=validation_runner,
                capabilities=SweepModelCapabilities(
                    min_train_size=7,
                    requires_validation=True,
                ),
            )
        )

        results = run_learning_curve_experiments(
            SweepDataset(mlip_features=X, targets=y),
            min_train=2,
            max_train=4,
            n_repeats=1,
            seed=3,
            use_trim=False,
            model_families=[baseline_family, skipped_family],
        )

        self.assertIsNotNone(results.ridge_df)
        self.assertEqual(results.ridge_df["n_train"].tolist(), [2, 3, 4])
        self.assertIsNotNone(results.weighted_linear_df)
        self.assertEqual(
            results.weighted_linear_df.columns.tolist(),
            ["n_train", "rmse_mean", "rmse_std"],
        )
        self.assertTrue(results.weighted_linear_df.empty)
        self.assertEqual(len(baseline_runner.payloads), 1)
        self.assertEqual(len(validation_runner.payloads), 1)

    def test_configured_family_routes_runner_inputs_with_validation(self) -> None:
        if not HAS_SKLEARN:
            self.skipTest("requires scikit-learn")

        class RecordingRunner:
            def run_with_validation(self, payload):
                self.last_payload = payload
                return pd.DataFrame(
                    {
                        "n_train": [3],
                        "rmse_mean": [0.2],
                        "rmse_std": [0.01],
                    }
                )

            def run_trimmed_with_validation(
                self,
                payload,
                *,
                z_thresh: float = 1.0,
            ):
                del z_thresh
                return self.run_with_validation(payload)

        family = ConfiguredSweepModelFamily(
            spec=SweepFamilySpec(
                result_field="ridge_df",
                trimmed_result_field=None,
                runner=RecordingRunner(),
                trim_z_thresh=1.0,
                capabilities=SweepModelCapabilities(
                    min_train_size=4,
                    requires_validation=True,
                ),
            )
        )
        payload = SweepRunPayload(
            dataset=SweepDataset(
                mlip_features=np.arange(21, dtype=float).reshape(7, 3),
                targets=np.arange(7, dtype=float),
            ),
            split_collection=SweepSplitCollection(
                splits=tuple(
                    generate_inner_validation_sweep_splits(
                        n_samples=7,
                        min_train=4,
                        max_train=4,
                        n_repeats=1,
                        rng=np.random.default_rng(5),
                    )
                ),
                planning_requirements=family.requirements(),
            ),
            use_trim=False,
        )

        result = family.run(payload)

        self.assertIsNotNone(result.ridge_df)
        self.assertIsInstance(family.spec.runner.last_payload, SweepRunnerPayload)
        self.assertEqual(len(family.spec.runner.last_payload.splits), 1)
        split = family.spec.runner.last_payload.splits[0]
        self.assertIsInstance(split, TrainValTestSweepRunnerInput)
        self.assertEqual(len(split.train_idx), 3)
        self.assertEqual(len(split.val_idx), 1)
        self.assertEqual(len(split.test_idx), 3)

    def test_prepare_parity_plot_data_extracts_render_inputs(self) -> None:
        df = pd.DataFrame(
            {
                "reference_ads_eng": [1.0, 2.0],
                "ridge_mlip_ads_eng_median": [1.1, 2.1],
                "lasso_mlip_ads_eng_median": [0.9, 1.9],
            }
        )

        plot_data = prepare_parity_plot_data(df)

        np.testing.assert_array_equal(plot_data.reference, np.array([1.0, 2.0]))
        self.assertEqual(set(plot_data.predictions), {"ridge", "lasso"})
        np.testing.assert_array_equal(
            plot_data.predictions["ridge"],
            np.array([1.1, 2.1]),
        )

    def test_run_learning_curve_experiments_from_config_uses_defaults_without_cfg(
        self,
    ) -> None:
        df = pd.DataFrame(
            {
                "reference_ads_eng": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "ridge_mlip_ads_eng_median": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
                "lasso_mlip_ads_eng_median": [0.9, 1.9, 2.9, 3.9, 4.9, 5.9],
            }
        )

        if not HAS_SKLEARN:
            self.skipTest("requires scikit-learn")

        results = run_learning_curve_experiments_from_config(df, cfg=None)

        self.assertIsInstance(results, LearningCurveResults)
        self.assertIsNotNone(results.ridge_df)
        self.assertEqual(results.ridge_df["n_train"].tolist(), [5])

    def test_run_learning_curve_experiments_from_frame_runs_weighted_baselines(
        self,
    ) -> None:
        df = pd.DataFrame(
            {
                "reference_ads_eng": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "ridge_mlip_ads_eng_median": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
                "lasso_mlip_ads_eng_median": [0.9, 1.9, 2.9, 3.9, 4.9, 5.9],
                "elastic_mlip_ads_eng_median": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )

        if not HAS_SKLEARN:
            self.skipTest("requires scikit-learn")

        results = run_learning_curve_experiments_from_frame(
            df,
            min_train=2,
            max_train=4,
            n_repeats=2,
            seed=17,
            use_trim=False,
            enabled_model_names=[
                "weighted_linear",
                "weighted_simplex",
            ],
        )

        self.assertIsNone(results.ridge_df)
        self.assertIsNone(results.resid_df)
        self.assertIsNotNone(results.weighted_linear_df)
        self.assertIsNotNone(results.weighted_simplex_df)
        self.assertEqual(results.weighted_linear_df["n_train"].tolist(), [2, 3, 4])
        self.assertEqual(results.weighted_simplex_df["n_train"].tolist(), [2, 3, 4])

    def test_run_learning_curve_experiments_from_config_honors_weighted_baseline_flags(
        self,
    ) -> None:
        df = pd.DataFrame(
            {
                "reference_ads_eng": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "ridge_mlip_ads_eng_median": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
                "lasso_mlip_ads_eng_median": [0.9, 1.9, 2.9, 3.9, 4.9, 5.9],
                "elastic_mlip_ads_eng_median": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )

        if not HAS_SKLEARN:
            self.skipTest("requires scikit-learn")

        cfg = SimpleNamespace(
            seed=19,
            plot=SimpleNamespace(
                min_train=2,
                max_train=4,
                n_repeats=2,
                trim=False,
                use_ridge=True,
                use_kernel_ridge=False,
                use_lasso=False,
                use_elastic_net=False,
                use_residual=False,
                use_weighted_linear=True,
                use_weighted_simplex=False,
            ),
        )

        results = run_learning_curve_experiments_from_config(df, cfg=cfg)

        self.assertIsNotNone(results.ridge_df)
        self.assertIsNotNone(results.weighted_linear_df)
        self.assertIsNone(results.weighted_simplex_df)
        self.assertIsNone(results.lasso_df)
        self.assertIsNone(results.resid_df)
