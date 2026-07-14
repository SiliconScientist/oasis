from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ase import Atoms
import numpy as np
import pandas as pd

try:
    import polars  # noqa: F401

    HAS_POLARS = True
except ModuleNotFoundError:
    HAS_POLARS = False

from oasis.exp import (
    build_sweep_dataset_from_config,
    build_sweep_dataset_from_frame,
    build_sweep_split_collection,
    generate_screening_sweep_splits,
    generate_screening_sweep_splits_with_validation,
    generate_inner_validation_sweep_splits,
    load_or_run_learning_curve_results_from_config,
    generate_sweep_splits,
    generate_sweep_splits_with_validation,
    inner_validation_size_for_sweep,
    outer_train_size_if_screening_feasible,
    prepare_parity_plot_data,
    resolve_configured_sweep_sizes,
    run_learning_curve_experiments,
    run_learning_curve_experiments_from_config,
    run_learning_curve_experiments_from_frame,
    screening_holdout_size_for_budget,
    validation_size_if_sweep_feasible,
)
from oasis.experiment.repeat_metrics import (
    load_learning_curve_repeat_metrics_artifact,
    repeat_metrics_artifact_path,
)
from oasis.graphs import atoms_to_graph_dataset_view, save_aligned_graph_dataset_parquet
from oasis.learning_curve.results_io import (
    LearningCurveSweepMetadata,
    load_learning_curve_results_artifact,
    save_learning_curve_results_artifact,
)
from oasis.plot import learning_curve_plot
from oasis.sweep import (
    GraphDatasetView,
    GraphRecord,
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
    from oasis.learning_curve.execution import sweep_results_frame
    from oasis.learning_curve.runners import (
        ConfiguredSweepModelFamily,
        LearnedModelSweepRunner,
        SupervisedModelSweepRunner,
        SweepFamilyRunArtifacts,
        SweepFamilySpec,
    )

    HAS_METHOD = True
except ModuleNotFoundError:
    HAS_METHOD = False

try:
    import sklearn  # noqa: F401

    HAS_SKLEARN = True
except ModuleNotFoundError:
    HAS_SKLEARN = False


class _GenerateSweepSplitsTests(unittest.TestCase):
    def test_resolve_configured_sweep_sizes_from_fractions(self) -> None:
        self.assertEqual(
            resolve_configured_sweep_sizes(
                100,
                min_train=None,
                max_train=None,
                sweep_fractions=[0.1, 0.2, 0.9],
            ),
            (10, 20, 90),
        )

    def test_resolve_configured_sweep_sizes_unions_explicit_and_fractional_budgets(
        self,
    ) -> None:
        self.assertEqual(
            resolve_configured_sweep_sizes(
                100,
                min_train=None,
                max_train=None,
                sweep_sizes=[1, 2, 3, 20],
                sweep_fractions=[0.05, 0.1, 0.2],
            ),
            (1, 2, 3, 5, 10, 20),
        )

    def test_resolve_configured_sweep_sizes_deduplicates_mixed_sources(self) -> None:
        self.assertEqual(
            resolve_configured_sweep_sizes(
                100,
                min_train=None,
                max_train=None,
                sweep_sizes=[5, 10, 20],
                sweep_fractions=[0.05, 0.1, 0.2],
            ),
            (5, 10, 20),
        )

    def test_generate_sweep_splits_honors_requested_sweep_sizes(self) -> None:
        splits = list(
            generate_sweep_splits(
                n_samples=12,
                min_train=1,
                max_train=1,
                n_repeats=2,
                rng=np.random.default_rng(7),
                requested_sweep_sizes=(2, 5, 9),
            )
        )

        self.assertEqual(
            [split.sweep_size for split in splits],
            [2, 2, 5, 5, 9, 9],
        )

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

    def test_generate_sweep_splits_skips_points_when_test_too_small(self) -> None:
        splits = list(
            generate_sweep_splits(
                n_samples=7,
                min_train=4,
                max_train=6,
                n_repeats=2,
                rng=np.random.default_rng(7),
                min_test_size=2,
            )
        )

        self.assertEqual([split.sweep_size for split in splits], [4, 4, 5, 5])
        self.assertEqual([len(split.test_idx) for split in splits], [3, 3, 2, 2])

    def test_same_seed_gives_same_splits_with_test_size_guard(self) -> None:
        splits_a = list(
            generate_sweep_splits(
                n_samples=7,
                min_train=4,
                max_train=6,
                n_repeats=2,
                rng=np.random.default_rng(42),
                min_test_size=2,
            )
        )
        splits_b = list(
            generate_sweep_splits(
                n_samples=7,
                min_train=4,
                max_train=6,
                n_repeats=2,
                rng=np.random.default_rng(42),
                min_test_size=2,
            )
        )

        self.assertEqual(len(splits_a), len(splits_b))
        for split_a, split_b in zip(splits_a, splits_b, strict=True):
            self.assertEqual(split_a.sweep_size, split_b.sweep_size)
            np.testing.assert_array_equal(split_a.train_idx, split_b.train_idx)
            np.testing.assert_array_equal(split_a.test_idx, split_b.test_idx)

    def test_screening_holdout_size_for_budget_uses_fraction_with_minimum(self) -> None:
        self.assertEqual(
            screening_holdout_size_for_budget(
                10,
                screen_fraction=0.2,
                min_screen_size=1,
            ),
            2,
        )
        self.assertEqual(
            screening_holdout_size_for_budget(
                5,
                screen_fraction=0.2,
                min_screen_size=2,
            ),
            2,
        )

    def test_generate_screening_sweep_splits_uses_fixed_budgeted_holdout(self) -> None:
        rng = np.random.default_rng(123)

        splits = list(
            generate_screening_sweep_splits(
                n_samples=10,
                min_train=5,
                max_train=6,
                n_repeats=2,
                rng=rng,
                screen_fraction=0.4,
                min_screen_size=2,
            )
        )

        self.assertEqual([split.sweep_size for split in splits], [5, 5, 6, 6, 6])
        for split in splits:
            self.assertEqual(
                len(split.test_idx),
                screening_holdout_size_for_budget(
                    split.sweep_size,
                    screen_fraction=0.4,
                    min_screen_size=2,
                ),
            )
            self.assertEqual(
                len(split.train_idx),
                outer_train_size_if_screening_feasible(
                    split.sweep_size,
                    screen_fraction=0.4,
                    min_screen_size=2,
                ),
            )
            self.assertEqual(
                len(np.intersect1d(split.train_idx, split.test_idx)),
                0,
            )
            self.assertEqual(len(np.union1d(split.train_idx, split.test_idx)), split.sweep_size)
            self.assertLess(len(split.train_idx) + len(split.test_idx), 10)
            self.assertIsNone(split.val_idx)

    def test_generate_screening_sweep_splits_preserves_holdout_size_when_repeats_is_one(
        self,
    ) -> None:
        splits = list(
            generate_screening_sweep_splits(
                n_samples=7,
                min_train=4,
                max_train=4,
                n_repeats=1,
                rng=np.random.default_rng(42),
                screen_fraction=0.25,
                min_screen_size=1,
            )
        )

        self.assertEqual([split.sweep_size for split in splits], [4, 4, 4, 4])
        self.assertEqual([len(split.test_idx) for split in splits], [1, 1, 1, 1])
        self.assertEqual([len(split.train_idx) for split in splits], [3, 3, 3, 3])

    def test_same_seed_gives_same_screening_splits(self) -> None:
        splits_a = list(
            generate_screening_sweep_splits(
                n_samples=10,
                min_train=5,
                max_train=6,
                n_repeats=2,
                rng=np.random.default_rng(42),
                screen_fraction=0.4,
                min_screen_size=2,
            )
        )
        splits_b = list(
            generate_screening_sweep_splits(
                n_samples=10,
                min_train=5,
                max_train=6,
                n_repeats=2,
                rng=np.random.default_rng(42),
                screen_fraction=0.4,
                min_screen_size=2,
            )
        )

        self.assertEqual(len(splits_a), len(splits_b))
        for split_a, split_b in zip(splits_a, splits_b, strict=True):
            self.assertEqual(split_a.sweep_size, split_b.sweep_size)
            np.testing.assert_array_equal(split_a.train_idx, split_b.train_idx)
            np.testing.assert_array_equal(split_a.test_idx, split_b.test_idx)

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


class _BuildSweepDatasetFromFrameTests(unittest.TestCase):
    def test_build_sweep_dataset_from_frame_without_graphs_uses_frame_order(
        self,
    ) -> None:
        df = pd.DataFrame(
            {
                "reaction": ["rxn-b", "rxn-a"],
                "reference_ads_eng": [2.0, 1.0],
                "model_a_mlip_ads_eng_median": [2.2, 1.1],
                "model_b_mlip_ads_eng_median": [1.8, 0.9],
            }
        )

        dataset = build_sweep_dataset_from_frame(df)

        np.testing.assert_array_equal(dataset.sample_ids, np.array(["rxn-b", "rxn-a"]))
        np.testing.assert_array_equal(
            dataset.mlip_features,
            np.array([[2.2, 1.8], [1.1, 0.9]]),
        )
        np.testing.assert_array_equal(dataset.targets, np.array([2.0, 1.0]))
        self.assertFalse(dataset.has_graphs)

    def test_build_sweep_dataset_from_frame_with_graphs_aligns_to_frame_order(
        self,
    ) -> None:
        df = pd.DataFrame(
            {
                "reaction": ["rxn-c", "rxn-a", "rxn-b"],
                "reference_ads_eng": [3.0, 1.0, 2.0],
                "model_a_mlip_ads_eng_median": [3.3, 1.1, 2.2],
            }
        )
        graph_view = GraphDatasetView.from_records(
            (
                GraphRecord(
                    sample_id="rxn-a",
                    node_features=np.array([[1.0]]),
                    edge_index=np.empty((2, 0), dtype=np.int64),
                ),
                GraphRecord(
                    sample_id="rxn-b",
                    node_features=np.array([[2.0]]),
                    edge_index=np.empty((2, 0), dtype=np.int64),
                ),
                GraphRecord(
                    sample_id="rxn-c",
                    node_features=np.array([[3.0]]),
                    edge_index=np.empty((2, 0), dtype=np.int64),
                ),
            )
        )

        dataset = build_sweep_dataset_from_frame(df, graph_view=graph_view)

        np.testing.assert_array_equal(
            dataset.sample_ids,
            np.array(["rxn-c", "rxn-a", "rxn-b"]),
        )
        np.testing.assert_array_equal(
            dataset.mlip_features,
            np.array([[3.3], [1.1], [2.2]]),
        )
        np.testing.assert_array_equal(dataset.targets, np.array([3.0, 1.0, 2.0]))
        self.assertTrue(dataset.has_graphs)
        self.assertEqual(dataset.graphs.sample_ids, ("rxn-c", "rxn-a", "rxn-b"))
        self.assertIs(dataset.sample(0).graph, graph_view["rxn-c"])
        self.assertIs(dataset.sample(1).graph, graph_view["rxn-a"])
        self.assertIs(dataset.sample(2).graph, graph_view["rxn-b"])

    def test_build_sweep_dataset_from_config_loads_configured_graph_view(self) -> None:
        df = pd.DataFrame(
            {
                "reaction": ["rxn-c", "rxn-a", "rxn-b"],
                "reference_ads_eng": [3.0, 1.0, 2.0],
                "model_a_mlip_ads_eng_median": [3.3, 1.1, 2.2],
            }
        )
        graph_view = GraphDatasetView.from_records(
            (
                GraphRecord(
                    sample_id="rxn-a",
                    node_features=np.array([[1.0]]),
                    edge_index=np.empty((2, 0), dtype=np.int64),
                ),
                GraphRecord(
                    sample_id="rxn-b",
                    node_features=np.array([[2.0]]),
                    edge_index=np.empty((2, 0), dtype=np.int64),
                ),
                GraphRecord(
                    sample_id="rxn-c",
                    node_features=np.array([[3.0]]),
                    edge_index=np.empty((2, 0), dtype=np.int64),
                ),
            )
        )
        graph_dataset_cfg = SimpleNamespace(
            path="data/graphs/reactions.json",
            join_key="reaction",
        )
        cfg = SimpleNamespace(
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(graph_dataset=graph_dataset_cfg)
            )
        )

        with patch(
            "oasis.graphs.load_configured_graph_dataset_view",
            return_value=graph_view,
        ) as mock_load:
            dataset = build_sweep_dataset_from_config(df, cfg)

        self.assertIs(mock_load.call_args.args[0], graph_dataset_cfg)
        self.assertTrue(dataset.has_graphs)
        self.assertEqual(dataset.graphs.sample_ids, ("rxn-c", "rxn-a", "rxn-b"))

    def test_build_sweep_dataset_from_config_prefers_explicit_graph_view(self) -> None:
        df = pd.DataFrame(
            {
                "reaction": ["rxn-b", "rxn-a"],
                "reference_ads_eng": [2.0, 1.0],
                "model_a_mlip_ads_eng_median": [2.2, 1.1],
            }
        )
        graph_view = GraphDatasetView.from_records(
            (
                GraphRecord(
                    sample_id="rxn-a",
                    node_features=np.array([[1.0]]),
                    edge_index=np.empty((2, 0), dtype=np.int64),
                ),
                GraphRecord(
                    sample_id="rxn-b",
                    node_features=np.array([[2.0]]),
                    edge_index=np.empty((2, 0), dtype=np.int64),
                ),
            )
        )
        cfg = SimpleNamespace(
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(
                    graph_dataset=SimpleNamespace(
                        path="data/graphs/reactions.json",
                        join_key="reaction",
                    )
                )
            )
        )

        with patch("oasis.graphs.load_configured_graph_dataset_view") as mock_load:
            dataset = build_sweep_dataset_from_config(df, cfg, graph_view=graph_view)

        mock_load.assert_not_called()
        self.assertTrue(dataset.has_graphs)
        self.assertEqual(dataset.graphs.sample_ids, ("rxn-b", "rxn-a"))

    def test_build_sweep_dataset_from_config_honors_configured_graph_join_key(
        self,
    ) -> None:
        df = pd.DataFrame(
            {
                "reaction_id": ["rxn-c", "rxn-a", "rxn-b"],
                "reference_ads_eng": [3.0, 1.0, 2.0],
                "model_a_mlip_ads_eng_median": [3.3, 1.1, 2.2],
            }
        )
        graph_view = GraphDatasetView.from_records(
            (
                GraphRecord(
                    sample_id="rxn-a",
                    node_features=np.array([[1.0]]),
                    edge_index=np.empty((2, 0), dtype=np.int64),
                ),
                GraphRecord(
                    sample_id="rxn-b",
                    node_features=np.array([[2.0]]),
                    edge_index=np.empty((2, 0), dtype=np.int64),
                ),
                GraphRecord(
                    sample_id="rxn-c",
                    node_features=np.array([[3.0]]),
                    edge_index=np.empty((2, 0), dtype=np.int64),
                ),
            )
        )
        cfg = SimpleNamespace(
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(
                    graph_dataset=SimpleNamespace(
                        path="data/graphs/reactions.json",
                        join_key="reaction_id",
                    )
                )
            )
        )

        with patch("oasis.graphs.load_configured_graph_dataset_view") as mock_load:
            dataset = build_sweep_dataset_from_config(df, cfg, graph_view=graph_view)

        mock_load.assert_not_called()
        self.assertEqual(dataset.graphs.sample_ids, ("rxn-c", "rxn-a", "rxn-b"))

    def test_build_sweep_dataset_from_config_loads_existing_graph_artifact_when_enabled(
        self,
    ) -> None:
        df = pd.DataFrame(
            {
                "reaction": ["rxn-b", "rxn-a"],
                "reference_ads_eng": [2.0, 1.0],
                "model_a_mlip_ads_eng_median": [2.2, 1.1],
            }
        )
        dataset = SweepDataset(
            mlip_features=np.array([[2.2], [1.1]]),
            targets=np.array([2.0, 1.0]),
            sample_ids=np.array(["rxn-b", "rxn-a"]),
        )
        cfg = SimpleNamespace(
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(
                    graph_dataset=SimpleNamespace(
                        path="data/graphs/reactions.parquet",
                        join_key="reaction",
                    )
                )
            )
        )

        with patch("pathlib.Path.is_file", return_value=True):
            with patch(
                "oasis.graphs.load_sweep_dataset_from_graph_artifact",
                return_value=dataset,
            ) as mock_load_dataset:
                with patch(
                    "oasis.graphs.load_configured_graph_dataset_view"
                ) as mock_load_graphs:
                    built_dataset = build_sweep_dataset_from_config(df, cfg)

        self.assertIs(built_dataset, dataset)
        self.assertEqual(mock_load_dataset.call_args.kwargs["join_key"], "reaction")
        mock_load_graphs.assert_not_called()

    def test_build_sweep_dataset_from_frame_passes_auxiliary_views(self) -> None:
        df = pd.DataFrame(
            {
                "reaction": ["rxn-a", "rxn-b"],
                "reference_ads_eng": [1.0, 2.0],
                "model_a_mlip_ads_eng_median": [1.1, 2.2],
            }
        )
        extra = pd.DataFrame({"x": [10.0, 20.0]})

        dataset = build_sweep_dataset_from_frame(df, auxiliary_views={"latent": extra})

        self.assertIn("latent", dataset.auxiliary_views)
        pd.testing.assert_frame_equal(dataset.auxiliary_views["latent"], extra)

    def test_build_sweep_dataset_from_config_passes_auxiliary_views_through(
        self,
    ) -> None:
        df = pd.DataFrame(
            {
                "reaction": ["rxn-a", "rxn-b"],
                "reference_ads_eng": [1.0, 2.0],
                "model_a_mlip_ads_eng_median": [1.1, 2.2],
            }
        )
        latent_df = pd.DataFrame({"feature": [0.5, 0.6]})
        cfg = SimpleNamespace(
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(graph_dataset=None, models=None)
            )
        )

        with patch(
            "oasis.graphs.load_configured_graph_dataset_view", return_value=None
        ):
            dataset = build_sweep_dataset_from_config(
                df, cfg, auxiliary_views={"latent": latent_df}
            )

        self.assertIn("latent", dataset.auxiliary_views)
        pd.testing.assert_frame_equal(dataset.auxiliary_views["latent"], latent_df)

    def test_build_sweep_dataset_from_config_masks_strict_inference_anomalies(
        self,
    ) -> None:
        df = pd.DataFrame(
            {
                "reaction": ["rxn-a", "rxn-b"],
                "reference_ads_eng": [1.0, 2.0],
                "model_a_mlip_ads_eng_median": [1.0, 2.0],
                "model_b_mlip_ads_eng_median": [3.0, 4.0],
                "model_a_slab_conv": [0, 0],
                "model_a_ads_conv": [0, 0],
                "model_a_slab_move": [0, 0],
                "model_a_ads_move": [0, 0],
                "model_a_slab_seed": [0, 0],
                "model_a_ads_seed": [0, 0],
                "model_a_ads_eng_seed": [0, 0],
                "model_a_adsorbate_migration": [0, 0],
                "model_b_slab_conv": [1, 0],
                "model_b_ads_conv": [0, 0],
                "model_b_slab_move": [0, 0],
                "model_b_ads_move": [0, 0],
                "model_b_slab_seed": [0, 0],
                "model_b_ads_seed": [0, 0],
                "model_b_ads_eng_seed": [0, 0],
                "model_b_adsorbate_migration": [0, 0],
            }
        )
        cfg = SimpleNamespace(
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(
                    graph_dataset=None,
                    models=None,
                    mlip_selection=SimpleNamespace(
                        exclude_anomalous=True,
                        strict_inference_anomaly=True,
                    ),
                )
            )
        )

        with patch(
            "oasis.graphs.load_configured_graph_dataset_view", return_value=None
        ):
            dataset = build_sweep_dataset_from_config(df, cfg)

        np.testing.assert_array_equal(
            dataset.auxiliary_views["mlip_validity_mask"],
            np.array([[True, False], [True, True]]),
        )
        np.testing.assert_allclose(
            dataset.mlip_features,
            np.array([[1.0, 1.0], [2.0, 4.0]]),
        )

    def test_build_sweep_dataset_from_config_skips_latent_when_disabled(self) -> None:
        df = pd.DataFrame(
            {
                "reaction": ["rxn-a"],
                "reference_ads_eng": [1.0],
                "model_a_mlip_ads_eng_median": [1.1],
            }
        )
        cfg = SimpleNamespace(
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(
                    graph_dataset=None,
                    models=SimpleNamespace(use_latent=False, latent=None),
                )
            )
        )

        with patch(
            "oasis.graphs.load_configured_graph_dataset_view", return_value=None
        ):
            dataset = build_sweep_dataset_from_config(df, cfg)

        self.assertNotIn("latent", dataset.auxiliary_views or {})


class _GenerateSweepSplitsWithValidationTests(unittest.TestCase):
    def test_inner_validation_size_for_sweep_uses_fraction_policy(self) -> None:
        self.assertEqual(
            inner_validation_size_for_sweep(
                1,
                validation_fraction=0.2,
                min_val_size=1,
            ),
            1,
        )
        self.assertEqual(
            inner_validation_size_for_sweep(
                4,
                validation_fraction=0.2,
                min_val_size=1,
            ),
            1,
        )
        self.assertEqual(
            inner_validation_size_for_sweep(
                5,
                validation_fraction=0.2,
                min_val_size=1,
            ),
            1,
        )
        self.assertEqual(
            inner_validation_size_for_sweep(
                10,
                validation_fraction=0.2,
                min_val_size=1,
            ),
            2,
        )

    def test_inner_validation_size_for_sweep_uses_minimum_policy(self) -> None:
        self.assertEqual(
            inner_validation_size_for_sweep(
                4,
                validation_fraction=0.2,
                min_val_size=2,
            ),
            2,
        )
        self.assertEqual(
            inner_validation_size_for_sweep(
                10,
                validation_fraction=0.0,
                min_val_size=3,
            ),
            3,
        )
        self.assertEqual(
            inner_validation_size_for_sweep(
                14,
                validation_fraction=0.2,
                min_val_size=2,
            ),
            2,
        )
        self.assertEqual(
            inner_validation_size_for_sweep(
                15,
                validation_fraction=0.2,
                min_val_size=2,
            ),
            3,
        )

    def test_inner_validation_size_for_sweep_uses_tuning_minimum_policy(self) -> None:
        self.assertEqual(
            inner_validation_size_for_sweep(
                5,
                validation_fraction=0.2,
                min_val_size=1,
                min_tuning_val_size=3,
            ),
            3,
        )
        self.assertEqual(
            inner_validation_size_for_sweep(
                20,
                validation_fraction=0.2,
                min_val_size=2,
                min_tuning_val_size=3,
            ),
            4,
        )

    def test_inner_validation_size_for_sweep_rejects_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "sweep_size must be positive"):
            inner_validation_size_for_sweep(
                0,
                validation_fraction=0.2,
                min_val_size=1,
            )

        with self.assertRaisesRegex(
            ValueError,
            "validation_fraction must be non-negative",
        ):
            inner_validation_size_for_sweep(
                5,
                validation_fraction=-0.1,
                min_val_size=1,
            )

        with self.assertRaisesRegex(ValueError, "min_val_size must be positive"):
            inner_validation_size_for_sweep(
                5,
                validation_fraction=0.2,
                min_val_size=0,
            )

        with self.assertRaisesRegex(
            ValueError,
            "min_tuning_val_size must be positive",
        ):
            inner_validation_size_for_sweep(
                5,
                validation_fraction=0.2,
                min_val_size=1,
                min_tuning_val_size=0,
            )

    def test_validation_size_if_sweep_feasible_centralizes_feasibility_rule(
        self,
    ) -> None:
        self.assertIsNone(
            validation_size_if_sweep_feasible(
                4,
                validation_fraction=0.2,
                min_val_size=1,
                min_tuning_val_size=3,
                min_inner_train_size=2,
            )
        )
        self.assertEqual(
            validation_size_if_sweep_feasible(
                5,
                validation_fraction=0.2,
                min_val_size=1,
                min_tuning_val_size=3,
                min_inner_train_size=2,
            ),
            3,
        )

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
                    np.concatenate([split.train_idx, split.val_idx, split.test_idx])
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

    def test_generate_sweep_splits_with_validation_skips_points_when_test_too_small(
        self,
    ) -> None:
        splits = list(
            generate_sweep_splits_with_validation(
                n_samples=8,
                min_train=4,
                max_train=6,
                n_val=2,
                n_repeats=2,
                rng=np.random.default_rng(7),
                min_test_size=3,
            )
        )

        self.assertEqual([split.sweep_size for split in splits], [4, 4, 5, 5])
        self.assertEqual([len(split.test_idx) for split in splits], [4, 4, 3, 3])

    def test_same_seed_gives_same_validation_splits_with_test_size_guard(self) -> None:
        splits_a = list(
            generate_sweep_splits_with_validation(
                n_samples=8,
                min_train=4,
                max_train=6,
                n_val=2,
                n_repeats=2,
                rng=np.random.default_rng(42),
                min_test_size=3,
            )
        )
        splits_b = list(
            generate_sweep_splits_with_validation(
                n_samples=8,
                min_train=4,
                max_train=6,
                n_val=2,
                n_repeats=2,
                rng=np.random.default_rng(42),
                min_test_size=3,
            )
        )

        self.assertEqual(len(splits_a), len(splits_b))
        for split_a, split_b in zip(splits_a, splits_b, strict=True):
            self.assertEqual(split_a.sweep_size, split_b.sweep_size)
            np.testing.assert_array_equal(split_a.train_idx, split_b.train_idx)
            np.testing.assert_array_equal(split_a.val_idx, split_b.val_idx)
            np.testing.assert_array_equal(split_a.test_idx, split_b.test_idx)

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

    def test_generate_sweep_splits_with_validation_requires_outer_train_to_fit_min_inner_train(
        self,
    ) -> None:
        splits = list(
            generate_sweep_splits_with_validation(
                n_samples=7,
                min_train=1,
                max_train=4,
                n_val=2,
                n_repeats=1,
                rng=np.random.default_rng(7),
                min_inner_train_size=2,
            )
        )

        self.assertEqual([split.sweep_size for split in splits], [4])
        self.assertTrue(all(len(split.train_idx) == 2 for split in splits))

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

        with self.assertRaisesRegex(
            ValueError,
            "min_inner_train_size must be positive",
        ):
            list(
                generate_sweep_splits_with_validation(
                    n_samples=6,
                    min_train=2,
                    max_train=3,
                    n_val=1,
                    n_repeats=1,
                    rng=np.random.default_rng(1),
                    min_inner_train_size=0,
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

    def test_generate_inner_validation_sweep_splits_honors_minimum_policy(self) -> None:
        splits = list(
            generate_inner_validation_sweep_splits(
                n_samples=12,
                min_train=4,
                max_train=10,
                n_repeats=1,
                rng=np.random.default_rng(7),
                validation_fraction=0.2,
                min_val_size=2,
            )
        )

        self.assertEqual([split.sweep_size for split in splits], [4, 5, 6, 7, 8, 9, 10])
        self.assertEqual(
            [len(split.val_idx) for split in splits], [2, 2, 2, 2, 2, 2, 2]
        )
        self.assertEqual(
            [len(split.train_idx) for split in splits], [2, 3, 4, 5, 6, 7, 8]
        )

    def test_generate_inner_validation_sweep_splits_honors_tuning_validation_minimum(
        self,
    ) -> None:
        splits = list(
            generate_inner_validation_sweep_splits(
                n_samples=12,
                min_train=4,
                max_train=10,
                n_repeats=1,
                rng=np.random.default_rng(7),
                validation_fraction=0.2,
                min_val_size=1,
                min_tuning_val_size=3,
            )
        )

        self.assertEqual([split.sweep_size for split in splits], [4, 5, 6, 7, 8, 9, 10])
        self.assertEqual(
            [len(split.val_idx) for split in splits], [3, 3, 3, 3, 3, 3, 3]
        )
        self.assertEqual(
            [len(split.train_idx) for split in splits], [1, 2, 3, 4, 5, 6, 7]
        )

    def test_generate_inner_validation_sweep_splits_exposes_one_point_validation_pathology(
        self,
    ) -> None:
        splits = list(
            generate_inner_validation_sweep_splits(
                n_samples=8,
                min_train=2,
                max_train=6,
                n_repeats=1,
                rng=np.random.default_rng(7),
                validation_fraction=0.2,
                min_val_size=1,
            )
        )

        self.assertEqual([split.sweep_size for split in splits], [2, 3, 4, 5, 6])
        self.assertEqual([len(split.val_idx) for split in splits], [1, 1, 1, 1, 1])
        self.assertEqual([len(split.train_idx) for split in splits], [1, 2, 3, 4, 5])

    def test_generate_inner_validation_sweep_splits_skips_points_with_insufficient_tuning_validation(
        self,
    ) -> None:
        splits = list(
            generate_inner_validation_sweep_splits(
                n_samples=8,
                min_train=2,
                max_train=6,
                n_repeats=1,
                rng=np.random.default_rng(7),
                validation_fraction=0.2,
                min_val_size=1,
                min_tuning_val_size=3,
            )
        )

        self.assertEqual([split.sweep_size for split in splits], [4, 5, 6])
        self.assertEqual([len(split.val_idx) for split in splits], [3, 3, 3])

    def test_generate_inner_validation_sweep_splits_honors_minimum_inner_train_policy(
        self,
    ) -> None:
        splits = list(
            generate_inner_validation_sweep_splits(
                n_samples=8,
                min_train=2,
                max_train=6,
                n_repeats=1,
                rng=np.random.default_rng(7),
                validation_fraction=0.2,
                min_val_size=1,
                min_inner_train_size=2,
            )
        )

        self.assertEqual([split.sweep_size for split in splits], [3, 4, 5, 6])
        self.assertEqual([len(split.val_idx) for split in splits], [1, 1, 1, 1])
        self.assertEqual([len(split.train_idx) for split in splits], [2, 3, 4, 5])

    def test_generate_inner_validation_sweep_splits_keeps_outer_test_disjoint(
        self,
    ) -> None:
        splits = list(
            generate_inner_validation_sweep_splits(
                n_samples=8,
                min_train=4,
                max_train=6,
                n_repeats=2,
                rng=np.random.default_rng(7),
                min_test_size=3,
            )
        )

        full_idx = np.arange(8)
        self.assertEqual([split.sweep_size for split in splits], [4, 4, 5, 5])
        for split in splits:
            self.assertIsNotNone(split.val_idx)
            self.assertGreaterEqual(len(split.test_idx), 3)
            self.assertEqual(len(np.intersect1d(split.train_idx, split.test_idx)), 0)
            self.assertEqual(len(np.intersect1d(split.val_idx, split.test_idx)), 0)
            self.assertEqual(len(np.intersect1d(split.train_idx, split.val_idx)), 0)
            np.testing.assert_array_equal(
                np.sort(
                    np.concatenate([split.train_idx, split.val_idx, split.test_idx])
                ),
                full_idx,
            )

    def test_generate_inner_validation_sweep_splits_skips_infeasible_policy_points(
        self,
    ) -> None:
        splits = list(
            generate_inner_validation_sweep_splits(
                n_samples=10,
                min_train=1,
                max_train=4,
                n_repeats=2,
                rng=np.random.default_rng(7),
                validation_fraction=0.6,
                min_val_size=3,
            )
        )

        self.assertEqual([split.sweep_size for split in splits], [4, 4])
        self.assertTrue(all(len(split.val_idx) == 3 for split in splits))
        self.assertTrue(all(len(split.train_idx) == 1 for split in splits))

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

    def test_generate_inner_validation_sweep_splits_returns_no_splits_when_min_val_consumes_budget(
        self,
    ) -> None:
        splits = list(
            generate_inner_validation_sweep_splits(
                n_samples=10,
                min_train=1,
                max_train=3,
                n_repeats=2,
                rng=np.random.default_rng(7),
                validation_fraction=0.0,
                min_val_size=3,
            )
        )

        self.assertEqual(splits, [])

    def test_generate_inner_validation_sweep_splits_rejects_invalid_min_inner_train_size(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "min_inner_train_size must be positive",
        ):
            list(
                generate_inner_validation_sweep_splits(
                    n_samples=10,
                    min_train=1,
                    max_train=3,
                    n_repeats=1,
                    rng=np.random.default_rng(7),
                    min_inner_train_size=0,
                )
            )

    def test_generate_sweep_splits_with_validation_returns_no_splits_when_min_test_size_leaves_too_little_room(
        self,
    ) -> None:
        splits = list(
            generate_sweep_splits_with_validation(
                n_samples=6,
                min_train=3,
                max_train=5,
                n_val=2,
                n_repeats=2,
                rng=np.random.default_rng(7),
                min_test_size=4,
            )
        )

        self.assertEqual(splits, [])

    def test_generate_screening_sweep_splits_with_validation_uses_budgeted_screen_holdout(
        self,
    ) -> None:
        splits = list(
            generate_screening_sweep_splits_with_validation(
                n_samples=12,
                min_train=6,
                max_train=7,
                n_repeats=2,
                rng=np.random.default_rng(7),
                screen_fraction=0.25,
                min_screen_size=2,
                validation_fraction=0.25,
                min_val_size=1,
                min_inner_train_size=2,
            )
        )

        self.assertEqual([split.sweep_size for split in splits], [6, 6, 6, 7, 7, 7])
        for split in splits:
            self.assertIsNotNone(split.val_idx)
            self.assertEqual(
                len(np.union1d(np.union1d(split.train_idx, split.val_idx), split.test_idx)),
                split.sweep_size,
            )
            self.assertLess(
                len(split.train_idx) + len(split.val_idx) + len(split.test_idx),
                12,
            )
            self.assertEqual(
                len(split.test_idx),
                screening_holdout_size_for_budget(
                    split.sweep_size,
                    screen_fraction=0.25,
                    min_screen_size=2,
                ),
            )
            outer_train_size = split.sweep_size - len(split.test_idx)
            self.assertEqual(len(split.train_idx) + len(split.val_idx), outer_train_size)

    def test_build_sweep_split_collection_honors_min_test_size(self) -> None:
        split_collection = build_sweep_split_collection(
            n_samples=7,
            min_train=4,
            max_train=6,
            n_repeats=1,
            seed=3,
            min_test_size=2,
        )

        self.assertEqual(
            [split.sweep_size for split in split_collection.splits], [4, 5]
        )
        self.assertEqual(
            [len(split.test_idx) for split in split_collection.splits], [3, 2]
        )

    def test_build_sweep_split_collection_screening_mode_uses_fixed_holdout(self) -> None:
        split_collection = build_sweep_split_collection(
            n_samples=10,
            min_train=5,
            max_train=7,
            n_repeats=1,
            seed=3,
            budget_mode="screening_fraction",
            screen_fraction=0.4,
            min_screen_size=2,
        )

        self.assertEqual(
            [split.sweep_size for split in split_collection.splits],
            [5, 5, 6, 6, 6, 7, 7, 7],
        )
        self.assertEqual(
            [len(split.test_idx) for split in split_collection.splits],
            [2, 2, 2, 2, 2, 2, 2, 2],
        )
        self.assertEqual(
            [len(split.train_idx) for split in split_collection.splits],
            [3, 3, 4, 4, 4, 5, 5, 5],
        )

    def test_build_sweep_split_collection_honors_requested_sparse_sizes(self) -> None:
        split_collection = build_sweep_split_collection(
            n_samples=20,
            min_train=None,
            max_train=None,
            requested_sweep_sizes=(2, 10, 18),
            n_repeats=2,
            seed=11,
        )

        self.assertEqual(
            sorted({split.sweep_size for split in split_collection.splits}),
            [2, 10, 18],
        )

    def test_build_sweep_split_collection_screening_mode_honors_validation_requirements(
        self,
    ) -> None:
        split_collection = build_sweep_split_collection(
            n_samples=12,
            min_train=4,
            max_train=7,
            n_repeats=1,
            seed=3,
            requirements=SweepFamilyRequirements(
                min_train_size=0,
                requires_inner_validation=True,
            ),
            budget_mode="screening_fraction",
            screen_fraction=0.25,
            min_screen_size=2,
            validation_fraction=0.2,
            min_val_size=1,
            min_tuning_val_size=2,
            min_inner_train_size=2,
        )

        self.assertEqual(
            [split.sweep_size for split in split_collection.splits], [6, 6, 6, 7, 7, 7]
        )
        self.assertEqual(
            [len(split.test_idx) for split in split_collection.splits],
            [2, 2, 2, 2, 2, 2],
        )
        self.assertEqual(
            [len(split.val_idx) for split in split_collection.splits],
            [2, 2, 2, 2, 2, 2],
        )

    def test_build_sweep_split_collection_screening_mode_requires_fraction(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "screen_fraction must be provided",
        ):
            build_sweep_split_collection(
                n_samples=10,
                min_train=5,
                max_train=7,
                n_repeats=1,
                seed=3,
                budget_mode="screening_fraction",
            )

    def test_build_sweep_split_collection_rejects_invalid_planner_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "n_samples must be positive"):
            build_sweep_split_collection(
                n_samples=0,
                min_train=1,
                max_train=2,
                n_repeats=1,
                seed=3,
            )

        with self.assertRaisesRegex(ValueError, "min_train must be positive"):
            build_sweep_split_collection(
                n_samples=6,
                min_train=0,
                max_train=2,
                n_repeats=1,
                seed=3,
            )

        with self.assertRaisesRegex(ValueError, "max_train must be positive"):
            build_sweep_split_collection(
                n_samples=6,
                min_train=1,
                max_train=0,
                n_repeats=1,
                seed=3,
            )

        with self.assertRaisesRegex(ValueError, "n_repeats must be positive"):
            build_sweep_split_collection(
                n_samples=6,
                min_train=1,
                max_train=2,
                n_repeats=0,
                seed=3,
            )

        with self.assertRaisesRegex(ValueError, "min_test_size must be positive"):
            build_sweep_split_collection(
                n_samples=6,
                min_train=1,
                max_train=2,
                n_repeats=1,
                seed=3,
                min_test_size=0,
            )

    def test_build_sweep_split_collection_preserves_train_only_requirements(
        self,
    ) -> None:
        requirements = SweepFamilyRequirements(
            min_train_size=5,
            requires_inner_validation=False,
        )

        split_collection = build_sweep_split_collection(
            n_samples=9,
            min_train=2,
            max_train=8,
            n_repeats=1,
            seed=3,
            requirements=requirements,
            min_test_size=2,
        )

        self.assertEqual(split_collection.planning_requirements, requirements)
        self.assertEqual(
            [split.sweep_size for split in split_collection.splits], [5, 6, 7]
        )
        self.assertTrue(all(split.val_idx is None for split in split_collection.splits))

    def test_build_sweep_split_collection_returns_empty_train_only_region_when_test_minimum_blocks_all_splits(
        self,
    ) -> None:
        requirements = SweepFamilyRequirements(
            min_train_size=5,
            requires_inner_validation=False,
        )

        split_collection = build_sweep_split_collection(
            n_samples=6,
            min_train=4,
            max_train=6,
            n_repeats=2,
            seed=3,
            requirements=requirements,
            min_test_size=2,
        )

        self.assertEqual(split_collection.splits, ())
        self.assertEqual(split_collection.planning_requirements, requirements)

    def test_build_sweep_split_collection_uses_policy_sized_validation_budget(
        self,
    ) -> None:
        requirements = SweepFamilyRequirements(
            min_train_size=2,
            requires_inner_validation=True,
        )

        split_collection = build_sweep_split_collection(
            n_samples=12,
            min_train=1,
            max_train=7,
            n_repeats=1,
            seed=3,
            requirements=requirements,
            validation_fraction=0.34,
            min_val_size=2,
            min_test_size=2,
        )

        self.assertEqual(split_collection.planning_requirements, requirements)
        self.assertEqual(
            [split.sweep_size for split in split_collection.splits],
            [3, 4, 5, 6, 7],
        )
        for split in split_collection.splits:
            expected_n_val = inner_validation_size_for_sweep(
                split.sweep_size,
                validation_fraction=0.34,
                min_val_size=2,
            )
            self.assertEqual(len(split.val_idx), expected_n_val)
            self.assertEqual(
                len(split.train_idx) + len(split.val_idx), split.sweep_size
            )

    def test_build_sweep_split_collection_honors_tuning_validation_minimum(
        self,
    ) -> None:
        split_collection = build_sweep_split_collection(
            n_samples=10,
            min_train=1,
            max_train=7,
            n_repeats=1,
            seed=3,
            requirements=SweepFamilyRequirements(
                min_train_size=0,
                requires_inner_validation=True,
            ),
            validation_fraction=0.2,
            min_val_size=1,
            min_tuning_val_size=3,
            min_test_size=2,
        )

        self.assertEqual(
            [split.sweep_size for split in split_collection.splits],
            [4, 5, 6, 7],
        )
        self.assertEqual(
            [len(split.val_idx) for split in split_collection.splits],
            [3, 3, 3, 3],
        )

    def test_build_sweep_split_collection_honors_minimum_inner_train_policy(
        self,
    ) -> None:
        split_collection = build_sweep_split_collection(
            n_samples=8,
            min_train=1,
            max_train=6,
            n_repeats=1,
            seed=3,
            requirements=SweepFamilyRequirements(
                min_train_size=0,
                requires_inner_validation=True,
            ),
            validation_fraction=0.2,
            min_val_size=1,
            min_inner_train_size=2,
            min_test_size=2,
        )

        self.assertEqual(
            [split.sweep_size for split in split_collection.splits],
            [3, 4, 5, 6],
        )
        self.assertEqual(
            [len(split.train_idx) for split in split_collection.splits],
            [2, 3, 4, 5],
        )

    def test_build_sweep_split_collection_returns_no_validation_sweeps_when_guards_consume_budget(
        self,
    ) -> None:
        split_collection = build_sweep_split_collection(
            n_samples=6,
            min_train=1,
            max_train=5,
            n_repeats=1,
            seed=3,
            requirements=SweepFamilyRequirements(
                min_train_size=0,
                requires_inner_validation=True,
            ),
            validation_fraction=0.0,
            min_val_size=2,
            min_test_size=4,
        )

        self.assertEqual(split_collection.splits, ())
        self.assertEqual(
            split_collection.planning_requirements,
            SweepFamilyRequirements(
                min_train_size=0,
                requires_inner_validation=True,
            ),
        )

    def test_build_sweep_split_collection_returns_empty_validation_region_when_minima_cannot_fit_together(
        self,
    ) -> None:
        requirements = SweepFamilyRequirements(
            min_train_size=0,
            requires_inner_validation=True,
        )

        split_collection_a = build_sweep_split_collection(
            n_samples=8,
            min_train=1,
            max_train=7,
            n_repeats=2,
            seed=3,
            requirements=requirements,
            validation_fraction=0.2,
            min_val_size=2,
            min_tuning_val_size=4,
            min_inner_train_size=3,
            min_test_size=2,
        )
        split_collection_b = build_sweep_split_collection(
            n_samples=8,
            min_train=1,
            max_train=7,
            n_repeats=2,
            seed=99,
            requirements=requirements,
            validation_fraction=0.2,
            min_val_size=2,
            min_tuning_val_size=4,
            min_inner_train_size=3,
            min_test_size=2,
        )

        self.assertEqual(split_collection_a.splits, ())
        self.assertEqual(split_collection_b.splits, ())
        self.assertEqual(split_collection_a.planning_requirements, requirements)
        self.assertEqual(split_collection_b.planning_requirements, requirements)

    def test_run_learning_curve_experiments_honors_minimum_inner_train_policy(
        self,
    ) -> None:
        X = np.arange(21, dtype=float).reshape(7, 3)
        y = np.arange(7, dtype=float)
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
                    min_train_size=0,
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
            validation_fraction=0.2,
            min_val_size=1,
            min_inner_train_size=2,
            model_families=[family],
        )

        self.assertIs(results.ridge_df, result_df)
        self.assertEqual(
            [split.sweep_size for split in family.last_payload.split_collection.splits],
            [3, 4, 5],
        )
        self.assertEqual(
            [
                len(split.train_idx)
                for split in family.last_payload.split_collection.splits
            ],
            [2, 3, 4],
        )

    def test_run_learning_curve_experiments_honors_tuning_validation_minimum(
        self,
    ) -> None:
        X = np.arange(24, dtype=float).reshape(8, 3)
        y = np.arange(8, dtype=float)
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
                    min_train_size=0,
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
            validation_fraction=0.2,
            min_val_size=1,
            min_tuning_val_size=3,
            model_families=[family],
        )

        self.assertIs(results.ridge_df, result_df)
        self.assertEqual(
            [split.sweep_size for split in family.last_payload.split_collection.splits],
            [4, 5],
        )
        self.assertEqual(
            [
                len(split.val_idx)
                for split in family.last_payload.split_collection.splits
            ],
            [3, 3],
        )


@unittest.skipUnless(HAS_METHOD, "requires method dependencies")
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
            "oasis.experiment.orchestration.run_learning_curve_experiments",
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
            )

        self.assertIsNotNone(results.ridge_df)
        dataset = run_mock.call_args.args[0]
        self.assertIsInstance(dataset, SweepDataset)
        np.testing.assert_array_equal(
            dataset.sample_ids,
            np.array(["rxn-0", "rxn-1", "rxn-2", "rxn-3", "rxn-4", "rxn-5"]),
        )
        self.assertIsNone(dataset.auxiliary_views)
        self.assertEqual(dataset.mlip_features.shape, (6, 2))
        self.assertIs(dataset.inputs.mlip_features, dataset.mlip_features)
        np.testing.assert_array_equal(
            dataset.targets,
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        )

    def test_run_learning_curve_experiments_from_frame_can_build_graph_backed_dataset(
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
        graph_view = GraphDatasetView.from_records(
            tuple(
                GraphRecord(
                    sample_id=f"rxn-{idx}",
                    node_features=np.arange(4, dtype=float).reshape(2, 2),
                    edge_index=np.array([[0], [1]], dtype=np.int64),
                )
                for idx in range(6)
            )
        )
        result_df = pd.DataFrame(
            {
                "n_train": [2, 3, 4],
                "rmse_mean": [0.4, 0.3, 0.2],
                "rmse_std": [0.05, 0.04, 0.03],
            }
        )

        with patch(
            "oasis.experiment.orchestration.run_learning_curve_experiments",
            autospec=True,
        ) as run_mock:
            run_mock.return_value = LearningCurveResults.from_mapping(
                {"ridge_df": result_df}
            )

            run_learning_curve_experiments_from_frame(
                df,
                min_train=2,
                max_train=4,
                n_repeats=1,
                graph_view=graph_view,
            )

        dataset = run_mock.call_args.args[0]
        self.assertIsInstance(dataset, SweepDataset)
        self.assertTrue(dataset.has_graphs)
        self.assertEqual(
            dataset.graphs.sample_ids,
            ("rxn-0", "rxn-1", "rxn-2", "rxn-3", "rxn-4", "rxn-5"),
        )
        self.assertIs(dataset.inputs.graph_view_required(), dataset.graph_view)
        self.assertIs(dataset.modalities.mlip_features, dataset.mlip_features)
        self.assertIs(dataset.modalities.graphs, dataset.graphs)
        self.assertIsNone(dataset.auxiliary_views)

    def test_run_learning_curve_experiments_from_frame_plumbs_atoms_derived_graphs(
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
        sample_ids = df["reaction"].tolist()
        sample_atoms = [
            Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74 + 0.01 * idx, 0.0, 0.0]])
            for idx in range(6)
        ]
        graph_view = atoms_to_graph_dataset_view(sample_ids, sample_atoms)
        result_df = pd.DataFrame(
            {
                "n_train": [2, 3, 4],
                "rmse_mean": [0.4, 0.3, 0.2],
                "rmse_std": [0.05, 0.04, 0.03],
            }
        )

        with patch(
            "oasis.experiment.orchestration.run_learning_curve_experiments",
            autospec=True,
        ) as run_mock:
            run_mock.return_value = LearningCurveResults.from_mapping(
                {"ridge_df": result_df}
            )

            run_learning_curve_experiments_from_frame(
                df,
                min_train=2,
                max_train=4,
                n_repeats=1,
                graph_view=graph_view,
            )

        dataset = run_mock.call_args.args[0]
        self.assertIsInstance(dataset, SweepDataset)
        self.assertTrue(dataset.has_graphs)
        self.assertEqual(
            dataset.graphs.sample_ids,
            ("rxn-0", "rxn-1", "rxn-2", "rxn-3", "rxn-4", "rxn-5"),
        )
        np.testing.assert_array_equal(
            dataset.sample_ids,
            np.array(sample_ids),
        )
        np.testing.assert_allclose(
            dataset.graphs["rxn-3"].node_positions,
            sample_atoms[3].positions,
        )
        np.testing.assert_array_equal(
            dataset.graphs["rxn-3"].node_features,
            np.array([[1.0], [1.0]]),
        )
        self.assertEqual(dataset.graphs["rxn-3"].edge_index.dtype, np.int64)
        self.assertIsNone(dataset.auxiliary_views)

    def test_run_learning_curve_experiments_dispatches_standard_workflow(self) -> None:
        dataset = SweepDataset(
            mlip_features=np.arange(12, dtype=float).reshape(6, 2),
            targets=np.arange(6, dtype=float),
        )
        expected = LearningCurveResults.from_mapping(
            {
                "ridge_df": pd.DataFrame(
                    {
                        "n_train": [2, 3],
                        "rmse_mean": [0.4, 0.3],
                        "rmse_std": [0.05, 0.04],
                    }
                )
            }
        )

        with patch(
            "oasis.experiment.orchestration.run_standard_learning_curve_experiments",
            autospec=True,
        ) as standard_mock, patch(
            "oasis.experiment.orchestration.run_screening_learning_curve_experiments",
            autospec=True,
        ) as screening_mock:
            standard_mock.return_value = expected

            result = run_learning_curve_experiments(
                dataset,
                min_train=2,
                max_train=3,
                n_repeats=1,
            )

        self.assertIs(result, expected)
        standard_mock.assert_called_once()
        screening_mock.assert_not_called()

    def test_run_learning_curve_experiments_dispatches_screening_workflow(self) -> None:
        dataset = SweepDataset(
            mlip_features=np.arange(12, dtype=float).reshape(6, 2),
            targets=np.arange(6, dtype=float),
        )
        expected = LearningCurveResults.from_mapping(
            {
                "ridge_df": pd.DataFrame(
                    {
                        "n_train": [4, 5],
                        "rmse_mean": [0.4, 0.3],
                        "rmse_std": [0.05, 0.04],
                    }
                )
            }
        )

        with patch(
            "oasis.experiment.orchestration.run_standard_learning_curve_experiments",
            autospec=True,
        ) as standard_mock, patch(
            "oasis.experiment.orchestration.run_screening_learning_curve_experiments",
            autospec=True,
        ) as screening_mock:
            screening_mock.return_value = expected

            result = run_learning_curve_experiments(
                dataset,
                min_train=4,
                max_train=5,
                n_repeats=1,
                budget_mode="screening_fraction",
                screen_fraction=0.25,
                min_screen_size=1,
            )

        self.assertIs(result, expected)
        screening_mock.assert_called_once()
        standard_mock.assert_not_called()

    def test_run_learning_curve_experiments_from_frame_routes_screening_mode(self) -> None:
        df = pd.DataFrame(
            {
                "reaction": [f"rxn-{idx}" for idx in range(6)],
                "reference_ads_eng": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "ridge_mlip_ads_eng_median": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
            }
        )
        expected = LearningCurveResults.from_mapping(
            {
                "ridge_df": pd.DataFrame(
                    {
                        "n_train": [4],
                        "rmse_mean": [0.3],
                        "rmse_std": [0.04],
                    }
                )
            }
        )

        with patch(
            "oasis.experiment.orchestration.run_screening_learning_curve_experiments",
            autospec=True,
        ) as screening_mock:
            screening_mock.return_value = expected

            result = run_learning_curve_experiments_from_frame(
                df,
                min_train=4,
                max_train=4,
                n_repeats=1,
                budget_mode="screening_fraction",
                screen_fraction=0.25,
            )

        self.assertIs(result, expected)
        screening_mock.assert_called_once()
        dataset = screening_mock.call_args.args[0]
        self.assertIsInstance(dataset, SweepDataset)
        np.testing.assert_array_equal(dataset.sample_ids, np.array(df["reaction"]))

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
            model_families=[family],
        )

        self.assertIs(results.ridge_df, result_df)
        self.assertEqual(dataset.n_samples, 5)
        self.assertEqual(
            [split.sweep_size for split in family.last_payload.split_collection.splits],
            [2, 3, 4],
        )
        self.assertTrue(
            all(
                split.val_idx is None
                for split in family.last_payload.split_collection.splits
            )
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
                return LearningCurveResults.from_mapping({self.field_name: result_df})

        ridge_family = StubFamily("ridge_df")
        weighted_family = StubFamily("weighted_linear_df")

        results = run_learning_curve_experiments(
            SweepDataset(mlip_features=X, targets=y),
            min_train=2,
            max_train=4,
            n_repeats=1,
            seed=9,
            model_families=[ridge_family, weighted_family],
        )

        self.assertIs(results.ridge_df, result_df)
        self.assertIs(results.weighted_linear_df, result_df)
        self.assertIsNone(results.kernel_ridge_df)
        self.assertEqual(ridge_family.calls, 1)
        self.assertEqual(weighted_family.calls, 1)
        self.assertIsInstance(ridge_family.last_payload, SweepRunPayload)
        self.assertEqual(
            [
                split.sweep_size
                for split in ridge_family.last_payload.split_collection.splits
            ],
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
            model_families=[family],
        )

        self.assertIs(results.ridge_df, result_df)
        self.assertEqual(
            [split.sweep_size for split in family.last_payload.split_collection.splits],
            [4, 5],
        )
        self.assertTrue(
            all(
                split.val_idx is not None
                for split in family.last_payload.split_collection.splits
            )
        )
        self.assertEqual(
            [
                len(split.train_idx)
                for split in family.last_payload.split_collection.splits
            ],
            [3, 4],
        )
        self.assertEqual(
            [
                len(split.val_idx)
                for split in family.last_payload.split_collection.splits
            ],
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
        self.assertTrue(
            all(split.val_idx is not None for split in split_collection.splits)
        )

    def test_build_sweep_split_collection_heavy_family_minimum_dominates(self) -> None:
        split_collection = build_sweep_split_collection(
            n_samples=10,
            min_train=2,
            max_train=8,
            n_repeats=1,
            seed=3,
            requirements=SweepFamilyRequirements(
                min_train_size=6,
                requires_inner_validation=True,
            ),
            validation_fraction=0.2,
            min_val_size=2,
            min_test_size=2,
        )

        self.assertEqual(
            [split.sweep_size for split in split_collection.splits],
            [6, 7, 8],
        )
        self.assertEqual(
            [len(split.val_idx) for split in split_collection.splits],
            [2, 2, 2],
        )
        self.assertEqual(
            [len(split.test_idx) for split in split_collection.splits],
            [4, 3, 2],
        )

    def test_build_sweep_split_collection_family_minimum_and_new_guards_compose(
        self,
    ) -> None:
        split_collection = build_sweep_split_collection(
            n_samples=10,
            min_train=2,
            max_train=8,
            n_repeats=1,
            seed=3,
            requirements=SweepFamilyRequirements(
                min_train_size=4,
                requires_inner_validation=True,
            ),
            validation_fraction=0.2,
            min_val_size=2,
            min_tuning_val_size=3,
            min_inner_train_size=2,
            min_test_size=2,
        )

        self.assertEqual(
            [split.sweep_size for split in split_collection.splits],
            [5, 6, 7, 8],
        )
        self.assertEqual(
            [len(split.val_idx) for split in split_collection.splits],
            [3, 3, 3, 3],
        )
        self.assertEqual(
            [len(split.train_idx) for split in split_collection.splits],
            [2, 3, 4, 5],
        )
        self.assertEqual(
            [len(split.test_idx) for split in split_collection.splits],
            [5, 4, 3, 2],
        )

    def test_build_sweep_split_collection_validation_and_test_guards_dominate(
        self,
    ) -> None:
        split_collection = build_sweep_split_collection(
            n_samples=7,
            min_train=1,
            max_train=6,
            n_repeats=1,
            seed=3,
            requirements=SweepFamilyRequirements(
                min_train_size=0,
                requires_inner_validation=True,
            ),
            validation_fraction=0.2,
            min_val_size=3,
            min_test_size=3,
        )

        self.assertEqual(
            [split.sweep_size for split in split_collection.splits],
            [4],
        )
        split = split_collection.splits[0]
        self.assertEqual(len(split.val_idx), 3)
        self.assertEqual(len(split.train_idx), 1)
        self.assertEqual(len(split.test_idx), 3)

    def test_build_sweep_split_collection_skips_undersized_points_deterministically(
        self,
    ) -> None:
        split_collection_a = build_sweep_split_collection(
            n_samples=8,
            min_train=1,
            max_train=7,
            n_repeats=2,
            seed=11,
            requirements=SweepFamilyRequirements(
                min_train_size=0,
                requires_inner_validation=True,
            ),
            validation_fraction=0.2,
            min_val_size=2,
            min_test_size=3,
        )
        split_collection_b = build_sweep_split_collection(
            n_samples=8,
            min_train=1,
            max_train=7,
            n_repeats=2,
            seed=11,
            requirements=SweepFamilyRequirements(
                min_train_size=0,
                requires_inner_validation=True,
            ),
            validation_fraction=0.2,
            min_val_size=2,
            min_test_size=3,
        )

        self.assertEqual(
            [split.sweep_size for split in split_collection_a.splits],
            [3, 3, 4, 4, 5, 5],
        )
        self.assertEqual(
            [split.sweep_size for split in split_collection_b.splits],
            [3, 3, 4, 4, 5, 5],
        )
        for split_a, split_b in zip(
            split_collection_a.splits,
            split_collection_b.splits,
            strict=True,
        ):
            np.testing.assert_array_equal(split_a.train_idx, split_b.train_idx)
            np.testing.assert_array_equal(split_a.val_idx, split_b.val_idx)
            np.testing.assert_array_equal(split_a.test_idx, split_b.test_idx)

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
            model_families=[baseline_family, validation_family],
        )

        self.assertIs(results.ridge_df, baseline_result)
        self.assertIs(results.weighted_linear_df, validation_result)
        self.assertEqual(
            [
                split.sweep_size
                for split in baseline_family.last_payload.split_collection.splits
            ],
            [2, 3, 4, 5],
        )
        self.assertTrue(
            all(
                split.val_idx is None
                for split in baseline_family.last_payload.split_collection.splits
            )
        )
        self.assertEqual(
            [
                split.sweep_size
                for split in validation_family.last_payload.split_collection.splits
            ],
            [4, 5],
        )
        self.assertTrue(
            all(
                split.val_idx is not None
                for split in validation_family.last_payload.split_collection.splits
            )
        )

    def test_run_learning_curve_experiments_honors_split_policy_knobs(self) -> None:
        X = np.arange(21, dtype=float).reshape(7, 3)
        y = np.arange(7, dtype=float)
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
            max_train=6,
            n_repeats=1,
            seed=3,
            validation_fraction=0.2,
            min_val_size=2,
            min_test_size=2,
            model_families=[family],
        )

        self.assertIs(results.ridge_df, result_df)
        self.assertEqual(
            [split.sweep_size for split in family.last_payload.split_collection.splits],
            [4, 5],
        )
        self.assertEqual(
            [
                len(split.val_idx)
                for split in family.last_payload.split_collection.splits
            ],
            [2, 2],
        )
        self.assertEqual(
            [
                len(split.test_idx)
                for split in family.last_payload.split_collection.splits
            ],
            [3, 2],
        )

    def test_run_learning_curve_experiments_mixed_families_preserve_outer_test_isolation_under_guards(
        self,
    ) -> None:
        X = np.arange(24, dtype=float).reshape(8, 3)
        y = np.arange(8, dtype=float)
        baseline_result = pd.DataFrame(
            {"n_train": [2, 3, 4, 5], "rmse_mean": [0.5] * 4, "rmse_std": [0.05] * 4}
        )
        guarded_result = pd.DataFrame(
            {"n_train": [3, 4, 5], "rmse_mean": [0.4] * 3, "rmse_std": [0.04] * 3}
        )
        heavy_result = pd.DataFrame(
            {"n_train": [5], "rmse_mean": [0.3], "rmse_std": [0.03]}
        )

        class BaselineFamily:
            def requirements(self) -> SweepFamilyRequirements:
                return SweepFamilyRequirements()

            def run(self, payload):
                self.last_payload = payload
                return LearningCurveResults.from_mapping({"ridge_df": baseline_result})

        class GuardedValidationFamily:
            def requirements(self) -> SweepFamilyRequirements:
                return SweepFamilyRequirements(
                    min_train_size=0,
                    requires_inner_validation=True,
                )

            def run(self, payload):
                self.last_payload = payload
                return LearningCurveResults.from_mapping(
                    {"weighted_linear_df": guarded_result}
                )

        class HeavyValidationFamily:
            def requirements(self) -> SweepFamilyRequirements:
                return SweepFamilyRequirements(
                    min_train_size=5,
                    requires_inner_validation=True,
                )

            def run(self, payload):
                self.last_payload = payload
                return LearningCurveResults.from_mapping(
                    {"weighted_simplex_df": heavy_result}
                )

        baseline_family = BaselineFamily()
        guarded_family = GuardedValidationFamily()
        heavy_family = HeavyValidationFamily()

        results = run_learning_curve_experiments(
            SweepDataset(mlip_features=X, targets=y),
            min_train=2,
            max_train=6,
            n_repeats=1,
            seed=7,
            validation_fraction=0.2,
            min_val_size=2,
            min_test_size=3,
            model_families=[baseline_family, guarded_family, heavy_family],
        )

        self.assertIs(results.ridge_df, baseline_result)
        self.assertIs(results.weighted_linear_df, guarded_result)
        self.assertIs(results.weighted_simplex_df, heavy_result)
        self.assertEqual(
            [
                split.sweep_size
                for split in baseline_family.last_payload.split_collection.splits
            ],
            [2, 3, 4, 5],
        )
        self.assertEqual(
            [
                split.sweep_size
                for split in guarded_family.last_payload.split_collection.splits
            ],
            [3, 4, 5],
        )
        self.assertEqual(
            [
                split.sweep_size
                for split in heavy_family.last_payload.split_collection.splits
            ],
            [5],
        )
        for payload in (
            baseline_family.last_payload,
            guarded_family.last_payload,
            heavy_family.last_payload,
        ):
            for split in payload.split_collection.splits:
                self.assertGreaterEqual(len(split.test_idx), 3)
                self.assertEqual(
                    len(np.intersect1d(split.train_idx, split.test_idx)), 0
                )
                if split.val_idx is not None:
                    self.assertEqual(
                        len(np.intersect1d(split.val_idx, split.test_idx)), 0
                    )
                    self.assertEqual(
                        len(np.intersect1d(split.train_idx, split.val_idx)), 0
                    )

    def test_run_learning_curve_experiments_family_requirements_compose_with_new_guards(
        self,
    ) -> None:
        X = np.arange(30, dtype=float).reshape(10, 3)
        y = np.arange(10, dtype=float)
        result_df = pd.DataFrame(
            {
                "n_train": [5, 6, 7, 8],
                "rmse_mean": [0.4, 0.35, 0.3, 0.25],
                "rmse_std": [0.05, 0.04, 0.03, 0.02],
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
            max_train=8,
            n_repeats=1,
            seed=3,
            validation_fraction=0.2,
            min_val_size=2,
            min_tuning_val_size=3,
            min_inner_train_size=2,
            min_test_size=2,
            model_families=[family],
        )

        self.assertIs(results.ridge_df, result_df)
        self.assertEqual(
            [split.sweep_size for split in family.last_payload.split_collection.splits],
            [5, 6, 7, 8],
        )
        self.assertEqual(
            [
                len(split.val_idx)
                for split in family.last_payload.split_collection.splits
            ],
            [3, 3, 3, 3],
        )
        self.assertEqual(
            [
                len(split.train_idx)
                for split in family.last_payload.split_collection.splits
            ],
            [2, 3, 4, 5],
        )
        self.assertTrue(
            all(
                len(split.test_idx) >= 2
                for split in family.last_payload.split_collection.splits
            )
        )
        self.assertEqual(
            family.last_payload.split_collection.planning_requirements,
            SweepFamilyRequirements(
                min_train_size=4,
                requires_inner_validation=True,
            ),
        )

    def test_run_learning_curve_experiments_skips_learned_family_when_combined_guards_make_range_infeasible(
        self,
    ) -> None:
        X = np.arange(24, dtype=float).reshape(8, 3)
        y = np.arange(8, dtype=float)

        class ValidationAwareStubFamily:
            def requirements(self) -> SweepFamilyRequirements:
                return SweepFamilyRequirements(
                    min_train_size=4,
                    requires_inner_validation=True,
                )

            def run(self, payload):
                self.last_payload = payload
                return LearningCurveResults.empty()

        family = ValidationAwareStubFamily()

        results = run_learning_curve_experiments(
            SweepDataset(mlip_features=X, targets=y),
            min_train=2,
            max_train=6,
            n_repeats=1,
            seed=3,
            validation_fraction=0.2,
            min_val_size=2,
            min_tuning_val_size=4,
            min_inner_train_size=3,
            min_test_size=2,
            model_families=[family],
        )

        self.assertIsNone(results.ridge_df)
        self.assertEqual(family.last_payload.split_collection.splits, ())
        self.assertEqual(
            family.last_payload.split_collection.planning_requirements,
            SweepFamilyRequirements(
                min_train_size=4,
                requires_inner_validation=True,
            ),
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
                runner=baseline_runner,
            )
        )
        validation_family = ConfiguredSweepModelFamily(
            spec=SweepFamilySpec(
                result_field="weighted_linear_df",
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
            model_families=[baseline_family, validation_family],
        )

        self.assertEqual(
            results.ridge_df.columns.tolist(),
            ["n_train", "rmse_mean", "rmse_std"],
        )
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

    def test_run_learning_curve_experiments_supports_sklearn_and_learned_runners(
        self,
    ) -> None:
        class SklearnLikeModel:
            fit_calls = 0

            def fit(self, X, y) -> None:
                type(self).fit_calls += 1
                self.coef_, *_ = np.linalg.lstsq(X, y, rcond=None)

            def predict(self, X):
                return X @ self.coef_

        seen_splits: list[TrainTestSweepRunnerInput] = []

        class SplitAwareModel:
            def fit(self, split: TrainTestSweepRunnerInput) -> None:
                seen_splits.append(split)
                X_train = split.dataset.mlip_features[split.train_idx]
                y_train = split.dataset.targets[split.train_idx]
                self.coef_, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)

            def predict(self, X):
                return X @ self.coef_

        X = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 1.0],
                [1.0, 2.0],
                [2.0, 2.0],
            ]
        )
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        sklearn_family = ConfiguredSweepModelFamily(
            spec=SweepFamilySpec(
                result_field="ridge_df",
                runner=SupervisedModelSweepRunner(SklearnLikeModel),
            )
        )
        learned_family = ConfiguredSweepModelFamily(
            spec=SweepFamilySpec(
                result_field="weighted_linear_df",
                runner=LearnedModelSweepRunner(SplitAwareModel),
            )
        )

        results = run_learning_curve_experiments(
            SweepDataset(mlip_features=X, targets=y),
            min_train=3,
            max_train=3,
            n_repeats=1,
            seed=5,
            model_families=[sklearn_family, learned_family],
        )

        self.assertEqual(results.ridge_df["n_train"].tolist(), [3])
        self.assertEqual(results.weighted_linear_df["n_train"].tolist(), [3])
        self.assertEqual(SklearnLikeModel.fit_calls, 1)
        self.assertEqual(len(seen_splits), 1)
        self.assertIsInstance(seen_splits[0], TrainTestSweepRunnerInput)

    def test_configured_family_rejects_validation_splits_for_train_test_runner(
        self,
    ) -> None:
        family = ConfiguredSweepModelFamily(
            spec=SweepFamilySpec(
                result_field="ridge_df",
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

        class EmptyAwareValidationRunner:
            def __init__(self) -> None:
                self.payloads: list[SweepRunnerPayload] = []

            def run_with_validation(
                self,
                payload: SweepRunnerPayload,
            ) -> pd.DataFrame:
                self.payloads.append(payload)
                if payload.splits != ():
                    raise AssertionError(
                        "expected skipped validation family to receive no splits"
                    )
                return sweep_results_frame({})

        X = np.arange(18, dtype=float).reshape(6, 3)
        y = np.arange(6, dtype=float)
        baseline_runner = BaselineRunner()
        baseline_family = ConfiguredSweepModelFamily(
            spec=SweepFamilySpec(
                result_field="ridge_df",
                runner=baseline_runner,
            )
        )
        validation_runner = EmptyAwareValidationRunner()
        skipped_family = ConfiguredSweepModelFamily(
            spec=SweepFamilySpec(
                result_field="weighted_linear_df",
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

    def test_run_learning_curve_experiments_returns_empty_result_for_only_skipped_validation_family(
        self,
    ) -> None:
        class EmptyAwareValidationRunner:
            def __init__(self) -> None:
                self.payloads: list[SweepRunnerPayload] = []

            def run_with_validation(
                self,
                payload: SweepRunnerPayload,
            ) -> pd.DataFrame:
                self.payloads.append(payload)
                if payload.splits != ():
                    raise AssertionError(
                        "expected no valid validation-aware sweep sizes"
                    )
                return sweep_results_frame({})

        validation_runner = EmptyAwareValidationRunner()
        skipped_family = ConfiguredSweepModelFamily(
            spec=SweepFamilySpec(
                result_field="weighted_linear_df",
                runner=validation_runner,
                capabilities=SweepModelCapabilities(
                    min_train_size=5,
                    requires_validation=True,
                ),
            )
        )

        results = run_learning_curve_experiments(
            SweepDataset(
                mlip_features=np.arange(18, dtype=float).reshape(6, 3),
                targets=np.arange(6, dtype=float),
            ),
            min_train=2,
            max_train=4,
            n_repeats=1,
            seed=3,
            validation_fraction=0.2,
            min_val_size=2,
            min_test_size=3,
            model_families=[skipped_family],
        )

        self.assertIsNotNone(results.weighted_linear_df)
        self.assertTrue(results.weighted_linear_df.empty)
        self.assertEqual(
            results.weighted_linear_df.columns.tolist(),
            ["n_train", "rmse_mean", "rmse_std"],
        )
        self.assertEqual(len(validation_runner.payloads), 1)
        self.assertEqual(validation_runner.payloads[0].splits, ())

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

        family = ConfiguredSweepModelFamily(
            spec=SweepFamilySpec(
                result_field="ridge_df",
                runner=RecordingRunner(),
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

    def test_run_learning_curve_experiments_from_config_uses_default_models_with_plot_only_cfg(
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

        cfg = SimpleNamespace(
            seed=23,
            plot=SimpleNamespace(
                output_dir="unused",
            ),
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(
                    min_train=2,
                    max_train=4,
                    n_repeats=2,
                    models=None,
                )
            ),
        )

        results = run_learning_curve_experiments_from_config(df, cfg=cfg)

        self.assertIsNotNone(results.ridge_df)
        self.assertIsNotNone(results.lasso_df)
        self.assertEqual(results.ridge_df["n_train"].tolist(), [3, 4])

    def test_run_learning_curve_experiments_from_config_selects_models_without_plot(
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
            seed=29,
            plot=None,
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(
                    min_train=2,
                    max_train=4,
                    n_repeats=2,
                    models=SimpleNamespace(
                        use_ridge=False,
                        use_kernel_ridge=False,
                        use_lasso=False,
                        use_elastic_net=False,
                        use_residual=True,
                        use_weighted_linear=True,
                        use_weighted_simplex=False,
                    ),
                )
            ),
        )

        results = run_learning_curve_experiments_from_config(df, cfg=cfg)

        self.assertIsNone(results.ridge_df)
        self.assertIsNone(results.lasso_df)
        self.assertIsNone(results.elastic_df)
        self.assertIsNotNone(results.resid_df)
        self.assertIsNotNone(results.resid_uq_df)
        self.assertIsNotNone(results.weighted_linear_df)
        self.assertIsNone(results.weighted_simplex_df)
        self.assertEqual(
            results.resid_uq_df["n_train"].tolist(),
            results.resid_df["n_train"].tolist(),
        )

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
            enabled_model_names=[
                "weighted_linear",
                "weighted_simplex",
            ],
        )

        self.assertIsNone(results.ridge_df)
        self.assertIsNone(results.resid_df)
        self.assertIsNotNone(results.weighted_linear_df)
        self.assertIsNotNone(results.weighted_simplex_df)
        self.assertIsNotNone(results.weighted_simplex_uq_df)
        self.assertEqual(results.weighted_linear_df["n_train"].tolist(), [2, 3, 4])
        self.assertEqual(results.weighted_simplex_df["n_train"].tolist(), [2, 3, 4])
        self.assertEqual(
            results.weighted_simplex_uq_df["n_train"].tolist(),
            [2, 3, 4],
        )

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
            plot=SimpleNamespace(output_dir="unused"),
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(
                    min_train=2,
                    max_train=4,
                    n_repeats=2,
                    models=SimpleNamespace(
                        use_ridge=True,
                        use_kernel_ridge=False,
                        use_lasso=False,
                        use_elastic_net=False,
                        use_residual=False,
                        use_weighted_linear=True,
                        use_weighted_simplex=False,
                    ),
                )
            ),
        )

        results = run_learning_curve_experiments_from_config(df, cfg=cfg)

        self.assertIsNotNone(results.ridge_df)
        self.assertIsNotNone(results.ridge_uq_df)
        self.assertIsNotNone(results.weighted_linear_df)
        self.assertIsNone(results.weighted_simplex_df)
        self.assertIsNone(results.lasso_df)
        self.assertIsNone(results.resid_df)
        self.assertEqual(
            results.ridge_uq_df["n_train"].tolist(),
            results.ridge_df["n_train"].tolist(),
        )
        self.assertTrue(
            (
                results.ridge_uq_df["uncertainty_note"]
                == "post-hoc scalar calibrated spread"
            ).all()
        )

    def test_run_learning_curve_experiments_from_config_saves_bundle_artifact(
        self,
    ) -> None:
        df = pd.DataFrame(
            {
                "reference_ads_eng": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "ridge_mlip_ads_eng_median": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
            }
        )
        result_df = pd.DataFrame(
            {
                "n_train": [2, 3],
                "rmse_mean": [0.4, 0.3],
                "rmse_std": [0.05, 0.04],
            }
        )
        selection_df = pd.DataFrame(
            {
                "n_train": [2, 3],
                "alpha": [0.1, 1.0],
            }
        )

        class RecordingFamily:
            def requirements(self) -> SweepFamilyRequirements:
                return SweepFamilyRequirements()

            def run(self, payload):
                self.last_payload = payload
                del payload
                return LearningCurveResults.from_mapping(
                    {
                        "ridge_df": result_df,
                        "ridge_selection_df": selection_df,
                    }
                )

        family = RecordingFamily()

        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = Path(tmp_dir) / "learning_curve_results.json"
            cfg = SimpleNamespace(
                seed=23,
                dataset_profile=SimpleNamespace(tag="mamun_oh"),
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        min_train=2,
                        max_train=3,
                        step=1,
                        n_repeats=1,
                        validation_fraction=0.2,
                        min_val_size=1,
                        min_tuning_val_size=1,
                        min_inner_train_size=1,
                        min_test_size=1,
                        results_bundle_path=bundle_path,
                        models=SimpleNamespace(
                            use_ridge=True,
                            use_kernel_ridge=False,
                            use_lasso=False,
                            use_elastic_net=False,
                            use_residual=False,
                            use_weighted_linear=False,
                            use_weighted_simplex=False,
                            use_graph_mean=False,
                            use_latent=False,
                            moe=SimpleNamespace(enabled=False),
                            probe_gnn=SimpleNamespace(enabled=False),
                            gnn_direct=SimpleNamespace(enabled=False),
                        ),
                    )
                ),
            )

            results = run_learning_curve_experiments_from_config(
                df,
                cfg=cfg,
                model_families=[family],
            )

            self.assertTrue(bundle_path.is_file())
            artifact = load_learning_curve_results_artifact(bundle_path)

        pd.testing.assert_frame_equal(results.ridge_df, result_df)
        pd.testing.assert_frame_equal(artifact.results.ridge_df, result_df)
        pd.testing.assert_frame_equal(
            artifact.results.ridge_selection_df,
            selection_df,
        )
        self.assertEqual(artifact.metadata.dataset_tag, "mamun_oh")
        self.assertEqual(artifact.metadata.dataset_size, len(df))

    def test_load_or_run_learning_curve_results_from_config_reuses_bundle_artifact(
        self,
    ) -> None:
        df = pd.DataFrame(
            {
                "reference_ads_eng": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "ridge_mlip_ads_eng_median": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
            }
        )
        ridge_df = pd.DataFrame(
            {
                "n_train": [2, 3],
                "rmse_mean": [0.4, 0.3],
                "rmse_std": [0.05, 0.04],
            }
        )
        ridge_selection_df = pd.DataFrame(
            {
                "n_train": [2, 3],
                "alpha": [0.1, 0.1],
            }
        )
        ridge_uq_df = pd.DataFrame(
            {
                "n_train": [2, 3],
                "miscalibration_area": [0.2, 0.1],
                "sharpness": [0.3, 0.2],
                "dispersion": [0.4, 0.3],
                "uncertainty_kind": ["calibrated", "calibrated"],
            }
        )
        weighted_linear_df = pd.DataFrame(
            {
                "n_train": [2, 3],
                "rmse_mean": [0.35, 0.25],
                "rmse_std": [0.04, 0.03],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = Path(tmp_dir) / "learning_curve_results.json"
            save_learning_curve_results_artifact(
                LearningCurveResults(
                    ridge_df=ridge_df,
                    ridge_selection_df=ridge_selection_df,
                    ridge_uq_df=ridge_uq_df,
                    weighted_linear_df=weighted_linear_df,
                ),
                LearningCurveSweepMetadata(
                    seed=23,
                    min_train=2,
                    max_train=3,
                    step=1,
                    n_repeats=1,
                    enabled_models=("ridge", "weighted_linear"),
                    mlip_feature_names=("ridge",),
                ),
                bundle_path,
            )
            cfg = SimpleNamespace(
                seed=23,
                plot=SimpleNamespace(filters=None),
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        min_train=2,
                        max_train=3,
                        step=1,
                        n_repeats=1,
                        validation_fraction=0.2,
                        min_val_size=1,
                        min_tuning_val_size=1,
                        min_inner_train_size=1,
                        min_test_size=1,
                        results_bundle_path=bundle_path,
                        reuse_results=True,
                        models=SimpleNamespace(
                            use_ridge=True,
                            use_kernel_ridge=False,
                            use_lasso=False,
                            use_elastic_net=False,
                            use_residual=False,
                            use_weighted_linear=False,
                            use_weighted_simplex=False,
                            use_graph_mean=False,
                            use_latent=False,
                            moe=SimpleNamespace(enabled=False),
                            probe_gnn=SimpleNamespace(enabled=False),
                            gnn_direct=SimpleNamespace(enabled=False),
                        ),
                    )
                ),
            )

            with patch(
                "oasis.experiment.orchestration.run_learning_curve_experiments_from_config"
            ) as mock_run:
                results = load_or_run_learning_curve_results_from_config(df, cfg=cfg)

        pd.testing.assert_frame_equal(results.ridge_df, ridge_df)
        self.assertFalse(mock_run.called)

    def test_load_or_run_learning_curve_results_from_config_does_not_reuse_bundle_when_required_uq_frame_is_missing(
        self,
    ) -> None:
        df = pd.DataFrame(
            {
                "reference_ads_eng": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "ridge_mlip_ads_eng_median": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
            }
        )
        kernel_ridge_df = pd.DataFrame(
            {
                "n_train": [3],
                "rmse_mean": [0.3],
                "rmse_std": [0.04],
            }
        )
        fresh_results = LearningCurveResults(
            kernel_ridge_df=kernel_ridge_df,
            kernel_ridge_selection_df=pd.DataFrame(
                {"n_train": [3], "alpha": [1.0], "gamma": [0.1], "kernel": ["rbf"]}
            ),
            kernel_ridge_uq_df=pd.DataFrame(
                {
                    "n_train": [3],
                    "miscalibration_area": [0.12],
                    "sharpness": [0.22],
                    "dispersion": [0.32],
                    "uncertainty_kind": ["calibrated"],
                }
            ),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = Path(tmp_dir) / "learning_curve_results.json"
            save_learning_curve_results_artifact(
                LearningCurveResults(kernel_ridge_df=kernel_ridge_df),
                LearningCurveSweepMetadata(
                    seed=23,
                    min_train=3,
                    max_train=3,
                    step=1,
                    n_repeats=1,
                    enabled_models=("kernel_ridge",),
                    mlip_feature_names=("ridge",),
                ),
                bundle_path,
            )
            cfg = SimpleNamespace(
                seed=23,
                plot=SimpleNamespace(filters=None),
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        min_train=2,
                        max_train=4,
                        step=1,
                        n_repeats=1,
                        validation_fraction=0.2,
                        min_val_size=1,
                        min_tuning_val_size=1,
                        min_inner_train_size=1,
                        min_test_size=1,
                        results_bundle_path=bundle_path,
                        reuse_results=True,
                        models=SimpleNamespace(
                            use_ridge=False,
                            use_kernel_ridge=True,
                            use_lasso=False,
                            use_elastic_net=False,
                            use_residual=False,
                            use_weighted_linear=False,
                            use_weighted_simplex=False,
                            use_graph_mean=False,
                            use_latent=False,
                            moe=SimpleNamespace(enabled=False),
                            probe_gnn=SimpleNamespace(enabled=False),
                            gnn_direct=SimpleNamespace(enabled=False),
                        ),
                    )
                ),
            )

            with patch(
                "oasis.experiment.orchestration.run_learning_curve_experiments_from_config",
                return_value=fresh_results,
            ) as mock_run:
                results = load_or_run_learning_curve_results_from_config(df, cfg=cfg)

        self.assertTrue(mock_run.called)
        pd.testing.assert_frame_equal(results.kernel_ridge_df, fresh_results.kernel_ridge_df)
        pd.testing.assert_frame_equal(
            results.kernel_ridge_uq_df,
            fresh_results.kernel_ridge_uq_df,
        )

    def test_load_or_run_learning_curve_results_from_config_reuses_bundle_when_requested_bounds_are_infeasible(
        self,
    ) -> None:
        df = pd.DataFrame(
            {
                "reference_ads_eng": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "ridge_mlip_ads_eng_median": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
            }
        )
        ridge_df = pd.DataFrame(
            {
                "n_train": [2, 3],
                "rmse_mean": [0.4, 0.3],
                "rmse_std": [0.05, 0.04],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = Path(tmp_dir) / "learning_curve_results.json"
            save_learning_curve_results_artifact(
                LearningCurveResults(ridge_df=ridge_df),
                LearningCurveSweepMetadata(
                    seed=23,
                    min_train=1,
                    max_train=20,
                    step=1,
                    n_repeats=1,
                    enabled_models=("ridge",),
                    adsorbate_filter="OH",
                    mlip_feature_names=("ridge",),
                ),
                bundle_path,
            )
            cfg = SimpleNamespace(
                seed=23,
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        min_train=20,
                        max_train=40,
                        step=5,
                        n_repeats=1,
                        validation_fraction=0.2,
                        min_val_size=1,
                        min_tuning_val_size=1,
                        min_inner_train_size=1,
                        min_test_size=1,
                        results_bundle_path=bundle_path,
                        reuse_results=True,
                        models=SimpleNamespace(
                            use_ridge=True,
                            use_kernel_ridge=False,
                            use_lasso=False,
                            use_elastic_net=False,
                            use_residual=False,
                            use_weighted_linear=False,
                            use_weighted_simplex=False,
                            use_graph_mean=False,
                            use_latent=False,
                            moe=SimpleNamespace(enabled=False),
                            probe_gnn=SimpleNamespace(enabled=False),
                            gnn_direct=SimpleNamespace(enabled=False),
                        ),
                    )
                ),
            )

            with patch(
                "oasis.experiment.orchestration.run_learning_curve_experiments_from_config",
            ) as mock_run:
                results = load_or_run_learning_curve_results_from_config(df, cfg=cfg)

        pd.testing.assert_frame_equal(results.ridge_df, ridge_df)
        self.assertFalse(mock_run.called)

    def test_load_or_run_learning_curve_results_from_config_reuses_bundle_across_repeat_counts(
        self,
    ) -> None:
        df = pd.DataFrame(
            {
                "reference_ads_eng": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "ridge_mlip_ads_eng_median": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
            }
        )
        ridge_df = pd.DataFrame(
            {
                "n_train": [1, 2, 3, 4, 5],
                "rmse_mean": [0.5, 0.4, 0.3, 0.25, 0.2],
                "rmse_std": [0.06, 0.05, 0.04, 0.035, 0.03],
            }
        )
        ridge_selection_df = pd.DataFrame(
            {
                "n_train": [1, 2, 3, 4, 5],
                "alpha": [0.1, 0.1, 0.1, 0.1, 0.1],
            }
        )
        ridge_uq_df = pd.DataFrame(
            {
                "n_train": [1, 2, 3, 4, 5],
                "miscalibration_area": [0.5, 0.4, 0.3, 0.25, 0.2],
                "sharpness": [0.6, 0.5, 0.4, 0.35, 0.3],
                "dispersion": [0.7, 0.6, 0.5, 0.45, 0.4],
                "uncertainty_kind": ["calibrated"] * 5,
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = Path(tmp_dir) / "learning_curve_results.json"
            save_learning_curve_results_artifact(
                LearningCurveResults(
                    ridge_df=ridge_df,
                    ridge_selection_df=ridge_selection_df,
                    ridge_uq_df=ridge_uq_df,
                ),
                LearningCurveSweepMetadata(
                    seed=23,
                    min_train=1,
                    max_train=20,
                    step=1,
                    n_repeats=30,
                    enabled_models=("ridge",),
                    adsorbate_filter="OH",
                    mlip_feature_names=("ridge",),
                ),
                bundle_path,
            )
            cfg = SimpleNamespace(
                seed=23,
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        min_train=1,
                        max_train=20,
                        step=1,
                        n_repeats=10,
                        validation_fraction=0.2,
                        min_val_size=1,
                        min_tuning_val_size=1,
                        min_inner_train_size=1,
                        min_test_size=1,
                        results_bundle_path=bundle_path,
                        reuse_results=True,
                        models=SimpleNamespace(
                            use_ridge=True,
                            use_kernel_ridge=False,
                            use_lasso=False,
                            use_elastic_net=False,
                            use_residual=False,
                            use_weighted_linear=False,
                            use_weighted_simplex=False,
                            use_graph_mean=False,
                            use_latent=False,
                            moe=SimpleNamespace(enabled=False),
                            probe_gnn=SimpleNamespace(enabled=False),
                            gnn_direct=SimpleNamespace(enabled=False),
                        ),
                    )
                ),
            )

            with patch(
                "oasis.experiment.orchestration.run_learning_curve_experiments_from_config"
            ) as mock_run:
                results = load_or_run_learning_curve_results_from_config(df, cfg=cfg)

        pd.testing.assert_frame_equal(results.ridge_df, ridge_df)
        self.assertFalse(mock_run.called)

    def test_load_or_run_learning_curve_results_from_config_does_not_reuse_bundle_when_mlip_set_changes(
        self,
    ) -> None:
        df = pd.DataFrame(
            {
                "reference_ads_eng": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "ridge_mlip_ads_eng_median": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
                "uma_mlip_ads_eng_median": [0.9, 1.9, 2.9, 3.9, 4.9, 5.9],
            }
        )
        ridge_df = pd.DataFrame(
            {
                "n_train": [1, 2, 3, 4, 5],
                "rmse_mean": [0.5, 0.4, 0.3, 0.25, 0.2],
                "rmse_std": [0.06, 0.05, 0.04, 0.035, 0.03],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = Path(tmp_dir) / "learning_curve_results.json"
            save_learning_curve_results_artifact(
                LearningCurveResults(ridge_df=ridge_df),
                LearningCurveSweepMetadata(
                    seed=23,
                    min_train=1,
                    max_train=20,
                    step=1,
                    n_repeats=1,
                    enabled_models=("ridge",),
                    mlip_feature_names=("ridge", "orb"),
                ),
                bundle_path,
            )
            cfg = SimpleNamespace(
                seed=23,
                plot=SimpleNamespace(filters=None),
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        min_train=1,
                        max_train=20,
                        step=1,
                        n_repeats=1,
                        validation_fraction=0.2,
                        min_val_size=1,
                        min_tuning_val_size=1,
                        min_inner_train_size=1,
                        min_test_size=1,
                        results_bundle_path=bundle_path,
                        reuse_results=True,
                        models=SimpleNamespace(
                            use_ridge=True,
                            use_kernel_ridge=False,
                            use_lasso=False,
                            use_elastic_net=False,
                            use_residual=False,
                            use_weighted_linear=False,
                            use_weighted_simplex=False,
                            use_graph_mean=False,
                            use_latent=False,
                            moe=SimpleNamespace(enabled=False),
                            probe_gnn=SimpleNamespace(enabled=False),
                            gnn_direct=SimpleNamespace(enabled=False),
                        ),
                    )
                ),
            )

            with patch(
                "oasis.experiment.orchestration.run_learning_curve_experiments_from_config"
            ) as mock_run:
                load_or_run_learning_curve_results_from_config(df, cfg=cfg)

        self.assertTrue(mock_run.called)

    def test_load_or_run_learning_curve_results_from_config_reuses_sparse_bundle_across_methods(
        self,
    ) -> None:
        df = pd.DataFrame(
            {
                "reference_ads_eng": list(range(1, 9)),
                "ridge_mlip_ads_eng_median": [value + 0.1 for value in range(1, 9)],
                "linear_mlip_ads_eng_median": [value - 0.1 for value in range(1, 9)],
            }
        )
        ridge_df = pd.DataFrame(
            {
                "n_train": [1, 2, 3, 4],
                "rmse_mean": [0.6, 0.5, 0.4, 0.3],
                "rmse_std": [0.06, 0.05, 0.04, 0.03],
            }
        )
        weighted_simplex_df = pd.DataFrame(
            {
                "n_train": [3, 4],
                "rmse_mean": [0.35, 0.25],
                "rmse_std": [0.045, 0.035],
            }
        )

        class RidgeFamily:
            def __init__(self) -> None:
                self.calls = 0
                self.method_name = "ridge"

            def requirements(self) -> SweepFamilyRequirements:
                return SweepFamilyRequirements()

            def run(self, payload):
                self.calls += 1
                del payload
                return LearningCurveResults.from_mapping({"ridge_df": ridge_df})

        class WeightedSimplexFamily:
            def __init__(self) -> None:
                self.calls = 0
                self.method_name = "weighted_simplex"

            def requirements(self) -> SweepFamilyRequirements:
                return SweepFamilyRequirements(min_train_size=3)

            def run(self, payload):
                self.calls += 1
                del payload
                return LearningCurveResults.from_mapping(
                    {"weighted_simplex_df": weighted_simplex_df}
                )

        ridge_family = RidgeFamily()
        weighted_simplex_family = WeightedSimplexFamily()

        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = Path(tmp_dir) / "learning_curve_results.json"
            plot_path = Path(tmp_dir) / "learning_curve.png"
            save_learning_curve_results_artifact(
                LearningCurveResults(
                    ridge_df=ridge_df,
                    weighted_simplex_df=weighted_simplex_df,
                ),
                LearningCurveSweepMetadata(
                    seed=23,
                    min_train=1,
                    max_train=4,
                    step=1,
                    n_repeats=1,
                    enabled_models=("ridge", "weighted_simplex"),
                    mlip_feature_names=("ridge", "linear"),
                ),
                bundle_path,
            )
            cfg = SimpleNamespace(
                seed=23,
                plot=SimpleNamespace(filters=None),
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        min_train=1,
                        max_train=4,
                        step=1,
                        n_repeats=1,
                        validation_fraction=0.2,
                        min_val_size=1,
                        min_tuning_val_size=1,
                        min_inner_train_size=1,
                        min_test_size=1,
                        results_bundle_path=bundle_path,
                        reuse_results=True,
                        force_refresh_methods=[],
                        force_refresh_train_sizes={},
                        models=SimpleNamespace(
                            use_ridge=True,
                            use_kernel_ridge=False,
                            use_lasso=False,
                            use_elastic_net=False,
                            use_residual=False,
                            use_weighted_linear=False,
                            use_weighted_simplex=True,
                            use_graph_mean=False,
                            use_latent=False,
                            moe=SimpleNamespace(enabled=False),
                            probe_gnn=SimpleNamespace(enabled=False),
                            gnn_direct=SimpleNamespace(enabled=False),
                        ),
                    )
                ),
            )

            results = load_or_run_learning_curve_results_from_config(
                df,
                cfg=cfg,
                model_families=[ridge_family, weighted_simplex_family],
            )
            learning_curve_plot(results, output_path=plot_path)

            self.assertTrue(plot_path.exists())

        self.assertEqual(ridge_family.calls, 0)
        self.assertEqual(weighted_simplex_family.calls, 0)
        pd.testing.assert_frame_equal(results.ridge_df, ridge_df)
        pd.testing.assert_frame_equal(results.weighted_simplex_df, weighted_simplex_df)

    def test_run_learning_curve_experiments_from_config_reuses_partial_bundle_cache(
        self,
    ) -> None:
        df = pd.DataFrame(
            {
                "reference_ads_eng": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "ridge_mlip_ads_eng_median": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
                "linear_mlip_ads_eng_median": [0.9, 1.9, 2.9, 3.9, 4.9, 5.9],
            }
        )
        ridge_df = pd.DataFrame(
            {
                "n_train": [2, 3],
                "rmse_mean": [0.4, 0.3],
                "rmse_std": [0.05, 0.04],
            }
        )
        weighted_linear_df = pd.DataFrame(
            {
                "n_train": [2, 3],
                "rmse_mean": [0.35, 0.25],
                "rmse_std": [0.04, 0.03],
            }
        )

        class StubFamily:
            def __init__(
                self,
                method_name: str,
                result_field: str,
                result_df: pd.DataFrame,
            ) -> None:
                self.method_name = method_name
                self.result_field = result_field
                self.result_df = result_df
                self.calls = 0

            def requirements(self) -> SweepFamilyRequirements:
                return SweepFamilyRequirements()

            def run(self, payload):
                self.calls += 1
                del payload
                return LearningCurveResults.from_mapping(
                    {self.result_field: self.result_df}
                )

        ridge_family = StubFamily("ridge", "ridge_df", ridge_df)
        weighted_linear_family = StubFamily(
            "weighted_linear",
            "weighted_linear_df",
            weighted_linear_df,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = Path(tmp_dir) / "learning_curve_results.json"
            save_learning_curve_results_artifact(
                LearningCurveResults.from_mapping({"ridge_df": ridge_df}),
                LearningCurveSweepMetadata(
                    seed=23,
                    min_train=2,
                    max_train=3,
                    step=1,
                    n_repeats=1,
                    enabled_models=("ridge", "weighted_linear"),
                    mlip_feature_names=("ridge", "linear"),
                ),
                bundle_path,
            )
            cfg = SimpleNamespace(
                seed=23,
                plot=SimpleNamespace(filters=None),
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        min_train=2,
                        max_train=3,
                        step=1,
                        n_repeats=1,
                        validation_fraction=0.2,
                        min_val_size=1,
                        min_tuning_val_size=1,
                        min_inner_train_size=1,
                        min_test_size=1,
                        results_bundle_path=bundle_path,
                        reuse_results=True,
                        models=SimpleNamespace(
                            use_ridge=True,
                            use_kernel_ridge=False,
                            use_lasso=False,
                            use_elastic_net=False,
                            use_residual=False,
                            use_weighted_linear=True,
                            use_weighted_simplex=False,
                            use_graph_mean=False,
                            use_latent=False,
                            moe=SimpleNamespace(enabled=False),
                            probe_gnn=SimpleNamespace(enabled=False),
                            gnn_direct=SimpleNamespace(enabled=False),
                        ),
                    )
                ),
            )

            results = run_learning_curve_experiments_from_config(
                df,
                cfg=cfg,
                model_families=[ridge_family, weighted_linear_family],
            )
            plot_path = Path(tmp_dir) / "learning_curve.png"
            learning_curve_plot(results, output_path=plot_path)
            self.assertTrue(plot_path.exists())

        self.assertEqual(ridge_family.calls, 0)
        self.assertEqual(weighted_linear_family.calls, 1)
        pd.testing.assert_frame_equal(results.ridge_df, ridge_df)
        pd.testing.assert_frame_equal(results.weighted_linear_df, weighted_linear_df)

    def test_run_learning_curve_experiments_from_config_gap_fills_missing_train_sizes_within_method(
        self,
    ) -> None:
        df = pd.DataFrame(
            {
                "reference_ads_eng": list(range(1, 45)),
                "ridge_mlip_ads_eng_median": [value + 0.1 for value in range(1, 45)],
            }
        )
        cached_ridge_df = pd.DataFrame(
            {
                "n_train": list(range(1, 21)),
                "rmse_mean": [0.5] * 20,
                "rmse_std": [0.05] * 20,
            }
        )
        fresh_ridge_df = pd.DataFrame(
            {
                "n_train": list(range(21, 41)),
                "rmse_mean": [0.4] * 20,
                "rmse_std": [0.04] * 20,
            }
        )

        class RecordingFamily:
            def __init__(self) -> None:
                self.calls = 0
                self.last_payload = None
                self.method_name = "ridge"

            def requirements(self) -> SweepFamilyRequirements:
                return SweepFamilyRequirements()

            def run(self, payload):
                self.calls += 1
                self.last_payload = payload
                return LearningCurveResults.from_mapping({"ridge_df": fresh_ridge_df})

        family = RecordingFamily()

        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = Path(tmp_dir) / "learning_curve_results.json"
            save_learning_curve_results_artifact(
                LearningCurveResults.from_mapping({"ridge_df": cached_ridge_df}),
                LearningCurveSweepMetadata(
                    seed=23,
                    min_train=1,
                    max_train=20,
                    step=1,
                    n_repeats=1,
                    enabled_models=("ridge",),
                    mlip_feature_names=("ridge",),
                ),
                bundle_path,
            )
            cfg = SimpleNamespace(
                seed=23,
                plot=SimpleNamespace(filters=None),
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        min_train=1,
                        max_train=40,
                        step=1,
                        n_repeats=1,
                        validation_fraction=0.2,
                        min_val_size=1,
                        min_tuning_val_size=1,
                        min_inner_train_size=1,
                        min_test_size=1,
                        results_bundle_path=bundle_path,
                        reuse_results=True,
                        force_refresh_methods=[],
                        models=SimpleNamespace(
                            use_ridge=True,
                            use_kernel_ridge=False,
                            use_lasso=False,
                            use_elastic_net=False,
                            use_residual=False,
                            use_weighted_linear=False,
                            use_weighted_simplex=False,
                            use_graph_mean=False,
                            use_latent=False,
                            moe=SimpleNamespace(enabled=False),
                            probe_gnn=SimpleNamespace(enabled=False),
                            gnn_direct=SimpleNamespace(enabled=False),
                        ),
                    )
                ),
            )

            results = run_learning_curve_experiments_from_config(
                df,
                cfg=cfg,
                model_families=[family],
            )
            artifact = load_learning_curve_results_artifact(bundle_path)

        self.assertEqual(family.calls, 1)
        self.assertEqual(
            [split.sweep_size for split in family.last_payload.split_collection.splits],
            list(range(21, 41)),
        )
        self.assertEqual(results.ridge_df["n_train"].tolist(), list(range(1, 41)))
        self.assertEqual(artifact.results.ridge_df["n_train"].tolist(), list(range(1, 41)))

    def test_run_learning_curve_experiments_from_config_gap_fills_mixed_sweep_union(
        self,
    ) -> None:
        df = pd.DataFrame(
            {
                "reference_ads_eng": list(range(1, 101)),
                "ridge_mlip_ads_eng_median": [value + 0.1 for value in range(1, 101)],
            }
        )
        cached_ridge_df = pd.DataFrame(
            {
                "n_train": [1, 2, 3, 5],
                "rmse_mean": [0.51, 0.5, 0.49, 0.47],
                "rmse_std": [0.05, 0.05, 0.05, 0.045],
            }
        )
        fresh_ridge_df = pd.DataFrame(
            {
                "n_train": [10, 20],
                "rmse_mean": [0.42, 0.36],
                "rmse_std": [0.04, 0.035],
            }
        )

        class RecordingFamily:
            def __init__(self) -> None:
                self.calls = 0
                self.last_payload = None
                self.method_name = "ridge"

            def requirements(self) -> SweepFamilyRequirements:
                return SweepFamilyRequirements()

            def run(self, payload):
                self.calls += 1
                self.last_payload = payload
                return LearningCurveResults.from_mapping({"ridge_df": fresh_ridge_df})

        family = RecordingFamily()

        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = Path(tmp_dir) / "learning_curve_results.json"
            save_learning_curve_results_artifact(
                LearningCurveResults.from_mapping({"ridge_df": cached_ridge_df}),
                LearningCurveSweepMetadata(
                    seed=23,
                    min_train=1,
                    max_train=5,
                    step=0,
                    n_repeats=1,
                    enabled_models=("ridge",),
                    requested_sweep_sizes=(1, 2, 3, 5),
                    requested_sweep_fractions=(0.05,),
                    mlip_feature_names=("ridge",),
                ),
                bundle_path,
            )
            cfg = SimpleNamespace(
                seed=23,
                plot=SimpleNamespace(filters=None),
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        min_train=None,
                        max_train=None,
                        step=1,
                        sweep_sizes=[1, 2, 3, 20],
                        sweep_fractions=[0.05, 0.1, 0.2],
                        n_repeats=1,
                        validation_fraction=0.2,
                        min_val_size=1,
                        min_tuning_val_size=1,
                        min_inner_train_size=1,
                        min_test_size=1,
                        results_bundle_path=bundle_path,
                        reuse_results=True,
                        force_refresh_methods=[],
                        models=SimpleNamespace(
                            use_ridge=True,
                            use_kernel_ridge=False,
                            use_lasso=False,
                            use_elastic_net=False,
                            use_residual=False,
                            use_weighted_linear=False,
                            use_weighted_simplex=False,
                            use_graph_mean=False,
                            use_latent=False,
                            moe=SimpleNamespace(enabled=False),
                            probe_gnn=SimpleNamespace(enabled=False),
                            gnn_direct=SimpleNamespace(enabled=False),
                        ),
                    )
                ),
            )

            results = run_learning_curve_experiments_from_config(
                df,
                cfg=cfg,
                model_families=[family],
            )
            artifact = load_learning_curve_results_artifact(bundle_path)

        self.assertEqual(family.calls, 1)
        self.assertEqual(
            [split.sweep_size for split in family.last_payload.split_collection.splits],
            [10, 20],
        )
        self.assertEqual(
            results.ridge_df["n_train"].tolist(),
            [1, 2, 3, 5, 10, 20],
        )
        self.assertEqual(
            artifact.results.ridge_df["n_train"].tolist(),
            [1, 2, 3, 5, 10, 20],
        )

    def test_run_learning_curve_experiments_from_config_force_refreshes_selected_methods_in_bundle(
        self,
    ) -> None:
        df = pd.DataFrame(
            {
                "reference_ads_eng": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "ridge_mlip_ads_eng_median": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
                "linear_mlip_ads_eng_median": [0.9, 1.9, 2.9, 3.9, 4.9, 5.9],
            }
        )
        cached_ridge_df = pd.DataFrame(
            {
                "n_train": [2, 3],
                "rmse_mean": [0.8, 0.7],
                "rmse_std": [0.07, 0.06],
            }
        )
        cached_ridge_uq_df = pd.DataFrame(
            {
                "n_train": [2, 3],
                "miscalibration_area": [0.4, 0.3],
                "sharpness": [0.5, 0.4],
                "dispersion": [0.6, 0.5],
                "uncertainty_kind": ["spread_only", "spread_only"],
                "uncertainty_note": [
                    "spread-only; not probabilistically interpretable",
                    "spread-only; not probabilistically interpretable",
                ],
            }
        )
        refreshed_ridge_df = pd.DataFrame(
            {
                "n_train": [2, 3],
                "rmse_mean": [0.4, 0.3],
                "rmse_std": [0.05, 0.04],
            }
        )
        refreshed_ridge_uq_df = pd.DataFrame(
            {
                "n_train": [2, 3],
                "miscalibration_area": [0.14, 0.13],
                "sharpness": [0.24, 0.23],
                "dispersion": [0.34, 0.33],
                "uncertainty_kind": ["spread_only", "spread_only"],
                "uncertainty_note": [
                    "spread-only; not probabilistically interpretable",
                    "spread-only; not probabilistically interpretable",
                ],
            }
        )
        cached_weighted_linear_df = pd.DataFrame(
            {
                "n_train": [2, 3],
                "rmse_mean": [0.35, 0.25],
                "rmse_std": [0.04, 0.03],
            }
        )

        class StubFamily:
            def __init__(
                self,
                method_name: str,
                result_field: str,
                result_df: pd.DataFrame,
            ) -> None:
                self.method_name = method_name
                self.result_field = result_field
                self.result_df = result_df
                self.calls = 0

            def requirements(self) -> SweepFamilyRequirements:
                return SweepFamilyRequirements()

            def run(self, payload):
                self.calls += 1
                del payload
                return LearningCurveResults.from_mapping(
                    {
                        self.result_field: self.result_df,
                        "ridge_uq_df": refreshed_ridge_uq_df
                        if self.result_field == "ridge_df"
                        else None,
                    }
                )

        ridge_family = StubFamily("ridge", "ridge_df", refreshed_ridge_df)
        weighted_linear_family = StubFamily(
            "weighted_linear",
            "weighted_linear_df",
            cached_weighted_linear_df,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = Path(tmp_dir) / "learning_curve_results.json"
            save_learning_curve_results_artifact(
                LearningCurveResults(
                    ridge_df=cached_ridge_df,
                    ridge_uq_df=cached_ridge_uq_df,
                    weighted_linear_df=cached_weighted_linear_df,
                ),
                LearningCurveSweepMetadata(
                    seed=23,
                    min_train=2,
                    max_train=3,
                    step=1,
                    n_repeats=1,
                    enabled_models=("ridge", "weighted_linear"),
                    mlip_feature_names=("ridge", "linear"),
                ),
                bundle_path,
            )
            cfg = SimpleNamespace(
                seed=23,
                plot=SimpleNamespace(filters=None),
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        min_train=2,
                        max_train=3,
                        step=1,
                        n_repeats=1,
                        validation_fraction=0.2,
                        min_val_size=1,
                        min_tuning_val_size=1,
                        min_inner_train_size=1,
                        min_test_size=1,
                        results_bundle_path=bundle_path,
                        reuse_results=True,
                        force_refresh_methods=["ridge"],
                        models=SimpleNamespace(
                            use_ridge=True,
                            use_kernel_ridge=False,
                            use_lasso=False,
                            use_elastic_net=False,
                            use_residual=False,
                            use_weighted_linear=True,
                            use_weighted_simplex=False,
                            use_graph_mean=False,
                            use_latent=False,
                            moe=SimpleNamespace(enabled=False),
                            probe_gnn=SimpleNamespace(enabled=False),
                            gnn_direct=SimpleNamespace(enabled=False),
                        ),
                    )
                ),
            )

            results = run_learning_curve_experiments_from_config(
                df,
                cfg=cfg,
                model_families=[ridge_family, weighted_linear_family],
            )
            bundle_artifact = load_learning_curve_results_artifact(bundle_path)

        self.assertEqual(ridge_family.calls, 1)
        self.assertEqual(weighted_linear_family.calls, 0)
        pd.testing.assert_frame_equal(results.ridge_df, refreshed_ridge_df)
        pd.testing.assert_frame_equal(results.ridge_uq_df, refreshed_ridge_uq_df)
        pd.testing.assert_frame_equal(
            results.weighted_linear_df,
            cached_weighted_linear_df,
        )
        pd.testing.assert_frame_equal(
            bundle_artifact.results.ridge_df,
            refreshed_ridge_df,
        )
        pd.testing.assert_frame_equal(
            bundle_artifact.results.ridge_uq_df,
            refreshed_ridge_uq_df,
        )
        pd.testing.assert_frame_equal(
            bundle_artifact.results.weighted_linear_df,
            cached_weighted_linear_df,
        )

    def test_run_learning_curve_experiments_from_config_replaces_cached_method_rows_when_uq_frame_is_missing(
        self,
    ) -> None:
        df = pd.DataFrame(
            {
                "reference_ads_eng": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "ridge_mlip_ads_eng_median": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
            }
        )
        cached_kernel_ridge_df = pd.DataFrame(
            {
                "n_train": [3],
                "rmse_mean": [0.8],
                "rmse_std": [0.07],
            }
        )
        refreshed_kernel_ridge_df = pd.DataFrame(
            {
                "n_train": [3],
                "rmse_mean": [0.4],
                "rmse_std": [0.05],
            }
        )
        refreshed_kernel_ridge_selection_df = pd.DataFrame(
            {"n_train": [3], "alpha": [1.0], "gamma": [0.1], "kernel": ["rbf"]}
        )
        refreshed_kernel_ridge_uq_df = pd.DataFrame(
            {
                "n_train": [3],
                "miscalibration_area": [0.14],
                "sharpness": [0.24],
                "dispersion": [0.34],
                "uncertainty_kind": ["calibrated"],
            }
        )

        class StubFamily:
            method_name = "kernel_ridge"
            result_field = "kernel_ridge_df"
            selection_metadata_field = "kernel_ridge_selection_df"
            uq_summary_field = "kernel_ridge_uq_df"

            def __init__(self) -> None:
                self.calls = 0

            def requirements(self) -> SweepFamilyRequirements:
                return SweepFamilyRequirements()

            def run(self, payload):
                self.calls += 1
                del payload
                return LearningCurveResults(
                    kernel_ridge_df=refreshed_kernel_ridge_df,
                    kernel_ridge_selection_df=refreshed_kernel_ridge_selection_df,
                    kernel_ridge_uq_df=refreshed_kernel_ridge_uq_df,
                )

        family = StubFamily()

        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = Path(tmp_dir) / "learning_curve_results.json"
            save_learning_curve_results_artifact(
                LearningCurveResults(kernel_ridge_df=cached_kernel_ridge_df),
                LearningCurveSweepMetadata(
                    seed=23,
                    min_train=3,
                    max_train=3,
                    step=1,
                    n_repeats=1,
                    enabled_models=("kernel_ridge",),
                    mlip_feature_names=("ridge",),
                ),
                bundle_path,
            )
            cfg = SimpleNamespace(
                seed=23,
                plot=SimpleNamespace(filters=None),
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        min_train=3,
                        max_train=3,
                        step=1,
                        n_repeats=1,
                        validation_fraction=0.2,
                        min_val_size=1,
                        min_tuning_val_size=1,
                        min_inner_train_size=1,
                        min_test_size=1,
                        results_bundle_path=bundle_path,
                        reuse_results=True,
                        force_refresh_methods=[],
                        force_refresh_train_sizes={},
                        models=SimpleNamespace(
                            use_ridge=False,
                            use_kernel_ridge=True,
                            use_lasso=False,
                            use_elastic_net=False,
                            use_residual=False,
                            use_weighted_linear=False,
                            use_weighted_simplex=False,
                            use_graph_mean=False,
                            use_latent=False,
                            moe=SimpleNamespace(enabled=False),
                            probe_gnn=SimpleNamespace(enabled=False),
                            gnn_direct=SimpleNamespace(enabled=False),
                        ),
                    )
                ),
            )

            results = run_learning_curve_experiments_from_config(
                df,
                cfg=cfg,
                model_families=[family],
            )
            bundle_artifact = load_learning_curve_results_artifact(bundle_path)

        self.assertEqual(family.calls, 1)
        pd.testing.assert_frame_equal(
            results.kernel_ridge_df,
            refreshed_kernel_ridge_df,
        )
        pd.testing.assert_frame_equal(
            results.kernel_ridge_uq_df,
            refreshed_kernel_ridge_uq_df,
        )
        pd.testing.assert_frame_equal(
            bundle_artifact.results.kernel_ridge_df,
            refreshed_kernel_ridge_df,
        )
        pd.testing.assert_frame_equal(
            bundle_artifact.results.kernel_ridge_uq_df,
            refreshed_kernel_ridge_uq_df,
        )

    def test_run_learning_curve_experiments_from_config_force_refreshes_selected_train_sizes_in_bundle(
        self,
    ) -> None:
        df = pd.DataFrame(
            {
                "reference_ads_eng": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "ridge_mlip_ads_eng_median": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
                "linear_mlip_ads_eng_median": [0.9, 1.9, 2.9, 3.9, 4.9, 5.9],
            }
        )
        cached_ridge_df = pd.DataFrame(
            {
                "n_train": [2, 3, 4],
                "rmse_mean": [0.8, 0.7, 0.6],
                "rmse_std": [0.07, 0.06, 0.05],
            }
        )
        cached_ridge_uq_df = pd.DataFrame(
            {
                "n_train": [2, 3, 4],
                "miscalibration_area": [0.32, 0.22, 0.12],
                "sharpness": [0.42, 0.32, 0.22],
                "dispersion": [0.52, 0.42, 0.32],
                "uncertainty_kind": ["spread_only", "spread_only", "spread_only"],
                "uncertainty_note": [
                    "spread-only; not probabilistically interpretable",
                    "spread-only; not probabilistically interpretable",
                    "spread-only; not probabilistically interpretable",
                ],
            }
        )
        refreshed_ridge_df = pd.DataFrame(
            {
                "n_train": [3],
                "rmse_mean": [0.35],
                "rmse_std": [0.035],
            }
        )
        refreshed_ridge_uq_df = pd.DataFrame(
            {
                "n_train": [3],
                "miscalibration_area": [0.11],
                "sharpness": [0.21],
                "dispersion": [0.31],
                "uncertainty_kind": ["spread_only"],
                "uncertainty_note": ["spread-only; not probabilistically interpretable"],
            }
        )
        cached_weighted_linear_df = pd.DataFrame(
            {
                "n_train": [2, 3, 4],
                "rmse_mean": [0.45, 0.4, 0.35],
                "rmse_std": [0.05, 0.045, 0.04],
            }
        )

        class StubFamily:
            def __init__(
                self,
                method_name: str,
                result_field: str,
                result_df: pd.DataFrame,
            ) -> None:
                self.method_name = method_name
                self.result_field = result_field
                self.result_df = result_df
                self.calls = 0
                self.last_payload = None

            def requirements(self) -> SweepFamilyRequirements:
                return SweepFamilyRequirements()

            def run(self, payload):
                self.calls += 1
                self.last_payload = payload
                return LearningCurveResults.from_mapping(
                    {
                        self.result_field: self.result_df,
                        "ridge_uq_df": refreshed_ridge_uq_df
                        if self.result_field == "ridge_df"
                        else None,
                    }
                )

        ridge_family = StubFamily("ridge", "ridge_df", refreshed_ridge_df)
        weighted_linear_family = StubFamily(
            "weighted_linear",
            "weighted_linear_df",
            cached_weighted_linear_df,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = Path(tmp_dir) / "learning_curve_results.json"
            save_learning_curve_results_artifact(
                LearningCurveResults(
                    ridge_df=cached_ridge_df,
                    ridge_uq_df=cached_ridge_uq_df,
                    weighted_linear_df=cached_weighted_linear_df,
                ),
                LearningCurveSweepMetadata(
                    seed=23,
                    min_train=2,
                    max_train=4,
                    step=1,
                    n_repeats=1,
                    enabled_models=("ridge", "weighted_linear"),
                    mlip_feature_names=("ridge", "linear"),
                ),
                bundle_path,
            )
            cfg = SimpleNamespace(
                seed=23,
                plot=SimpleNamespace(filters=None),
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        min_train=2,
                        max_train=4,
                        step=1,
                        n_repeats=1,
                        validation_fraction=0.2,
                        min_val_size=1,
                        min_tuning_val_size=1,
                        min_inner_train_size=1,
                        min_test_size=1,
                        results_bundle_path=bundle_path,
                        reuse_results=True,
                        force_refresh_methods=[],
                        force_refresh_train_sizes={"ridge": [3]},
                        models=SimpleNamespace(
                            use_ridge=True,
                            use_kernel_ridge=False,
                            use_lasso=False,
                            use_elastic_net=False,
                            use_residual=False,
                            use_weighted_linear=True,
                            use_weighted_simplex=False,
                            use_graph_mean=False,
                            use_latent=False,
                            moe=SimpleNamespace(enabled=False),
                            probe_gnn=SimpleNamespace(enabled=False),
                            gnn_direct=SimpleNamespace(enabled=False),
                        ),
                    )
                ),
            )

            results = run_learning_curve_experiments_from_config(
                df,
                cfg=cfg,
                model_families=[ridge_family, weighted_linear_family],
            )
            bundle_artifact = load_learning_curve_results_artifact(bundle_path)

        self.assertEqual(ridge_family.calls, 1)
        self.assertEqual(weighted_linear_family.calls, 0)
        self.assertEqual(
            [split.sweep_size for split in ridge_family.last_payload.split_collection.splits],
            [3],
        )
        pd.testing.assert_frame_equal(
            results.ridge_df,
            pd.DataFrame(
                {
                    "n_train": [2, 3, 4],
                    "rmse_mean": [0.8, 0.35, 0.6],
                    "rmse_std": [0.07, 0.035, 0.05],
                }
            ),
        )
        pd.testing.assert_frame_equal(
            results.ridge_uq_df,
            pd.DataFrame(
                {
                    "n_train": [2, 3, 4],
                    "miscalibration_area": [0.32, 0.11, 0.12],
                    "sharpness": [0.42, 0.21, 0.22],
                    "dispersion": [0.52, 0.31, 0.32],
                    "uncertainty_kind": [
                        "spread_only",
                        "spread_only",
                        "spread_only",
                    ],
                    "uncertainty_note": [
                        "spread-only; not probabilistically interpretable",
                        "spread-only; not probabilistically interpretable",
                        "spread-only; not probabilistically interpretable",
                    ],
                }
            ),
        )
        pd.testing.assert_frame_equal(
            results.weighted_linear_df,
            cached_weighted_linear_df,
        )
        pd.testing.assert_frame_equal(
            bundle_artifact.results.ridge_df,
            pd.DataFrame(
                {
                    "n_train": [2, 3, 4],
                    "rmse_mean": [0.8, 0.35, 0.6],
                    "rmse_std": [0.07, 0.035, 0.05],
                }
            ),
        )
        pd.testing.assert_frame_equal(
            bundle_artifact.results.ridge_uq_df,
            results.ridge_uq_df,
        )

    def test_run_learning_curve_experiments_from_config_preserves_prior_bundle_methods(
        self,
    ) -> None:
        df = pd.DataFrame(
            {
                "reference_ads_eng": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "linear_mlip_ads_eng_median": [0.9, 1.9, 2.9, 3.9, 4.9, 5.9],
            }
        )
        ridge_df = pd.DataFrame(
            {
                "n_train": [2, 3],
                "rmse_mean": [0.4, 0.3],
                "rmse_std": [0.05, 0.04],
            }
        )
        weighted_simplex_df = pd.DataFrame(
            {
                "n_train": [2, 3],
                "rmse_mean": [0.32, 0.27],
                "rmse_std": [0.04, 0.03],
            }
        )

        class StubFamily:
            def __init__(self, result_df: pd.DataFrame) -> None:
                self.calls = 0
                self.result_df = result_df

            def requirements(self) -> SweepFamilyRequirements:
                return SweepFamilyRequirements()

            def run(self, payload):
                self.calls += 1
                del payload
                return LearningCurveResults.from_mapping(
                    {"weighted_simplex_df": self.result_df}
                )

        family = StubFamily(weighted_simplex_df)

        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = Path(tmp_dir) / "learning_curve_results.json"
            save_learning_curve_results_artifact(
                LearningCurveResults.from_mapping({"ridge_df": ridge_df}),
                LearningCurveSweepMetadata(
                    seed=23,
                    min_train=2,
                    max_train=3,
                    step=1,
                    n_repeats=1,
                    enabled_models=("ridge",),
                    mlip_feature_names=("linear",),
                ),
                bundle_path,
            )
            cfg = SimpleNamespace(
                seed=23,
                plot=SimpleNamespace(filters=None),
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        min_train=2,
                        max_train=3,
                        step=1,
                        n_repeats=1,
                        validation_fraction=0.2,
                        min_val_size=1,
                        min_tuning_val_size=1,
                        min_inner_train_size=1,
                        min_test_size=1,
                        results_bundle_path=bundle_path,
                        reuse_results=True,
                        models=SimpleNamespace(
                            use_ridge=False,
                            use_kernel_ridge=False,
                            use_lasso=False,
                            use_elastic_net=False,
                            use_residual=False,
                            use_weighted_linear=False,
                            use_weighted_simplex=True,
                            use_graph_mean=False,
                            use_latent=False,
                            moe=SimpleNamespace(enabled=False),
                            probe_gnn=SimpleNamespace(enabled=False),
                            gnn_direct=SimpleNamespace(enabled=False),
                        ),
                    )
                ),
            )

            results = run_learning_curve_experiments_from_config(
                df,
                cfg=cfg,
                model_families=[family],
            )
            artifact = load_learning_curve_results_artifact(bundle_path)

        self.assertEqual(family.calls, 1)
        self.assertIsNone(results.ridge_df)
        pd.testing.assert_frame_equal(results.weighted_simplex_df, weighted_simplex_df)
        pd.testing.assert_frame_equal(artifact.results.ridge_df, ridge_df)
        pd.testing.assert_frame_equal(
            artifact.results.weighted_simplex_df,
            weighted_simplex_df,
        )
        self.assertEqual(artifact.metadata.dataset_tag, None)
        self.assertEqual(artifact.metadata.dataset_size, len(df))

    def test_run_learning_curve_experiments_from_config_merges_method_rows_across_bounds(
        self,
    ) -> None:
        df = pd.DataFrame(
            {
                "reference_ads_eng": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "ridge_mlip_ads_eng_median": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
            }
        )
        cached_ridge_df = pd.DataFrame(
            {
                "n_train": [1, 10, 20],
                "rmse_mean": [0.8, 0.5, 0.4],
                "rmse_std": [0.1, 0.07, 0.05],
            }
        )
        fresh_ridge_df = pd.DataFrame(
            {
                "n_train": [20, 30, 40],
                "rmse_mean": [0.39, 0.35, 0.3],
                "rmse_std": [0.045, 0.04, 0.03],
            }
        )

        class StubFamily:
            def __init__(self, result_df: pd.DataFrame) -> None:
                self.calls = 0
                self.result_df = result_df

            def requirements(self) -> SweepFamilyRequirements:
                return SweepFamilyRequirements()

            def run(self, payload):
                self.calls += 1
                del payload
                return LearningCurveResults.from_mapping({"ridge_df": self.result_df})

        family = StubFamily(fresh_ridge_df)

        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = Path(tmp_dir) / "learning_curve_results.json"
            save_learning_curve_results_artifact(
                LearningCurveResults.from_mapping({"ridge_df": cached_ridge_df}),
                LearningCurveSweepMetadata(
                    seed=23,
                    min_train=1,
                    max_train=20,
                    step=1,
                    n_repeats=30,
                    enabled_models=("ridge",),
                    mlip_feature_names=("ridge",),
                ),
                bundle_path,
                run_id="cached-run",
                run_timestamp_utc="2026-05-26T00:00:00+00:00",
            )
            cfg = SimpleNamespace(
                seed=23,
                plot=SimpleNamespace(filters=None),
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        min_train=20,
                        max_train=40,
                        step=10,
                        n_repeats=10,
                        validation_fraction=0.2,
                        min_val_size=1,
                        min_tuning_val_size=1,
                        min_inner_train_size=1,
                        min_test_size=1,
                        results_bundle_path=bundle_path,
                        reuse_results=False,
                        force_refresh_methods=["ridge"],
                        models=SimpleNamespace(
                            use_ridge=True,
                            use_kernel_ridge=False,
                            use_lasso=False,
                            use_elastic_net=False,
                            use_residual=False,
                            use_weighted_linear=False,
                            use_weighted_simplex=False,
                            use_graph_mean=False,
                            use_latent=False,
                            moe=SimpleNamespace(enabled=False),
                            probe_gnn=SimpleNamespace(enabled=False),
                            gnn_direct=SimpleNamespace(enabled=False),
                        ),
                    )
                ),
            )

            results = run_learning_curve_experiments_from_config(
                df,
                cfg=cfg,
                model_families=[family],
            )
            artifact = load_learning_curve_results_artifact(bundle_path)

        self.assertEqual(family.calls, 1)
        pd.testing.assert_frame_equal(
            results.ridge_df,
            pd.DataFrame(
                {
                    "n_train": [1, 10, 20, 30, 40],
                    "rmse_mean": [0.8, 0.5, 0.39, 0.35, 0.3],
                    "rmse_std": [0.1, 0.07, 0.045, 0.04, 0.03],
                }
            ),
        )
        pd.testing.assert_frame_equal(artifact.results.ridge_df, results.ridge_df)
        provenance = artifact.point_provenance["ridge_df"]
        self.assertEqual(provenance["n_train"].tolist(), [1, 10, 20, 30, 40])
        self.assertEqual(provenance["seed"].tolist(), [23, 23, 23, 23, 23])
        self.assertEqual(provenance["n_repeats"].tolist(), [30, 30, 10, 10, 10])
        self.assertEqual(provenance["sweep_min_train"].tolist(), [1, 1, 20, 20, 20])
        self.assertEqual(provenance["sweep_max_train"].tolist(), [20, 20, 40, 40, 40])
        self.assertEqual(provenance["sweep_step"].tolist(), [1, 1, 10, 10, 10])
        self.assertEqual(provenance["run_id"].iloc[0], "cached-run")
        self.assertEqual(provenance["run_id"].iloc[1], "cached-run")
        self.assertEqual(
            provenance["run_timestamp_utc"].iloc[0],
            "2026-05-26T00:00:00+00:00",
        )
        self.assertEqual(
            provenance["run_timestamp_utc"].iloc[1],
            "2026-05-26T00:00:00+00:00",
        )
        self.assertNotEqual(provenance["run_id"].iloc[2], "cached-run")
        self.assertEqual(len(set(provenance["run_id"].iloc[2:])), 1)
        self.assertEqual(len(set(provenance["run_timestamp_utc"].iloc[2:])), 1)

    def test_run_learning_curve_experiments_from_config_saves_repeat_metrics_sidecar(
        self,
    ) -> None:
        df = pd.DataFrame(
            {
                "reference_ads_eng": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "ridge_mlip_ads_eng_median": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
            }
        )
        result_df = pd.DataFrame(
            {
                "n_train": [4],
                "rmse_mean": [0.3],
                "rmse_std": [0.04],
            }
        )
        repeat_metrics_df = pd.DataFrame(
            {
                "n_train": [4, 4],
                "repeat": [0, 1],
                "outer_test_rmse": [0.32, 0.28],
            }
        )

        class StubFamily:
            method_name = "ridge"

            def capabilities(self):
                return SweepModelCapabilities()

            def requirements(self):
                return SweepFamilyRequirements()

            def run(self, payload):
                del payload
                return LearningCurveResults.from_mapping({"ridge_df": result_df})

            def run_with_artifacts(self, payload):
                del payload
                return SweepFamilyRunArtifacts(
                    results=LearningCurveResults.from_mapping({"ridge_df": result_df}),
                    repeat_metrics=repeat_metrics_df,
                )

        family = StubFamily()

        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = Path(tmp_dir) / "learning_curve_results.json"
            cfg = SimpleNamespace(
                seed=23,
                plot=SimpleNamespace(filters=None),
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        min_train=4,
                        max_train=4,
                        step=1,
                        n_repeats=2,
                        budget_mode="full_remainder_test",
                        validation_fraction=0.2,
                        min_val_size=1,
                        min_tuning_val_size=1,
                        calibration_enabled=False,
                        calibration_fraction=0.2,
                        min_cal_size=1,
                        min_inner_train_size=1,
                        min_test_size=1,
                        results_bundle_path=bundle_path,
                        reuse_results=False,
                        force_refresh_methods=[],
                        force_refresh_train_sizes={},
                        models=SimpleNamespace(
                            use_ridge=True,
                            use_kernel_ridge=False,
                            use_lasso=False,
                            use_elastic_net=False,
                            use_residual=False,
                            use_weighted_linear=False,
                            use_weighted_simplex=False,
                            use_graph_mean=False,
                            use_latent=False,
                            moe=SimpleNamespace(enabled=False),
                            probe_gnn=SimpleNamespace(enabled=False),
                            gnn_direct=SimpleNamespace(enabled=False),
                        ),
                    )
                ),
            )

            results = run_learning_curve_experiments_from_config(
                df,
                cfg=cfg,
                model_families=[family],
            )
            sidecar_path = repeat_metrics_artifact_path(bundle_path)
            sidecar_exists = sidecar_path.is_file()
            repeat_metrics_artifact = load_learning_curve_repeat_metrics_artifact(sidecar_path)

        self.assertTrue(sidecar_exists)
        pd.testing.assert_frame_equal(results.ridge_df, result_df)
        pd.testing.assert_frame_equal(
            repeat_metrics_artifact.repeat_metrics_df,
            pd.DataFrame(
                {
                    "method": pd.Series(["ridge", "ridge"], dtype="string"),
                    "budget": pd.Series([4, 4], dtype="Int64"),
                    "repeat": pd.Series([0, 1], dtype="Int64"),
                    "outer_test_rmse": pd.Series([0.32, 0.28], dtype="Float64"),
                }
            ),
        )

    def test_run_learning_curve_experiments_from_config_overwrites_enabled_overlap_when_reuse_disabled(
        self,
    ) -> None:
        df = pd.DataFrame(
            {
                "reference_ads_eng": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "ridge_mlip_ads_eng_median": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
                "linear_mlip_ads_eng_median": [0.9, 1.9, 2.9, 3.9, 4.9, 5.9],
            }
        )
        cached_ridge_df = pd.DataFrame(
            {
                "n_train": [10, 20],
                "rmse_mean": [0.5, 0.4],
                "rmse_std": [0.07, 0.05],
            }
        )
        fresh_ridge_df = pd.DataFrame(
            {
                "n_train": [20, 30],
                "rmse_mean": [0.39, 0.35],
                "rmse_std": [0.045, 0.04],
            }
        )
        cached_weighted_linear_df = pd.DataFrame(
            {
                "n_train": [10, 20],
                "rmse_mean": [0.45, 0.42],
                "rmse_std": [0.05, 0.04],
            }
        )

        class StubFamily:
            def run(self, payload):
                del payload
                return LearningCurveResults.from_mapping({"ridge_df": fresh_ridge_df})

        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = Path(tmp_dir) / "learning_curve_results.json"
            save_learning_curve_results_artifact(
                LearningCurveResults(
                    ridge_df=cached_ridge_df,
                    weighted_linear_df=cached_weighted_linear_df,
                ),
                LearningCurveSweepMetadata(
                    seed=23,
                    min_train=10,
                    max_train=20,
                    step=10,
                    n_repeats=30,
                    enabled_models=("ridge", "weighted_linear"),
                    mlip_feature_names=("ridge", "linear"),
                ),
                bundle_path,
            )
            cfg = SimpleNamespace(
                seed=23,
                plot=SimpleNamespace(filters=None),
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        min_train=20,
                        max_train=30,
                        step=10,
                        n_repeats=10,
                        validation_fraction=0.2,
                        min_val_size=1,
                        min_tuning_val_size=1,
                        min_inner_train_size=1,
                        min_test_size=1,
                        results_bundle_path=bundle_path,
                        reuse_results=False,
                        force_refresh_methods=[],
                        models=SimpleNamespace(
                            use_ridge=True,
                            use_kernel_ridge=False,
                            use_lasso=False,
                            use_elastic_net=False,
                            use_residual=False,
                            use_weighted_linear=False,
                            use_weighted_simplex=False,
                            use_graph_mean=False,
                            use_latent=False,
                            moe=SimpleNamespace(enabled=False),
                            probe_gnn=SimpleNamespace(enabled=False),
                            gnn_direct=SimpleNamespace(enabled=False),
                        ),
                    )
                ),
            )

            results = run_learning_curve_experiments_from_config(
                df,
                cfg=cfg,
                model_families=[StubFamily()],
            )
            artifact = load_learning_curve_results_artifact(bundle_path)

        pd.testing.assert_frame_equal(
            results.ridge_df,
            pd.DataFrame(
                {
                    "n_train": [10, 20, 30],
                    "rmse_mean": [0.5, 0.39, 0.35],
                    "rmse_std": [0.07, 0.045, 0.04],
                }
            ),
        )
        self.assertIsNone(results.weighted_linear_df)
        pd.testing.assert_frame_equal(
            artifact.results.ridge_df,
            pd.DataFrame(
                {
                    "n_train": [10, 20, 30],
                    "rmse_mean": [0.5, 0.39, 0.35],
                    "rmse_std": [0.07, 0.045, 0.04],
                }
            ),
        )
        pd.testing.assert_frame_equal(
            artifact.results.weighted_linear_df,
            cached_weighted_linear_df,
        )

    def test_run_learning_curve_experiments_from_config_applies_split_policy_knobs(
        self,
    ) -> None:
        df = pd.DataFrame(
            {
                "reference_ads_eng": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                "ridge_mlip_ads_eng_median": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1],
            }
        )
        result_df = pd.DataFrame(
            {
                "n_train": [5],
                "rmse_mean": [0.3],
                "rmse_std": [0.04],
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
        cfg = SimpleNamespace(
            seed=23,
            plot=None,
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(
                    min_train=2,
                    max_train=6,
                    n_repeats=1,
                    validation_fraction=0.2,
                    min_val_size=2,
                    min_tuning_val_size=3,
                    min_inner_train_size=2,
                    min_test_size=2,
                    models=None,
                )
            ),
        )

        results = run_learning_curve_experiments_from_config(
            df,
            cfg=cfg,
            model_families=[family],
        )

        self.assertIs(results.ridge_df, result_df)
        self.assertEqual(
            [split.sweep_size for split in family.last_payload.split_collection.splits],
            [5],
        )
        self.assertEqual(
            [
                len(split.val_idx)
                for split in family.last_payload.split_collection.splits
            ],
            [3],
        )
        self.assertEqual(
            [
                len(split.train_idx)
                for split in family.last_payload.split_collection.splits
            ],
            [2],
        )
        self.assertEqual(
            [
                len(split.test_idx)
                for split in family.last_payload.split_collection.splits
            ],
            [2],
        )

    def test_run_learning_curve_experiments_from_config_keeps_fraction_sweep_sizes_for_multiple_families(
        self,
    ) -> None:
        df = pd.DataFrame(
            {
                "reference_ads_eng": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "ridge_mlip_ads_eng_median": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1],
            }
        )

        class StubFamily:
            def __init__(self, result_field: str) -> None:
                self.result_field = result_field
                self.last_payload = None

            def run(self, payload):
                self.last_payload = payload
                return LearningCurveResults.from_mapping(
                    {
                        self.result_field: pd.DataFrame(
                            {
                                "n_train": [split.sweep_size for split in payload.split_collection.splits],
                                "rmse_mean": [0.3] * len(payload.split_collection.splits),
                                "rmse_std": [0.04] * len(payload.split_collection.splits),
                            }
                        )
                    }
                )

        ridge_family = StubFamily("ridge_df")
        residual_family = StubFamily("resid_df")
        cfg = SimpleNamespace(
            seed=23,
            plot=None,
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(
                    min_train=None,
                    max_train=None,
                    sweep_fractions=[0.25, 0.5],
                    n_repeats=1,
                    validation_fraction=0.2,
                    min_val_size=1,
                    min_tuning_val_size=1,
                    min_inner_train_size=1,
                    min_test_size=1,
                    models=None,
                )
            ),
        )

        results = run_learning_curve_experiments_from_config(
            df,
            cfg=cfg,
            model_families=[ridge_family, residual_family],
        )

        self.assertIsNotNone(results.ridge_df)
        self.assertIsNotNone(results.resid_df)
        self.assertEqual(
            [split.sweep_size for split in ridge_family.last_payload.split_collection.splits],
            [2, 4],
        )
        self.assertEqual(
            [split.sweep_size for split in residual_family.last_payload.split_collection.splits],
            [2, 4],
        )

    def test_run_learning_curve_experiments_from_config_routes_screening_mode(
        self,
    ) -> None:
        df = pd.DataFrame(
            {
                "reference_ads_eng": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                "ridge_mlip_ads_eng_median": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1],
            }
        )
        expected = LearningCurveResults.from_mapping(
            {
                "ridge_df": pd.DataFrame(
                    {
                        "n_train": [4],
                        "rmse_mean": [0.3],
                        "rmse_std": [0.04],
                    }
                )
            }
        )

        cfg = SimpleNamespace(
            seed=23,
            plot=None,
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(
                    min_train=4,
                    max_train=4,
                    n_repeats=1,
                    budget_mode="screening_fraction",
                    screen_fraction=0.25,
                    min_screen_size=1,
                    validation_fraction=0.2,
                    min_val_size=1,
                    min_tuning_val_size=1,
                    min_inner_train_size=1,
                    min_test_size=1,
                    models=None,
                    results_bundle_path=None,
                    reuse_results=False,
                    force_refresh_methods=[],
                    force_refresh_train_sizes={},
                )
            ),
        )

        with patch(
            "oasis.experiment.orchestration.run_screening_learning_curve_experiments",
            autospec=True,
        ) as screening_mock, patch(
            "oasis.experiment.orchestration.run_standard_learning_curve_experiments",
            autospec=True,
        ) as standard_mock:
            screening_mock.return_value = expected

            results = run_learning_curve_experiments_from_config(df, cfg=cfg)

        self.assertIsNotNone(results.ridge_df)
        pd.testing.assert_frame_equal(results.ridge_df, expected.ridge_df)
        screening_mock.assert_called_once()
        standard_mock.assert_not_called()

    def test_run_learning_curve_experiments_screening_results_include_budget_columns(
        self,
    ) -> None:
        df = pd.DataFrame(
            {
                "reference_ads_eng": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                "ridge_mlip_ads_eng_median": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1],
            }
        )
        result_df = pd.DataFrame(
            {
                "n_train": [4],
                "rmse_mean": [0.3],
                "rmse_std": [0.04],
            }
        )

        class StubFamily:
            def run(self, payload):
                self.last_payload = payload
                return LearningCurveResults.from_mapping({"ridge_df": result_df})

        family = StubFamily()

        results = run_learning_curve_experiments_from_frame(
            df,
            min_train=4,
            max_train=4,
            n_repeats=1,
            budget_mode="screening_fraction",
            screen_fraction=0.25,
            model_families=[family],
        )

        assert results.ridge_df is not None
        self.assertEqual(
            results.ridge_df.columns.tolist(),
            [
                "n_budget",
                "n_train",
                "n_screen",
                "screen_fraction",
                "n_cv_folds",
                "cv_rmse_mean",
                "cv_rmse_std",
            ],
        )
        self.assertEqual(results.ridge_df["n_budget"].tolist(), [4])
        self.assertEqual(results.ridge_df["n_train"].tolist(), [3])
        self.assertEqual(results.ridge_df["n_screen"].tolist(), [1])
        self.assertEqual(results.ridge_df["n_cv_folds"].tolist(), [4])

    def test_run_learning_curve_experiments_from_config_forwards_graph_view(
        self,
    ) -> None:
        df = pd.DataFrame(
            {
                "reaction": ["s0", "s1", "s2", "s3", "s4", "s5"],
                "reference_ads_eng": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "ridge_mlip_ads_eng_median": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
            }
        )
        graph_view = GraphDatasetView.from_records(
            (
                GraphRecord(
                    sample_id="s0",
                    node_features=np.array([[1.0]]),
                    edge_index=np.empty((2, 0), dtype=np.int64),
                ),
                GraphRecord(
                    sample_id="s1",
                    node_features=np.array([[2.0]]),
                    edge_index=np.empty((2, 0), dtype=np.int64),
                ),
                GraphRecord(
                    sample_id="s2",
                    node_features=np.array([[3.0]]),
                    edge_index=np.empty((2, 0), dtype=np.int64),
                ),
                GraphRecord(
                    sample_id="s3",
                    node_features=np.array([[4.0]]),
                    edge_index=np.empty((2, 0), dtype=np.int64),
                ),
                GraphRecord(
                    sample_id="s4",
                    node_features=np.array([[5.0]]),
                    edge_index=np.empty((2, 0), dtype=np.int64),
                ),
                GraphRecord(
                    sample_id="s5",
                    node_features=np.array([[6.0]]),
                    edge_index=np.empty((2, 0), dtype=np.int64),
                ),
            )
        )
        cfg = SimpleNamespace(
            seed=23,
            plot=None,
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(
                    min_train=2,
                    max_train=4,
                    n_repeats=1,
                    validation_fraction=0.2,
                    min_val_size=1,
                    min_test_size=1,
                    models=None,
                )
            ),
        )

        with patch("oasis.experiment.orchestration.build_sweep_dataset_from_config") as mock_build:
            with patch("oasis.experiment.orchestration.run_learning_curve_experiments") as mock_run:
                mock_build.return_value = SweepDataset(
                    mlip_features=np.array([[1.0]]),
                    targets=np.array([1.0]),
                    sample_ids=np.array(["s0"]),
                    graph_view=GraphDatasetView.from_records((graph_view["s0"],)),
                )
                mock_run.return_value = LearningCurveResults.empty()

                result = run_learning_curve_experiments_from_config(
                    df,
                    cfg=cfg,
                    graph_view=graph_view,
                )

        self.assertEqual(result, LearningCurveResults.empty())
        self.assertIs(mock_build.call_args.kwargs["graph_view"], graph_view)
        self.assertIs(mock_run.call_args.args[0], mock_build.return_value)

    def test_run_learning_curve_experiments_from_config_loads_graph_view_from_config(
        self,
    ) -> None:
        df = pd.DataFrame(
            {
                "reaction": ["s0", "s1", "s2", "s3", "s4", "s5"],
                "reference_ads_eng": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "ridge_mlip_ads_eng_median": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
            }
        )
        graph_view = GraphDatasetView.from_records(
            (
                GraphRecord(
                    sample_id="s0",
                    node_features=np.array([[1.0]]),
                    edge_index=np.empty((2, 0), dtype=np.int64),
                ),
            )
        )
        graph_dataset_cfg = SimpleNamespace(
            path="data/graphs/reactions.json",
            join_key="reaction_id",
        )
        cfg = SimpleNamespace(
            seed=23,
            plot=None,
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(
                    min_train=2,
                    max_train=4,
                    n_repeats=1,
                    validation_fraction=0.2,
                    min_val_size=1,
                    min_test_size=1,
                    graph_dataset=graph_dataset_cfg,
                    models=None,
                )
            ),
        )

        with patch(
            "oasis.graphs.load_configured_graph_dataset_view",
            return_value=graph_view,
        ) as mock_load:
            with patch("oasis.experiment.orchestration.build_sweep_dataset_from_config") as mock_build:
                with patch("oasis.experiment.orchestration.run_learning_curve_experiments") as mock_run:
                    mock_build.return_value = SweepDataset(
                        mlip_features=np.array([[1.0]]),
                        targets=np.array([1.0]),
                        sample_ids=np.array(["s0"]),
                        graph_view=graph_view,
                    )
                    mock_run.return_value = LearningCurveResults.empty()

                    result = run_learning_curve_experiments_from_config(df, cfg=cfg)

        self.assertEqual(result, LearningCurveResults.empty())
        self.assertIs(mock_build.call_args.args[1], cfg)
        self.assertIsNone(mock_build.call_args.kwargs["graph_view"])
        self.assertIs(mock_run.call_args.args[0].graph_view, graph_view)
        self.assertFalse(mock_load.called)

    @unittest.skipUnless(HAS_POLARS, "requires polars")
    def test_run_learning_curve_experiments_from_config_runs_with_saved_graph_artifact(
        self,
    ) -> None:
        df = pd.DataFrame(
            {
                "reaction": ["s0", "s1", "s2", "s3", "s4", "s5"],
                "reference_ads_eng": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "ridge_mlip_ads_eng_median": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
            }
        )
        graph_view = GraphDatasetView.from_records(
            tuple(
                GraphRecord(
                    sample_id=f"s{idx}",
                    node_features=np.array([[float(idx + 1)]]),
                    edge_index=np.empty((2, 0), dtype=np.int64),
                    node_positions=np.array([[float(idx), 0.0, 0.0]]),
                )
                for idx in range(6)
            )
        )
        result_df = pd.DataFrame(
            {
                "n_train": [2, 3],
                "rmse_mean": [0.4, 0.3],
                "rmse_std": [0.05, 0.04],
            }
        )

        class RecordingFamily:
            def requirements(self) -> SweepFamilyRequirements:
                return SweepFamilyRequirements()

            def run(self, payload):
                self.last_payload = payload
                return LearningCurveResults.from_mapping({"ridge_df": result_df})

        family = RecordingFamily()

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / "aligned_graphs.parquet"
            save_aligned_graph_dataset_parquet(df, graph_view, artifact_path)
            cfg = SimpleNamespace(
                seed=23,
                plot=None,
                experiment=SimpleNamespace(
                    learning_curve=SimpleNamespace(
                        min_train=2,
                        max_train=3,
                        n_repeats=1,
                        validation_fraction=0.2,
                        min_val_size=1,
                        min_test_size=1,
                        graph_dataset=SimpleNamespace(
                            path=artifact_path,
                            join_key="reaction",
                        ),
                        models=None,
                    )
                ),
            )

            results = run_learning_curve_experiments_from_config(
                df,
                cfg=cfg,
                model_families=[family],
            )

        self.assertIs(results.ridge_df, result_df)
        dataset = family.last_payload.dataset
        self.assertTrue(dataset.has_graphs)
        np.testing.assert_array_equal(
            dataset.sample_ids,
            np.array(["s0", "s1", "s2", "s3", "s4", "s5"]),
        )
        self.assertEqual(
            dataset.graphs.sample_ids, ("s0", "s1", "s2", "s3", "s4", "s5")
        )
        np.testing.assert_allclose(
            dataset.graphs["s3"].node_positions,
            np.array([[3.0, 0.0, 0.0]]),
        )
        np.testing.assert_array_equal(
            dataset.mlip_features[:, 0],
            np.array([1.1, 2.1, 3.1, 4.1, 5.1, 6.1]),
        )
