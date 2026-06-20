from __future__ import annotations

from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import pandas as pd

from oasis.learning_curve.results_io import (
    LearningCurveSweepMetadata,
    build_learning_curve_point_provenance,
    dump_learning_curve_method_artifact,
    dump_learning_curve_results,
    learning_curve_uq_field_for_method_name,
    learning_curve_sweep_metadata_from_config,
    load_learning_curve_method_artifact,
    load_learning_curve_method_artifact_mapping,
    load_learning_curve_results_artifact,
    load_learning_curve_results,
    load_learning_curve_results_from_method_artifacts,
    load_learning_curve_results_mapping,
    select_learning_curve_results_methods,
    save_learning_curve_method_artifacts,
    save_learning_curve_results_artifact,
    save_learning_curve_results,
)
from oasis.sweep import LearningCurveResults


class LearningCurveResultsIoTests(unittest.TestCase):
    def test_select_learning_curve_results_methods_keeps_uq_companions(self) -> None:
        results = LearningCurveResults(
            ridge_df=pd.DataFrame(
                {
                    "n_train": [4, 8],
                    "rmse_mean": [0.41, 0.32],
                    "rmse_std": [0.06, 0.03],
                }
            ),
            ridge_selection_df=pd.DataFrame(
                {
                    "n_train": [4, 8],
                    "alpha": [0.1, 1.0],
                }
            ),
            ridge_uq_df=pd.DataFrame(
                {
                    "n_train": [4, 8],
                    "miscalibration_area": [0.13, 0.09],
                    "sharpness": [0.2, 0.16],
                    "dispersion": [0.4, 0.3],
                    "uncertainty_kind": ["spread_only", "spread_only"],
                }
            ),
            weighted_linear_df=pd.DataFrame(
                {
                    "n_train": [4, 8],
                    "rmse_mean": [0.35, 0.25],
                    "rmse_std": [0.04, 0.03],
                }
            ),
        )

        selected = select_learning_curve_results_methods(results, ("ridge",))

        pd.testing.assert_frame_equal(selected.ridge_df, results.ridge_df)
        pd.testing.assert_frame_equal(
            selected.ridge_selection_df,
            results.ridge_selection_df,
        )
        pd.testing.assert_frame_equal(selected.ridge_uq_df, results.ridge_uq_df)
        self.assertIsNone(selected.weighted_linear_df)

    def test_learning_curve_uq_field_for_method_name_returns_registered_uq_field(
        self,
    ) -> None:
        self.assertEqual(learning_curve_uq_field_for_method_name("ridge"), "ridge_uq_df")
        self.assertEqual(learning_curve_uq_field_for_method_name("moe"), "moe_uq_df")
        self.assertIsNone(learning_curve_uq_field_for_method_name("missing"))

    def test_round_trip_serialization_preserves_mixed_result_frames(self) -> None:
        results = LearningCurveResults(
            ridge_df=pd.DataFrame(
                {
                    "n_train": [4, 8],
                    "rmse_mean": [0.41, 0.32],
                    "rmse_std": [0.06, 0.03],
                }
            ),
            moe_df=pd.DataFrame(
                {
                    "n_train": [8],
                    "rmse_mean": [0.28],
                    "rmse_std": [0.02],
                }
            ),
            ridge_selection_df=pd.DataFrame(
                {
                    "n_train": [4, 8],
                    "alpha": [0.1, 1.0],
                    "best_validation_score": [0.52, 0.44],
                }
            ),
            ridge_uq_df=pd.DataFrame(
                {
                    "n_train": [4, 8],
                    "miscalibration_area": [0.13, 0.09],
                    "sharpness": [0.2, 0.16],
                    "dispersion": [0.4, 0.3],
                    "uncertainty_kind": ["spread_only", "spread_only"],
                }
            ),
            probe_gnn_selection_df=pd.DataFrame(
                {
                    "n_train": [8],
                    "hidden_dim": [32],
                    "aggregator": ["mean"],
                }
            ),
        )

        restored = load_learning_curve_results_mapping(
            dump_learning_curve_results(results)
        )

        self.assertIsNone(restored.lasso_df)
        pd.testing.assert_frame_equal(restored.ridge_df, results.ridge_df)
        pd.testing.assert_frame_equal(restored.moe_df, results.moe_df)
        pd.testing.assert_frame_equal(
            restored.ridge_selection_df,
            results.ridge_selection_df,
        )
        pd.testing.assert_frame_equal(
            restored.ridge_uq_df,
            results.ridge_uq_df,
        )
        pd.testing.assert_frame_equal(
            restored.probe_gnn_selection_df,
            results.probe_gnn_selection_df,
        )

    def test_save_and_load_learning_curve_results_bundle(self) -> None:
        results = LearningCurveResults(
            weighted_simplex_df=pd.DataFrame(
                {
                    "n_train": [3, 5],
                    "rmse_mean": [0.7, 0.55],
                    "rmse_std": [0.1, 0.08],
                }
            )
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "learning_curve_results.json"
            saved_path = save_learning_curve_results(results, path)
            restored = load_learning_curve_results(saved_path)

        self.assertEqual(saved_path, path)
        pd.testing.assert_frame_equal(
            restored.weighted_simplex_df,
            results.weighted_simplex_df,
        )

    def test_save_and_load_per_method_artifacts_with_metadata(self) -> None:
        results = LearningCurveResults(
            ridge_df=pd.DataFrame(
                {
                    "n_train": [4, 8],
                    "rmse_mean": [0.41, 0.32],
                    "rmse_std": [0.06, 0.03],
                }
            ),
            ridge_selection_df=pd.DataFrame(
                {
                    "n_train": [4, 8],
                    "alpha": [0.1, 1.0],
                }
            ),
            ridge_uq_df=pd.DataFrame(
                {
                    "n_train": [4, 8],
                    "miscalibration_area": [0.13, 0.09],
                    "sharpness": [0.2, 0.16],
                    "dispersion": [0.4, 0.3],
                    "uncertainty_kind": ["spread_only", "spread_only"],
                }
            ),
            moe_df=pd.DataFrame(
                {
                    "n_train": [8],
                    "rmse_mean": [0.28],
                    "rmse_std": [0.02],
                }
            ),
        )
        metadata = LearningCurveSweepMetadata(
            seed=17,
            min_train=2,
            max_train=8,
            step=2,
            n_repeats=3,
            enabled_models=("moe", "ridge"),
            dataset_tag="mamun_oh",
            dataset_size=42,
            adsorbate_filter="OH",
            anomaly_filter="!inference_anomaly",
            reaction_contains_filter=("Pt",),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            saved_paths = save_learning_curve_method_artifacts(
                results,
                metadata,
                tmp_dir,
            )
            ridge_artifact = load_learning_curve_method_artifact(
                saved_paths["ridge"],
                expected_metadata=metadata,
            )
            restored = load_learning_curve_results_from_method_artifacts(
                tmp_dir,
                expected_metadata=metadata,
            )

        self.assertEqual(set(saved_paths), {"ridge", "moe"})
        self.assertTrue(saved_paths["ridge"].name.endswith("ridge.json"))
        self.assertEqual(ridge_artifact.metadata.dataset_tag, "mamun_oh")
        self.assertEqual(ridge_artifact.metadata.dataset_size, 42)
        pd.testing.assert_frame_equal(restored.ridge_df, results.ridge_df)
        pd.testing.assert_frame_equal(
            restored.ridge_selection_df,
            results.ridge_selection_df,
        )
        pd.testing.assert_frame_equal(
            restored.ridge_uq_df,
            results.ridge_uq_df,
        )
        pd.testing.assert_frame_equal(restored.moe_df, results.moe_df)

    def test_incompatible_method_artifact_metadata_is_rejected(self) -> None:
        results = LearningCurveResults(
            ridge_df=pd.DataFrame(
                {
                    "n_train": [4],
                    "rmse_mean": [0.41],
                    "rmse_std": [0.06],
                }
            )
        )
        stored_metadata = LearningCurveSweepMetadata(
            seed=17,
            min_train=2,
            max_train=8,
            step=1,
            n_repeats=3,
            enabled_models=("ridge",),
        )
        expected_metadata = LearningCurveSweepMetadata(
            seed=19,
            min_train=2,
            max_train=8,
            step=1,
            n_repeats=3,
            enabled_models=("ridge",),
        )
        payload = dump_learning_curve_method_artifact(
            "ridge",
            results,
            stored_metadata,
        )

        with self.assertRaisesRegex(
            ValueError,
            "metadata is incompatible: seed: expected 19, got 17",
        ):
            load_learning_curve_method_artifact_mapping(
                payload,
                expected_metadata=expected_metadata,
            )

    def test_method_artifact_load_rejects_incompatible_directory_metadata(self) -> None:
        results = LearningCurveResults(
            ridge_df=pd.DataFrame(
                {
                    "n_train": [4],
                    "rmse_mean": [0.41],
                    "rmse_std": [0.06],
                }
            )
        )
        stored_metadata = LearningCurveSweepMetadata(
            seed=17,
            min_train=2,
            max_train=8,
            step=1,
            n_repeats=3,
            enabled_models=("ridge",),
        )
        expected_metadata = LearningCurveSweepMetadata(
            seed=17,
            min_train=2,
            max_train=8,
            step=1,
            n_repeats=3,
            enabled_models=("moe", "ridge"),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_learning_curve_method_artifacts(results, stored_metadata, tmp_dir)
            with self.assertRaisesRegex(
                ValueError,
                "enabled_models: expected \\['moe', 'ridge'\\], got \\['ridge'\\]",
            ):
                load_learning_curve_results_from_method_artifacts(
                    tmp_dir,
                    expected_metadata=expected_metadata,
                )

    def test_method_artifact_load_allows_enabled_model_superset(self) -> None:
        results = LearningCurveResults(
            ridge_df=pd.DataFrame(
                {
                    "n_train": [4],
                    "rmse_mean": [0.41],
                    "rmse_std": [0.06],
                }
            )
        )
        stored_metadata = LearningCurveSweepMetadata(
            seed=17,
            min_train=2,
            max_train=8,
            step=1,
            n_repeats=3,
            enabled_models=("ridge", "weighted_linear"),
        )
        expected_metadata = LearningCurveSweepMetadata(
            seed=17,
            min_train=2,
            max_train=8,
            step=1,
            n_repeats=3,
            enabled_models=("ridge",),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_learning_curve_method_artifacts(results, stored_metadata, tmp_dir)
            restored = load_learning_curve_results_from_method_artifacts(
                tmp_dir,
                expected_metadata=expected_metadata,
                allow_enabled_model_superset=True,
            )

        pd.testing.assert_frame_equal(restored.ridge_df, results.ridge_df)

    def test_results_bundle_artifact_can_ignore_enabled_models(self) -> None:
        results = LearningCurveResults(
            ridge_df=pd.DataFrame(
                {
                    "n_train": [4],
                    "rmse_mean": [0.41],
                    "rmse_std": [0.06],
                }
            )
        )
        stored_metadata = LearningCurveSweepMetadata(
            seed=17,
            min_train=2,
            max_train=8,
            step=1,
            n_repeats=3,
            enabled_models=("ridge",),
        )
        expected_metadata = LearningCurveSweepMetadata(
            seed=17,
            min_train=2,
            max_train=8,
            step=1,
            n_repeats=3,
            enabled_models=("weighted_simplex",),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = Path(tmp_dir) / "learning_curve_results.json"
            save_learning_curve_results_artifact(
                results,
                stored_metadata,
                bundle_path,
            )
            restored = load_learning_curve_results_artifact(
                bundle_path,
                expected_metadata=expected_metadata,
                ignore_enabled_models=True,
            )

        pd.testing.assert_frame_equal(restored.results.ridge_df, results.ridge_df)

    def test_results_bundle_artifact_can_ignore_train_grid(self) -> None:
        results = LearningCurveResults(
            ridge_df=pd.DataFrame(
                {
                    "n_train": [4],
                    "rmse_mean": [0.41],
                    "rmse_std": [0.06],
                }
            )
        )
        stored_metadata = LearningCurveSweepMetadata(
            seed=17,
            min_train=1,
            max_train=20,
            step=1,
            n_repeats=3,
            enabled_models=("ridge",),
            adsorbate_filter="OH",
        )
        expected_metadata = LearningCurveSweepMetadata(
            seed=17,
            min_train=20,
            max_train=40,
            step=5,
            n_repeats=3,
            enabled_models=("weighted_simplex",),
            adsorbate_filter="OH",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = Path(tmp_dir) / "learning_curve_results.json"
            save_learning_curve_results_artifact(
                results,
                stored_metadata,
                bundle_path,
            )
            restored = load_learning_curve_results_artifact(
                bundle_path,
                expected_metadata=expected_metadata,
                ignore_enabled_models=True,
                ignore_train_grid=True,
            )

        pd.testing.assert_frame_equal(restored.results.ridge_df, results.ridge_df)

    def test_results_bundle_artifact_can_ignore_repeat_count(self) -> None:
        results = LearningCurveResults(
            ridge_df=pd.DataFrame(
                {
                    "n_train": [4],
                    "rmse_mean": [0.41],
                    "rmse_std": [0.06],
                }
            )
        )
        stored_metadata = LearningCurveSweepMetadata(
            seed=17,
            min_train=1,
            max_train=20,
            step=1,
            n_repeats=30,
            enabled_models=("ridge",),
            adsorbate_filter="OH",
        )
        expected_metadata = LearningCurveSweepMetadata(
            seed=17,
            min_train=1,
            max_train=20,
            step=1,
            n_repeats=10,
            enabled_models=("weighted_simplex",),
            adsorbate_filter="OH",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = Path(tmp_dir) / "learning_curve_results.json"
            save_learning_curve_results_artifact(
                results,
                stored_metadata,
                bundle_path,
            )
            restored = load_learning_curve_results_artifact(
                bundle_path,
                expected_metadata=expected_metadata,
                ignore_enabled_models=True,
                ignore_repeat_count=True,
            )

        pd.testing.assert_frame_equal(restored.results.ridge_df, results.ridge_df)

    def test_results_bundle_artifact_allows_enabled_model_superset(self) -> None:
        results = LearningCurveResults(
            ridge_df=pd.DataFrame(
                {
                    "n_train": [4],
                    "rmse_mean": [0.41],
                    "rmse_std": [0.06],
                }
            ),
            weighted_linear_df=pd.DataFrame(
                {
                    "n_train": [4],
                    "rmse_mean": [0.33],
                    "rmse_std": [0.04],
                }
            ),
        )
        stored_metadata = LearningCurveSweepMetadata(
            seed=17,
            min_train=2,
            max_train=8,
            step=1,
            n_repeats=3,
            enabled_models=("ridge", "weighted_linear"),
        )
        expected_metadata = LearningCurveSweepMetadata(
            seed=17,
            min_train=2,
            max_train=8,
            step=1,
            n_repeats=3,
            enabled_models=("ridge",),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / "learning_curve_results.json"
            save_learning_curve_results_artifact(results, stored_metadata, artifact_path)
            restored = load_learning_curve_results_artifact(
                artifact_path,
                expected_metadata=expected_metadata,
                allow_enabled_model_superset=True,
            )

        pd.testing.assert_frame_equal(restored.results.ridge_df, results.ridge_df)

    def test_results_bundle_artifact_preserves_point_provenance(self) -> None:
        results = LearningCurveResults(
            ridge_df=pd.DataFrame(
                {
                    "n_train": [4, 8],
                    "rmse_mean": [0.41, 0.32],
                    "rmse_std": [0.06, 0.03],
                }
            ),
            ridge_selection_df=pd.DataFrame(
                {
                    "n_train": [4, 8],
                    "alpha": [0.1, 1.0],
                }
            ),
            ridge_uq_df=pd.DataFrame(
                {
                    "n_train": [4, 8],
                    "miscalibration_area": [0.13, 0.09],
                    "sharpness": [0.2, 0.16],
                    "dispersion": [0.4, 0.3],
                    "uncertainty_kind": ["spread_only", "spread_only"],
                }
            ),
        )
        metadata = LearningCurveSweepMetadata(
            seed=17,
            min_train=2,
            max_train=8,
            step=2,
            n_repeats=3,
            enabled_models=("ridge",),
        )
        point_provenance = build_learning_curve_point_provenance(
            results,
            metadata,
            run_id="run-a",
            run_timestamp_utc="2026-05-26T00:00:00+00:00",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / "learning_curve_results.json"
            save_learning_curve_results_artifact(
                results,
                metadata,
                artifact_path,
                point_provenance=point_provenance,
            )
            restored = load_learning_curve_results_artifact(artifact_path)

        pd.testing.assert_frame_equal(
            restored.point_provenance["ridge_df"],
            point_provenance["ridge_df"],
        )
        pd.testing.assert_frame_equal(
            restored.point_provenance["ridge_selection_df"],
            point_provenance["ridge_selection_df"],
        )
        pd.testing.assert_frame_equal(
            restored.point_provenance["ridge_uq_df"],
            point_provenance["ridge_uq_df"],
        )

    def test_single_method_artifact_file_round_trip_preserves_point_provenance(self) -> None:
        results = LearningCurveResults(
            ridge_df=pd.DataFrame(
                {
                    "n_train": [4],
                    "rmse_mean": [0.41],
                    "rmse_std": [0.06],
                }
            ),
            ridge_selection_df=pd.DataFrame({"n_train": [4], "alpha": [0.1]}),
        )
        metadata = LearningCurveSweepMetadata(
            seed=17,
            min_train=2,
            max_train=8,
            step=1,
            n_repeats=3,
            enabled_models=("ridge",),
        )
        point_provenance = build_learning_curve_point_provenance(
            results,
            metadata,
            run_id="run-b",
            run_timestamp_utc="2026-05-26T01:00:00+00:00",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_learning_curve_method_artifacts(
                results,
                metadata,
                tmp_dir,
                point_provenance=point_provenance,
            )
            artifact = load_learning_curve_method_artifact(
                Path(tmp_dir) / "ridge.json",
                expected_metadata=metadata,
            )

        pd.testing.assert_frame_equal(
            artifact.point_provenance["ridge_df"],
            point_provenance["ridge_df"],
        )
        pd.testing.assert_frame_equal(
            artifact.point_provenance["ridge_selection_df"],
            point_provenance["ridge_selection_df"],
        )

    def test_sweep_metadata_from_config_collects_enabled_models(self) -> None:
        cfg = SimpleNamespace(
            seed=23,
            dataset_profile=SimpleNamespace(tag="mamun_oh"),
            plot=None,
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(
                    min_train=2,
                    max_train=6,
                    step=2,
                    n_repeats=4,
                    models=SimpleNamespace(
                        use_ridge=True,
                        use_kernel_ridge=False,
                        use_lasso=False,
                        use_elastic_net=False,
                        use_residual=True,
                        use_weighted_linear=False,
                        use_weighted_simplex=False,
                        use_graph_mean=False,
                        use_latent=False,
                        moe=SimpleNamespace(enabled=True),
                        probe_gnn=SimpleNamespace(enabled=False),
                        gnn_direct=SimpleNamespace(enabled=False),
                    ),
                )
            ),
        )

        metadata = learning_curve_sweep_metadata_from_config(
            cfg,
            dataset_size=12,
            mlip_feature_names=("ridge", "linear"),
        )

        self.assertEqual(metadata.seed, 23)
        self.assertEqual(metadata.min_train, 2)
        self.assertEqual(metadata.max_train, 6)
        self.assertEqual(metadata.step, 2)
        self.assertEqual(metadata.n_repeats, 4)
        self.assertEqual(metadata.enabled_models, ("moe", "residual", "ridge"))
        self.assertEqual(metadata.dataset_tag, "mamun_oh")
        self.assertEqual(metadata.dataset_size, 12)
        self.assertEqual(metadata.mlip_feature_names, ("ridge", "linear"))

    def test_results_artifact_rejects_mismatched_mlip_feature_names(self) -> None:
        results = LearningCurveResults(
            ridge_df=pd.DataFrame(
                {
                    "n_train": [4],
                    "rmse_mean": [0.41],
                    "rmse_std": [0.06],
                }
            )
        )
        stored_metadata = LearningCurveSweepMetadata(
            seed=17,
            min_train=2,
            max_train=8,
            step=1,
            n_repeats=3,
            enabled_models=("ridge",),
            mlip_feature_names=("mace", "uma"),
        )
        expected_metadata = LearningCurveSweepMetadata(
            seed=17,
            min_train=2,
            max_train=8,
            step=1,
            n_repeats=3,
            enabled_models=("ridge",),
            mlip_feature_names=("mace", "orb"),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / "learning_curve_results.json"
            save_learning_curve_results_artifact(results, stored_metadata, artifact_path)
            with self.assertRaisesRegex(
                ValueError,
                "mlip_feature_names: expected \\['mace', 'orb'\\], got \\['mace', 'uma'\\]",
            ):
                load_learning_curve_results_artifact(
                    artifact_path,
                    expected_metadata=expected_metadata,
                )

    def test_sweep_metadata_from_config_includes_screening_settings(self) -> None:
        cfg = SimpleNamespace(
            seed=23,
            dataset_profile=None,
            plot=None,
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(
                    min_train=4,
                    max_train=8,
                    step=2,
                    n_repeats=4,
                    budget_mode="screening_fraction",
                    screen_fraction=0.25,
                    min_screen_size=2,
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

        metadata = learning_curve_sweep_metadata_from_config(cfg, dataset_size=12)

        self.assertEqual(metadata.budget_mode, "screening_fraction")
        self.assertEqual(metadata.screen_fraction, 0.25)
        self.assertEqual(metadata.min_screen_size, 2)

    def test_sweep_metadata_from_config_records_explicit_fraction_sweep(self) -> None:
        cfg = SimpleNamespace(
            seed=23,
            dataset_profile=None,
            plot=None,
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(
                    min_train=None,
                    max_train=None,
                    step=1,
                    sweep_fractions=[0.1, 0.3, 0.9],
                    sweep_sizes=[],
                    n_repeats=4,
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

        metadata = learning_curve_sweep_metadata_from_config(cfg, dataset_size=20)

        self.assertEqual(metadata.min_train, 2)
        self.assertEqual(metadata.max_train, 18)
        self.assertEqual(metadata.step, 0)
        self.assertEqual(metadata.requested_sweep_sizes, (2, 6, 18))
        self.assertEqual(metadata.requested_sweep_fractions, (0.1, 0.3, 0.9))

    def test_point_provenance_preserves_screening_columns(self) -> None:
        results = LearningCurveResults(
            ridge_df=pd.DataFrame(
                {
                    "n_budget": [5, 6],
                    "n_train": [3, 4],
                    "n_screen": [2, 2],
                    "screen_fraction": [0.4, 1 / 3],
                    "rmse_mean": [0.41, 0.32],
                    "rmse_std": [0.06, 0.03],
                }
            )
        )
        metadata = LearningCurveSweepMetadata(
            seed=17,
            min_train=4,
            max_train=8,
            step=1,
            n_repeats=3,
            enabled_models=("ridge",),
            budget_mode="screening_fraction",
            screen_fraction=0.25,
            min_screen_size=2,
        )

        provenance = build_learning_curve_point_provenance(results, metadata)

        self.assertEqual(provenance["ridge_df"]["n_budget"].tolist(), [5, 6])
        self.assertEqual(provenance["ridge_df"]["n_screen"].tolist(), [2, 2])
        self.assertNotIn("n_repeats", provenance["ridge_df"].columns)

    def test_results_bundle_artifact_screening_provenance_omits_repeat_count(self) -> None:
        results = LearningCurveResults(
            ridge_df=pd.DataFrame(
                {
                    "n_budget": [5],
                    "n_train": [3],
                    "n_screen": [2],
                    "screen_fraction": [0.4],
                    "cv_rmse_mean": [0.41],
                    "cv_rmse_std": [0.06],
                }
            )
        )
        metadata = LearningCurveSweepMetadata(
            seed=17,
            min_train=4,
            max_train=8,
            step=1,
            n_repeats=3,
            enabled_models=("ridge",),
            budget_mode="screening_fraction",
            screen_fraction=0.25,
            min_screen_size=2,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = Path(tmp_dir) / "screening_results.json"
            save_learning_curve_results_artifact(results, metadata, bundle_path)
            payload = json.loads(bundle_path.read_text(encoding="utf-8"))
            restored = load_learning_curve_results_artifact(bundle_path)

        stored_provenance = pd.read_json(
            StringIO(payload["point_provenance"]["ridge_df"]),
            orient="table",
        )
        self.assertNotIn("n_repeats", stored_provenance.columns)
        self.assertIn("n_repeats", restored.point_provenance["ridge_df"].columns)
        self.assertTrue(restored.point_provenance["ridge_df"]["n_repeats"].isna().all())

    def test_single_method_artifact_file_round_trip(self) -> None:
        results = LearningCurveResults(
            ridge_df=pd.DataFrame(
                {
                    "n_train": [4],
                    "rmse_mean": [0.41],
                    "rmse_std": [0.06],
                }
            ),
            ridge_selection_df=pd.DataFrame({"n_train": [4], "alpha": [0.1]}),
            moe_df=pd.DataFrame(
                {
                    "n_train": [8],
                    "rmse_mean": [0.28],
                    "rmse_std": [0.02],
                }
            ),
        )
        metadata = LearningCurveSweepMetadata(
            seed=17,
            min_train=2,
            max_train=8,
            step=1,
            n_repeats=3,
            enabled_models=("ridge", "moe"),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_learning_curve_method_artifacts(results, metadata, tmp_dir)
            artifact = load_learning_curve_method_artifact(
                Path(tmp_dir) / "ridge.json",
                expected_metadata=metadata,
            )

        self.assertEqual(artifact.method_name, "ridge")
        self.assertEqual(artifact.metadata, metadata)
        pd.testing.assert_frame_equal(artifact.results.ridge_df, results.ridge_df)
        pd.testing.assert_frame_equal(
            artifact.results.ridge_selection_df,
            results.ridge_selection_df,
        )
        self.assertIsNone(artifact.results.moe_df)
