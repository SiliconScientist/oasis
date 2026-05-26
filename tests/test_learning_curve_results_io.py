from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import pandas as pd

from oasis.learning_curve.results_io import (
    LearningCurveSweepMetadata,
    dump_learning_curve_method_artifact,
    dump_learning_curve_results,
    learning_curve_sweep_metadata_from_config,
    load_learning_curve_method_artifact,
    load_learning_curve_method_artifact_mapping,
    load_learning_curve_results_artifact,
    load_learning_curve_results,
    load_learning_curve_results_from_method_artifacts,
    load_learning_curve_results_mapping,
    save_learning_curve_method_artifacts,
    save_learning_curve_results_artifact,
    save_learning_curve_results,
)
from oasis.sweep import LearningCurveResults


class LearningCurveResultsIoTests(unittest.TestCase):
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
            restored = load_learning_curve_results_from_method_artifacts(
                tmp_dir,
                expected_metadata=metadata,
            )

        self.assertEqual(set(saved_paths), {"ridge", "moe"})
        self.assertTrue(saved_paths["ridge"].name.endswith("ridge.json"))
        pd.testing.assert_frame_equal(restored.ridge_df, results.ridge_df)
        pd.testing.assert_frame_equal(
            restored.ridge_selection_df,
            results.ridge_selection_df,
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

    def test_sweep_metadata_from_config_collects_enabled_models_and_filters(self) -> None:
        cfg = SimpleNamespace(
            seed=23,
            plot=SimpleNamespace(
                filters=SimpleNamespace(
                    adsorbate="OH",
                    anomaly_label="!inference_anomaly",
                    reaction_contains=["Pt", "", "Ni"],
                )
            ),
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

        metadata = learning_curve_sweep_metadata_from_config(cfg)

        self.assertEqual(metadata.seed, 23)
        self.assertEqual(metadata.min_train, 2)
        self.assertEqual(metadata.max_train, 6)
        self.assertEqual(metadata.step, 2)
        self.assertEqual(metadata.n_repeats, 4)
        self.assertEqual(metadata.enabled_models, ("moe", "residual", "ridge"))
        self.assertEqual(metadata.adsorbate_filter, "OH")
        self.assertEqual(metadata.anomaly_filter, "!inference_anomaly")
        self.assertEqual(metadata.reaction_contains_filter, ("Pt", "Ni"))

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
