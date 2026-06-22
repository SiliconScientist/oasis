from __future__ import annotations

import unittest

import pandas as pd

from oasis.learning_curve.time_accuracy import (
    GenerationTimingAggregate,
    aggregate_generation_timing,
    build_fixed_split_time_accuracy_table,
    build_time_accuracy_table,
)
from oasis.mlip.timing import MlipGenerationTimingSummary
from oasis.sweep import LearningCurveResults


class LearningCurveTimeAccuracyTests(unittest.TestCase):
    def test_aggregate_generation_timing_sums_selected_mlips(self) -> None:
        aggregate = aggregate_generation_timing(
            {
                "mace": MlipGenerationTimingSummary(
                    model_name="mace",
                    reaction_count=2,
                    generation_time_total_s=10.0,
                    generation_time_slab_s=2.0,
                    generation_time_adslab_s=8.0,
                    generation_steps_total=20,
                    generation_steps_slab=4,
                    generation_steps_adslab=16,
                    time_per_step_s=0.5,
                ),
                "orb": MlipGenerationTimingSummary(
                    model_name="orb",
                    reaction_count=2,
                    generation_time_total_s=5.0,
                    generation_time_slab_s=1.0,
                    generation_time_adslab_s=4.0,
                    generation_steps_total=10,
                    generation_steps_slab=2,
                    generation_steps_adslab=8,
                    time_per_step_s=0.5,
                ),
            },
            mlip_feature_names=("mace", "orb"),
        )

        self.assertEqual(aggregate.generation_time_s, 15.0)
        self.assertEqual(aggregate.generation_time_slab_s, 3.0)
        self.assertEqual(aggregate.generation_time_adslab_s, 12.0)
        self.assertEqual(aggregate.generation_steps_total, 30)
        self.assertEqual(aggregate.time_per_step_s, 0.5)

    def test_build_time_accuracy_table_combines_generation_and_training_timings(
        self,
    ) -> None:
        results = LearningCurveResults(
            ridge_df=pd.DataFrame(
                {
                    "n_train": [4, 8],
                    "rmse_mean": [0.41, 0.32],
                    "rmse_std": [0.06, 0.03],
                    "fit_time_mean_s": [0.12, 0.34],
                    "fit_time_std_s": [0.01, 0.02],
                }
            ),
            weighted_linear_df=pd.DataFrame(
                {
                    "n_train": [4, 8],
                    "rmse_mean": [0.35, 0.25],
                    "rmse_std": [0.04, 0.03],
                    "fit_time_mean_s": [0.09, 0.18],
                    "fit_time_std_s": [0.005, 0.01],
                }
            ),
            graph_mean_df=pd.DataFrame(
                {
                    "n_train": [4, 8],
                    "rmse_mean": [0.5, 0.4],
                    "rmse_std": [0.07, 0.05],
                }
            ),
            probe_gnn_df=pd.DataFrame(
                {
                    "n_train": [4],
                    "rmse_mean": [0.2],
                    "rmse_std": [0.02],
                    "fit_time_mean_s": [9.9],
                    "fit_time_std_s": [0.1],
                }
            ),
            latent_df=pd.DataFrame(
                {
                    "n_train": [4],
                    "rmse_mean": [0.3],
                    "rmse_std": [0.03],
                    "fit_time_mean_s": [8.8],
                    "fit_time_std_s": [0.2],
                }
            ),
        )
        generation_summaries = {
            "mace": MlipGenerationTimingSummary(
                model_name="mace",
                reaction_count=2,
                generation_time_total_s=10.0,
                generation_time_slab_s=2.0,
                generation_time_adslab_s=8.0,
                generation_steps_total=20,
                generation_steps_slab=4,
                generation_steps_adslab=16,
                time_per_step_s=0.5,
            ),
            "orb": MlipGenerationTimingSummary(
                model_name="orb",
                reaction_count=2,
                generation_time_total_s=5.0,
                generation_time_slab_s=1.0,
                generation_time_adslab_s=4.0,
                generation_steps_total=10,
                generation_steps_slab=2,
                generation_steps_adslab=8,
                time_per_step_s=0.5,
            ),
        }

        table = build_time_accuracy_table(
            results,
            generation_summaries,
            mlip_feature_names=("mace", "orb"),
        )

        self.assertEqual(
            table.columns.tolist(),
            [
                "method",
                "n_train",
                "rmse_mean",
                "generation_time_s",
                "training_time_s",
                "total_time_s",
            ],
        )
        self.assertEqual(table["method"].tolist(), ["ridge", "ridge", "weighted_linear", "weighted_linear"])
        self.assertEqual(table["n_train"].tolist(), [4, 8, 4, 8])
        self.assertEqual(table["generation_time_s"].tolist(), [15.0, 15.0, 15.0, 15.0])
        self.assertEqual(table["training_time_s"].tolist(), [0.12, 0.34, 0.09, 0.18])
        self.assertEqual(table["total_time_s"].tolist(), [15.12, 15.34, 15.09, 15.18])

    def test_build_time_accuracy_table_raises_when_selected_mlip_is_missing(self) -> None:
        results = LearningCurveResults(
            ridge_df=pd.DataFrame(
                {
                    "n_train": [4],
                    "rmse_mean": [0.41],
                    "rmse_std": [0.06],
                    "fit_time_mean_s": [0.12],
                    "fit_time_std_s": [0.01],
                }
            )
        )
        generation_summaries = {
            "mace": MlipGenerationTimingSummary(
                model_name="mace",
                reaction_count=2,
                generation_time_total_s=10.0,
                generation_time_slab_s=2.0,
                generation_time_adslab_s=8.0,
                generation_steps_total=20,
                generation_steps_slab=4,
                generation_steps_adslab=16,
                time_per_step_s=0.5,
            )
        }

        with self.assertRaisesRegex(
            KeyError,
            "missing generation timing summaries for MLIP features",
        ):
            build_time_accuracy_table(
                results,
                generation_summaries,
                mlip_feature_names=("mace", "orb"),
            )

    def test_build_time_accuracy_table_uses_method_specific_generation_timing(
        self,
    ) -> None:
        results = LearningCurveResults(
            probe_gnn_df=pd.DataFrame(
                {
                    "n_train": [4],
                    "rmse_mean": [0.2],
                    "rmse_std": [0.02],
                    "fit_time_mean_s": [9.9],
                    "fit_time_std_s": [0.1],
                }
            ),
            latent_df=pd.DataFrame(
                {
                    "n_train": [4],
                    "rmse_mean": [0.3],
                    "rmse_std": [0.03],
                    "fit_time_mean_s": [8.8],
                    "fit_time_std_s": [0.2],
                }
            ),
        )

        table = build_time_accuracy_table(
            results,
            {},
            generation_timing_by_method={
                "probe_gnn": GenerationTimingAggregate(
                    generation_time_s=7.0,
                    generation_time_slab_s=2.0,
                    generation_time_adslab_s=5.0,
                    generation_steps_total=14,
                    generation_steps_slab=4,
                    generation_steps_adslab=10,
                    time_per_step_s=0.5,
                    mlip_feature_names=("probe_features",),
                ),
                "latent": GenerationTimingAggregate(
                    generation_time_s=11.0,
                    generation_time_slab_s=0.0,
                    generation_time_adslab_s=0.0,
                    generation_steps_total=0,
                    generation_steps_slab=0,
                    generation_steps_adslab=0,
                    time_per_step_s=None,
                    mlip_feature_names=("latent_csv",),
                ),
            },
            method_names=("probe_gnn", "latent"),
        )

        self.assertEqual(table["method"].tolist(), ["probe_gnn", "latent"])
        self.assertEqual(table["generation_time_s"].tolist(), [7.0, 11.0])
        self.assertEqual(table["training_time_s"].tolist(), [9.9, 8.8])
        self.assertEqual(table["total_time_s"].tolist(), [16.9, 19.8])

    def test_build_fixed_split_time_accuracy_table_selects_one_benchmark_point_per_method(
        self,
    ) -> None:
        results = LearningCurveResults(
            ridge_df=pd.DataFrame(
                {
                    "n_train": [4, 8],
                    "rmse_mean": [0.41, 0.32],
                    "rmse_std": [0.06, 0.03],
                    "fit_time_mean_s": [0.12, 0.34],
                    "fit_time_std_s": [0.01, 0.02],
                }
            ),
            weighted_linear_df=pd.DataFrame(
                {
                    "n_train": [4, 8],
                    "rmse_mean": [0.35, 0.25],
                    "rmse_std": [0.04, 0.03],
                    "fit_time_mean_s": [0.09, 0.18],
                    "fit_time_std_s": [0.005, 0.01],
                }
            ),
            graph_mean_df=pd.DataFrame(
                {
                    "n_train": [8],
                    "rmse_mean": [0.5],
                    "rmse_std": [0.07],
                }
            ),
        )
        generation_summaries = {
            "mace": MlipGenerationTimingSummary(
                model_name="mace",
                reaction_count=2,
                generation_time_total_s=10.0,
                generation_time_slab_s=2.0,
                generation_time_adslab_s=8.0,
                generation_steps_total=20,
                generation_steps_slab=4,
                generation_steps_adslab=16,
                time_per_step_s=0.5,
            ),
            "orb": MlipGenerationTimingSummary(
                model_name="orb",
                reaction_count=2,
                generation_time_total_s=5.0,
                generation_time_slab_s=1.0,
                generation_time_adslab_s=4.0,
                generation_steps_total=10,
                generation_steps_slab=2,
                generation_steps_adslab=8,
                time_per_step_s=0.5,
            ),
        }

        table = build_fixed_split_time_accuracy_table(
            results,
            generation_summaries,
            dataset_size=10,
            train_fraction=0.8,
            mlip_feature_names=("mace", "orb"),
        )

        self.assertEqual(
            table.columns.tolist(),
            [
                "method",
                "n_train",
                "rmse_mean",
                "rmse_std",
                "generation_time_s",
                "training_time_mean_s",
                "training_time_std_s",
                "total_time_mean_s",
                "total_time_std_s",
            ],
        )
        self.assertEqual(table["method"].tolist(), ["ridge", "weighted_linear"])
        self.assertEqual(table["n_train"].tolist(), [8, 8])
        self.assertEqual(table["rmse_mean"].tolist(), [0.32, 0.25])
        self.assertEqual(table["rmse_std"].tolist(), [0.03, 0.03])
        self.assertEqual(table["generation_time_s"].tolist(), [15.0, 15.0])
        self.assertEqual(table["training_time_mean_s"].tolist(), [0.34, 0.18])
        self.assertEqual(table["training_time_std_s"].tolist(), [0.02, 0.01])
        self.assertEqual(table["total_time_mean_s"].tolist(), [15.34, 15.18])
        self.assertEqual(table["total_time_std_s"].tolist(), [0.02, 0.01])

    def test_build_fixed_split_time_accuracy_table_uses_method_specific_generation_timing(
        self,
    ) -> None:
        results = LearningCurveResults(
            probe_gnn_df=pd.DataFrame(
                {
                    "n_train": [8],
                    "rmse_mean": [0.2],
                    "rmse_std": [0.02],
                    "fit_time_mean_s": [9.9],
                    "fit_time_std_s": [0.1],
                }
            ),
            latent_df=pd.DataFrame(
                {
                    "n_train": [8],
                    "rmse_mean": [0.3],
                    "rmse_std": [0.03],
                    "fit_time_mean_s": [8.8],
                    "fit_time_std_s": [0.2],
                }
            ),
        )

        table = build_fixed_split_time_accuracy_table(
            results,
            {},
            dataset_size=10,
            train_fraction=0.8,
            generation_timing_by_method={
                "probe_gnn": GenerationTimingAggregate(
                    generation_time_s=7.0,
                    generation_time_slab_s=2.0,
                    generation_time_adslab_s=5.0,
                    generation_steps_total=14,
                    generation_steps_slab=4,
                    generation_steps_adslab=10,
                    time_per_step_s=0.5,
                    mlip_feature_names=("probe_features",),
                ),
                "latent": GenerationTimingAggregate(
                    generation_time_s=11.0,
                    generation_time_slab_s=0.0,
                    generation_time_adslab_s=0.0,
                    generation_steps_total=0,
                    generation_steps_slab=0,
                    generation_steps_adslab=0,
                    time_per_step_s=None,
                    mlip_feature_names=("latent_csv",),
                ),
            },
            method_names=("probe_gnn", "latent"),
        )

        self.assertEqual(table["method"].tolist(), ["probe_gnn", "latent"])
        self.assertEqual(table["generation_time_s"].tolist(), [7.0, 11.0])
        self.assertEqual(table["training_time_mean_s"].tolist(), [9.9, 8.8])
        self.assertEqual(table["total_time_mean_s"].tolist(), [16.9, 19.8])
