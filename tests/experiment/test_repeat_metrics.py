from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from oasis.experiment.repeat_metrics import (
    LearningCurveRepeatMetricsArtifact,
    load_learning_curve_repeat_metrics_artifact,
    merge_learning_curve_repeat_metrics,
    repeat_metrics_artifact_path,
    save_learning_curve_repeat_metrics_artifact,
)
from oasis.learning_curve.results_io import LearningCurveSweepMetadata


class LearningCurveRepeatMetricsTests(unittest.TestCase):
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

    def test_repeat_metrics_artifact_path_uses_bundle_stem(self) -> None:
        self.assertEqual(
            repeat_metrics_artifact_path("data/results/example.json"),
            Path("data/results/example_repeat_metrics.json"),
        )

    def test_merge_learning_curve_repeat_metrics_deduplicates_by_method_budget_repeat(
        self,
    ) -> None:
        left = pd.DataFrame(
            {
                "method": ["ridge", "ridge"],
                "budget": [4, 8],
                "repeat": [0, 0],
                "outer_test_rmse": [0.31, 0.21],
            }
        )
        right = pd.DataFrame(
            {
                "method": ["ridge", "weighted_linear"],
                "budget": [4, 8],
                "repeat": [0, 0],
                "outer_test_rmse": [0.29, 0.2],
            }
        )

        merged = merge_learning_curve_repeat_metrics(left, right)

        assert merged is not None
        self.assertEqual(merged["method"].tolist(), ["ridge", "ridge", "weighted_linear"])
        self.assertEqual(merged["outer_test_rmse"].tolist(), [0.29, 0.21, 0.2])

    def test_repeat_metrics_artifact_round_trip(self) -> None:
        artifact = LearningCurveRepeatMetricsArtifact(
            metadata=self._metadata(),
            repeat_metrics_df=pd.DataFrame(
                {
                    "method": ["weighted_linear", "ridge", "ridge"],
                    "budget": [8, 4, 8],
                    "repeat": [0, 0, 1],
                    "outer_test_rmse": [0.2, 0.31, 0.24],
                }
            ),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "repeat_metrics.json"
            saved_path = save_learning_curve_repeat_metrics_artifact(artifact, path)
            restored = load_learning_curve_repeat_metrics_artifact(
                saved_path,
                expected_metadata=self._metadata(),
            )

        self.assertEqual(saved_path, path)
        self.assertEqual(restored.metadata, artifact.metadata)
        pd.testing.assert_frame_equal(restored.repeat_metrics_df, artifact.repeat_metrics_df)
