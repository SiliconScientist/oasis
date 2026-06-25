from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import unittest

import numpy as np

from oasis.config import MoETrainingConfig
from oasis.learning_curve.families.gnn_gate import GnnGateTuningSpec
from oasis.sweep import (
    GraphDatasetView,
    GraphRecord,
    SweepDataset,
    SweepFamilyRequirements,
    SweepRunnerPayload,
    TrainValCalTestSweepRunnerInput,
)
from oasis.tune import sweep_learned_trial_model_selection


def _make_graph(sample_id: int) -> GraphRecord:
    node_features = np.array(
        [[float(sample_id), 0.0], [float(sample_id) + 1.0, 1.0]],
        dtype=np.float32,
    )
    edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)
    edge_features = np.ones((2, 1), dtype=np.float32)
    return GraphRecord(
        sample_id=sample_id,
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
    )


@dataclass
class _GnnMockTrial:
    lr: float = 1e-3
    weight_decay: float = 1e-4

    def __post_init__(self) -> None:
        self.params: dict[str, Any] = {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
        }

    def suggest_float(self, name: str, low: float, high: float, **kwargs: Any) -> float:
        del low, high, kwargs
        return self.params[name]


class MoeCalibrationTests(unittest.TestCase):
    def test_gnn_gate_learned_selection_marks_calibrated_uq_when_calibration_split_present(
        self,
    ) -> None:
        n_samples = 7
        dataset = SweepDataset(
            mlip_features=np.array(
                [
                    [0.0, 2.0],
                    [1.0, 3.0],
                    [2.0, 4.0],
                    [3.0, 5.0],
                    [4.0, 6.0],
                    [5.0, 7.0],
                    [6.0, 8.0],
                ],
                dtype=np.float32,
            ),
            targets=np.array([0.5, 1.4, 2.6, 3.5, 4.7, 5.5, 6.6], dtype=np.float32),
            sample_ids=np.arange(n_samples),
            graph_view=GraphDatasetView.from_records([_make_graph(i) for i in range(n_samples)]),
        )
        split = TrainValCalTestSweepRunnerInput(
            dataset=dataset,
            sweep_size=5,
            train_idx=np.array([0, 1, 2]),
            val_idx=np.array([3]),
            cal_idx=np.array([4]),
            test_idx=np.array([5, 6]),
        )
        payload = SweepRunnerPayload(
            splits=(split,),
            planning_requirements=SweepFamilyRequirements(
                requires_inner_validation=True,
                requires_calibration=True,
            ),
        )
        spec = GnnGateTuningSpec(
            training_cfg=MoETrainingConfig(epochs=2, seed=0, device="cpu"),
            hidden_dims=(8,),
        )

        result = sweep_learned_trial_model_selection(
            payload,
            spec,
            trial_factory=lambda split: (_GnnMockTrial(),),
        )

        assert result.uq_summary is not None
        self.assertEqual(result.uq_summary["uncertainty_kind"].tolist(), ["calibrated"])
        self.assertTrue(
            (result.uq_summary["uncertainty_note"] == "post-hoc scalar calibrated spread").all()
        )


if __name__ == "__main__":
    unittest.main()
