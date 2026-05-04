from __future__ import annotations

import unittest

import torch

from oasis.model import (
    BaselineMLPGatedMoE,
    MLPGatingNetwork,
    weighted_expert_prediction,
)


class BaselineModelTests(unittest.TestCase):
    def test_mlp_gate_output_shape(self) -> None:
        gate = MLPGatingNetwork(input_dim=3, hidden_dims=(8,))
        mlip_energies = torch.tensor(
            [[-1.0, -0.8, -1.2], [-0.3, -0.4, -0.2]],
            dtype=torch.float32,
        )

        logits = gate(mlip_energies)
        self.assertEqual(tuple(logits.shape), (2, 3))

    def test_weighted_prediction_matches_manual_sum(self) -> None:
        mlip_energies = torch.tensor(
            [[-1.0, -0.8, -1.2], [-0.3, -0.4, -0.2]],
            dtype=torch.float32,
        )
        weights = torch.tensor(
            [[0.2, 0.5, 0.3], [0.1, 0.2, 0.7]],
            dtype=torch.float32,
        )

        prediction = weighted_expert_prediction(mlip_energies, weights)
        expected = torch.tensor(
            [
                (-1.0 * 0.2) + (-0.8 * 0.5) + (-1.2 * 0.3),
                (-0.3 * 0.1) + (-0.4 * 0.2) + (-0.2 * 0.7),
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(prediction, expected))

    def test_baseline_moe_outputs_normalized_weights(self) -> None:
        model = BaselineMLPGatedMoE(n_experts=3, hidden_dims=(8, 8))
        mlip_energies = torch.tensor(
            [[-1.0, -0.8, -1.2], [-0.3, -0.4, -0.2]],
            dtype=torch.float32,
        )

        output = model(mlip_energies)
        self.assertEqual(tuple(output.logits.shape), (2, 3))
        self.assertEqual(tuple(output.weights.shape), (2, 3))
        self.assertEqual(tuple(output.prediction.shape), (2,))
        self.assertTrue(
            torch.allclose(
                output.weights.sum(dim=-1),
                torch.ones(2, dtype=torch.float32),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                output.prediction,
                weighted_expert_prediction(mlip_energies, output.weights),
            )
        )


if __name__ == "__main__":
    unittest.main()
