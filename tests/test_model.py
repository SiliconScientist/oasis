from __future__ import annotations

import unittest

from ase import Atoms
import polars as pl
import torch

from oasis.model import (
    BaselineMLPGatedMoE,
    MLPGatingNetwork,
    MixtureOfExpertsPredictionHead,
    SchNetGatedMoE,
    SchNetStructureEncoder,
    weighted_expert_prediction,
)
from oasis.graph import batch_adsorption_graphs, build_adsorption_graphs


class BaselineModelTests(unittest.TestCase):
    def _example_graph_batch(self):
        wide_df = pl.DataFrame(
            {
                "reaction": ["CuOH_* -> OH*", "CuH_* -> H*"],
                "adsorbate": ["OH", "H"],
                "reference_ads_eng": [-1.25, -0.45],
                "mace_mlip_ads_eng_median": [-1.10, -0.40],
                "mace_label": ["normal", "normal"],
                "orb_mlip_ads_eng_median": [-1.30, -0.50],
                "orb_label": ["normal", "energy_anomaly"],
            }
        )
        atoms_list = [
            Atoms(
                "Cu2OH",
                positions=[
                    [0.0, 0.0, 0.0],
                    [2.5, 0.0, 0.0],
                    [1.2, 0.1, 1.3],
                    [1.2, 0.1, 2.2],
                ],
                cell=[12.0, 12.0, 12.0],
                pbc=[False, False, False],
            ),
            Atoms(
                "Cu2H",
                positions=[
                    [0.0, 0.0, 0.0],
                    [2.5, 0.0, 0.0],
                    [1.3, 0.0, 1.1],
                ],
                cell=[12.0, 12.0, 12.0],
                pbc=[False, False, False],
            ),
        ]
        graphs = build_adsorption_graphs(wide_df, atoms_list, cutoff=3.0)
        return batch_adsorption_graphs(graphs)

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

    def test_moe_prediction_head_normalizes_logits_to_simplex(self) -> None:
        head = MixtureOfExpertsPredictionHead()
        logits = torch.tensor(
            [[0.0, 1.0, -1.0], [2.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        mlip_energies = torch.tensor(
            [[-1.0, -0.8, -1.2], [-0.3, -0.4, -0.2]],
            dtype=torch.float32,
        )

        output = head(logits, mlip_energies)
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
            torch.all(output.weights >= 0.0).item()
        )

    def test_schnet_structure_encoder_output_shape(self) -> None:
        graph_batch = self._example_graph_batch()
        encoder = SchNetStructureEncoder(
            hidden_dim=16,
            n_interactions=2,
            n_rbf=8,
            cutoff=3.0,
        )
        embedding = encoder(graph_batch)
        self.assertEqual(tuple(embedding.shape), (2, 16))

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

    def test_schnet_gated_moe_forward_shapes(self) -> None:
        graph_batch = self._example_graph_batch()
        model = SchNetGatedMoE(
            n_experts=2,
            structure_hidden_dim=16,
            n_interactions=2,
            n_rbf=8,
            cutoff=3.0,
            gate_hidden_dims=(16,),
        )
        output = model(graph_batch, graph_batch.mlip_energies)
        self.assertEqual(tuple(output.logits.shape), (2, 2))
        self.assertEqual(tuple(output.weights.shape), (2, 2))
        self.assertEqual(tuple(output.prediction.shape), (2,))
        self.assertTrue(
            torch.allclose(
                output.weights.sum(dim=-1),
                torch.ones(2, dtype=torch.float32),
                atol=1e-6,
            )
        )


if __name__ == "__main__":
    unittest.main()
