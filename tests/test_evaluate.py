from __future__ import annotations

import math
import unittest

from ase import Atoms
import polars as pl
import torch

from oasis.dataset import GatingDataset, collate_gating_samples
from oasis.evaluate import (
    default_model_baselines,
    evaluate_baselines,
    evaluate_models_and_baselines,
    mean_ensemble_prediction,
)
from oasis.graph import build_adsorption_graphs
from oasis.train import build_gating_dataloaders


def _example_dataset() -> GatingDataset:
    wide_df = pl.DataFrame(
        {
            "reaction": [
                "CuOH_* -> OH*",
                "CuH_* -> H*",
                "NiOH_* -> OH*",
                "NiH_* -> H*",
            ],
            "adsorbate": ["OH", "H", "OH", "H"],
            "reference_ads_eng": [-1.25, -0.45, -1.10, -0.35],
            "mace_mlip_ads_eng_median": [-1.10, -0.40, -1.05, -0.30],
            "mace_label": ["normal", "normal", "normal", "normal"],
            "orb_mlip_ads_eng_median": [-1.30, -0.50, -1.20, -0.45],
            "orb_label": ["normal", "energy_anomaly", "normal", "normal"],
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
        Atoms(
            "Ni2OH",
            positions=[
                [0.0, 0.0, 0.0],
                [2.4, 0.0, 0.0],
                [1.1, 0.1, 1.2],
                [1.1, 0.1, 2.1],
            ],
            cell=[12.0, 12.0, 12.0],
            pbc=[False, False, False],
        ),
        Atoms(
            "Ni2H",
            positions=[
                [0.0, 0.0, 0.0],
                [2.4, 0.0, 0.0],
                [1.2, 0.0, 1.0],
            ],
            cell=[12.0, 12.0, 12.0],
            pbc=[False, False, False],
        ),
    ]
    graphs = build_adsorption_graphs(wide_df, atoms_list, cutoff=3.0)
    return GatingDataset(graphs, wide_df)


class EvaluationTests(unittest.TestCase):
    def test_mean_ensemble_prediction(self) -> None:
        mlip_energies = torch.tensor(
            [[-1.0, -0.8], [-0.4, -0.2]],
            dtype=torch.float32,
        )
        prediction = mean_ensemble_prediction(mlip_energies)
        self.assertTrue(
            torch.allclose(
                prediction,
                torch.tensor([-0.9, -0.3], dtype=torch.float32),
            )
        )

    def test_evaluate_baselines(self) -> None:
        dataset = _example_dataset()
        batch = collate_gating_samples([dataset[i] for i in range(len(dataset))])
        evaluation = evaluate_baselines(batch)

        self.assertEqual(
            evaluation.best_single_expert_name,
            "orb_mlip_ads_eng_median",
        )
        self.assertTrue(
            math.isclose(
                evaluation.best_single_expert.rmse,
                0.07905693914812899,
                rel_tol=1e-6,
            )
        )
        self.assertTrue(
            math.isclose(
                evaluation.mean_ensemble.rmse,
                0.030618599580468097,
                rel_tol=1e-6,
            )
        )

    def test_evaluate_models_and_baselines(self) -> None:
        dataset = _example_dataset()
        batch = collate_gating_samples([dataset[i] for i in range(len(dataset))])
        train_loader, _ = build_gating_dataloaders(
            dataset,
            batch_size=2,
            val_fraction=0.25,
            seed=7,
        )
        models = default_model_baselines(n_experts=2)
        report = evaluate_models_and_baselines(
            batch,
            data_loader=train_loader,
            models=models,
            device="cpu",
        )

        self.assertEqual(len(report.models), 2)
        self.assertEqual(report.models[0].model_name, "mlp_gate")
        self.assertEqual(report.models[1].model_name, "schnet_gate")
        for model_eval in report.models:
            self.assertGreaterEqual(model_eval.metrics.mae, 0.0)
            self.assertGreaterEqual(model_eval.metrics.rmse, 0.0)


if __name__ == "__main__":
    unittest.main()
