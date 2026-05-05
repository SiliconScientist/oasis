from __future__ import annotations

import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ase import Atoms
import polars as pl
import torch

from oasis.dataset import GatingDataset
from oasis.evaluate import evaluate_gating_model
from oasis.graph import build_adsorption_graphs
from oasis.model import BaselineMLPGatedMoE
from oasis.train import (
    TrainConfig,
    build_gating_dataloaders,
    split_gating_dataset,
    train_gating_model,
)


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


class TrainingTests(unittest.TestCase):
    def test_dataset_split_has_train_and_val_samples(self) -> None:
        dataset = _example_dataset()
        train_subset, val_subset = split_gating_dataset(
            dataset,
            val_fraction=0.25,
            seed=7,
        )
        self.assertEqual(len(train_subset), 3)
        self.assertEqual(len(val_subset), 1)

    def test_baseline_training_loop_writes_checkpoints(self) -> None:
        dataset = _example_dataset()
        train_loader, val_loader = build_gating_dataloaders(
            dataset,
            batch_size=2,
            val_fraction=0.25,
            seed=7,
        )
        model = BaselineMLPGatedMoE(n_experts=2, hidden_dims=(8,))

        with TemporaryDirectory() as tmpdir:
            result = train_gating_model(
                model,
                train_loader,
                val_loader,
                config=TrainConfig(
                    batch_size=2,
                    epochs=2,
                    learning_rate=1e-2,
                    val_fraction=0.25,
                    random_seed=7,
                    checkpoint_dir=tmpdir,
                    device="cpu",
                ),
            )

            self.assertEqual(len(result.history), 2)
            self.assertIsNotNone(result.best_epoch)
            self.assertTrue(math.isfinite(result.best_val_loss))
            self.assertIsNotNone(result.best_checkpoint_path)
            self.assertIsNotNone(result.latest_checkpoint_path)
            self.assertTrue(Path(result.best_checkpoint_path).is_file())
            self.assertTrue(Path(result.latest_checkpoint_path).is_file())
            self.assertEqual(
                result.best_epoch,
                min(result.history, key=lambda metric: metric.val_loss).epoch,
            )

            best_checkpoint = torch.load(
                result.best_checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            best_state_dict = best_checkpoint["model_state_dict"]
            for name, tensor in model.state_dict().items():
                self.assertTrue(torch.equal(tensor.cpu(), best_state_dict[name]))

    def test_training_and_evaluation_are_reproducible_for_fixed_seed(self) -> None:
        dataset = _example_dataset()

        def _run_once() -> tuple[list[tuple[float, float]], float]:
            torch.manual_seed(23)
            train_loader, val_loader = build_gating_dataloaders(
                dataset,
                batch_size=2,
                val_fraction=0.25,
                seed=7,
            )
            model = BaselineMLPGatedMoE(n_experts=2, hidden_dims=(8,))
            result = train_gating_model(
                model,
                train_loader,
                val_loader,
                config=TrainConfig(
                    batch_size=2,
                    epochs=2,
                    learning_rate=1e-2,
                    val_fraction=0.25,
                    random_seed=7,
                    checkpoint_dir=None,
                    device="cpu",
                ),
            )
            metrics = evaluate_gating_model(model, val_loader, device="cpu")
            history = [
                (round(epoch.train_loss, 10), round(epoch.val_loss, 10))
                for epoch in result.history
            ]
            return history, round(metrics.rmse, 10)

        first_history, first_rmse = _run_once()
        second_history, second_rmse = _run_once()

        self.assertEqual(first_history, second_history)
        self.assertEqual(first_rmse, second_rmse)


if __name__ == "__main__":
    unittest.main()
