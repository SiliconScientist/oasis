from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ase import Atoms
import polars as pl

from oasis.dataset import GatingDataset
from oasis.exp import (
    GatingMethodSpec,
    build_learning_curve_sweeps,
    default_tabular_method_specs,
    run_all_method_sweeps,
    run_data_fraction_sweep,
    save_method_sweep_rows_csv,
    save_sweep_points_csv,
)
from oasis.graph import build_adsorption_graphs
from oasis.model import BaselineMLPGatedMoE
from oasis.train import TrainConfig


def _example_dataset() -> tuple[GatingDataset, pl.DataFrame]:
    wide_df = pl.DataFrame(
        {
            "reaction": [
                "CuOH_* -> OH*",
                "CuH_* -> H*",
                "NiOH_* -> OH*",
                "NiH_* -> H*",
                "PtOH_* -> OH*",
                "PtH_* -> H*",
            ],
            "adsorbate": ["OH", "H", "OH", "H", "OH", "H"],
            "reference_ads_eng": [-1.25, -0.45, -1.10, -0.35, -0.95, -0.25],
            "mace_mlip_ads_eng_median": [-1.10, -0.40, -1.05, -0.30, -0.90, -0.20],
            "mace_label": ["normal"] * 6,
            "orb_mlip_ads_eng_median": [-1.30, -0.50, -1.20, -0.45, -1.00, -0.35],
            "orb_label": ["normal", "energy_anomaly", "normal", "normal", "normal", "normal"],
        }
    )
    atoms_list = [
        Atoms("Cu2OH", positions=[[0, 0, 0], [2.5, 0, 0], [1.2, 0.1, 1.3], [1.2, 0.1, 2.2]], cell=[12, 12, 12], pbc=[False, False, False]),
        Atoms("Cu2H", positions=[[0, 0, 0], [2.5, 0, 0], [1.3, 0, 1.1]], cell=[12, 12, 12], pbc=[False, False, False]),
        Atoms("Ni2OH", positions=[[0, 0, 0], [2.4, 0, 0], [1.1, 0.1, 1.2], [1.1, 0.1, 2.1]], cell=[12, 12, 12], pbc=[False, False, False]),
        Atoms("Ni2H", positions=[[0, 0, 0], [2.4, 0, 0], [1.2, 0, 1.0]], cell=[12, 12, 12], pbc=[False, False, False]),
        Atoms("Pt2OH", positions=[[0, 0, 0], [2.6, 0, 0], [1.3, 0.1, 1.4], [1.3, 0.1, 2.3]], cell=[12, 12, 12], pbc=[False, False, False]),
        Atoms("Pt2H", positions=[[0, 0, 0], [2.6, 0, 0], [1.4, 0, 1.1]], cell=[12, 12, 12], pbc=[False, False, False]),
    ]
    graphs = build_adsorption_graphs(wide_df, atoms_list, cutoff=3.0)
    return GatingDataset(graphs, wide_df), wide_df


class ExperimentTests(unittest.TestCase):
    def test_data_fraction_sweep_and_csv(self) -> None:
        dataset, _ = _example_dataset()
        with TemporaryDirectory() as tmpdir:
            sweep_points, train_results = run_data_fraction_sweep(
                dataset,
                model_factory=lambda: BaselineMLPGatedMoE(n_experts=2, hidden_dims=(8,)),
                fractions=[0.34, 0.67, 1.0],
                train_config=TrainConfig(
                    batch_size=2,
                    epochs=1,
                    learning_rate=1e-2,
                    val_fraction=0.33,
                    random_seed=11,
                    checkpoint_dir=tmpdir,
                    device="cpu",
                ),
            )
            self.assertEqual(len(sweep_points), 3)
            self.assertEqual(len(train_results), 3)
            self.assertEqual([p.train_size for p in sweep_points], sorted([p.train_size for p in sweep_points]))
            self.assertEqual(sweep_points[-1].val_size, 2)

            csv_path = save_sweep_points_csv(sweep_points, Path(tmpdir) / "sweep.csv")
            self.assertTrue(csv_path.is_file())
            df = pl.read_csv(csv_path)
            self.assertEqual(df.height, 3)
            self.assertEqual(df["train_size"].to_list(), [1, 3, 4])

    def test_build_learning_curve_sweeps(self) -> None:
        _, wide_df = _example_dataset()
        sweep_results = build_learning_curve_sweeps(
            wide_df,
            min_train=2,
            max_train=3,
            n_repeats=1,
            use_trim=False,
            use_ridge=True,
            use_kernel_ridge=False,
            use_lasso=False,
            use_elastic=False,
            use_residual=True,
            use_linearization=True,
        )
        self.assertIsNotNone(sweep_results["ridge_df"])
        self.assertIsNone(sweep_results["kernel_ridge_df"])
        self.assertIsNotNone(sweep_results["resid_df"])
        self.assertIsNotNone(sweep_results["linear_df"])
        self.assertEqual(sweep_results["ridge_df"]["n_train"].tolist(), [2, 3])

    def test_run_all_method_sweeps_and_save_csv(self) -> None:
        dataset, wide_df = _example_dataset()
        tabular_methods = default_tabular_method_specs(
            use_trim=False,
            use_ridge=True,
            use_kernel_ridge=False,
            use_lasso=False,
            use_elastic=False,
            use_residual=True,
            use_linearization=False,
            n_repeats=1,
        )
        gating_methods = [
            GatingMethodSpec(
                name="moe_baseline",
                model_factory=lambda: BaselineMLPGatedMoE(
                    n_experts=2,
                    hidden_dims=(8,),
                ),
                train_config=TrainConfig(
                    batch_size=2,
                    epochs=1,
                    learning_rate=1e-2,
                    checkpoint_dir=None,
                    device="cpu",
                ),
                n_repeats=1,
                seed=7,
            )
        ]
        rows = run_all_method_sweeps(
            wide_df=wide_df,
            gating_dataset=dataset,
            tabular_methods=tabular_methods,
            gating_methods=gating_methods,
            tabular_train_sizes=[2, 3],
            gating_train_sizes=[2],
        )
        self.assertTrue(any(row.method == "ridge" for row in rows))
        self.assertTrue(any(row.method == "residual" for row in rows))
        self.assertTrue(any(row.method == "moe_baseline" for row in rows))

        with TemporaryDirectory() as tmpdir:
            csv_path = save_method_sweep_rows_csv(rows, Path(tmpdir) / "all_methods.csv")
            self.assertTrue(csv_path.is_file())
            df = pl.read_csv(csv_path)
            self.assertIn("method", df.columns)
            self.assertIn("rmse_mean", df.columns)


if __name__ == "__main__":
    unittest.main()
