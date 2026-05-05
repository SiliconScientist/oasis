from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from ase import Atoms
import numpy as np
import polars as pl
import torch

from oasis.dataset import GatingDataset
from oasis.exp import (
    GatingMethodSpec,
    SweepSplit,
    TabularMethodSpec,
    build_learning_curve_sweeps,
    build_train_size_splits,
    default_tabular_method_specs,
    run_all_method_sweeps,
    run_data_fraction_sweep,
    run_single_split_comparison,
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
    def test_build_train_size_splits(self) -> None:
        splits = build_train_size_splits(
            6,
            train_sizes=[2, 3],
            n_repeats=2,
            seed=13,
        )
        self.assertEqual(len(splits), 4)
        self.assertTrue(all(isinstance(split, SweepSplit) for split in splits))
        self.assertEqual([split.size for split in splits], [2, 2, 3, 3])
        self.assertTrue(all(split.axis == "train_size" for split in splits))
        self.assertTrue(all(len(split.train_idx) == split.size for split in splits))
        self.assertTrue(
            all(
                len(set(split.train_idx).intersection(split.eval_idx)) == 0
                for split in splits
            )
        )
        self.assertTrue(
            all(
                sorted(split.train_idx + split.eval_idx) == list(range(6))
                for split in splits
            )
        )

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
            use_ridge=True,
            use_residual=True,
        )
        self.assertIsNotNone(sweep_results["ridge_df"])
        self.assertIsNotNone(sweep_results["resid_df"])
        self.assertEqual(sweep_results["ridge_df"]["n_train"].tolist(), [2, 3])

    def test_run_all_method_sweeps_and_save_csv(self) -> None:
        dataset, wide_df = _example_dataset()
        tabular_methods = default_tabular_method_specs(
            use_ridge=True,
            use_residual=True,
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
            shared_train_seed=17,
        )
        self.assertTrue(any(row.method == "ridge" for row in rows))
        self.assertTrue(any(row.method == "residual" for row in rows))
        self.assertTrue(any(row.method == "moe_baseline" for row in rows))

        with TemporaryDirectory() as tmpdir:
            csv_path = save_method_sweep_rows_csv(rows, Path(tmpdir) / "all_methods.csv")
            self.assertTrue(csv_path.is_file())
            df = pl.read_csv(csv_path)
            self.assertIn("method", df.columns)
            self.assertIn("repeat", df.columns)
            self.assertIn("split_id", df.columns)
            self.assertIn("rmse", df.columns)

    def test_shared_splits_across_ridge_residual_and_moe(self) -> None:
        dataset, wide_df = _example_dataset()
        X = wide_df.select(
            [
                "mace_mlip_ads_eng_median",
                "orb_mlip_ads_eng_median",
            ]
        ).to_numpy()
        row_lookup = {tuple(row.tolist()): idx for idx, row in enumerate(X)}

        def _rows_to_indices(rows: np.ndarray) -> tuple[int, ...]:
            return tuple(sorted(row_lookup[tuple(row.tolist())] for row in rows))

        tabular_calls: dict[str, list[tuple[tuple[int, ...], tuple[int, ...]]]] = {
            "ridge": [],
            "residual": [],
        }

        def _tabular_evaluator(method_name: str):
            def _evaluate(
                X_train: np.ndarray,
                y_train: np.ndarray,
                X_eval: np.ndarray,
                y_eval: np.ndarray,
            ) -> float:
                del y_train, y_eval
                tabular_calls[method_name].append(
                    (_rows_to_indices(X_train), _rows_to_indices(X_eval))
                )
                return 0.0

            return _evaluate

        tabular_methods = [
            TabularMethodSpec(
                name="ridge",
                sweep_axis="train_size",
                evaluator=_tabular_evaluator("ridge"),
                n_repeats=2,
                seed=41,
            ),
            TabularMethodSpec(
                name="residual",
                sweep_axis="train_size",
                evaluator=_tabular_evaluator("residual"),
                n_repeats=2,
                seed=999,
            ),
        ]

        gating_calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

        class _DummyMetrics:
            rmse = 0.0

        def _fake_build_loaders(
            dataset: GatingDataset,
            *,
            batch_size: int,
            train_pool_indices: tuple[int, ...],
            test_indices: tuple[int, ...],
            val_fraction: float,
            seed: int,
        ) -> tuple[object, object, object]:
            del dataset, batch_size, val_fraction, seed
            gating_calls.append(
                (
                    tuple(sorted(int(i) for i in train_pool_indices)),
                    tuple(sorted(int(i) for i in test_indices)),
                )
            )
            return object(), object(), object()

        expected_splits = build_train_size_splits(
            len(dataset),
            train_sizes=[2, 3],
            n_repeats=2,
            seed=17,
        )
        expected_by_size = {
            size: [
                (
                    tuple(sorted(split.train_idx)),
                    tuple(sorted(split.eval_idx)),
                )
                for split in expected_splits
                if split.size == size
            ]
            for size in (2, 3)
        }

        with (
            patch("oasis.exp.build_nested_gating_dataloaders", _fake_build_loaders),
            patch("oasis.exp.train_gating_model", lambda *args, **kwargs: None),
            patch("oasis.exp.evaluate_gating_model", lambda *args, **kwargs: _DummyMetrics()),
        ):
            run_all_method_sweeps(
                wide_df=wide_df,
                gating_dataset=dataset,
                tabular_methods=tabular_methods,
                gating_methods=[
                    GatingMethodSpec(
                        name="moe_baseline",
                        model_factory=lambda: object(),
                        train_config=TrainConfig(
                            batch_size=2,
                            epochs=1,
                            learning_rate=1e-2,
                            checkpoint_dir=None,
                            device="cpu",
                        ),
                        n_repeats=2,
                        seed=7,
                    )
                ],
                tabular_train_sizes=[2, 3],
                gating_train_sizes=[2, 3],
                shared_train_seed=17,
            )

        for method_name in ("ridge", "residual"):
            self.assertEqual(
                sorted(tabular_calls[method_name]),
                sorted(expected_by_size[2] + expected_by_size[3]),
            )
        self.assertEqual(
            sorted(gating_calls),
            sorted(expected_by_size[2] + expected_by_size[3]),
        )

    def test_all_methods_share_outer_split_ids(self) -> None:
        dataset, wide_df = _example_dataset()
        rows = run_all_method_sweeps(
            wide_df=wide_df,
            gating_dataset=dataset,
            tabular_methods=default_tabular_method_specs(
                use_ridge=True,
                use_residual=True,
                n_repeats=2,
            ),
            gating_methods=[
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
                    n_repeats=2,
                    seed=7,
                )
            ],
            tabular_train_sizes=[2, 3],
            gating_train_sizes=[2, 3],
            shared_train_seed=17,
        )

        split_to_methods: dict[str, set[str]] = {}
        for row in rows:
            split_to_methods.setdefault(row.split_id, set()).add(row.method)

        self.assertEqual(
            split_to_methods,
            {
                "train_size:2:repeat:0": {"ridge", "residual", "moe_baseline"},
                "train_size:2:repeat:1": {"ridge", "residual", "moe_baseline"},
                "train_size:3:repeat:0": {"ridge", "residual", "moe_baseline"},
                "train_size:3:repeat:1": {"ridge", "residual", "moe_baseline"},
            },
        )

    def test_moe_uses_inner_val_and_outer_test_separately(self) -> None:
        dataset, wide_df = _example_dataset()
        split = SweepSplit(
            train_idx=(0, 1, 2),
            eval_idx=(3, 4, 5),
            size=3,
            repeat=0,
            axis="train_size",
        )
        observed_inner_train_indices: list[tuple[int, ...]] = []
        observed_inner_val_indices: list[tuple[int, ...]] = []
        observed_outer_test_indices: list[tuple[int, ...]] = []

        class _DummyMetrics:
            rmse = 0.0

        def _train_spy(model, train_loader, val_loader, *, config):
            del model, config
            train_subset = train_loader.dataset
            val_subset = val_loader.dataset
            outer_train_subset = train_subset.dataset
            observed_inner_train_indices.append(
                tuple(
                    sorted(
                        int(outer_train_subset.indices[i]) for i in train_subset.indices
                    )
                )
            )
            observed_inner_val_indices.append(
                tuple(
                    sorted(
                        int(outer_train_subset.indices[i]) for i in val_subset.indices
                    )
                )
            )
            return None

        def _eval_spy(model, data_loader, *, device):
            del model, device
            subset = data_loader.dataset
            observed_outer_test_indices.append(
                tuple(sorted(int(i) for i in subset.indices))
            )
            return _DummyMetrics()

        with (
            patch("oasis.exp.train_gating_model", _train_spy),
            patch("oasis.exp.evaluate_gating_model", _eval_spy),
        ):
            run_single_split_comparison(
                wide_df=wide_df,
                gating_dataset=dataset,
                tabular_methods=[],
                gating_methods=[
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
                ],
                split=split,
            )

        self.assertEqual(len(observed_inner_train_indices), 1)
        self.assertEqual(len(observed_inner_val_indices), 1)
        self.assertEqual(observed_outer_test_indices, [tuple(sorted(split.eval_idx))])
        self.assertEqual(
            set(observed_inner_train_indices[0]).intersection(split.eval_idx),
            set(),
        )
        self.assertEqual(
            set(observed_inner_val_indices[0]).intersection(split.eval_idx),
            set(),
        )
        self.assertEqual(
            set(observed_inner_train_indices[0]).intersection(
                observed_inner_val_indices[0]
            ),
            set(),
        )
        self.assertEqual(
            set(observed_inner_train_indices[0]).union(observed_inner_val_indices[0]),
            set(split.train_idx),
        )

    def test_run_single_split_comparison(self) -> None:
        dataset, wide_df = _example_dataset()
        split = SweepSplit(
            train_idx=(0, 1),
            eval_idx=(2, 3, 4, 5),
            size=2,
            repeat=0,
            axis="train_size",
        )

        class _DummyMetrics:
            rmse = 0.123

        with (
            patch("oasis.exp.train_gating_model", lambda *args, **kwargs: None),
            patch("oasis.exp.evaluate_gating_model", lambda *args, **kwargs: _DummyMetrics()),
        ):
            rows = run_single_split_comparison(
                wide_df=wide_df,
                gating_dataset=dataset,
                tabular_methods=default_tabular_method_specs(
                    use_ridge=True,
                    use_residual=True,
                    n_repeats=1,
                ),
                gating_methods=[
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
                ],
                split=split,
            )

        self.assertEqual([row.method for row in rows], ["ridge", "residual", "moe_baseline"])
        self.assertTrue(all(row.size == 2 for row in rows))
        self.assertTrue(all(row.repeat == 0 for row in rows))
        self.assertTrue(all(row.split_id == "train_size:2:repeat:0" for row in rows))
        self.assertEqual(rows[-1].rmse, 0.123)


if __name__ == "__main__":
    unittest.main()
