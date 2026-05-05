from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import polars as pl

from oasis.analysis import filter_wide_predictions
from oasis.config import get_config
from oasis.dataset import GatingDataset
from oasis.exp import (
    GatingMethodSpec,
    default_tabular_method_specs,
    run_all_method_sweeps,
    save_method_sweep_rows_csv,
)
from oasis.graph import build_adsorption_graphs
from oasis.io import find_result_files, load_corresponding_atoms, load_wide_predictions
from oasis.model import BaselineMLPGatedMoE, SchNetGatedMoE
from oasis.plot import learning_curve_plot, moe_learning_speed_plot, parity_plot
from oasis.mlip.cli import main as mlip_main
from oasis.train import TrainConfig


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "mlip":
        # Forward remaining args to mlip CLI
        mlip_main(sys.argv[2:])
        return

    cfg = get_config()
    if cfg.moe is None:
        raise ValueError("config.toml must define a [moe] section")
    moe_cfg = cfg.moe
    base_dir = cfg.analysis.base_dir if cfg.analysis else Path("data/mlips")
    result_files = find_result_files(base_dir)
    wide_df = load_wide_predictions(result_files)
    atoms_list = load_corresponding_atoms(wide_df, cfg.mlip.dataset)
    graphs = build_adsorption_graphs(
        wide_df,
        atoms_list,
        cutoff=moe_cfg.graph.cutoff,
        max_neighbors=moe_cfg.graph.max_neighbors,
    )
    gating_dataset = GatingDataset(graphs, wide_df)
    output_dir = cfg.plot.output_dir if cfg.plot else Path("data/results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_methods_csv = output_dir / "all_method_sweeps.csv"

    if cfg.train:
        moe_train_config = TrainConfig(
            batch_size=moe_cfg.train.batch_size,
            epochs=moe_cfg.train.epochs,
            learning_rate=moe_cfg.train.learning_rate,
            weight_decay=moe_cfg.train.weight_decay,
            val_fraction=moe_cfg.train.val_fraction,
            random_seed=cfg.seed or 0,
            checkpoint_dir=None,
            device=moe_cfg.train.device,
        )
        n_experts = len(gating_dataset[0].graph.mlip_names)
        tabular_methods = default_tabular_method_specs(
            use_trim=cfg.plot.trim if cfg.plot else True,
            use_ridge=cfg.plot.use_ridge if cfg.plot else True,
            use_kernel_ridge=cfg.plot.use_kernel_ridge if cfg.plot else True,
            use_lasso=cfg.plot.use_lasso if cfg.plot else True,
            use_elastic=cfg.plot.use_elastic_net if cfg.plot else True,
            use_residual=cfg.plot.use_residual if cfg.plot else True,
            use_linearization=cfg.plot.use_linearization if cfg.plot else True,
            n_repeats=cfg.plot.n_repeats if cfg.plot else 30,
        )
        available_train = len(gating_dataset) - 1
        moe_train_sizes = sorted(
            {
                max(1, int(round(available_train * fraction)))
                for fraction in (0.1, 0.25, 0.5, 0.75, 1.0)
            }
        )
        moe_repeats = max(1, min(5, (cfg.plot.n_repeats if cfg.plot else 30) // 5))
        gating_methods = [
            GatingMethodSpec(
                name="moe_baseline",
                model_factory=lambda: BaselineMLPGatedMoE(
                    n_experts=n_experts,
                    hidden_dims=tuple(moe_cfg.baseline_gate.hidden_dims),
                    dropout=moe_cfg.baseline_gate.dropout,
                ),
                train_config=moe_train_config,
                n_repeats=moe_repeats,
                seed=(cfg.seed or 0) + 101,
            ),
            GatingMethodSpec(
                name="moe_schnet",
                model_factory=lambda: SchNetGatedMoE(
                    n_experts=n_experts,
                    structure_hidden_dim=moe_cfg.schnet_gate.structure_hidden_dim,
                    n_interactions=moe_cfg.schnet_gate.n_interactions,
                    n_rbf=moe_cfg.graph.n_rbf,
                    cutoff=moe_cfg.graph.cutoff,
                    gate_hidden_dims=tuple(moe_cfg.schnet_gate.gate_hidden_dims),
                    dropout=moe_cfg.schnet_gate.dropout,
                ),
                train_config=replace(moe_train_config, checkpoint_dir=None),
                n_repeats=moe_repeats,
                seed=(cfg.seed or 0) + 202,
            ),
        ]
        all_method_rows = run_all_method_sweeps(
            wide_df=wide_df,
            gating_dataset=gating_dataset,
            tabular_methods=tabular_methods,
            gating_methods=gating_methods,
            tabular_train_sizes=list(range(cfg.plot.min_train, cfg.plot.max_train + 1))
            if cfg.plot
            else list(range(2, 11)),
            gating_train_sizes=moe_train_sizes,
        )
        all_methods_csv = save_method_sweep_rows_csv(
            all_method_rows,
            all_methods_csv,
        )
        print(f"Saved combined method sweep CSV: {all_methods_csv}")

    if all_methods_csv.is_file():
        sweep_df = pl.read_csv(all_methods_csv)
        baseline_df = sweep_df.filter(pl.col("method") == "moe_baseline")
        schnet_df = sweep_df.filter(pl.col("method") == "moe_schnet")
        if baseline_df.height > 0:
            baseline_plot = moe_learning_speed_plot(
                baseline_df,
                output_dir / "moe_baseline_learning_speed.png",
                x_col="size",
                metric="rmse_mean",
                title="MOE baseline gate learning speed",
            )
            print(f"Saved MOE baseline sweep plot: {baseline_plot}")
        if schnet_df.height > 0:
            schnet_plot = moe_learning_speed_plot(
                schnet_df,
                output_dir / "moe_schnet_learning_speed.png",
                x_col="size",
                metric="rmse_mean",
                title="MOE SchNet gate learning speed",
            )
            print(f"Saved MOE SchNet sweep plot: {schnet_plot}")

    adsorbate_filter = cfg.plot.adsorbate if cfg.plot else None
    anomaly_filter = cfg.plot.anomaly_label if cfg.plot else None
    reaction_contains_filter = cfg.plot.reaction_contains if cfg.plot else None
    if reaction_contains_filter is not None:
        reaction_contains_filter = [s for s in reaction_contains_filter if s]
        if not reaction_contains_filter:
            reaction_contains_filter = None
    wide_df = filter_wide_predictions(
        wide_df,
        adsorbate_filter=adsorbate_filter,
        anomaly_filter=anomaly_filter,
        reaction_contains_filter=reaction_contains_filter,
    )

    suffix_parts: list[str] = []
    if adsorbate_filter:
        suffix_parts.append(f"adsorbate_{adsorbate_filter}")
    if anomaly_filter:
        suffix_parts.append(f"anomaly_{anomaly_filter}")
    if reaction_contains_filter:
        joined = "-".join(reaction_contains_filter)
        suffix_parts.append(f"reaction_contains_{joined}")
    suffix = f"_{'_'.join(suffix_parts)}" if suffix_parts else ""
    output_path = output_dir / f"mlips_vs_dft_parity{suffix}.png"
    saved_path = parity_plot(wide_df, output_path=output_path)
    learning_curve_plot(
        cfg=cfg,
        df=wide_df,
        output_path=output_dir / "learning_curve.png",
    )
    print(f"Saved parity plot: {saved_path}")
    print(f"Saved learning curve plot: {output_dir / 'learning_curve.png'}")


if __name__ == "__main__":
    main()
