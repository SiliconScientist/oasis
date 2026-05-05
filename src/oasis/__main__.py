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
from oasis.plot import learning_curve_plot, parity_plot
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
    plot_moe_baseline = cfg.plot.plot_moe_baseline if cfg.plot else False
    plot_moe_schnet = cfg.plot.plot_moe_schnet if cfg.plot else False

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
            use_ridge=cfg.plot.use_ridge if cfg.plot else True,
            use_residual=cfg.plot.use_residual if cfg.plot else True,
            n_repeats=cfg.plot.n_repeats if cfg.plot else 30,
        )
        tabular_train_sizes = (
            list(range(cfg.plot.min_train, cfg.plot.max_train + 1))
            if cfg.plot
            else list(range(2, 11))
        )
        available_train = len(gating_dataset) - 1
        moe_train_sizes = [
            size for size in tabular_train_sizes if 0 < size <= available_train
        ]
        moe_repeats = cfg.plot.n_repeats if cfg.plot else 30
        gating_methods: list[GatingMethodSpec] = []
        if plot_moe_baseline:
            gating_methods.append(
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
                )
            )
        if plot_moe_schnet:
            gating_methods.append(
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
                )
            )
        all_method_rows = run_all_method_sweeps(
            wide_df=wide_df,
            gating_dataset=gating_dataset,
            tabular_methods=tabular_methods,
            gating_methods=gating_methods,
            tabular_train_sizes=tabular_train_sizes,
            gating_train_sizes=moe_train_sizes,
            shared_train_seed=cfg.seed or 0,
        )
        all_methods_csv = save_method_sweep_rows_csv(
            all_method_rows,
            all_methods_csv,
        )
        print(f"Saved combined method sweep CSV: {all_methods_csv}")

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

    ###

    method_sweeps = pl.read_csv(all_methods_csv) if all_methods_csv.is_file() else None
    if method_sweeps is not None:
        excluded_methods: list[str] = []
        if cfg.plot and not cfg.plot.use_ridge:
            excluded_methods.append("ridge")
        if cfg.plot and not cfg.plot.use_residual:
            excluded_methods.append("residual")
        if not plot_moe_baseline:
            excluded_methods.append("moe_baseline")
        if not plot_moe_schnet:
            excluded_methods.append("moe_schnet")
        if excluded_methods:
            method_sweeps = method_sweeps.filter(
                ~pl.col("method").is_in(excluded_methods)
            )
    learning_curve_plot(
        cfg=cfg,
        df=wide_df,
        method_sweeps=method_sweeps,
        output_path=output_dir / "learning_curve.png",
    )
    print(f"Saved parity plot: {saved_path}")
    print(f"Saved learning curve plot: {output_dir / 'learning_curve.png'}")


if __name__ == "__main__":
    main()
