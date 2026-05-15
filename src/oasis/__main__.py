from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

from oasis.analysis import filter_wide_predictions
from oasis.config import get_config
from oasis.exp import run_learning_curve_experiments_from_config
from oasis.graphs import atoms_to_graph_dataset_view, load_probe_graph_dataset_view, save_aligned_graph_dataset_parquet
from oasis.io import (
    find_result_files,
    load_sample_atoms_for_wide_df,
    load_wide_predictions,
)
from oasis.plot import learning_curve_plot, parity_plot
from oasis.mlip.cli import main as mlip_main
from oasis.probe import build_probe_dataset, updated_dataset_output_path
from oasis.probe_features import add_mlip_feature_matrices_to_dataset


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "mlip":
        # Forward remaining args to mlip CLI
        mlip_main(sys.argv[2:])
        return

    cfg = get_config()
    dataset_path = Path(cfg.mlip.dataset)
    if not updated_dataset_output_path(dataset_path).exists():
        build_probe_dataset(cfg)
        print(f"Built probe dataset from {dataset_path}")

    probe_cfg = cfg.probe_features
    if probe_cfg is not None and probe_cfg.mlip_results_dir.exists():
        add_mlip_feature_matrices_to_dataset(
            dataset_path=probe_cfg.dataset_path,
            mlip_results_dir=probe_cfg.mlip_results_dir,
        )
        print(f"Embedded probe feature matrices from {probe_cfg.mlip_results_dir}")

    base_dir = cfg.analysis.base_dir if cfg.analysis else Path("data/mlips")
    result_files = find_result_files(base_dir)
    wide_df = load_wide_predictions(result_files)

    plot_filters = cfg.plot.filters if cfg.plot else None
    adsorbate_filter = plot_filters.adsorbate if plot_filters else None
    anomaly_filter = plot_filters.anomaly_label if plot_filters else None
    reaction_contains_filter = plot_filters.reaction_contains if plot_filters else None
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

    auxiliary_views: dict = {}
    models_cfg = (
        cfg.experiment.learning_curve.models
        if cfg.experiment and cfg.experiment.learning_curve
        else None
    )
    if models_cfg is not None and getattr(models_cfg, "use_latent", False):
        latent_cfg = models_cfg.latent
        if latent_cfg is not None:
            latent_df = pd.read_csv(latent_cfg.csv_path)
            csv_energies = set(latent_df["adsorption_energy"].tolist())
            wide_df = wide_df.filter(
                wide_df.get_column("reference_ads_eng").is_in(list(csv_energies))
            )
            energy_order = wide_df.get_column("reference_ads_eng").to_list()
            latent_df = (
                latent_df.set_index("adsorption_energy").loc[energy_order].reset_index()
            )
            auxiliary_views["latent"] = latent_df
            print(
                f"Latent filter: {len(wide_df)} samples aligned to {latent_cfg.csv_path.name}"
            )

    probe_gnn_cfg = getattr(models_cfg, "probe_gnn", None)
    if getattr(probe_gnn_cfg, "enabled", False):
        if probe_cfg is None or not probe_cfg.dataset_path.exists():
            print(
                "Warning: probe_gnn.enabled=true but probe_features.dataset_path is "
                "not configured or does not exist — probe graphs not loaded."
            )
        else:
            probe_graph_view = load_probe_graph_dataset_view(probe_cfg.dataset_path)
            reactions = wide_df.get_column("reaction").to_list()
            auxiliary_views["probe_gnn_records"] = [
                probe_graph_view[r] for r in reactions
            ]
            print(
                f"Loaded {len(reactions)} probe-augmented graphs from {probe_cfg.dataset_path}"
            )
    output_dir = cfg.plot.output_dir if cfg.plot else Path("data/results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
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
    print(
        f"Processed {len(result_files)} MLIP files"
        f"{f' with adsorbate={adsorbate_filter}' if adsorbate_filter else ''}"
        f"{f' with anomaly_label={anomaly_filter}' if anomaly_filter else ''}"
        f"{f' with reaction_contains={reaction_contains_filter}' if reaction_contains_filter else ''}"
        f" -> parity plot: {saved_path}"
    )
    print(f"Rows in combined parity dataset: {len(wide_df)}")

    graph_dataset_cfg = (
        cfg.experiment.learning_curve.graph_dataset
        if cfg.experiment and cfg.experiment.learning_curve
        else None
    )
    reuse_existing_graph_dataset = (
        graph_dataset_cfg is not None and graph_dataset_cfg.path.is_file()
    )
    graph_view = None
    if reuse_existing_graph_dataset:
        print(f"Using existing aligned graph dataset from {graph_dataset_cfg.path}")
    else:
        sample_atoms = load_sample_atoms_for_wide_df(wide_df, cfg)
        graph_view = atoms_to_graph_dataset_view(
            wide_df.get_column("reaction").to_list(),
            sample_atoms,
        )
        print(f"Loaded {len(sample_atoms)} adsorbed structures from {cfg.mlip.dataset}")
        print(f"Built {len(graph_view)} graph artifacts from sampled structures")
        if graph_dataset_cfg is not None:
            saved_graph_dataset_path = save_aligned_graph_dataset_parquet(
                wide_df,
                graph_view,
                graph_dataset_cfg.path,
                join_key=graph_dataset_cfg.join_key,
            )
            print(f"Saved aligned graph dataset to {saved_graph_dataset_path}")

    learning_curve_results = run_learning_curve_experiments_from_config(
        wide_df,
        cfg,
        graph_view=graph_view,
        auxiliary_views=auxiliary_views,
    )

    learning_curve_plot(
        results=learning_curve_results,
        output_path=output_dir / "learning_curve.png",
    )


if __name__ == "__main__":
    main()
