from __future__ import annotations

from pathlib import Path

import pandas as pd

from oasis.analysis import (
    filter_anomalous_mlip_columns,
    filter_wide_predictions,
)
from oasis.config import get_config
from oasis.exp import load_or_run_learning_curve_results_from_config
from oasis.experiment_data import (
    atoms_to_graph_dataset_view,
    build_probe_dataset,
    graph_artifact_matches_frame,
    load_probe_graph_dataset_view,
    save_aligned_graph_dataset_parquet,
    updated_dataset_output_path,
)
from oasis.io import load_sample_atoms_for_wide_df
from oasis.mlip.artifacts import (
    find_result_files,
    load_wide_predictions,
)
from oasis.plot import learning_curve_plot, parity_plot, screening_budget_plot
from oasis.probe_features import add_mlip_feature_matrices_to_dataset


def _frame_height(frame: object) -> int:
    return int(getattr(frame, "height", len(frame)))


def _can_reuse_graph_artifact(
    artifact_path: Path,
    *,
    wide_df: object,
    join_key: str = "reaction",
) -> bool:
    if not artifact_path.is_file():
        return False
    return graph_artifact_matches_frame(artifact_path, wide_df, join_key=join_key)


def ensure_probe_artifacts(cfg: object) -> bool:
    dataset_path = Path(cfg.mlip.dataset)
    probe_cfg = cfg.probe_features
    models_cfg = (
        cfg.experiment.learning_curve.models
        if cfg.experiment and cfg.experiment.learning_curve
        else None
    )
    probe_gnn_cfg = getattr(models_cfg, "probe_gnn", None)
    probe_gnn_enabled = bool(getattr(probe_gnn_cfg, "enabled", False))

    if probe_gnn_enabled and probe_cfg is not None:
        if not updated_dataset_output_path(dataset_path).exists():
            build_probe_dataset(cfg)
            print(f"Built probe dataset from {dataset_path}")

    if probe_gnn_enabled and probe_cfg is not None and probe_cfg.mlip_results_dir.exists():
        add_mlip_feature_matrices_to_dataset(
            dataset_path=probe_cfg.dataset_path,
            mlip_results_dir=probe_cfg.mlip_results_dir,
        )
        print(f"Embedded probe feature matrices from {probe_cfg.mlip_results_dir}")

    return probe_gnn_enabled


def load_filtered_wide_predictions(cfg: object):
    base_dir = cfg.analysis.base_dir if cfg.analysis else Path("data/mlips")
    result_files = find_result_files(base_dir)
    wide_df = load_wide_predictions(result_files)
    print(f"Loaded combined wide_df with {_frame_height(wide_df)} rows before filters")

    plot_filters = cfg.plot.filters if cfg.plot else None
    adsorbate_filter = plot_filters.adsorbate if plot_filters else None
    anomaly_filter = plot_filters.anomaly_label if plot_filters else None
    reaction_contains_filter = plot_filters.reaction_contains if plot_filters else None
    if reaction_contains_filter is not None:
        reaction_contains_filter = [s for s in reaction_contains_filter if s]
        if not reaction_contains_filter:
            reaction_contains_filter = None

    pre_filter_rows = _frame_height(wide_df)
    wide_df = filter_wide_predictions(
        wide_df,
        adsorbate_filter=adsorbate_filter,
        anomaly_filter=anomaly_filter,
        reaction_contains_filter=reaction_contains_filter,
    )
    print(
        "Applied plot filters"
        f" adsorbate={adsorbate_filter!r}"
        f" anomaly_label={anomaly_filter!r}"
        f" reaction_contains={reaction_contains_filter!r}"
        f": {pre_filter_rows} -> {_frame_height(wide_df)} rows"
    )
    learning_curve_cfg = (
        cfg.experiment.learning_curve
        if cfg.experiment and cfg.experiment.learning_curve
        else None
    )
    mlip_selection_cfg = getattr(learning_curve_cfg, "mlip_selection", None)
    wide_df = filter_anomalous_mlip_columns(
        wide_df,
        enabled=bool(
            getattr(mlip_selection_cfg, "exclude_anomalous", False)
        ),
        label_allowlist=(
            list(mlip_selection_cfg.label_allowlist)
            if mlip_selection_cfg is not None
            else None
        ),
        strict_inference_anomaly=bool(
            getattr(mlip_selection_cfg, "strict_inference_anomaly", False)
        ),
    )
    return wide_df, result_files, adsorbate_filter, anomaly_filter, reaction_contains_filter


def build_auxiliary_views(cfg: object, wide_df: object, probe_gnn_enabled: bool) -> dict:
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
            pre_latent_rows = _frame_height(wide_df)
            wide_df = wide_df.filter(
                wide_df.get_column("reference_ads_eng").is_in(list(csv_energies))
            )
            energy_order = wide_df.get_column("reference_ads_eng").to_list()
            latent_df = (
                latent_df.set_index("adsorption_energy").loc[energy_order].reset_index()
            )
            auxiliary_views["latent"] = latent_df
            print(
                "Applied latent alignment filter"
                f" against {latent_cfg.csv_path.name}:"
                f" {pre_latent_rows} -> {_frame_height(wide_df)} rows"
            )

    probe_cfg = cfg.probe_features
    if probe_gnn_enabled and probe_cfg is not None and probe_cfg.dataset_path.exists():
        probe_graph_view = load_probe_graph_dataset_view(probe_cfg.dataset_path)
        reactions = wide_df.get_column("reaction").to_list()
        auxiliary_views["probe_gnn_records"] = [probe_graph_view[r] for r in reactions]
        print(
            f"Loaded {len(reactions)} probe-augmented graphs from {probe_cfg.dataset_path}"
        )

    return auxiliary_views


def write_parity_plot(
    cfg: object,
    wide_df: object,
    result_files: list[Path],
    adsorbate_filter: object,
    anomaly_filter: object,
    reaction_contains_filter: object,
) -> Path:
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
    return saved_path


def prepare_graph_view(cfg: object, wide_df: object):
    graph_dataset_cfg = (
        cfg.experiment.learning_curve.graph_dataset
        if cfg.experiment and cfg.experiment.learning_curve
        else None
    )
    reaction_ids = wide_df.get_column("reaction").to_list()
    graph_join_key = graph_dataset_cfg.join_key if graph_dataset_cfg is not None else "reaction"
    reuse_existing_graph_dataset = (
        graph_dataset_cfg is not None
        and _can_reuse_graph_artifact(
            graph_dataset_cfg.path,
            wide_df=wide_df,
            join_key=graph_join_key,
        )
    )
    if reuse_existing_graph_dataset:
        print(f"Using existing aligned graph dataset from {graph_dataset_cfg.path}")
        return None

    if graph_dataset_cfg is not None and graph_dataset_cfg.path.is_file():
        print(
            "Existing aligned graph dataset does not match the current filtered "
            f"rows; rebuilding {graph_dataset_cfg.path}"
        )
    sample_atoms = load_sample_atoms_for_wide_df(wide_df, cfg)
    graph_view = atoms_to_graph_dataset_view(reaction_ids, sample_atoms)
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
    return graph_view


def run_experiment(cfg: object):
    probe_gnn_enabled = ensure_probe_artifacts(cfg)
    wide_df, result_files, adsorbate_filter, anomaly_filter, reaction_contains_filter = (
        load_filtered_wide_predictions(cfg)
    )
    auxiliary_views = build_auxiliary_views(cfg, wide_df, probe_gnn_enabled)
    write_parity_plot(
        cfg,
        wide_df,
        result_files,
        adsorbate_filter,
        anomaly_filter,
        reaction_contains_filter,
    )
    graph_view = prepare_graph_view(cfg, wide_df)
    learning_curve_results = load_or_run_learning_curve_results_from_config(
        wide_df,
        cfg,
        graph_view=graph_view,
        auxiliary_views=auxiliary_views,
    )
    output_dir = cfg.plot.output_dir if cfg.plot else Path("data/results/plots")
    curve_window_cfg = getattr(cfg.plot, "curve_window", None) if cfg.plot else None
    plot_kwargs = {
        "min_x": getattr(curve_window_cfg, "min_x", None),
        "max_x": getattr(curve_window_cfg, "max_x", None),
        "include_x": getattr(curve_window_cfg, "include_x", None),
    }
    learning_curve_cfg = (
        cfg.experiment.learning_curve if cfg.experiment is not None else None
    )
    budget_mode = getattr(learning_curve_cfg, "budget_mode", "full_remainder_test")
    if budget_mode == "screening_fraction":
        screening_budget_plot(
            results=learning_curve_results,
            output_path=output_dir / "screening_budget.png",
            **plot_kwargs,
        )
    else:
        learning_curve_plot(
            results=learning_curve_results,
            output_path=output_dir / "learning_curve.png",
            **plot_kwargs,
        )
    return learning_curve_results


def run_experiment_from_config(config_paths=None):
    return run_experiment(get_config(config_paths))
