from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from oasis.analysis import (
    filter_anomalous_mlip_columns,
    filter_structures_with_insufficient_valid_mlips,
)
from oasis.config import get_config
from oasis.exp import (
    load_or_run_learning_curve_results_from_config,
    prepare_parity_plot_data,
)
from oasis.experiment.splits import resolve_configured_sweep_sizes
from oasis.figure import learning_screening_figure, uq_summary_figure
from oasis.learning_curve.execution import (
    dispersion_from_spread,
    miscalibration_area,
    sharpness_from_spread,
)
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
from oasis.plot import (
    dispersion_plot,
    learning_curve_plot,
    miscalibration_area_plot,
    parity_plot,
    screening_budget_plot,
    sharpness_plot,
)
from oasis.probe_features import add_mlip_feature_matrices_to_dataset


def _frame_height(frame: object) -> int:
    return int(getattr(frame, "height", len(frame)))


def _mlip_selection_cfg(cfg: object):
    learning_curve_cfg = (
        cfg.experiment.learning_curve
        if getattr(cfg, "experiment", None) and cfg.experiment.learning_curve
        else None
    )
    return getattr(learning_curve_cfg, "mlip_selection", None)


def _anomaly_aware_run_suffix(cfg: object) -> str:
    mlip_selection_cfg = _mlip_selection_cfg(cfg)
    enabled = bool(getattr(mlip_selection_cfg, "exclude_anomalous", False))
    return "anomalyaware_on" if enabled else "anomalyaware_off"


def _append_output_suffix(path: Path, suffix: str) -> Path:
    if path.stem.endswith(f"_{suffix}"):
        return path
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def _apply_run_output_suffixes(cfg: object) -> str:
    suffix = _anomaly_aware_run_suffix(cfg)

    experiment_cfg = getattr(cfg, "experiment", None)
    learning_curve_cfg = (
        experiment_cfg.learning_curve
        if experiment_cfg is not None and experiment_cfg.learning_curve
        else None
    )
    screening_cfg = (
        experiment_cfg.screening
        if experiment_cfg is not None and getattr(experiment_cfg, "screening", None)
        else None
    )
    for curve_cfg in (learning_curve_cfg, screening_cfg):
        if curve_cfg is None:
            continue
        results_bundle_path = getattr(curve_cfg, "results_bundle_path", None)
        if results_bundle_path is not None:
            curve_cfg.results_bundle_path = _append_output_suffix(
                Path(results_bundle_path),
                suffix,
            )

    graph_dataset_cfg = getattr(learning_curve_cfg, "graph_dataset", None)
    graph_dataset_path = getattr(graph_dataset_cfg, "path", None)
    if graph_dataset_cfg is not None and graph_dataset_path is not None:
        graph_dataset_cfg.path = _append_output_suffix(
            Path(graph_dataset_path),
            suffix,
        )

    analysis_cfg = getattr(cfg, "analysis", None)
    comparison_plot_path = getattr(analysis_cfg, "comparison_plot_path", None)
    if comparison_plot_path is not None:
        analysis_cfg.comparison_plot_path = _append_output_suffix(
            Path(comparison_plot_path),
            suffix,
        )

    return suffix


def _derived_screening_learning_curve_cfg(cfg: object):
    experiment_cfg = getattr(cfg, "experiment", None)
    if experiment_cfg is None:
        return None
    derive = getattr(experiment_cfg, "derived_screening_learning_curve", None)
    if callable(derive):
        return derive()

    screening_cfg = getattr(experiment_cfg, "screening", None)
    learning_curve_cfg = getattr(experiment_cfg, "learning_curve", None)
    if screening_cfg is None:
        return None
    if learning_curve_cfg is None:
        raise ValueError(
            "experiment.screening requires experiment.learning_curve to define "
            "the training grid and model families."
        )
    derived_cfg = copy.deepcopy(learning_curve_cfg)
    for field_name in (
        "budget_mode",
        "screen_fraction",
        "min_screen_size",
        "validation_fraction",
        "min_val_size",
        "min_tuning_val_size",
        "min_inner_train_size",
        "results_bundle_path",
        "reuse_results",
        "force_refresh_methods",
        "force_refresh_train_sizes",
    ):
        setattr(derived_cfg, field_name, copy.deepcopy(getattr(screening_cfg, field_name)))
    return derived_cfg


def _screening_run_cfg(cfg: object):
    screening_learning_curve_cfg = _derived_screening_learning_curve_cfg(cfg)
    if screening_learning_curve_cfg is None:
        return None
    screening_cfg = copy.deepcopy(cfg)
    screening_cfg.experiment.learning_curve = screening_learning_curve_cfg
    return screening_cfg


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
    print(f"Loaded combined wide_df with {_frame_height(wide_df)} rows")
    mlip_selection_cfg = _mlip_selection_cfg(cfg)
    wide_df = filter_structures_with_insufficient_valid_mlips(
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
    return wide_df, result_files


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
    run_suffix: str,
) -> Path:
    output_dir = cfg.plot.output_dir if cfg.plot else Path("data/results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix_parts: list[str] = [run_suffix]
    suffix = f"_{'_'.join(suffix_parts)}" if suffix_parts else ""
    output_path = output_dir / f"mlips_vs_dft_parity{suffix}.png"
    saved_path = parity_plot(wide_df, output_path=output_path)
    print(
        f"Processed {len(result_files)} MLIP files"
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


def _has_uq_summary(results: object) -> bool:
    if results is None:
        return False
    for field_name in (
        "resid_uq_df",
        "weighted_simplex_uq_df",
        "ridge_uq_df",
        "moe_uq_df",
    ):
        frame = getattr(results, field_name, None)
        if frame is not None and not frame.empty:
            return True
    return False


def _zero_shot_uq_baselines(parity_plot_data: object) -> dict[str, float]:
    zero_shot_preds = np.mean(
        np.column_stack(list(parity_plot_data.predictions.values())),
        axis=1,
    )
    zero_shot_spread = np.std(
        np.column_stack(list(parity_plot_data.predictions.values())),
        axis=1,
    )
    return {
        "miscalibration_area": float(
            miscalibration_area(
                parity_plot_data.reference,
                zero_shot_preds,
                zero_shot_spread,
            )
        ),
        "sharpness": float(sharpness_from_spread(zero_shot_spread)),
        "dispersion": float(dispersion_from_spread(zero_shot_spread)),
    }


def run_experiment(cfg: object):
    run_suffix = _apply_run_output_suffixes(cfg)
    probe_gnn_enabled = ensure_probe_artifacts(cfg)
    wide_df, result_files = load_filtered_wide_predictions(cfg)
    auxiliary_views = build_auxiliary_views(cfg, wide_df, probe_gnn_enabled)
    write_parity_plot(
        cfg,
        wide_df,
        result_files,
        run_suffix,
    )
    graph_view = prepare_graph_view(cfg, wide_df)
    output_dir = cfg.plot.output_dir if cfg.plot else Path("data/results/plots")
    curve_window_cfg = getattr(cfg.plot, "curve_window", None) if cfg.plot else None
    use_full_curve_window = bool(
        getattr(curve_window_cfg, "full_dataset_window", False)
    ) or bool(getattr(curve_window_cfg, "all", False))
    configured_include_x = getattr(curve_window_cfg, "include_x", None)
    configured_include_fractions = getattr(
        curve_window_cfg, "include_fractions", None
    )
    resolved_include_fraction_x = (
        list(
            resolve_configured_sweep_sizes(
                _frame_height(wide_df),
                min_train=None,
                max_train=None,
                sweep_fractions=configured_include_fractions,
            )
        )
        if configured_include_fractions
        else None
    )
    resolved_include_x = None
    if configured_include_x or resolved_include_fraction_x:
        resolved_include_x = sorted(
            {
                int(value)
                for value in (
                    list(configured_include_x or [])
                    + list(resolved_include_fraction_x or [])
                )
            }
        )
    plot_kwargs = (
        {
            "min_x": None,
            "max_x": None,
            "include_x": resolved_include_x,
        }
        if use_full_curve_window
        else {
            "min_x": getattr(curve_window_cfg, "min_x", None),
            "max_x": getattr(curve_window_cfg, "max_x", None),
            "include_x": resolved_include_x,
        }
    )
    parity_plot_data = prepare_parity_plot_data(wide_df)
    zero_shot_preds = np.mean(
        np.column_stack(list(parity_plot_data.predictions.values())),
        axis=1,
    )
    zero_shot_rmse = float(
        np.sqrt(np.mean((parity_plot_data.reference - zero_shot_preds) ** 2))
    )
    zero_shot_uq = _zero_shot_uq_baselines(parity_plot_data)
    learning_curve_results = None
    learning_curve_plot_path = None
    learning_curve_cfg = getattr(getattr(cfg, "experiment", None), "learning_curve", None)
    if learning_curve_cfg is not None:
        learning_curve_results = load_or_run_learning_curve_results_from_config(
            wide_df,
            cfg,
            graph_view=graph_view,
            auxiliary_views=auxiliary_views,
        )
        learning_curve_plot_path = learning_curve_plot(
            results=learning_curve_results,
            output_path=output_dir / f"learning_curve_{run_suffix}.png",
            zero_shot_rmse=zero_shot_rmse,
            **plot_kwargs,
        )
        if _has_uq_summary(learning_curve_results):
            with tempfile.TemporaryDirectory() as tmpdir:
                miscalibration_path = miscalibration_area_plot(
                    learning_curve_results,
                    output_path=Path(tmpdir)
                    / f"miscalibration_area_panel_{run_suffix}.png",
                    show_xlabel=False,
                    zero_shot_value=zero_shot_uq["miscalibration_area"],
                    **plot_kwargs,
                )
                sharpness_path = sharpness_plot(
                    learning_curve_results,
                    output_path=Path(tmpdir) / f"sharpness_panel_{run_suffix}.png",
                    show_legend=False,
                    show_xlabel=False,
                    zero_shot_value=zero_shot_uq["sharpness"],
                    **plot_kwargs,
                )
                dispersion_path = dispersion_plot(
                    learning_curve_results,
                    output_path=Path(tmpdir) / f"dispersion_panel_{run_suffix}.png",
                    show_legend=False,
                    zero_shot_value=zero_shot_uq["dispersion"],
                    **plot_kwargs,
                )
                uq_summary_figure(
                    miscalibration_area_path=miscalibration_path,
                    sharpness_path=sharpness_path,
                    dispersion_path=dispersion_path,
                    output_path=output_dir / f"uq_summary_figure_{run_suffix}.png",
                )

    screening_run_cfg = _screening_run_cfg(cfg)
    screening_results = None
    screening_plot_path = None
    if screening_run_cfg is not None:
        screening_results = load_or_run_learning_curve_results_from_config(
            wide_df,
            screening_run_cfg,
            graph_view=graph_view,
            auxiliary_views=auxiliary_views,
        )
        screening_plot_path = screening_budget_plot(
            results=screening_results,
            output_path=output_dir / f"screening_budget_{run_suffix}.png",
            **plot_kwargs,
        )
    if learning_curve_plot_path is not None and screening_plot_path is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            screening_panel_path = screening_budget_plot(
                results=screening_results,
                output_path=Path(tmpdir) / f"screening_budget_panel_{run_suffix}.png",
                show_legend=False,
                **plot_kwargs,
            )
            learning_screening_figure(
                learning_curve_path=learning_curve_plot_path,
                screening_curve_path=screening_panel_path,
                output_path=output_dir / f"learning_screening_figure_{run_suffix}.png",
            )
    return learning_curve_results if learning_curve_results is not None else screening_results


def run_experiment_from_config(config_paths=None):
    return run_experiment(get_config(config_paths))
