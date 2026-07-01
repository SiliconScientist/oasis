from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from oasis.analysis import filter_structures_with_insufficient_valid_mlips
from oasis.config import get_config
from oasis.exp import (
    load_or_run_learning_curve_results_from_config,
    prepare_parity_plot_data,
)
from oasis.experiment.dataset import build_sweep_dataset_from_config, mlip_feature_names
from oasis.experiment.policy_diagnostic import (
    PolicySelectionDiagnosticArtifact,
    build_policy_selection_diagnostic_results,
    save_policy_selection_diagnostic_artifact,
)
from oasis.experiment.splits import resolve_configured_sweep_sizes
from oasis.figure import learning_screening_figure, uq_summary_figure
from oasis.learning_curve.time_accuracy import (
    GenerationTimingAggregate,
    aggregate_generation_timing,
)
from oasis.learning_curve.execution import (
    dispersion_from_spread,
    miscalibration_area,
    sharpness_from_spread,
)
from oasis.experiment_data import (
    atoms_to_graph_dataset_view,
    graph_artifact_matches_frame,
    load_probe_graph_dataset_view,
    save_aligned_graph_dataset_parquet,
)
from oasis.learning_curve.results_io import learning_curve_sweep_metadata_from_config
from oasis.experiment_config import derive_dataset_profile_paths
from oasis.io import load_sample_atoms_for_wide_df
from oasis.mlip.artifacts import (
    INFERENCE_DETAIL_COLUMNS,
    find_result_files,
    load_wide_predictions,
)
from oasis.mlip.timing import load_generation_timing_summaries
from oasis.mlip.timing import load_probe_generation_timing_summaries
from oasis.plot import (
    dispersion_plot,
    fixed_split_total_time_accuracy_plot,
    fixed_split_training_time_accuracy_plot,
    generation_time_accuracy_plot,
    learning_curve_plot,
    miscalibration_area_plot,
    parity_plot,
    screening_budget_plot,
    sharpness_plot,
    zero_shot_rmse_stage_plot,
)
_DEV_RUN_MAX_ROWS = 24
_DEV_RUN_SWEEP_SIZE = 8
_DEV_RUN_OPTUNA_TRIALS = 3


def _frame_height(frame: object) -> int:
    return int(getattr(frame, "height", len(frame)))


def _frame_head(frame: object, n_rows: int):
    if hasattr(frame, "head"):
        return frame.head(n_rows)
    if hasattr(frame, "slice"):
        return frame.slice(0, n_rows)
    if hasattr(frame, "iloc"):
        return frame.iloc[:n_rows]
    raise TypeError(f"Unsupported frame type for head(): {type(frame)!r}")


def _filter_frame_to_reactions(frame: object, reactions: list[object]):
    reaction_col = frame.get_column("reaction")
    return frame.filter(reaction_col.is_in(reactions))


def _zero_shot_rmse_from_frame(frame: object) -> float:
    parity_plot_data = prepare_parity_plot_data(frame)
    zero_shot_preds = np.mean(
        np.column_stack(list(parity_plot_data.predictions.values())),
        axis=1,
    )
    return float(
        np.sqrt(np.mean((parity_plot_data.reference - zero_shot_preds) ** 2))
    )


def _column_to_numpy(frame: object, column_name: str) -> np.ndarray:
    column = frame.get_column(column_name)
    if hasattr(column, "to_numpy"):
        return np.asarray(column.to_numpy(), dtype=float)
    return np.asarray(column, dtype=float)


def _strict_anomaly_aware_zero_shot_rmse_from_frame(frame: object) -> float:
    feature_names = mlip_feature_names(frame)
    if not feature_names:
        raise ValueError("No MLIP feature columns found for strict zero-shot RMSE.")

    prediction_matrix = np.column_stack(
        [
            _column_to_numpy(frame, f"{feature_name}_mlip_ads_eng_median")
            for feature_name in feature_names
        ]
    )
    valid_matrix = np.column_stack(
        [
            np.all(
                np.column_stack(
                    [
                        _column_to_numpy(frame, f"{feature_name}_{detail_name}") == 0.0
                        for detail_name in INFERENCE_DETAIL_COLUMNS
                        if f"{feature_name}_{detail_name}" in getattr(frame, "columns", ())
                    ]
                ),
                axis=1,
            )
            for feature_name in feature_names
        ]
    )
    if valid_matrix.shape != prediction_matrix.shape:
        raise ValueError("Prediction and validity matrices must have matching shapes.")
    valid_counts = valid_matrix.sum(axis=1)
    if np.any(valid_counts <= 0):
        raise ValueError("Strict zero-shot RMSE requires at least one valid MLIP per row.")
    masked_sum = np.where(valid_matrix, prediction_matrix, 0.0).sum(axis=1)
    zero_shot_preds = masked_sum / valid_counts
    reference = _column_to_numpy(frame, "reference_ads_eng")
    return float(np.sqrt(np.mean((reference - zero_shot_preds) ** 2)))


def _strict_validity_masks_by_mlip(frame: object) -> dict[str, np.ndarray]:
    feature_names = mlip_feature_names(frame)
    if not feature_names:
        raise ValueError("No MLIP feature columns found for strict validity masking.")

    validity_masks: dict[str, np.ndarray] = {}
    available_columns = set(getattr(frame, "columns", ()))
    for feature_name in feature_names:
        detail_column_names = [
            f"{feature_name}_{detail_name}"
            for detail_name in INFERENCE_DETAIL_COLUMNS
            if f"{feature_name}_{detail_name}" in available_columns
        ]
        if not detail_column_names:
            raise ValueError(
                "Strict validity masking requires detail columns for "
                f"{feature_name!r}."
            )
        detail_matrix = np.column_stack(
            [_column_to_numpy(frame, column_name) for column_name in detail_column_names]
        )
        validity_masks[feature_name] = np.all(detail_matrix == 0, axis=1)
    return validity_masks


def _learning_curve_zero_shot_rmse_from_frame(
    cfg: object,
    frame: object,
) -> float:
    if _exclude_anomalous_mlips_enabled(cfg):
        return _strict_anomaly_aware_zero_shot_rmse_from_frame(frame)
    return _zero_shot_rmse_from_frame(frame)


def _stage_filter_kwargs(cfg: object) -> dict[str, object]:
    return {
        "enabled": _minimum_quorum(cfg) > 0,
        "label_allowlist": None,
        "strict_inference_anomaly": True,
        "min_valid_mlips": _minimum_quorum(cfg),
    }


def _build_zero_shot_stage_rows(
    *,
    cfg: object,
    dataset_tag: str,
    dataset_label: str,
    raw_wide_df: object,
    selected_wide_df: object,
) -> list[dict[str, object]]:
    selected_reactions = selected_wide_df.get_column("reaction").to_list()
    if not selected_reactions:
        return []
    matched_subset_df = _filter_frame_to_reactions(raw_wide_df, selected_reactions)
    exclude_anomalous_mlips = _exclude_anomalous_mlips_enabled(cfg)
    return [
        {
            "dataset": dataset_tag,
            "dataset_label": dataset_label,
            "stage": "Full / all MLIPs",
            "rmse": _zero_shot_rmse_from_frame(raw_wide_df),
            "n_samples": _frame_height(raw_wide_df),
        },
        {
            "dataset": dataset_tag,
            "dataset_label": dataset_label,
            "stage": "Matched subset / all MLIPs",
            "rmse": _zero_shot_rmse_from_frame(matched_subset_df),
            "n_samples": _frame_height(matched_subset_df),
        },
        {
            "dataset": dataset_tag,
            "dataset_label": dataset_label,
            "stage": "Matched subset / anomaly-aware selection",
            "rmse": (
                _strict_anomaly_aware_zero_shot_rmse_from_frame(selected_wide_df)
                if exclude_anomalous_mlips
                else _zero_shot_rmse_from_frame(selected_wide_df)
            ),
            "n_samples": _frame_height(selected_wide_df),
        },
    ]


def _load_zero_shot_stage_rows_for_dataset(
    cfg: object,
    *,
    dataset_tag: str,
) -> list[dict[str, object]]:
    named_profile = getattr(cfg, "datasets", {}).get(dataset_tag)
    profile_paths = derive_dataset_profile_paths(dataset_tag, named_profile)
    dataset_label = (
        dataset_tag
        if named_profile is None
        else named_profile.mlip_run_dirname_or_default(dataset_tag)
    )
    base_dir = profile_paths.analysis_base_dir
    result_files = find_result_files(base_dir, enabled_models=_enabled_mlips(cfg))
    raw_wide_df = load_wide_predictions(result_files)
    filter_kwargs = _stage_filter_kwargs(cfg)
    selected_wide_df = filter_structures_with_insufficient_valid_mlips(
        raw_wide_df,
        **filter_kwargs,
    )
    raw_wide_df = _apply_dev_run_frame_cap(cfg, raw_wide_df)
    selected_wide_df = _apply_dev_run_frame_cap(cfg, selected_wide_df)
    return _build_zero_shot_stage_rows(
        cfg=cfg,
        dataset_tag=dataset_tag,
        dataset_label=dataset_label,
        raw_wide_df=raw_wide_df,
        selected_wide_df=selected_wide_df,
    )


def _preview_values(values: list[object], *, limit: int = 5) -> str:
    preview = ", ".join(repr(value) for value in values[:limit])
    if len(values) > limit:
        preview = f"{preview}, ..."
    return preview


def _mlip_selection_cfg(cfg: object):
    learning_curve_cfg = (
        cfg.experiment.learning_curve
        if getattr(cfg, "experiment", None) and cfg.experiment.learning_curve
        else None
    )
    return getattr(learning_curve_cfg, "mlip_selection", None)


def _exclude_anomalous_mlips_enabled(cfg: object) -> bool:
    mlip_selection_cfg = _mlip_selection_cfg(cfg)
    return bool(
        getattr(
            mlip_selection_cfg,
            "exclude_anomalous_mlips",
            getattr(mlip_selection_cfg, "exclude_anomalous", False),
        )
    )


def _minimum_quorum(cfg: object) -> int:
    mlip_selection_cfg = _mlip_selection_cfg(cfg)
    return int(getattr(mlip_selection_cfg, "minimum_quorum", 0))


def _enabled_mlips(cfg: object) -> list[str] | None:
    mlip_selection_cfg = _mlip_selection_cfg(cfg)
    enabled_mlips = getattr(mlip_selection_cfg, "enabled", None)
    if not enabled_mlips:
        return None
    return list(enabled_mlips)


def _resolved_dataset_path(cfg: object) -> Path | None:
    dataset_path = getattr(cfg, "resolved_dataset_path", None)
    if dataset_path is not None:
        return Path(dataset_path)
    mlip_cfg = getattr(cfg, "mlip", None)
    legacy_dataset_path = getattr(mlip_cfg, "dataset", None)
    if legacy_dataset_path is not None:
        return Path(legacy_dataset_path)
    return None


def _anomaly_aware_run_suffix(cfg: object) -> str:
    enabled = bool(_exclude_anomalous_mlips_enabled(cfg) or _minimum_quorum(cfg) > 0)
    return "anomalyaware_on" if enabled else "anomalyaware_off"


def _latent_run_suffix(cfg: object) -> str:
    experiment_cfg = getattr(cfg, "experiment", None)
    learning_curve_cfg = (
        experiment_cfg.learning_curve
        if experiment_cfg is not None and experiment_cfg.learning_curve
        else None
    )
    models_cfg = getattr(learning_curve_cfg, "models", None)
    enabled = bool(getattr(models_cfg, "use_latent", False))
    return "latent_on" if enabled else "latent_off"


def _append_output_suffix(path: Path, suffix: str) -> Path:
    if path.stem.endswith(f"_{suffix}"):
        return path
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def _plot_output_suffix(cfg: object) -> str:
    suffix_parts = [_anomaly_aware_run_suffix(cfg)]
    if bool(getattr(cfg, "dev_run", False)):
        suffix_parts.append("devrun_on")
    return "_".join(suffix_parts)


def _persistent_output_suffix(cfg: object, *, dataset_size: int) -> str:
    suffix_parts = [
        _anomaly_aware_run_suffix(cfg),
        _latent_run_suffix(cfg),
        f"n{dataset_size}",
    ]
    if bool(getattr(cfg, "dev_run", False)):
        suffix_parts.append("devrun_on")
    return "_".join(suffix_parts)


def _apply_run_output_suffixes(cfg: object, *, plot_suffix: str) -> None:
    analysis_cfg = getattr(cfg, "analysis", None)
    comparison_plot_path = getattr(analysis_cfg, "comparison_plot_path", None)
    if comparison_plot_path is not None:
        analysis_cfg.comparison_plot_path = _append_output_suffix(
            Path(comparison_plot_path),
            plot_suffix,
        )


def _apply_persistent_output_suffixes(
    cfg: object,
    *,
    dataset_size: int,
) -> str:
    suffix = _persistent_output_suffix(cfg, dataset_size=dataset_size)
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
    return suffix


def _apply_dev_run_frame_cap(cfg: object, wide_df: object):
    if not bool(getattr(cfg, "dev_run", False)):
        return wide_df
    pre_rows = _frame_height(wide_df)
    capped_df = _frame_head(wide_df, min(_DEV_RUN_MAX_ROWS, pre_rows))
    print(
        "Applied dev_run row cap:"
        f" {pre_rows} -> {_frame_height(capped_df)} rows"
    )
    return capped_df


def _apply_dev_run_curve_overrides(cfg: object, *, n_samples: int) -> None:
    if not bool(getattr(cfg, "dev_run", False)):
        return

    experiment_cfg = getattr(cfg, "experiment", None)
    if experiment_cfg is None:
        return

    learning_curve_cfg = getattr(experiment_cfg, "learning_curve", None)
    if learning_curve_cfg is None:
        return

    max_sweep_size = max(1, n_samples - 1)
    sweep_size = min(_DEV_RUN_SWEEP_SIZE, max_sweep_size)
    learning_curve_cfg.n_repeats = 1
    learning_curve_cfg.sweep_sizes = [sweep_size]
    if hasattr(learning_curve_cfg, "sweep_fractions"):
        learning_curve_cfg.sweep_fractions = []
    if hasattr(learning_curve_cfg, "min_train"):
        learning_curve_cfg.min_train = sweep_size
    if hasattr(learning_curve_cfg, "max_train"):
        learning_curve_cfg.max_train = sweep_size
    if hasattr(learning_curve_cfg, "step"):
        learning_curve_cfg.step = 1

    models_cfg = getattr(learning_curve_cfg, "models", None)
    optuna_overrides: list[str] = []
    if models_cfg is not None:
        for family_name in ("moe", "probe_gnn", "gnn_direct"):
            family_cfg = getattr(models_cfg, family_name, None)
            tuning_cfg = getattr(family_cfg, "tuning", None)
            optuna_cfg = getattr(tuning_cfg, "optuna", None)
            if optuna_cfg is None:
                continue
            pre_trials = int(optuna_cfg.n_trials)
            if pre_trials <= _DEV_RUN_OPTUNA_TRIALS:
                continue
            optuna_cfg.n_trials = _DEV_RUN_OPTUNA_TRIALS
            optuna_overrides.append(
                f"{family_name}.tuning.optuna.n_trials={pre_trials}->{_DEV_RUN_OPTUNA_TRIALS}"
            )
    print(
        "Applied dev_run learning-curve overrides:"
        f" n_repeats=1, sweep_sizes=[{sweep_size}]"
    )
    if optuna_overrides:
        print(
            "Applied dev_run Optuna overrides:"
            f" {', '.join(optuna_overrides)}"
        )


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


def _write_policy_selection_diagnostic(
    *,
    cfg: object,
    wide_df: object,
    graph_view: object,
    auxiliary_views: dict[str, object] | None,
    output_dir: Path,
    run_suffix: str,
) -> Path | None:
    experiment_cfg = getattr(cfg, "experiment", None)
    learning_curve_cfg = getattr(experiment_cfg, "learning_curve", None)
    screening_cfg = getattr(experiment_cfg, "screening", None)
    if learning_curve_cfg is None or screening_cfg is None:
        return None
    if getattr(screening_cfg, "screen_fraction", None) is None:
        return None

    from oasis.learning_curve.registry import (
        default_sweep_model_families,
        enabled_learning_curve_model_names_from_config,
    )

    dataset = build_sweep_dataset_from_config(
        wide_df,
        cfg,
        graph_view=graph_view,
        auxiliary_views=auxiliary_views,
    )
    enabled_model_names = enabled_learning_curve_model_names_from_config(
        getattr(learning_curve_cfg, "models", None)
    )
    model_families = default_sweep_model_families(
        enabled_model_names,
        config=getattr(learning_curve_cfg, "models", None),
    )
    requested_sweep_sizes = resolve_configured_sweep_sizes(
        dataset.n_samples,
        min_train=learning_curve_cfg.min_train,
        max_train=learning_curve_cfg.max_train,
        step=getattr(learning_curve_cfg, "step", 1),
        sweep_sizes=getattr(learning_curve_cfg, "sweep_sizes", ()),
        sweep_fractions=getattr(learning_curve_cfg, "sweep_fractions", ()),
    )
    diagnostic_results = build_policy_selection_diagnostic_results(
        dataset,
        min_train=learning_curve_cfg.min_train,
        max_train=learning_curve_cfg.max_train,
        step=getattr(learning_curve_cfg, "step", 1),
        n_repeats=learning_curve_cfg.n_repeats,
        seed=cfg.seed if getattr(cfg, "seed", None) is not None else 42,
        model_families=model_families,
        outer_validation_fraction=getattr(learning_curve_cfg, "validation_fraction", 0.2),
        outer_min_val_size=getattr(learning_curve_cfg, "min_val_size", 1),
        outer_min_tuning_val_size=getattr(learning_curve_cfg, "min_tuning_val_size", 1),
        outer_calibration_enabled=getattr(learning_curve_cfg, "calibration_enabled", False),
        outer_calibration_fraction=getattr(learning_curve_cfg, "calibration_fraction", 0.2),
        outer_min_cal_size=getattr(learning_curve_cfg, "min_cal_size", 1),
        outer_min_inner_train_size=getattr(learning_curve_cfg, "min_inner_train_size", 1),
        min_test_size=getattr(learning_curve_cfg, "min_test_size", 1),
        screening_fraction=screening_cfg.screen_fraction,
        min_screen_size=getattr(screening_cfg, "min_screen_size", 1),
        screening_validation_fraction=getattr(screening_cfg, "validation_fraction", 0.2),
        screening_min_val_size=getattr(screening_cfg, "min_val_size", 1),
        screening_min_tuning_val_size=getattr(screening_cfg, "min_tuning_val_size", 1),
        screening_calibration_enabled=getattr(screening_cfg, "calibration_enabled", False),
        screening_calibration_fraction=getattr(screening_cfg, "calibration_fraction", 0.2),
        screening_min_cal_size=getattr(screening_cfg, "min_cal_size", 1),
        screening_min_inner_train_size=getattr(screening_cfg, "min_inner_train_size", 1),
        requested_sweep_sizes=requested_sweep_sizes,
    )
    metadata = learning_curve_sweep_metadata_from_config(
        cfg,
        dataset_size=int(len(wide_df)),
        mlip_feature_names=mlip_feature_names(wide_df),
    )
    artifact_path = output_dir / f"policy_selection_diagnostic_{run_suffix}.json"
    save_policy_selection_diagnostic_artifact(
        PolicySelectionDiagnosticArtifact(
            metadata=metadata,
            results=diagnostic_results,
        ),
        artifact_path,
    )
    diagnostic_results.detail_df.to_csv(
        output_dir / f"policy_selection_diagnostic_detail_{run_suffix}.csv",
        index=False,
    )
    diagnostic_results.summary_df.to_csv(
        output_dir / f"policy_selection_diagnostic_summary_{run_suffix}.csv",
        index=False,
    )
    return artifact_path


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
    probe_cfg = cfg.probe_features
    models_cfg = (
        cfg.experiment.learning_curve.models
        if cfg.experiment and cfg.experiment.learning_curve
        else None
    )
    probe_gnn_cfg = getattr(models_cfg, "probe_gnn", None)
    probe_gnn_enabled = bool(getattr(probe_gnn_cfg, "enabled", False)) or bool(
        getattr(models_cfg, "use_probe_gnn", False)
    )

    if not probe_gnn_enabled:
        return False

    if probe_cfg is None:
        raise ValueError(
            "probe_gnn requires external probe artifacts; configure "
            "probe_features.dataset_path and probe_features.mlip_results_dir."
        )

    if not Path(probe_cfg.dataset_path).is_file():
        raise FileNotFoundError(
            "probe_gnn requires an existing external probe dataset at "
            f"{probe_cfg.dataset_path}"
        )

    if not Path(probe_cfg.mlip_results_dir).is_dir():
        raise FileNotFoundError(
            "probe_gnn requires an existing external probe MLIP results "
            f"directory at {probe_cfg.mlip_results_dir}"
        )

    return True


def load_filtered_wide_predictions(cfg: object):
    base_dir = getattr(cfg, "resolved_mlip_results_dir", None)
    if base_dir is None:
        base_dir = cfg.analysis.base_dir if cfg.analysis else Path("data/mlips")
    result_files = find_result_files(base_dir, enabled_models=_enabled_mlips(cfg))
    raw_wide_df = load_wide_predictions(result_files)
    print(f"Loaded combined wide_df with {_frame_height(raw_wide_df)} rows")
    filter_kwargs = _stage_filter_kwargs(cfg)
    wide_df = filter_structures_with_insufficient_valid_mlips(
        raw_wide_df,
        **filter_kwargs,
    )
    return wide_df, result_files, raw_wide_df


def write_zero_shot_rmse_stage_plot(
    *,
    cfg: object,
    raw_wide_df: object,
    selected_wide_df: object,
    output_dir: Path,
    run_suffix: str,
) -> Path | None:
    if not bool(_exclude_anomalous_mlips_enabled(cfg) or _minimum_quorum(cfg) > 0):
        return None

    dataset_tag = getattr(getattr(cfg, "dataset_profile", None), "tag", "dataset")
    named_profile = getattr(cfg, "datasets", {}).get(dataset_tag)
    dataset_label = (
        dataset_tag
        if named_profile is None
        else named_profile.mlip_run_dirname_or_default(dataset_tag)
    )
    stage_rows = _build_zero_shot_stage_rows(
        cfg=cfg,
        dataset_tag=dataset_tag,
        dataset_label=dataset_label,
        raw_wide_df=raw_wide_df,
        selected_wide_df=selected_wide_df,
    )
    if not stage_rows:
        print("Skipping zero-shot stage plot: selected dataset is empty")
        return None
    stage_df = pd.DataFrame(stage_rows)
    output_path = output_dir / f"zero_shot_rmse_stage_{run_suffix}.png"
    return zero_shot_rmse_stage_plot(stage_df, output_path=output_path)


def write_zero_shot_stage_parity_plots(
    *,
    cfg: object,
    selected_wide_df: object,
    output_dir: Path,
    run_suffix: str,
) -> list[Path]:
    if not bool(_exclude_anomalous_mlips_enabled(cfg) or _minimum_quorum(cfg) > 0):
        return []

    if _frame_height(selected_wide_df) == 0:
        print("Skipping zero-shot stage parity plots: selected dataset is empty")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = [
        parity_plot(
            selected_wide_df,
            output_path=output_dir
            / f"mlips_vs_dft_parity_matched_subset_{run_suffix}.png",
            title="Parity plot (matched subset / all MLIPs)",
        )
    ]
    if _exclude_anomalous_mlips_enabled(cfg):
        saved_paths.append(
            parity_plot(
                selected_wide_df,
                output_path=output_dir
                / f"mlips_vs_dft_parity_anomaly_aware_{run_suffix}.png",
                title="Parity plot (matched subset / anomaly-aware selection)",
                validity_mask_by_prediction=_strict_validity_masks_by_mlip(
                    selected_wide_df
                ),
            )
        )
    return saved_paths


def write_all_datasets_zero_shot_rmse_stage_plot(
    *,
    cfg: object,
    output_dir: Path,
    run_suffix: str,
) -> Path | None:
    if not bool(_exclude_anomalous_mlips_enabled(cfg) or _minimum_quorum(cfg) > 0):
        return None

    configured_tags = list(getattr(cfg, "datasets", {}))
    current_tag = getattr(getattr(cfg, "dataset_profile", None), "tag", None)
    if current_tag and current_tag not in configured_tags:
        configured_tags.insert(0, current_tag)
    if len(configured_tags) <= 1:
        return None

    stage_rows: list[dict[str, object]] = []
    for dataset_tag in configured_tags:
        stage_rows.extend(
            _load_zero_shot_stage_rows_for_dataset(cfg, dataset_tag=dataset_tag)
        )
    if not stage_rows:
        return None

    output_path = output_dir / f"zero_shot_rmse_stage_all_datasets_{run_suffix}.png"
    return zero_shot_rmse_stage_plot(
        pd.DataFrame(stage_rows),
        output_path=output_path,
    )


def build_auxiliary_views(
    cfg: object,
    wide_df: object,
    probe_gnn_enabled: bool,
) -> tuple[object, dict]:
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
        probe_graph_view = load_probe_graph_dataset_view(
            probe_cfg.dataset_path,
            mlip_results_dir=probe_cfg.mlip_results_dir,
        )
        reactions = wide_df.get_column("reaction").to_list()
        reaction_set = set(reactions)
        probe_sample_ids = set(probe_graph_view.sample_ids)
        missing_reactions = [r for r in reactions if r not in probe_sample_ids]
        if missing_reactions:
            extra_probe_reactions = [
                sample_id
                for sample_id in probe_graph_view.sample_ids
                if sample_id not in reaction_set
            ]
            message = (
                "Probe GNN dataset does not align with wide_df reactions: "
                f"{len(missing_reactions)} missing from "
                f"{probe_cfg.dataset_path}"
            )
            message += f" ({_preview_values(missing_reactions)})."
            if extra_probe_reactions:
                message += (
                    " Probe dataset also contains extra reactions with no "
                    f"wide_df match ({_preview_values(extra_probe_reactions)})."
                )
            raise KeyError(message)
        auxiliary_views["probe_gnn_records"] = [probe_graph_view[r] for r in reactions]
        print(
            f"Loaded {len(reactions)} probe-augmented graphs from {probe_cfg.dataset_path}"
        )

    return wide_df, auxiliary_views


def _method_generation_timing_overrides(
    cfg: object,
    wide_df: object,
) -> dict[str, GenerationTimingAggregate] | None:
    overrides: dict[str, GenerationTimingAggregate] = {}
    probe_cfg = getattr(cfg, "probe_features", None)
    learning_curve_cfg = (
        cfg.experiment.learning_curve
        if getattr(cfg, "experiment", None) and cfg.experiment.learning_curve
        else None
    )
    models_cfg = getattr(learning_curve_cfg, "models", None)
    probe_gnn_cfg = getattr(models_cfg, "probe_gnn", None)
    probe_gnn_enabled = bool(getattr(probe_gnn_cfg, "enabled", False)) or bool(
        getattr(models_cfg, "use_probe_gnn", False)
    )
    gnn_direct_cfg = getattr(models_cfg, "gnn_direct", None)
    gnn_direct_enabled = bool(getattr(gnn_direct_cfg, "enabled", False)) or bool(
        getattr(models_cfg, "use_gnn_direct", False)
    )
    feature_names = mlip_feature_names(wide_df)
    latent_cfg = getattr(models_cfg, "latent", None)
    latent_enabled = bool(getattr(models_cfg, "use_latent", False))

    if latent_enabled and latent_cfg is not None:
        latent_timing_path = Path(
            getattr(latent_cfg, "timing_path", None)
            or str(latent_cfg.csv_path) + ".timing.json"
        )
        if latent_timing_path.is_file():
            with latent_timing_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            generation_time_s = float(payload.get("generation_time_s", 0.0) or 0.0)
            overrides["latent"] = GenerationTimingAggregate(
                generation_time_s=generation_time_s,
                generation_time_slab_s=0.0,
                generation_time_adslab_s=0.0,
                generation_steps_total=0,
                generation_steps_slab=0,
                generation_steps_adslab=0,
                time_per_step_s=None,
                mlip_feature_names=("latent_csv",),
            )

    if (
        probe_gnn_enabled
        and probe_cfg is not None
        and getattr(probe_cfg, "mlip_results_dir", None) is not None
        and feature_names
    ):
        try:
            probe_result_files = find_result_files(
                probe_cfg.mlip_results_dir,
                pattern="*_result.json",
            )
            probe_generation_timing_by_mlip = load_probe_generation_timing_summaries(
                probe_result_files
            )
            overrides["probe_gnn"] = aggregate_generation_timing(
                probe_generation_timing_by_mlip,
                mlip_feature_names=feature_names,
            )
        except (FileNotFoundError, KeyError):
            pass

    if gnn_direct_enabled:
        sample_ids = wide_df.get_column("reaction").to_list()
        sample_atoms = load_sample_atoms_for_wide_df(wide_df, cfg)
        start_time = perf_counter()
        atoms_to_graph_dataset_view(sample_ids, sample_atoms)
        generation_time_s = perf_counter() - start_time
        generation_steps_total = len(sample_ids)
        overrides["gnn_direct"] = GenerationTimingAggregate(
            generation_time_s=generation_time_s,
            generation_time_slab_s=0.0,
            generation_time_adslab_s=0.0,
            generation_steps_total=generation_steps_total,
            generation_steps_slab=0,
            generation_steps_adslab=0,
            time_per_step_s=(
                None
                if generation_steps_total <= 0
                else generation_time_s / generation_steps_total
            ),
            mlip_feature_names=("atoms_to_graph",),
        )

    return overrides or None


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
    print(f"Loaded {len(sample_atoms)} adsorbed structures from {_resolved_dataset_path(cfg)}")
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


def write_uq_summary_figure(
    *,
    results: object,
    output_path: Path,
    run_suffix: str,
    zero_shot_uq: dict[str, float],
    panel_prefix: str = "",
    **plot_kwargs,
) -> Path | None:
    if not _has_uq_summary(results):
        return None

    panel_stem_prefix = f"{panel_prefix}_" if panel_prefix else ""
    with tempfile.TemporaryDirectory() as tmpdir:
        miscalibration_path = miscalibration_area_plot(
            results,
            output_path=Path(tmpdir)
            / f"{panel_stem_prefix}miscalibration_area_panel_{run_suffix}.png",
            show_xlabel=False,
            zero_shot_value=zero_shot_uq["miscalibration_area"],
            **plot_kwargs,
        )
        sharpness_path = sharpness_plot(
            results,
            output_path=Path(tmpdir)
            / f"{panel_stem_prefix}sharpness_panel_{run_suffix}.png",
            show_legend=False,
            show_xlabel=False,
            zero_shot_value=zero_shot_uq["sharpness"],
            **plot_kwargs,
        )
        dispersion_path = dispersion_plot(
            results,
            output_path=Path(tmpdir)
            / f"{panel_stem_prefix}dispersion_panel_{run_suffix}.png",
            show_legend=False,
            zero_shot_value=zero_shot_uq["dispersion"],
            **plot_kwargs,
        )
        return uq_summary_figure(
            miscalibration_area_path=miscalibration_path,
            sharpness_path=sharpness_path,
            dispersion_path=dispersion_path,
            output_path=output_path,
        )


def write_time_accuracy_plots(
    *,
    learning_curve_results: object,
    result_files: list[Path],
    wide_df: object,
    output_dir: Path,
    run_suffix: str,
    generation_timing_by_method: dict[str, GenerationTimingAggregate] | None = None,
) -> Path | None:
    if not result_files:
        print("Skipping time-accuracy plots: no MLIP result JSON files were found")
        return None
    generation_timing_by_mlip = load_generation_timing_summaries(result_files)
    if not generation_timing_by_mlip:
        print("Skipping time-accuracy plots: no MLIP timing summaries were loaded")
        return None
    feature_names = mlip_feature_names(wide_df)
    if not feature_names:
        print("Skipping time-accuracy plots: no MLIP feature columns were found")
        return None
    try:
        aggregate_generation_timing(
            generation_timing_by_mlip,
            mlip_feature_names=feature_names,
        )
    except KeyError as exc:
        print(f"Skipping time-accuracy plots: {exc}")
        return None
    generation_path = generation_time_accuracy_plot(
        results=learning_curve_results,
        generation_timing_by_mlip=generation_timing_by_mlip,
        mlip_feature_names=feature_names,
        generation_timing_by_method=generation_timing_by_method,
        output_path=output_dir / f"generation_time_accuracy_{run_suffix}.png",
    )
    return generation_path


def write_fixed_split_time_accuracy_plots(
    *,
    learning_curve_results: object,
    result_files: list[Path],
    wide_df: object,
    output_dir: Path,
    run_suffix: str,
    train_fraction: float = 0.8,
    generation_timing_by_method: dict[str, GenerationTimingAggregate] | None = None,
) -> tuple[Path, Path] | None:
    if not result_files:
        print("Skipping fixed-split time-accuracy plots: no MLIP result JSON files were found")
        return None
    generation_timing_by_mlip = load_generation_timing_summaries(result_files)
    if not generation_timing_by_mlip:
        print("Skipping fixed-split time-accuracy plots: no MLIP timing summaries were loaded")
        return None
    feature_names = mlip_feature_names(wide_df)
    if not feature_names:
        print("Skipping fixed-split time-accuracy plots: no MLIP feature columns were found")
        return None
    try:
        aggregate_generation_timing(
            generation_timing_by_mlip,
            mlip_feature_names=feature_names,
        )
    except KeyError as exc:
        print(f"Skipping fixed-split time-accuracy plots: {exc}")
        return None
    dataset_size = _frame_height(wide_df)
    training_path = fixed_split_training_time_accuracy_plot(
        results=learning_curve_results,
        generation_timing_by_mlip=generation_timing_by_mlip,
        dataset_size=dataset_size,
        train_fraction=train_fraction,
        mlip_feature_names=feature_names,
        generation_timing_by_method=generation_timing_by_method,
        output_path=output_dir / f"fixed_split_training_time_accuracy_{run_suffix}.png",
    )
    total_path = fixed_split_total_time_accuracy_plot(
        results=learning_curve_results,
        generation_timing_by_mlip=generation_timing_by_mlip,
        dataset_size=dataset_size,
        train_fraction=train_fraction,
        mlip_feature_names=feature_names,
        generation_timing_by_method=generation_timing_by_method,
        output_path=output_dir / f"fixed_split_total_time_accuracy_{run_suffix}.png",
    )
    return training_path, total_path


def run_experiment(cfg: object):
    run_suffix = _plot_output_suffix(cfg)
    _apply_run_output_suffixes(cfg, plot_suffix=run_suffix)
    probe_gnn_enabled = ensure_probe_artifacts(cfg)
    wide_df, result_files, raw_wide_df = load_filtered_wide_predictions(cfg)
    raw_wide_df = _apply_dev_run_frame_cap(cfg, raw_wide_df)
    wide_df = _apply_dev_run_frame_cap(cfg, wide_df)
    zero_shot_stage_selected_wide_df = wide_df
    wide_df, auxiliary_views = build_auxiliary_views(cfg, wide_df, probe_gnn_enabled)
    _apply_persistent_output_suffixes(cfg, dataset_size=_frame_height(wide_df))
    _apply_dev_run_curve_overrides(cfg, n_samples=_frame_height(wide_df))
    generation_timing_by_method = _method_generation_timing_overrides(cfg, wide_df)
    write_parity_plot(
        cfg,
        wide_df,
        result_files,
        run_suffix,
    )
    output_dir = cfg.plot.output_dir if cfg.plot else Path("data/results/plots")
    write_zero_shot_stage_parity_plots(
        cfg=cfg,
        selected_wide_df=zero_shot_stage_selected_wide_df,
        output_dir=output_dir,
        run_suffix=run_suffix,
    )
    graph_view = prepare_graph_view(cfg, wide_df)
    write_zero_shot_rmse_stage_plot(
        cfg=cfg,
        raw_wide_df=raw_wide_df,
        selected_wide_df=zero_shot_stage_selected_wide_df,
        output_dir=output_dir,
        run_suffix=run_suffix,
    )
    write_all_datasets_zero_shot_rmse_stage_plot(
        cfg=cfg,
        output_dir=output_dir,
        run_suffix=run_suffix,
    )
    curve_window_cfg = getattr(cfg.plot, "curve_window", None) if cfg.plot else None
    fixed_split_cfg = getattr(cfg.plot, "fixed_split", None) if cfg.plot else None
    fixed_split_train_fraction = float(
        getattr(fixed_split_cfg, "train_fraction", 0.8)
    )
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
    zero_shot_rmse = _learning_curve_zero_shot_rmse_from_frame(cfg, wide_df)
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
        write_time_accuracy_plots(
            learning_curve_results=learning_curve_results,
            result_files=result_files,
            wide_df=wide_df,
            output_dir=output_dir,
            run_suffix=run_suffix,
            generation_timing_by_method=generation_timing_by_method,
        )
        write_fixed_split_time_accuracy_plots(
            learning_curve_results=learning_curve_results,
            result_files=result_files,
            wide_df=wide_df,
            output_dir=output_dir,
            run_suffix=run_suffix,
            train_fraction=fixed_split_train_fraction,
            generation_timing_by_method=generation_timing_by_method,
        )
        write_uq_summary_figure(
            results=learning_curve_results,
            output_path=output_dir / f"uq_summary_figure_{run_suffix}.png",
            run_suffix=run_suffix,
            zero_shot_uq=zero_shot_uq,
            **plot_kwargs,
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
        write_uq_summary_figure(
            results=screening_results,
            output_path=output_dir / f"screening_uq_summary_figure_{run_suffix}.png",
            run_suffix=run_suffix,
            zero_shot_uq=zero_shot_uq,
            panel_prefix="screening",
            **plot_kwargs,
        )
    if learning_curve_cfg is not None and screening_run_cfg is not None:
        _write_policy_selection_diagnostic(
            cfg=cfg,
            wide_df=wide_df,
            graph_view=graph_view,
            auxiliary_views=auxiliary_views,
            output_dir=output_dir,
            run_suffix=run_suffix,
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
