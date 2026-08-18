from __future__ import annotations

from io import StringIO
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

if "MPLCONFIGDIR" not in os.environ:
    mplconfigdir = Path(tempfile.gettempdir()) / "oasis-matplotlib"
    mplconfigdir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mplconfigdir)

import matplotlib

if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from oasis.exp import prepare_parity_plot_data
from oasis.learning_curve.time_accuracy import (
    GenerationTimingAggregate,
    build_fixed_split_time_accuracy_table,
    build_mean_time_accuracy_table,
    build_time_accuracy_table,
)
from oasis.mlip.timing import MlipGenerationTimingSummary
from oasis.plot_style import get_plot_style
from oasis.sweep import LearningCurveResults

_MLIP_DISPLAY_NAMES = {
    "7net-omni": "7Net-Omni",
    "mace-mh-1": "MACE-MH-1",
    "mattersim-v1-5m": "MatterSim-v1-5M",
    "orb-v3-conservative-inf-omat": "ORB-v3\nconservative",
    "uma-s-1p1": "UMA-s-1p1",
}
_MLIP_SWARM_MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">", "8", "h", "*")
_DEFAULT_PLOT_FONTSIZE = 16
_DEFAULT_TICK_FONTSIZE = 8
_DEFAULT_LEGEND_FONTSIZE = 8
_METHOD_PLOT_STYLES = (
    ("ridge", "ridge_df", "ridge_uq_df", "Ridge", "o", "tab:blue"),
    (
        "kernel_ridge",
        "kernel_ridge_df",
        "kernel_ridge_uq_df",
        "Kernel Ridge",
        "X",
        "tab:cyan",
    ),
    ("lasso", "lasso_df", "lasso_uq_df", "Lasso", "s", "tab:orange"),
    ("elastic", "elastic_df", "elastic_uq_df", "Elastic Net", "D", "tab:purple"),
    ("residual", "resid_df", "resid_uq_df", "Residual", "^", "tab:green"),
    (
        "weighted_linear",
        "weighted_linear_df",
        "weighted_linear_uq_df",
        "Weighted linear",
        "*",
        "tab:gray",
    ),
    (
        "weighted_simplex",
        "weighted_simplex_df",
        "weighted_simplex_uq_df",
        "Weighted simplex",
        "8",
        "teal",
    ),
    ("graph_mean", "graph_mean_df", "graph_mean_uq_df", "Graph mean", "P", "tab:red"),
    ("moe", "moe_df", "moe_uq_df", "MoE", "*", "tab:purple"),
    ("gnn_direct", "gnn_direct_df", "gnn_direct_uq_df", "GNN direct", "s", "tab:cyan"),
    ("probe_gnn", "probe_gnn_df", "probe_gnn_uq_df", "Probe GNN", "D", "tab:olive"),
    ("latent", "latent_df", "latent_uq_df", "Latent", "v", "tab:brown"),
    (
        "fitted_latent",
        "fitted_latent_df",
        "fitted_latent_uq_df",
        "Fitted latent",
        "<",
        "sienna",
    ),
)
_METHOD_RESULT_FIELDS = {
    method_name: result_field for method_name, result_field, *_ in _METHOD_PLOT_STYLES
}
_METHOD_UQ_FIELDS = {
    method_name: uq_field for method_name, _, uq_field, *_ in _METHOD_PLOT_STYLES
}


def _tick_fontsize() -> int:
    return get_plot_style().tick_fontsize


def _mlip_display_name(mlip: str) -> str:
    default_name = _MLIP_DISPLAY_NAMES.get(mlip, mlip)
    return get_plot_style().mlip_alias(mlip, default_name)


def _mlip_marker_map(mlips: list[str] | tuple[str, ...]) -> dict[str, str]:
    return {
        mlip: _MLIP_SWARM_MARKERS[index % len(_MLIP_SWARM_MARKERS)]
        for index, mlip in enumerate(sorted(dict.fromkeys(mlips)))
    }


def _method_color(method_name: str, default: str) -> str:
    return get_plot_style().method_color(method_name, default)


def _dataset_color(dataset_name: str) -> str:
    return get_plot_style().dataset_color(dataset_name)


def _mlip_color(mlip_name: str) -> str:
    return get_plot_style().mlip_color(mlip_name)


def _policy_color(policy_name: str) -> str:
    return get_plot_style().policy_color(policy_name)


def _baseline_color(baseline_name: str) -> str:
    return get_plot_style().baseline_color(baseline_name)


def _stage_color(stage_name: str) -> str:
    return get_plot_style().stage_color(stage_name)


def _oracle_series_color() -> str:
    return _policy_color("oracle_best_held_out_rmse")


def _normalize_dataset_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _dataset_aliases(value: str) -> set[str]:
    raw = str(value).strip().lower()
    return {raw, _normalize_dataset_token(raw)}


def _ordered_learning_curve_frame(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    if frame is None or frame.empty:
        return frame
    if "n_train" not in frame.columns:
        raise ValueError("learning-curve result frames must contain an n_train column.")
    return frame.sort_values("n_train").reset_index(drop=True)


def oracle_learning_curve_frame(
    results: LearningCurveResults,
    *,
    enabled_method_names: list[str] | tuple[str, ...],
    dataset: str,
    dataset_label: str | None = None,
) -> pd.DataFrame:
    unknown_methods = sorted(
        method_name
        for method_name in enabled_method_names
        if method_name not in _METHOD_RESULT_FIELDS
    )
    if unknown_methods:
        raise ValueError(
            f"enabled_method_names contains unknown methods: {unknown_methods}"
        )

    oracle_rows: list[pd.DataFrame] = []
    for method_name in enabled_method_names:
        frame = getattr(results, _METHOD_RESULT_FIELDS[method_name])
        if frame is None or frame.empty:
            continue
        if "n_train" not in frame.columns or "rmse_mean" not in frame.columns:
            raise ValueError(
                f"{method_name!r} result frame must contain n_train and rmse_mean columns."
            )
        oracle_rows.append(
            frame.loc[:, ["n_train", "rmse_mean"]]
            .assign(oracle_method=method_name)
            .rename(columns={"rmse_mean": "oracle_rmse"})
        )

    if not oracle_rows:
        raise ValueError("No enabled learning-curve result frames were available.")

    candidates = (
        pd.concat(oracle_rows, ignore_index=True)
        .sort_values(["n_train", "oracle_rmse", "oracle_method"])
        .reset_index(drop=True)
    )
    oracle = (
        candidates.groupby("n_train", as_index=False, sort=True)
        .first()
        .loc[:, ["n_train", "oracle_rmse", "oracle_method"]]
    )
    oracle.insert(
        0, "dataset_label", dataset if dataset_label is None else dataset_label
    )
    oracle.insert(0, "dataset", dataset)
    return oracle


def oracle_uq_curve_frame(
    results: LearningCurveResults,
    *,
    enabled_method_names: list[str] | tuple[str, ...],
    dataset: str,
    dataset_label: str | None = None,
) -> pd.DataFrame:
    unknown_methods = sorted(
        method_name
        for method_name in enabled_method_names
        if method_name not in _METHOD_UQ_FIELDS
        or method_name not in _METHOD_RESULT_FIELDS
    )
    if unknown_methods:
        raise ValueError(
            f"enabled_method_names contains unknown methods: {unknown_methods}"
        )

    oracle_rows: list[pd.DataFrame] = []
    for method_name in enabled_method_names:
        result_frame = getattr(results, _METHOD_RESULT_FIELDS[method_name])
        uq_frame = getattr(results, _METHOD_UQ_FIELDS[method_name])
        if (
            result_frame is None
            or result_frame.empty
            or uq_frame is None
            or uq_frame.empty
        ):
            continue
        result_required_columns = {
            "n_train",
            "rmse_mean",
        }
        missing_result_columns = result_required_columns.difference(
            result_frame.columns
        )
        if missing_result_columns:
            raise ValueError(
                f"{method_name!r} result frame is missing required columns: "
                f"{sorted(missing_result_columns)}"
            )
        uq_required_columns = {
            "n_train",
            "miscalibration_area",
            "sharpness",
            "dispersion",
        }
        missing_uq_columns = uq_required_columns.difference(uq_frame.columns)
        if missing_uq_columns:
            raise ValueError(
                f"{method_name!r} UQ frame is missing required columns: "
                f"{sorted(missing_uq_columns)}"
            )
        oracle_rows.append(
            result_frame.loc[:, ["n_train", "rmse_mean"]]
            .merge(
                uq_frame.loc[
                    :,
                    ["n_train", "miscalibration_area", "sharpness", "dispersion"],
                ],
                on="n_train",
                how="inner",
            )
            .assign(oracle_method=method_name)
            .rename(
                columns={
                    "rmse_mean": "oracle_rmse",
                    "miscalibration_area": "oracle_miscalibration_area",
                    "sharpness": "oracle_sharpness",
                    "dispersion": "oracle_dispersion",
                }
            )
        )

    if not oracle_rows:
        raise ValueError("No enabled UQ result frames were available.")

    candidates = (
        pd.concat(oracle_rows, ignore_index=True)
        .sort_values(["n_train", "oracle_rmse", "oracle_method"])
        .reset_index(drop=True)
    )
    oracle = (
        candidates.groupby("n_train", as_index=False, sort=True)
        .first()
        .loc[
            :,
            [
                "n_train",
                "oracle_miscalibration_area",
                "oracle_sharpness",
                "oracle_dispersion",
                "oracle_method",
            ],
        ]
    )
    oracle.insert(
        0, "dataset_label", dataset if dataset_label is None else dataset_label
    )
    oracle.insert(0, "dataset", dataset)
    return oracle


def _ordered_screening_frame(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    if frame is None or frame.empty:
        return frame
    if "n_budget" not in frame.columns:
        raise ValueError("screening result frames must contain an n_budget column.")
    return frame.sort_values("n_budget").reset_index(drop=True)


def _filter_curve_frame(
    frame: pd.DataFrame | None,
    *,
    x_column: str,
    min_x: int | None,
    max_x: int | None,
    include_x: list[int] | tuple[int, ...] | None,
) -> pd.DataFrame | None:
    if frame is None or frame.empty:
        return frame
    filtered = frame
    if min_x is not None:
        filtered = filtered.loc[filtered[x_column] >= min_x]
    if max_x is not None:
        filtered = filtered.loc[filtered[x_column] <= max_x]
    if include_x:
        allowed_x = {int(value) for value in include_x}
        filtered = filtered.loc[filtered[x_column].isin(sorted(allowed_x))]
    return filtered.reset_index(drop=True)


def _screening_metric_columns(frame: pd.DataFrame) -> tuple[str, str]:
    if "cv_rmse_mean" in frame.columns and "cv_rmse_std" in frame.columns:
        return "cv_rmse_mean", "cv_rmse_std"
    if "rmse_mean" in frame.columns and "rmse_std" in frame.columns:
        return "rmse_mean", "rmse_std"
    raise ValueError("screening result frames must contain cv_rmse_mean/cv_rmse_std.")


def _set_integer_x_ticks(ax: Any) -> None:
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _format_train_fraction_label(train_fraction: float) -> str:
    percentage = 100.0 * float(train_fraction)
    return f"{percentage:g}%"


def _uq_x_axis_config(results: LearningCurveResults) -> tuple[str, str, str]:
    for _, _, uq_field, *_ in _METHOD_PLOT_STYLES:
        frame = getattr(results, uq_field)
        if frame is None or frame.empty:
            continue
        if "n_budget" in frame.columns:
            return (
                "n_budget",
                "budget",
                "Sample budget",
            )
        if "n_train" in frame.columns:
            return (
                "n_train",
                "train size",
                "Train size",
            )
    return ("n_train", "train size", "Train size")


def _plot_uq_metric_curve(
    results: LearningCurveResults,
    *,
    metric_column: str,
    output_path: str | Path,
    fontsize: int = _DEFAULT_PLOT_FONTSIZE,
    min_x: int | None = None,
    max_x: int | None = None,
    include_x: list[int] | tuple[int, ...] | None = None,
    show_legend: bool = True,
    legend_outside_right: bool = False,
    show_xlabel: bool = True,
    zero_shot_value: float | None = None,
    title_prefix: str,
    ylabel: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    _draw_uq_metric_curve(
        ax,
        results,
        metric_column=metric_column,
        fontsize=fontsize,
        min_x=min_x,
        max_x=max_x,
        include_x=include_x,
        show_legend=show_legend,
        legend_outside_right=legend_outside_right,
        show_xlabel=show_xlabel,
        zero_shot_value=zero_shot_value,
        title_prefix=title_prefix,
        ylabel=ylabel,
    )
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def _draw_uq_metric_curve(
    ax: Any,
    results: LearningCurveResults,
    *,
    metric_column: str,
    fontsize: int = _DEFAULT_PLOT_FONTSIZE,
    min_x: int | None = None,
    max_x: int | None = None,
    include_x: list[int] | tuple[int, ...] | None = None,
    show_legend: bool = True,
    legend_outside_right: bool = False,
    show_xlabel: bool = True,
    zero_shot_value: float | None = None,
    title_prefix: str,
    ylabel: str,
) -> None:
    x_column, title_axis_label, xlabel = _uq_x_axis_config(results)
    order_frame = (
        _ordered_screening_frame
        if x_column == "n_budget"
        else _ordered_learning_curve_frame
    )
    results = LearningCurveResults.from_mapping(
        {
            field_name: _filter_curve_frame(
                order_frame(frame),
                x_column=x_column,
                min_x=min_x,
                max_x=max_x,
                include_x=include_x,
            )
            for field_name, frame in results.to_mapping().items()
        }
    )
    for method_name, _, uq_field, display_name, marker, color in _METHOD_PLOT_STYLES:
        frame = getattr(results, uq_field)
        if frame is None or frame.empty or metric_column not in frame.columns:
            continue
        metric_std_column = f"{metric_column}_std"
        resolved_color = _method_color(method_name, color)
        ax.plot(
            frame[x_column],
            frame[metric_column],
            marker=marker,
            color=resolved_color,
            label=display_name,
        )
        if metric_std_column in frame.columns:
            ax.fill_between(
                frame[x_column],
                frame[metric_column] - frame[metric_std_column],
                frame[metric_column] + frame[metric_std_column],
                color=resolved_color,
                alpha=0.2,
            )
    if zero_shot_value is not None:
        x_min, x_max = ax.get_xlim()
        ax.hlines(
            y=zero_shot_value,
            xmin=x_min,
            xmax=x_max,
            colors="black",
            linestyles="--",
            linewidth=1.25,
            label="Zero-shot mean-MLIP baseline",
        )
        ax.set_xlim(x_min, x_max)
    if show_xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    else:
        ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(f"{title_prefix} vs {title_axis_label}", fontsize=fontsize)
    _set_integer_x_ticks(ax)
    ax.tick_params(axis="both", labelsize=_tick_fontsize())
    ax.grid(True, linestyle="--", alpha=0.3)
    if show_legend:
        if legend_outside_right:
            ax.legend(
                fontsize=_DEFAULT_LEGEND_FONTSIZE,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
            )
        else:
            ax.legend(fontsize=_DEFAULT_LEGEND_FONTSIZE)


def mae_comparison_plot(
    comparison_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    required_columns = {"MLIP_name", "MAE_total (eV)"}
    for label, df in (("comparison_df", comparison_df), ("summary_df", summary_df)):
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"{label} is missing required columns: {sorted(missing_columns)}"
            )

    comparison_mae = comparison_df.loc[:, ["MLIP_name", "MAE_total (eV)"]].rename(
        columns={"MAE_total (eV)": "control_mae"}
    )
    summary_mae = summary_df.loc[:, ["MLIP_name", "MAE_total (eV)"]].rename(
        columns={"MAE_total (eV)": "comparison_mae"}
    )
    merged = comparison_mae.merge(summary_mae, on="MLIP_name", how="inner")
    if merged.empty:
        raise ValueError(
            "No overlapping MLIP_name values found between the two dataframes."
        )

    merged["control_mae"] = pd.to_numeric(merged["control_mae"], errors="coerce")
    merged["comparison_mae"] = pd.to_numeric(merged["comparison_mae"], errors="coerce")
    merged = merged.dropna(subset=["control_mae", "comparison_mae"])
    if merged.empty:
        raise ValueError(
            "No numeric MAE_total (eV) values found for overlapping MLIPs."
        )

    merged["display_name"] = (
        merged["MLIP_name"].map(_MLIP_DISPLAY_NAMES).fillna(merged["MLIP_name"])
    )

    x = np.arange(len(merged))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, merged["control_mae"], width, label="Control")
    ax.bar(x + width / 2, merged["comparison_mae"], width, label="Constrained")

    ax.set_ylabel("MAE (eV$^2$)", fontsize=16)
    ax.set_xticks(x, merged["display_name"])
    ax.legend(frameon=False)

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def parity_plot(
    df: Any,
    output_path: str | Path,
    *,
    title: str | None = None,
    validity_mask_by_prediction: dict[str, np.ndarray] | None = None,
    show_legend: bool = True,
    legend_fontsize: int = _DEFAULT_LEGEND_FONTSIZE,
    legend_loc: str | None = None,
    legend_bbox_to_anchor: tuple[float, float, float, float] | None = None,
    legend_markerscale: float | None = None,
    metrics_position: tuple[float, float] = (0.03, 0.97),
    metrics_horizontalalignment: str = "center",
    metrics_verticalalignment: str = "center",
    metrics_fontsize: int = _DEFAULT_TICK_FONTSIZE + 1,
    y_label: str = "MLIP adsorption energy (eV)",
    y_label_fontsize: int = _DEFAULT_PLOT_FONTSIZE,
) -> Path:
    """
    Create a parity plot comparing reference adsorption energies to each MLIP prediction.

    Returns the path to the saved PNG.
    """
    plot_data = prepare_parity_plot_data(df)
    ref = plot_data.reference

    fig, ax = plt.subplots(figsize=(7, 7))
    plotted_ref_values: list[np.ndarray] = []
    plotted_prediction_values: list[np.ndarray] = []
    parity_markers = _mlip_marker_map(list(plot_data.predictions))

    for label, preds in plot_data.predictions.items():
        mask = None
        if validity_mask_by_prediction is not None:
            mask = np.asarray(validity_mask_by_prediction.get(label), dtype=bool)
            if mask.shape != preds.shape:
                raise ValueError(
                    "Prediction validity masks must match prediction array shapes."
                )
        ref_values = ref if mask is None else ref[mask]
        pred_values = preds if mask is None else preds[mask]
        if len(ref_values) == 0:
            continue
        ax.scatter(
            ref_values,
            pred_values,
            s=35,
            alpha=0.85,
            label=_mlip_display_name(label),
            color=_mlip_color(label),
            edgecolor="black",
            linewidth=0.5,
            marker=parity_markers[label],
        )
        plotted_ref_values.append(np.asarray(ref_values, dtype=float))
        plotted_prediction_values.append(np.asarray(pred_values, dtype=float))

    if not plotted_prediction_values:
        raise ValueError("No valid MLIP predictions remain to plot.")
    plotted_ref = np.concatenate(plotted_ref_values)
    mlip_vals = np.concatenate(plotted_prediction_values)
    rmse = float(np.sqrt(np.mean((mlip_vals - plotted_ref) ** 2)))
    ref_mean = float(np.mean(plotted_ref))
    ss_tot = float(np.sum((plotted_ref - ref_mean) ** 2))
    ss_res = float(np.sum((mlip_vals - plotted_ref) ** 2))
    r2_text = "nan" if ss_tot == 0.0 else f"{1.0 - ss_res / ss_tot:.3f}"
    min_val = min(plotted_ref.min(), mlip_vals.min())
    max_val = max(plotted_ref.max(), mlip_vals.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1, label="Parity")
    ax.text(
        metrics_position[0],
        metrics_position[1],
        f"RMSE = {rmse:.3f} eV\n$R^2$ = {r2_text}",
        transform=ax.transAxes,
        ha=metrics_horizontalalignment,
        va=metrics_verticalalignment,
        fontsize=metrics_fontsize,
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "white",
            "edgecolor": "black",
            "linewidth": 0.6,
            "alpha": 0.95,
        },
    )

    ax.set_xlabel("Reference adsorption energy (eV)", fontsize=_DEFAULT_PLOT_FONTSIZE)
    ax.set_ylabel(y_label, fontsize=y_label_fontsize)
    ax.tick_params(axis="both", labelsize=_tick_fontsize())
    ax.set_aspect("equal", adjustable="box")
    if show_legend:
        legend_kwargs = {"fontsize": legend_fontsize}
        if legend_loc is not None:
            legend_kwargs["loc"] = legend_loc
        if legend_bbox_to_anchor is not None:
            legend_kwargs["bbox_to_anchor"] = legend_bbox_to_anchor
            legend_kwargs["borderaxespad"] = 0.0
        if legend_markerscale is not None:
            legend_kwargs["markerscale"] = legend_markerscale
        ax.legend(**legend_kwargs)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def zero_shot_rmse_stage_plot(
    stage_df: pd.DataFrame,
    output_path: str | Path,
    *,
    fontsize: int = _DEFAULT_PLOT_FONTSIZE,
    show_lone_mlip_swarm: bool = True,
    show_lone_mlip_legend: bool = True,
    stage_legend_loc: str = "upper left",
    max_rmse: float | None = None,
    title: str | None = None,
) -> Path:
    required_columns = {"dataset", "stage", "rmse", "n_samples"}
    missing_columns = required_columns.difference(stage_df.columns)
    if missing_columns:
        raise ValueError(
            f"stage_df is missing required columns: {sorted(missing_columns)}"
        )
    if stage_df.empty:
        raise ValueError("stage_df must contain at least one row.")

    stage_order = [
        "Full / all MLIPs",
        "Matched subset / all MLIPs",
        "Matched subset / anomaly-aware selection",
    ]
    filtered = stage_df.loc[stage_df["stage"].isin(stage_order)].copy()
    if filtered.empty:
        raise ValueError("stage_df does not contain any recognized stage labels.")
    bar_rows = (
        filtered.loc[filtered["mlip"].isna()].copy()
        if "mlip" in filtered.columns
        else filtered.copy()
    )
    swarm_rows = (
        filtered.loc[filtered["mlip"].notna()].copy()
        if "mlip" in filtered.columns
        else filtered.iloc[0:0].copy()
    )
    if bar_rows.empty:
        raise ValueError("stage_df does not contain any bar rows to render.")

    dataset_order = list(dict.fromkeys(bar_rows["dataset"].tolist()))
    if "dataset_label" in bar_rows.columns:
        label_rows = bar_rows.loc[:, ["dataset", "dataset_label"]].drop_duplicates(
            subset=["dataset"],
            keep="first",
        )
        dataset_labels = (
            label_rows.set_index("dataset")
            .reindex(dataset_order)["dataset_label"]
            .fillna(pd.Series(dataset_order, index=dataset_order))
            .tolist()
        )
    else:
        dataset_labels = dataset_order
    x = np.arange(len(dataset_order), dtype=float) * 0.52
    width = 0.24
    plot_offsets = {
        "Full / all MLIPs": -width / 2,
        "Matched subset / all MLIPs": width / 2,
        "Matched subset / anomaly-aware selection": width / 2,
    }
    line_stage_name = "Matched subset / all MLIPs"
    clipped_any = False

    fig, ax = plt.subplots(figsize=(max(4.4, 0.82 + 1.05 * len(dataset_order)), 4.8))
    for stage_name in stage_order:
        stage_rows = (
            bar_rows.loc[bar_rows["stage"] == stage_name]
            .set_index("dataset")
            .reindex(dataset_order)
        )
        if stage_rows["rmse"].isna().all():
            continue
        plotted_rmse = stage_rows["rmse"].copy()
        clipped_mask = pd.Series(False, index=stage_rows.index)
        if max_rmse is not None:
            clipped_mask = plotted_rmse > max_rmse
            plotted_rmse = plotted_rmse.clip(upper=max_rmse)
        if stage_name == line_stage_name:
            valid_mask = plotted_rmse.notna().to_numpy()
            line_x = x[valid_mask] + plot_offsets[stage_name]
            for x_center, y_value in zip(
                line_x.tolist(),
                plotted_rmse.loc[plotted_rmse.notna()].to_numpy().tolist(),
                strict=True,
            ):
                ax.plot(
                    [x_center - width * 0.46, x_center, x_center + width * 0.46],
                    [y_value, y_value, y_value],
                    color="black",
                    linewidth=2.8,
                    zorder=5,
                )
            plotted_artists: list[tuple[float, float]] = list(
                zip(
                    line_x.tolist(),
                    plotted_rmse.loc[plotted_rmse.notna()].to_numpy().tolist(),
                    strict=True,
                )
            )
        else:
            bars = ax.bar(
                x + plot_offsets[stage_name],
                plotted_rmse,
                width,
                label=stage_name,
                color=_stage_color(stage_name),
                zorder=3,
            )
            plotted_artists = [
                (bar.get_x() + bar.get_width() / 2, float(bar.get_height()))
                for bar in bars
            ]
        for bar_index, ((x_center, height), n_samples) in enumerate(
            zip(plotted_artists, stage_rows["n_samples"], strict=True)
        ):
            if stage_name == "Matched subset / anomaly-aware selection":
                if bool(clipped_mask.iloc[bar_index]):
                    clipped_any = True
                    ax.text(
                        x_center,
                        max_rmse,
                        "↑",
                        ha="center",
                        va="bottom",
                        fontsize=_DEFAULT_TICK_FONTSIZE + 4,
                        color="black",
                        clip_on=False,
                        zorder=7,
                    )
                continue
            if pd.isna(n_samples) or pd.isna(height):
                continue
            label_y = max(height * 0.03, 0.02)
            ax.text(
                x_center,
                label_y,
                f"n={int(n_samples)}",
                ha="center",
                va="bottom",
                fontsize=max(_DEFAULT_TICK_FONTSIZE - 1, 6),
                zorder=6,
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "edgecolor": "black",
                    "linewidth": 0.6,
                    "alpha": 0.95,
                },
            )
            if bool(clipped_mask.iloc[bar_index]):
                clipped_any = True
                ax.text(
                    x_center,
                    max_rmse,
                    "↑",
                    ha="center",
                    va="bottom",
                    fontsize=_DEFAULT_TICK_FONTSIZE + 4,
                    color="black",
                    clip_on=False,
                    zorder=7,
                )

    if show_lone_mlip_swarm and not swarm_rows.empty:
        swarm_stage = "Full / all MLIPs"
        swarm_offset = plot_offsets[swarm_stage]
        swarm_mlips = sorted(swarm_rows["mlip"].dropna().unique().tolist())
        swarm_markers = _mlip_marker_map(swarm_mlips)
        swarm_colors = {mlip: _mlip_color(mlip) for mlip in swarm_mlips}
        for dataset_index, dataset_name in enumerate(dataset_order):
            dataset_swarm = swarm_rows.loc[
                (swarm_rows["dataset"] == dataset_name)
                & (swarm_rows["stage"] == swarm_stage)
            ].sort_values(["rmse", "mlip"])
            if dataset_swarm.empty:
                continue
            point_count = len(dataset_swarm)
            if point_count == 1:
                local_offsets = np.array([0.0])
            else:
                local_offsets = np.linspace(
                    -width * 0.28, width * 0.28, num=point_count
                )
            for local_offset, (_, row) in zip(
                local_offsets,
                dataset_swarm.iterrows(),
                strict=True,
            ):
                plotted_rmse = row["rmse"]
                is_clipped = False
                if max_rmse is not None and plotted_rmse > max_rmse:
                    plotted_rmse = max_rmse
                    is_clipped = True
                ax.scatter(
                    x[dataset_index] + swarm_offset + local_offset,
                    plotted_rmse,
                    s=40,
                    color=swarm_colors[row["mlip"]],
                    edgecolors="black",
                    linewidths=0.6,
                    marker=swarm_markers[row["mlip"]],
                    alpha=0.85,
                    zorder=4,
                )
                if is_clipped:
                    clipped_any = True
                    ax.text(
                        x[dataset_index] + swarm_offset + local_offset,
                        max_rmse,
                        "↑",
                        ha="center",
                        va="bottom",
                        fontsize=_DEFAULT_TICK_FONTSIZE + 4,
                        color=swarm_colors[row["mlip"]],
                        clip_on=False,
                        zorder=7,
                    )

    ax.set_xticks(x, dataset_labels)
    ax.margins(x=0.015)
    ax.set_ylabel("Zero-shot RMSE (eV)", fontsize=fontsize)
    if max_rmse is not None:
        ax.set_ylim(top=max_rmse)
    ax.tick_params(axis="x", labelsize=fontsize)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    tick_label_offset = mtransforms.ScaledTranslation(4 / 72, 0, fig.dpi_scale_trans)
    for label in ax.get_xticklabels():
        label.set_transform(label.get_transform() + tick_label_offset)
    ax.tick_params(axis="y", labelsize=_tick_fontsize())
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    stage_handles = [
        Line2D(
            [],
            [],
            color=_stage_color("Full / all MLIPs"),
            linewidth=8,
            solid_capstyle="butt",
            label="Full / all MLIPs",
        ),
        Line2D(
            [],
            [],
            color="black",
            linewidth=2.8,
            solid_capstyle="butt",
            label="Matched subset / all MLIPs",
        ),
        Line2D(
            [],
            [],
            color=_stage_color("Matched subset / anomaly-aware selection"),
            linewidth=8,
            solid_capstyle="butt",
            label="Matched subset / anomaly-aware selection",
        ),
    ]
    stage_legend = ax.legend(
        handles=stage_handles,
        fontsize=_DEFAULT_LEGEND_FONTSIZE,
        loc=stage_legend_loc,
    )
    if show_lone_mlip_swarm and show_lone_mlip_legend and not swarm_rows.empty:
        mlip_handles = [
            Line2D(
                [],
                [],
                linestyle="None",
                marker=swarm_markers[mlip],
                markerfacecolor=swarm_colors[mlip],
                markeredgecolor="black",
                markeredgewidth=0.6,
                color=swarm_colors[mlip],
                markersize=5,
                label=_mlip_display_name(mlip),
            )
            for mlip in swarm_mlips
        ]
        if max_rmse is not None and clipped_any:
            mlip_handles.append(
                Line2D(
                    [],
                    [],
                    linestyle="None",
                    marker="$↑$",
                    markerfacecolor="black",
                    markeredgecolor="black",
                    color="black",
                    markersize=8,
                    label=f"Clipped above {max_rmse:g} eV",
                )
            )
        ax.add_artist(stage_legend)
        ax.legend(
            handles=mlip_handles,
            title="Lone MLIPs",
            fontsize=_DEFAULT_LEGEND_FONTSIZE,
            title_fontsize=_DEFAULT_LEGEND_FONTSIZE,
            loc="upper right",
        )
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def oracle_learning_curve_plot(
    oracle_df: pd.DataFrame,
    output_path: str | Path,
    *,
    fontsize: int = _DEFAULT_PLOT_FONTSIZE,
    title: str | None = "Oracle learning curve by dataset",
    show_legend: bool = True,
    legend_outside_right: bool = False,
    legend_source_df: pd.DataFrame | None = None,
    log_x: bool = False,
    min_x: int | None = None,
    max_x: int | None = None,
    include_x: list[int] | tuple[int, ...] | None = None,
) -> Path:
    required_columns = {"dataset", "dataset_label", "n_train", "oracle_rmse"}
    missing_columns = required_columns.difference(oracle_df.columns)
    if missing_columns:
        raise ValueError(
            f"oracle_df is missing required columns: {sorted(missing_columns)}"
        )
    if oracle_df.empty:
        raise ValueError("oracle_df must contain at least one row.")

    filtered = _filter_curve_frame(
        oracle_df,
        x_column="n_train",
        min_x=min_x,
        max_x=max_x,
        include_x=include_x,
    )
    if filtered is None or filtered.empty:
        raise ValueError("oracle_df does not contain any rows after x-axis filtering.")

    legend_source = oracle_df if legend_source_df is None else legend_source_df
    dataset_order = list(dict.fromkeys(legend_source["dataset"].tolist()))
    plot_dataset_order = list(dict.fromkeys(filtered["dataset"].tolist()))
    label_rows = legend_source.loc[:, ["dataset", "dataset_label"]].drop_duplicates(
        subset=["dataset"],
        keep="first",
    )
    dataset_labels = (
        label_rows.set_index("dataset")
        .reindex(dataset_order)["dataset_label"]
        .fillna(pd.Series(dataset_order, index=dataset_order))
        .to_dict()
    )
    filtered = filtered.sort_values(["dataset", "n_train"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    for dataset in dataset_order:
        dataset_rows = filtered.loc[filtered["dataset"] == dataset]
        color = _dataset_color(dataset)
        if dataset in plot_dataset_order:
            ax.plot(
                dataset_rows["n_train"],
                dataset_rows["oracle_rmse"],
                marker="o",
                color=color,
                label=dataset_labels[dataset],
            )
        elif show_legend:
            ax.plot(
                [],
                [],
                marker="o",
                color=color,
                label=dataset_labels[dataset],
            )
    zero_shot_rows = (
        filtered.loc[:, ["dataset", "zero_shot_rmse"]]
        .dropna(subset=["zero_shot_rmse"])
        .drop_duplicates(subset=["dataset"], keep="first")
        if "zero_shot_rmse" in filtered.columns
        else pd.DataFrame(columns=["dataset", "zero_shot_rmse"])
    )
    if not zero_shot_rows.empty:
        yaxis_transform = mtransforms.blended_transform_factory(
            ax.transAxes, ax.transData
        )
        _, y_max = ax.get_ylim()
        clipped_zero_shot_rows = zero_shot_rows.loc[
            zero_shot_rows["zero_shot_rmse"].astype(float) > y_max
        ].reset_index(drop=True)
        clipped_x_positions = (
            np.linspace(-0.015, 0.015, num=len(clipped_zero_shot_rows))
            if len(clipped_zero_shot_rows) > 1
            else np.array([0.0]) if len(clipped_zero_shot_rows) == 1 else np.array([])
        )
        clipped_x_by_dataset = {
            str(row["dataset"]): float(x_pos)
            for (_, row), x_pos in zip(
                clipped_zero_shot_rows.iterrows(),
                clipped_x_positions,
                strict=True,
            )
        }
        for _, row in zero_shot_rows.iterrows():
            dataset = str(row["dataset"])
            zero_shot_rmse = float(row["zero_shot_rmse"])
            if zero_shot_rmse > y_max:
                ax.text(
                    clipped_x_by_dataset[dataset],
                    y_max,
                    "↑",
                    transform=yaxis_transform,
                    ha="center",
                    va="bottom",
                    fontsize=_DEFAULT_TICK_FONTSIZE + 4,
                    color=_dataset_color(dataset),
                    clip_on=False,
                    zorder=7,
                )
            else:
                ax.scatter(
                    [0.0],
                    [zero_shot_rmse],
                    marker="s",
                    s=36,
                    color=_dataset_color(dataset),
                    edgecolors="black",
                    linewidths=0.6,
                    transform=yaxis_transform,
                    clip_on=False,
                    zorder=5,
                )

    ax.set_xlabel("Train size", fontsize=fontsize)
    ax.set_ylabel("Oracle RMSE (eV)", fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    if log_x:
        ax.set_xscale("log")
    else:
        _set_integer_x_ticks(ax)
    ax.tick_params(axis="both", labelsize=_tick_fontsize())
    ax.grid(True, linestyle="--", alpha=0.3)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if not zero_shot_rows.empty:
            handles.append(
                Line2D(
                    [],
                    [],
                    linestyle="None",
                    marker="s",
                    markerfacecolor="black",
                    markeredgecolor="black",
                    color="black",
                    markersize=6,
                )
            )
            labels.append("Zero-shot")
        if legend_outside_right:
            ax.legend(
                handles,
                labels,
                fontsize=_DEFAULT_LEGEND_FONTSIZE,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
            )
        else:
            ax.legend(handles, labels, fontsize=_DEFAULT_LEGEND_FONTSIZE)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def learning_curve_plot(
    results: LearningCurveResults,
    output_path: str | Path,
    fontsize: int = _DEFAULT_PLOT_FONTSIZE,
    title: str | None = "Learning curve (ensemble vs sample size)",
    min_x: int | None = None,
    max_x: int | None = None,
    include_x: list[int] | tuple[int, ...] | None = None,
    zero_shot_rmse: float | None = None,
    show_legend: bool = True,
    legend_outside_right: bool = False,
    show_std_bands: bool = True,
) -> Path:
    results = LearningCurveResults.from_mapping(
        {
            field_name: _filter_curve_frame(
                _ordered_learning_curve_frame(frame),
                x_column="n_train",
                min_x=min_x,
                max_x=max_x,
                include_x=include_x,
            )
            for field_name, frame in results.to_mapping().items()
        }
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    if results.ridge_df is not None:
        ridge_color = _method_color("ridge", "tab:blue")
        ax.plot(
            results.ridge_df["n_train"],
            results.ridge_df["rmse_mean"],
            marker="o",
            color=ridge_color,
            label="Ridge mean",
        )
        if show_std_bands:
            ax.fill_between(
                results.ridge_df["n_train"],
                results.ridge_df["rmse_mean"] - results.ridge_df["rmse_std"],
                results.ridge_df["rmse_mean"] + results.ridge_df["rmse_std"],
                color=ridge_color,
                alpha=0.2,
                label="_nolegend_",
            )
    if results.kernel_ridge_df is not None:
        kernel_ridge_color = _method_color("kernel_ridge", "tab:cyan")
        ax.plot(
            results.kernel_ridge_df["n_train"],
            results.kernel_ridge_df["rmse_mean"],
            marker="X",
            color=kernel_ridge_color,
            label="Kernel Ridge mean",
        )
        if show_std_bands:
            ax.fill_between(
                results.kernel_ridge_df["n_train"],
                results.kernel_ridge_df["rmse_mean"] - results.kernel_ridge_df["rmse_std"],
                results.kernel_ridge_df["rmse_mean"] + results.kernel_ridge_df["rmse_std"],
                color=kernel_ridge_color,
                alpha=0.2,
                label="_nolegend_",
            )
    if results.lasso_df is not None:
        lasso_color = _method_color("lasso", "tab:orange")
        ax.plot(
            results.lasso_df["n_train"],
            results.lasso_df["rmse_mean"],
            marker="s",
            color=lasso_color,
            label="Lasso mean",
        )
        if show_std_bands:
            ax.fill_between(
                results.lasso_df["n_train"],
                results.lasso_df["rmse_mean"] - results.lasso_df["rmse_std"],
                results.lasso_df["rmse_mean"] + results.lasso_df["rmse_std"],
                color=lasso_color,
                alpha=0.2,
                label="_nolegend_",
            )
    if results.elastic_df is not None:
        elastic_color = _method_color("elastic", "tab:purple")
        ax.plot(
            results.elastic_df["n_train"],
            results.elastic_df["rmse_mean"],
            marker="D",
            color=elastic_color,
            label="Elastic Net mean",
        )
        if show_std_bands:
            ax.fill_between(
                results.elastic_df["n_train"],
                results.elastic_df["rmse_mean"] - results.elastic_df["rmse_std"],
                results.elastic_df["rmse_mean"] + results.elastic_df["rmse_std"],
                color=elastic_color,
                alpha=0.2,
                label="_nolegend_",
            )
    if results.resid_df is not None:
        residual_color = _method_color("residual", "tab:green")
        ax.plot(
            results.resid_df["n_train"],
            results.resid_df["rmse_mean"],
            marker="^",
            color=residual_color,
            label="Residual mean",
        )
        if show_std_bands:
            ax.fill_between(
                results.resid_df["n_train"],
                results.resid_df["rmse_mean"] - results.resid_df["rmse_std"],
                results.resid_df["rmse_mean"] + results.resid_df["rmse_std"],
                color=residual_color,
                alpha=0.2,
                label="_nolegend_",
            )
    if results.weighted_linear_df is not None:
        weighted_linear_color = _method_color("weighted_linear", "tab:gray")
        ax.plot(
            results.weighted_linear_df["n_train"],
            results.weighted_linear_df["rmse_mean"],
            marker="*",
            color=weighted_linear_color,
            label="Weighted linear mean",
        )
        if show_std_bands:
            ax.fill_between(
                results.weighted_linear_df["n_train"],
                results.weighted_linear_df["rmse_mean"]
                - results.weighted_linear_df["rmse_std"],
                results.weighted_linear_df["rmse_mean"]
                + results.weighted_linear_df["rmse_std"],
                color=weighted_linear_color,
                alpha=0.2,
                label="_nolegend_",
            )
    if results.weighted_simplex_df is not None:
        weighted_simplex_color = _method_color("weighted_simplex", "teal")
        ax.plot(
            results.weighted_simplex_df["n_train"],
            results.weighted_simplex_df["rmse_mean"],
            marker="8",
            color=weighted_simplex_color,
            label="Weighted simplex mean",
        )
        if show_std_bands:
            ax.fill_between(
                results.weighted_simplex_df["n_train"],
                results.weighted_simplex_df["rmse_mean"]
                - results.weighted_simplex_df["rmse_std"],
                results.weighted_simplex_df["rmse_mean"]
                + results.weighted_simplex_df["rmse_std"],
                color=weighted_simplex_color,
                alpha=0.2,
                label="_nolegend_",
            )
    if results.graph_mean_df is not None:
        graph_mean_color = _method_color("graph_mean", "tab:red")
        ax.plot(
            results.graph_mean_df["n_train"],
            results.graph_mean_df["rmse_mean"],
            marker="P",
            color=graph_mean_color,
            label="Graph mean mean",
        )
        if show_std_bands:
            ax.fill_between(
                results.graph_mean_df["n_train"],
                results.graph_mean_df["rmse_mean"] - results.graph_mean_df["rmse_std"],
                results.graph_mean_df["rmse_mean"] + results.graph_mean_df["rmse_std"],
                color=graph_mean_color,
                alpha=0.2,
                label="_nolegend_",
            )
    if results.moe_df is not None:
        moe_color = _method_color("moe", "tab:purple")
        ax.plot(
            results.moe_df["n_train"],
            results.moe_df["rmse_mean"],
            marker="*",
            color=moe_color,
            label="MoE mean",
        )
        if show_std_bands:
            ax.fill_between(
                results.moe_df["n_train"],
                results.moe_df["rmse_mean"] - results.moe_df["rmse_std"],
                results.moe_df["rmse_mean"] + results.moe_df["rmse_std"],
                color=moe_color,
                alpha=0.2,
                label="_nolegend_",
            )
    if results.gnn_direct_df is not None:
        gnn_direct_color = _method_color("gnn_direct", "tab:cyan")
        ax.plot(
            results.gnn_direct_df["n_train"],
            results.gnn_direct_df["rmse_mean"],
            marker="s",
            color=gnn_direct_color,
            label="GNN direct mean",
        )
        if show_std_bands:
            ax.fill_between(
                results.gnn_direct_df["n_train"],
                results.gnn_direct_df["rmse_mean"] - results.gnn_direct_df["rmse_std"],
                results.gnn_direct_df["rmse_mean"] + results.gnn_direct_df["rmse_std"],
                color=gnn_direct_color,
                alpha=0.2,
                label="_nolegend_",
            )
    if results.probe_gnn_df is not None:
        probe_gnn_color = _method_color("probe_gnn", "tab:olive")
        ax.plot(
            results.probe_gnn_df["n_train"],
            results.probe_gnn_df["rmse_mean"],
            marker="D",
            color=probe_gnn_color,
            label="Probe GNN mean",
        )
        if show_std_bands:
            ax.fill_between(
                results.probe_gnn_df["n_train"],
                results.probe_gnn_df["rmse_mean"] - results.probe_gnn_df["rmse_std"],
                results.probe_gnn_df["rmse_mean"] + results.probe_gnn_df["rmse_std"],
                color=probe_gnn_color,
                alpha=0.2,
                label="_nolegend_",
            )
    if results.latent_df is not None:
        latent_color = _method_color("latent", "tab:brown")
        ax.plot(
            results.latent_df["n_train"],
            results.latent_df["rmse_mean"],
            marker="v",
            color=latent_color,
            label="Latent mean",
        )
        if show_std_bands:
            ax.fill_between(
                results.latent_df["n_train"],
                results.latent_df["rmse_mean"] - results.latent_df["rmse_std"],
                results.latent_df["rmse_mean"] + results.latent_df["rmse_std"],
                color=latent_color,
                alpha=0.2,
                label="_nolegend_",
            )
    if results.fitted_latent_df is not None:
        fitted_latent_color = _method_color("fitted_latent", "sienna")
        ax.plot(
            results.fitted_latent_df["n_train"],
            results.fitted_latent_df["rmse_mean"],
            marker="<",
            color=fitted_latent_color,
            label="Fitted latent mean",
        )
        if show_std_bands:
            ax.fill_between(
                results.fitted_latent_df["n_train"],
                results.fitted_latent_df["rmse_mean"]
                - results.fitted_latent_df["rmse_std"],
                results.fitted_latent_df["rmse_mean"]
                + results.fitted_latent_df["rmse_std"],
                color=fitted_latent_color,
                alpha=0.2,
                label="_nolegend_",
            )
    if zero_shot_rmse is not None:
        x_min, x_max = ax.get_xlim()
        ax.hlines(
            y=zero_shot_rmse,
            xmin=x_min,
            xmax=x_max,
            colors="black",
            linestyles="--",
            linewidth=1.25,
            label="Zero-shot mean-MLIP RMSE",
        )
        ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Train size", fontsize=fontsize)
    ax.set_ylabel("RMSE (eV)", fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    _set_integer_x_ticks(ax)
    ax.tick_params(axis="both", labelsize=_tick_fontsize())
    ax.grid(True, linestyle="--", alpha=0.3)
    if show_legend:
        if legend_outside_right:
            ax.legend(
                fontsize=_DEFAULT_LEGEND_FONTSIZE,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
            )
        else:
            ax.legend(fontsize=_DEFAULT_LEGEND_FONTSIZE)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def miscalibration_area_plot(
    results: LearningCurveResults,
    output_path: str | Path,
    fontsize: int = _DEFAULT_PLOT_FONTSIZE,
    min_x: int | None = None,
    max_x: int | None = None,
    include_x: list[int] | tuple[int, ...] | None = None,
    show_legend: bool = True,
    legend_outside_right: bool = False,
    show_xlabel: bool = True,
    zero_shot_value: float | None = None,
) -> Path:
    return _plot_uq_metric_curve(
        results,
        metric_column="miscalibration_area",
        output_path=output_path,
        fontsize=fontsize,
        min_x=min_x,
        max_x=max_x,
        include_x=include_x,
        show_legend=show_legend,
        legend_outside_right=legend_outside_right,
        show_xlabel=show_xlabel,
        zero_shot_value=zero_shot_value,
        title_prefix="Miscalibration area",
        ylabel="Miscalibration area",
    )


def sharpness_plot(
    results: LearningCurveResults,
    output_path: str | Path,
    fontsize: int = _DEFAULT_PLOT_FONTSIZE,
    min_x: int | None = None,
    max_x: int | None = None,
    include_x: list[int] | tuple[int, ...] | None = None,
    show_legend: bool = True,
    legend_outside_right: bool = False,
    show_xlabel: bool = True,
    zero_shot_value: float | None = None,
) -> Path:
    return _plot_uq_metric_curve(
        results,
        metric_column="sharpness",
        output_path=output_path,
        fontsize=fontsize,
        min_x=min_x,
        max_x=max_x,
        include_x=include_x,
        show_legend=show_legend,
        legend_outside_right=legend_outside_right,
        show_xlabel=show_xlabel,
        zero_shot_value=zero_shot_value,
        title_prefix="Sharpness",
        ylabel="Sharpness",
    )


def dispersion_plot(
    results: LearningCurveResults,
    output_path: str | Path,
    fontsize: int = _DEFAULT_PLOT_FONTSIZE,
    min_x: int | None = None,
    max_x: int | None = None,
    include_x: list[int] | tuple[int, ...] | None = None,
    show_legend: bool = True,
    legend_outside_right: bool = False,
    show_xlabel: bool = True,
    zero_shot_value: float | None = None,
) -> Path:
    return _plot_uq_metric_curve(
        results,
        metric_column="dispersion",
        output_path=output_path,
        fontsize=fontsize,
        min_x=min_x,
        max_x=max_x,
        include_x=include_x,
        show_legend=show_legend,
        legend_outside_right=legend_outside_right,
        show_xlabel=show_xlabel,
        zero_shot_value=zero_shot_value,
        title_prefix="Dispersion",
        ylabel="Dispersion",
    )


def screening_budget_plot(
    results: LearningCurveResults,
    output_path: str | Path,
    fontsize: int = _DEFAULT_PLOT_FONTSIZE,
    title: str | None = "Screening curve (method performance vs budget)",
    min_x: int | None = None,
    max_x: int | None = None,
    include_x: list[int] | tuple[int, ...] | None = None,
    show_legend: bool = True,
) -> Path:
    results = LearningCurveResults.from_mapping(
        {
            field_name: _filter_curve_frame(
                _ordered_screening_frame(frame),
                x_column="n_budget",
                min_x=min_x,
                max_x=max_x,
                include_x=include_x,
            )
            for field_name, frame in results.to_mapping().items()
        }
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    if results.ridge_df is not None:
        mean_col, std_col = _screening_metric_columns(results.ridge_df)
        ridge_color = _method_color("ridge", "tab:blue")
        ax.plot(
            results.ridge_df["n_budget"],
            results.ridge_df[mean_col],
            marker="o",
            color=ridge_color,
            label="Ridge mean",
        )
        ax.fill_between(
            results.ridge_df["n_budget"],
            results.ridge_df[mean_col] - results.ridge_df[std_col],
            results.ridge_df[mean_col] + results.ridge_df[std_col],
            color=ridge_color,
            alpha=0.2,
            label="Ridge +/- 1sd",
        )
    if results.kernel_ridge_df is not None:
        mean_col, std_col = _screening_metric_columns(results.kernel_ridge_df)
        kernel_ridge_color = _method_color("kernel_ridge", "tab:cyan")
        ax.plot(
            results.kernel_ridge_df["n_budget"],
            results.kernel_ridge_df[mean_col],
            marker="X",
            color=kernel_ridge_color,
            label="Kernel Ridge mean",
        )
        ax.fill_between(
            results.kernel_ridge_df["n_budget"],
            results.kernel_ridge_df[mean_col] - results.kernel_ridge_df[std_col],
            results.kernel_ridge_df[mean_col] + results.kernel_ridge_df[std_col],
            color=kernel_ridge_color,
            alpha=0.2,
            label="Kernel Ridge +/- 1sd",
        )
    if results.lasso_df is not None:
        mean_col, std_col = _screening_metric_columns(results.lasso_df)
        lasso_color = _method_color("lasso", "tab:orange")
        ax.plot(
            results.lasso_df["n_budget"],
            results.lasso_df[mean_col],
            marker="s",
            color=lasso_color,
            label="Lasso mean",
        )
        ax.fill_between(
            results.lasso_df["n_budget"],
            results.lasso_df[mean_col] - results.lasso_df[std_col],
            results.lasso_df[mean_col] + results.lasso_df[std_col],
            color=lasso_color,
            alpha=0.2,
            label="Lasso +/- 1sd",
        )
    if results.elastic_df is not None:
        mean_col, std_col = _screening_metric_columns(results.elastic_df)
        elastic_color = _method_color("elastic", "tab:purple")
        ax.plot(
            results.elastic_df["n_budget"],
            results.elastic_df[mean_col],
            marker="D",
            color=elastic_color,
            label="Elastic Net mean",
        )
        ax.fill_between(
            results.elastic_df["n_budget"],
            results.elastic_df[mean_col] - results.elastic_df[std_col],
            results.elastic_df[mean_col] + results.elastic_df[std_col],
            color=elastic_color,
            alpha=0.2,
            label="Elastic Net +/- 1sd",
        )
    if results.resid_df is not None:
        mean_col, std_col = _screening_metric_columns(results.resid_df)
        residual_color = _method_color("residual", "tab:green")
        ax.plot(
            results.resid_df["n_budget"],
            results.resid_df[mean_col],
            marker="^",
            color=residual_color,
            label="Residual mean",
        )
        ax.fill_between(
            results.resid_df["n_budget"],
            results.resid_df[mean_col] - results.resid_df[std_col],
            results.resid_df[mean_col] + results.resid_df[std_col],
            color=residual_color,
            alpha=0.2,
            label="Residual +/- 1sd",
        )
    if results.weighted_linear_df is not None:
        mean_col, std_col = _screening_metric_columns(results.weighted_linear_df)
        weighted_linear_color = _method_color("weighted_linear", "tab:gray")
        ax.plot(
            results.weighted_linear_df["n_budget"],
            results.weighted_linear_df[mean_col],
            marker="*",
            color=weighted_linear_color,
            label="Weighted linear mean",
        )
        ax.fill_between(
            results.weighted_linear_df["n_budget"],
            results.weighted_linear_df[mean_col] - results.weighted_linear_df[std_col],
            results.weighted_linear_df[mean_col] + results.weighted_linear_df[std_col],
            color=weighted_linear_color,
            alpha=0.2,
            label="Weighted linear +/- 1sd",
        )
    if results.weighted_simplex_df is not None:
        mean_col, std_col = _screening_metric_columns(results.weighted_simplex_df)
        weighted_simplex_color = _method_color("weighted_simplex", "teal")
        ax.plot(
            results.weighted_simplex_df["n_budget"],
            results.weighted_simplex_df[mean_col],
            marker="8",
            color=weighted_simplex_color,
            label="Weighted simplex mean",
        )
        ax.fill_between(
            results.weighted_simplex_df["n_budget"],
            results.weighted_simplex_df[mean_col]
            - results.weighted_simplex_df[std_col],
            results.weighted_simplex_df[mean_col]
            + results.weighted_simplex_df[std_col],
            color=weighted_simplex_color,
            alpha=0.2,
            label="Weighted simplex +/- 1sd",
        )
    if results.graph_mean_df is not None:
        mean_col, std_col = _screening_metric_columns(results.graph_mean_df)
        graph_mean_color = _method_color("graph_mean", "tab:red")
        ax.plot(
            results.graph_mean_df["n_budget"],
            results.graph_mean_df[mean_col],
            marker="P",
            color=graph_mean_color,
            label="Graph mean mean",
        )
        ax.fill_between(
            results.graph_mean_df["n_budget"],
            results.graph_mean_df[mean_col] - results.graph_mean_df[std_col],
            results.graph_mean_df[mean_col] + results.graph_mean_df[std_col],
            color=graph_mean_color,
            alpha=0.2,
            label="Graph mean +/- 1sd",
        )
    if results.moe_df is not None:
        mean_col, std_col = _screening_metric_columns(results.moe_df)
        moe_color = _method_color("moe", "tab:purple")
        ax.plot(
            results.moe_df["n_budget"],
            results.moe_df[mean_col],
            marker="*",
            color=moe_color,
            label="MoE mean",
        )
        ax.fill_between(
            results.moe_df["n_budget"],
            results.moe_df[mean_col] - results.moe_df[std_col],
            results.moe_df[mean_col] + results.moe_df[std_col],
            color=moe_color,
            alpha=0.2,
            label="MoE +/- 1sd",
        )
    if results.gnn_direct_df is not None:
        mean_col, std_col = _screening_metric_columns(results.gnn_direct_df)
        gnn_direct_color = _method_color("gnn_direct", "tab:cyan")
        ax.plot(
            results.gnn_direct_df["n_budget"],
            results.gnn_direct_df[mean_col],
            marker="s",
            color=gnn_direct_color,
            label="GNN direct mean",
        )
        ax.fill_between(
            results.gnn_direct_df["n_budget"],
            results.gnn_direct_df[mean_col] - results.gnn_direct_df[std_col],
            results.gnn_direct_df[mean_col] + results.gnn_direct_df[std_col],
            color=gnn_direct_color,
            alpha=0.2,
            label="GNN direct +/- 1sd",
        )
    if results.probe_gnn_df is not None:
        mean_col, std_col = _screening_metric_columns(results.probe_gnn_df)
        probe_gnn_color = _method_color("probe_gnn", "tab:olive")
        ax.plot(
            results.probe_gnn_df["n_budget"],
            results.probe_gnn_df[mean_col],
            marker="D",
            color=probe_gnn_color,
            label="Probe GNN mean",
        )
        ax.fill_between(
            results.probe_gnn_df["n_budget"],
            results.probe_gnn_df[mean_col] - results.probe_gnn_df[std_col],
            results.probe_gnn_df[mean_col] + results.probe_gnn_df[std_col],
            color=probe_gnn_color,
            alpha=0.2,
            label="Probe GNN +/- 1sd",
        )
    if results.latent_df is not None:
        mean_col, std_col = _screening_metric_columns(results.latent_df)
        latent_color = _method_color("latent", "tab:brown")
        ax.plot(
            results.latent_df["n_budget"],
            results.latent_df[mean_col],
            marker="v",
            color=latent_color,
            label="Latent mean",
        )
        ax.fill_between(
            results.latent_df["n_budget"],
            results.latent_df[mean_col] - results.latent_df[std_col],
            results.latent_df[mean_col] + results.latent_df[std_col],
            color=latent_color,
            alpha=0.2,
            label="Latent +/- 1sd",
        )
    if results.fitted_latent_df is not None:
        mean_col, std_col = _screening_metric_columns(results.fitted_latent_df)
        fitted_latent_color = _method_color("fitted_latent", "sienna")
        ax.plot(
            results.fitted_latent_df["n_budget"],
            results.fitted_latent_df[mean_col],
            marker="<",
            color=fitted_latent_color,
            label="Fitted latent mean",
        )
        ax.fill_between(
            results.fitted_latent_df["n_budget"],
            results.fitted_latent_df[mean_col] - results.fitted_latent_df[std_col],
            results.fitted_latent_df[mean_col] + results.fitted_latent_df[std_col],
            color=fitted_latent_color,
            alpha=0.2,
            label="Fitted latent +/- 1sd",
        )
    ax.set_xlabel("Sample budget", fontsize=fontsize)
    ax.set_ylabel("CV RMSE (eV)", fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    _set_integer_x_ticks(ax)
    ax.tick_params(axis="both", labelsize=_tick_fontsize())
    ax.grid(True, linestyle="--", alpha=0.3)
    if show_legend:
        ax.legend(fontsize=_DEFAULT_LEGEND_FONTSIZE)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def _time_accuracy_scatter_plot(
    table: pd.DataFrame,
    *,
    x_column: str,
    output_path: str | Path,
    fontsize: int = _DEFAULT_PLOT_FONTSIZE,
    title: str,
    xlabel: str,
    ylabel: str = "RMSE (eV)",
    show_legend: bool = True,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    if not table.empty:
        for method_name, _, _, display_name, marker, color in _METHOD_PLOT_STYLES:
            method_table = table.loc[table["method"] == method_name]
            if method_table.empty or x_column not in method_table.columns:
                continue
            if "n_train" in method_table.columns:
                ordered = method_table.sort_values("n_train").reset_index(drop=True)
            else:
                ordered = method_table.reset_index(drop=True)
            resolved_color = _method_color(method_name, color)
            ax.scatter(
                ordered[x_column],
                ordered["rmse_mean"],
                marker=marker,
                color=resolved_color,
                label=display_name,
                s=55,
                alpha=0.9,
            )
            if len(ordered) > 1:
                ax.plot(
                    ordered[x_column],
                    ordered["rmse_mean"],
                    color=resolved_color,
                    alpha=0.35,
                    linewidth=1.0,
                )

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.tick_params(axis="both", labelsize=_tick_fontsize())
    ax.grid(True, linestyle="--", alpha=0.3)
    if show_legend and not table.empty:
        ax.legend(fontsize=_DEFAULT_LEGEND_FONTSIZE)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def generation_time_accuracy_plot(
    results: LearningCurveResults,
    generation_timing_by_mlip: dict[str, MlipGenerationTimingSummary],
    *,
    output_path: str | Path,
    mlip_feature_names: tuple[str, ...] | list[str] | None = None,
    generation_timing_by_method: dict[str, GenerationTimingAggregate] | None = None,
    method_names: tuple[str, ...] | list[str] | None = None,
    fontsize: int = _DEFAULT_PLOT_FONTSIZE,
    show_legend: bool = True,
) -> Path:
    table = build_mean_time_accuracy_table(
        results,
        generation_timing_by_mlip,
        mlip_feature_names=mlip_feature_names,
        generation_timing_by_method=generation_timing_by_method,
        method_names=method_names,
    )
    return _time_accuracy_scatter_plot(
        table,
        x_column="generation_time_s",
        output_path=output_path,
        fontsize=fontsize,
        title="Generation time vs RMSE",
        xlabel="Generation time (s)",
        show_legend=show_legend,
    )


def training_time_accuracy_plot(
    results: LearningCurveResults,
    generation_timing_by_mlip: dict[str, MlipGenerationTimingSummary],
    *,
    output_path: str | Path,
    mlip_feature_names: tuple[str, ...] | list[str] | None = None,
    generation_timing_by_method: dict[str, GenerationTimingAggregate] | None = None,
    method_names: tuple[str, ...] | list[str] | None = None,
    fontsize: int = _DEFAULT_PLOT_FONTSIZE,
    show_legend: bool = True,
) -> Path:
    table = build_time_accuracy_table(
        results,
        generation_timing_by_mlip,
        mlip_feature_names=mlip_feature_names,
        generation_timing_by_method=generation_timing_by_method,
        method_names=method_names,
    )
    return _time_accuracy_scatter_plot(
        table,
        x_column="training_time_s",
        output_path=output_path,
        fontsize=fontsize,
        title="Training time vs RMSE",
        xlabel="Training time (s)",
        show_legend=show_legend,
    )


def total_time_accuracy_plot(
    results: LearningCurveResults,
    generation_timing_by_mlip: dict[str, MlipGenerationTimingSummary],
    *,
    output_path: str | Path,
    mlip_feature_names: tuple[str, ...] | list[str] | None = None,
    generation_timing_by_method: dict[str, GenerationTimingAggregate] | None = None,
    method_names: tuple[str, ...] | list[str] | None = None,
    fontsize: int = _DEFAULT_PLOT_FONTSIZE,
    show_legend: bool = True,
) -> Path:
    table = build_time_accuracy_table(
        results,
        generation_timing_by_mlip,
        mlip_feature_names=mlip_feature_names,
        generation_timing_by_method=generation_timing_by_method,
        method_names=method_names,
    )
    return _time_accuracy_scatter_plot(
        table,
        x_column="total_time_s",
        output_path=output_path,
        fontsize=fontsize,
        title="Total time vs RMSE",
        xlabel="Total time (s)",
        show_legend=show_legend,
    )


def _fixed_split_time_accuracy_plot(
    table: pd.DataFrame,
    *,
    x_column: str,
    xerr_column: str,
    output_path: str | Path,
    fontsize: int = _DEFAULT_PLOT_FONTSIZE,
    title: str,
    xlabel: str,
    ylabel: str = "RMSE (eV)",
    show_legend: bool = True,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    if not table.empty:
        for method_name, _, _, display_name, marker, color in _METHOD_PLOT_STYLES:
            method_table = table.loc[table["method"] == method_name]
            if method_table.empty:
                continue
            row = method_table.iloc[0]
            resolved_color = _method_color(method_name, color)
            ax.errorbar(
                [row[x_column]],
                [row["rmse_mean"]],
                xerr=[row[xerr_column]],
                yerr=[row["rmse_std"]],
                fmt="none",
                ecolor=resolved_color,
                elinewidth=1.0,
                capsize=3,
                alpha=0.45,
            )
            ax.scatter(
                [row[x_column]],
                [row["rmse_mean"]],
                marker=marker,
                color=resolved_color,
                label=display_name,
                s=80,
                alpha=0.9,
            )

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.tick_params(axis="both", labelsize=_tick_fontsize())
    ax.grid(True, linestyle="--", alpha=0.3)
    if show_legend and not table.empty:
        ax.legend(fontsize=_DEFAULT_LEGEND_FONTSIZE)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def policy_selected_vs_oracle_plot(
    summary_df: pd.DataFrame,
    *,
    fixed_method_summary_df: pd.DataFrame | None = None,
    output_path: str | Path,
    min_x: int | None = None,
    max_x: int | None = None,
    include_x: list[int] | tuple[int, ...] | None = None,
    fontsize: int = _DEFAULT_PLOT_FONTSIZE,
    show_title: bool = True,
    show_legend: bool = True,
    legend_labels: dict[str, str] | None = None,
) -> Path:
    required_columns = {
        "budget",
        "oracle_outer_rmse_mean",
        "screening_selected_outer_rmse_mean",
    }
    if not required_columns.issubset(summary_df.columns):
        raise ValueError(
            "policy summary frame must contain budget, oracle_outer_rmse_mean, "
            "and screening_selected_outer_rmse_mean."
        )
    frame = _filter_curve_frame(
        summary_df.sort_values("budget").reset_index(drop=True),
        x_column="budget",
        min_x=min_x,
        max_x=max_x,
        include_x=include_x,
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    configured_legend_labels = {} if legend_labels is None else legend_labels
    oracle_frame = frame.sort_values("budget").drop_duplicates(
        subset=["budget"], keep="first"
    )
    ax.plot(
        oracle_frame["budget"],
        oracle_frame["oracle_outer_rmse_mean"],
        marker="o",
        color=_oracle_series_color(),
        label=configured_legend_labels.get("oracle", "Oracle best held-out RMSE"),
    )
    if "policy_name" in frame.columns:
        for policy_name, group in frame.groupby("policy_name", sort=True):
            ordered = group.sort_values("budget")
            policy_name = str(policy_name)
            ax.plot(
                ordered["budget"],
                ordered["screening_selected_outer_rmse_mean"],
                marker="s",
                color=_policy_color(policy_name),
                label=configured_legend_labels.get(
                    policy_name,
                    f"{policy_name} held-out RMSE",
                ),
            )
    else:
        ax.plot(
            frame["budget"],
            frame["screening_selected_outer_rmse_mean"],
            marker="s",
            color=_policy_color("screening_selected_held_out_rmse"),
            label=configured_legend_labels.get(
                "screening_selected",
                "Screening-selected held-out RMSE",
            ),
        )
    if fixed_method_summary_df is not None and not fixed_method_summary_df.empty:
        fixed_frame = _filter_curve_frame(
            fixed_method_summary_df.sort_values(
                ["baseline_name", "budget"]
            ).reset_index(drop=True),
            x_column="budget",
            min_x=min_x,
            max_x=max_x,
            include_x=include_x,
        )
        baseline_styles = [("^", ":"), ("D", "-."), ("P", "--"), ("X", "-")]
        for index, (baseline_name, group) in enumerate(
            fixed_frame.groupby("baseline_name", sort=True)
        ):
            marker, linestyle = baseline_styles[index % len(baseline_styles)]
            ordered = group.sort_values("budget")
            baseline_name = str(baseline_name)
            ax.plot(
                ordered["budget"],
                ordered["outer_rmse_mean"],
                marker=marker,
                color=_baseline_color(baseline_name),
                linestyle=linestyle,
                label=configured_legend_labels.get(
                    baseline_name,
                    f"{baseline_name} held-out RMSE",
                ),
            )
    ax.set_xlabel("Sample budget", fontsize=fontsize)
    ax.set_ylabel("Held-out RMSE", fontsize=fontsize)
    if show_title:
        ax.set_title("Oracle vs screening-selected held-out RMSE", fontsize=fontsize)
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(fontsize=_DEFAULT_LEGEND_FONTSIZE)
    _set_integer_x_ticks(ax)
    ax.tick_params(axis="both", labelsize=_tick_fontsize())
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def policy_regret_plot(
    summary_df: pd.DataFrame,
    *,
    fixed_method_summary_df: pd.DataFrame | None = None,
    output_path: str | Path,
    min_x: int | None = None,
    max_x: int | None = None,
    include_x: list[int] | tuple[int, ...] | None = None,
    fontsize: int = _DEFAULT_PLOT_FONTSIZE,
) -> Path:
    required_columns = {"budget", "mean_regret"}
    if not required_columns.issubset(summary_df.columns):
        raise ValueError("policy summary frame must contain budget and mean_regret.")
    frame = _filter_curve_frame(
        summary_df.sort_values("budget").reset_index(drop=True),
        x_column="budget",
        min_x=min_x,
        max_x=max_x,
        include_x=include_x,
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    if "policy_name" in frame.columns:
        for policy_name, group in frame.groupby("policy_name", sort=True):
            ordered = group.sort_values("budget")
            color = _policy_color(str(policy_name))
            ax.plot(
                ordered["budget"],
                ordered["mean_regret"],
                marker="o",
                color=color,
                label=f"{policy_name} mean regret",
            )
            if {"ci95_low", "ci95_high"}.issubset(ordered.columns):
                ax.fill_between(
                    ordered["budget"],
                    ordered["ci95_low"],
                    ordered["ci95_high"],
                    color=color,
                    alpha=0.2,
                )
            elif "std_regret" in ordered.columns:
                ax.fill_between(
                    ordered["budget"],
                    ordered["mean_regret"] - ordered["std_regret"],
                    ordered["mean_regret"] + ordered["std_regret"],
                    color=color,
                    alpha=0.2,
                )
    else:
        ax.plot(
            frame["budget"],
            frame["mean_regret"],
            marker="o",
            color="tab:red",
            label="Mean regret",
        )
        if {"ci95_low", "ci95_high"}.issubset(frame.columns):
            ax.fill_between(
                frame["budget"],
                frame["ci95_low"],
                frame["ci95_high"],
                color="tab:red",
                alpha=0.2,
                label="95% CI",
            )
        elif "std_regret" in frame.columns:
            ax.fill_between(
                frame["budget"],
                frame["mean_regret"] - frame["std_regret"],
                frame["mean_regret"] + frame["std_regret"],
                color="tab:red",
                alpha=0.2,
                label="Std. dev.",
            )
    if fixed_method_summary_df is not None and not fixed_method_summary_df.empty:
        fixed_frame = _filter_curve_frame(
            fixed_method_summary_df.sort_values(
                ["baseline_name", "budget"]
            ).reset_index(drop=True),
            x_column="budget",
            min_x=min_x,
            max_x=max_x,
            include_x=include_x,
        )
        baseline_styles = [("^", ":"), ("D", "-."), ("P", "--"), ("X", "-")]
        for index, (baseline_name, group) in enumerate(
            fixed_frame.groupby("baseline_name", sort=True)
        ):
            marker, linestyle = baseline_styles[index % len(baseline_styles)]
            ordered = group.sort_values("budget")
            ax.plot(
                ordered["budget"],
                ordered["mean_regret"],
                marker=marker,
                color=_baseline_color(str(baseline_name)),
                linestyle=linestyle,
                label=f"{baseline_name} regret",
            )
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Sample budget", fontsize=fontsize)
    ax.set_ylabel("Regret", fontsize=fontsize)
    ax.set_title("Screening policy regret", fontsize=fontsize)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=_DEFAULT_LEGEND_FONTSIZE)
    _set_integer_x_ticks(ax)
    ax.tick_params(axis="both", labelsize=_tick_fontsize())
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def all_datasets_policy_regret_plot(
    summary_df: pd.DataFrame,
    *,
    output_path: str | Path,
    min_x: int | None = None,
    max_x: int | None = None,
    include_x: list[int] | tuple[int, ...] | None = None,
    fontsize: int = _DEFAULT_PLOT_FONTSIZE,
    log_x: bool = False,
    show_uncertainty: bool = True,
    show_title: bool = True,
    show_legend: bool = True,
) -> Path:
    required_columns = {"dataset", "dataset_label", "budget", "mean_regret"}
    missing_columns = required_columns.difference(summary_df.columns)
    if missing_columns:
        raise ValueError(
            f"summary_df is missing required columns: {sorted(missing_columns)}"
        )
    if summary_df.empty:
        raise ValueError("summary_df must contain at least one row.")

    filtered = _filter_curve_frame(
        summary_df.sort_values(["dataset", "budget"]).reset_index(drop=True),
        x_column="budget",
        min_x=min_x,
        max_x=max_x,
        include_x=include_x,
    )
    if filtered is None or filtered.empty:
        raise ValueError("summary_df does not contain any rows after x-axis filtering.")

    dataset_order = list(dict.fromkeys(filtered["dataset"].tolist()))
    label_rows = summary_df.loc[:, ["dataset", "dataset_label"]].drop_duplicates(
        subset=["dataset"],
        keep="first",
    )
    dataset_labels = (
        label_rows.set_index("dataset")
        .reindex(dataset_order)["dataset_label"]
        .fillna(pd.Series(dataset_order, index=dataset_order))
        .to_dict()
    )
    multiple_policies = (
        "policy_name" in filtered.columns
        and filtered["policy_name"].nunique(dropna=True) > 1
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    for dataset in dataset_order:
        dataset_rows = filtered.loc[filtered["dataset"] == dataset]
        if "policy_name" in dataset_rows.columns:
            grouped = dataset_rows.groupby("policy_name", sort=True)
        else:
            grouped = [(None, dataset_rows)]
        for policy_name, group in grouped:
            ordered = group.sort_values("budget")
            label = dataset_labels[dataset]
            if multiple_policies and policy_name is not None:
                label = f"{label}: {policy_name}"
            color = _dataset_color(dataset)
            ax.plot(
                ordered["budget"],
                ordered["mean_regret"],
                marker="o",
                color=color,
                label=label,
            )
            if show_uncertainty and {"ci95_low", "ci95_high"}.issubset(ordered.columns):
                ax.fill_between(
                    ordered["budget"],
                    ordered["ci95_low"],
                    ordered["ci95_high"],
                    color=color,
                    alpha=0.2,
                )
            elif show_uncertainty and "std_regret" in ordered.columns:
                ax.fill_between(
                    ordered["budget"],
                    ordered["mean_regret"] - ordered["std_regret"],
                    ordered["mean_regret"] + ordered["std_regret"],
                    color=color,
                    alpha=0.2,
                )

    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Sample budget", fontsize=fontsize)
    ax.set_ylabel("Regret", fontsize=fontsize)
    if show_title:
        ax.set_title("Screening policy regret by dataset", fontsize=fontsize)
    if log_x:
        ax.set_xscale("log")
    else:
        _set_integer_x_ticks(ax)
    ax.tick_params(axis="both", labelsize=_tick_fontsize())
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(fontsize=_DEFAULT_LEGEND_FONTSIZE)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def all_datasets_uq_oracle_plot(
    oracle_df: pd.DataFrame,
    *,
    output_path: str | Path,
    metric_column: str,
    ylabel: str,
    title: str,
    min_x: int | None = None,
    max_x: int | None = None,
    include_x: list[int] | tuple[int, ...] | None = None,
    fontsize: int = _DEFAULT_PLOT_FONTSIZE,
    log_x: bool = False,
    show_legend: bool = True,
    legend_outside_right: bool = False,
    legend_source_df: pd.DataFrame | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    _draw_all_datasets_uq_oracle(
        ax,
        oracle_df,
        metric_column=metric_column,
        ylabel=ylabel,
        title=title,
        min_x=min_x,
        max_x=max_x,
        include_x=include_x,
        fontsize=fontsize,
        log_x=log_x,
        show_legend=show_legend,
        legend_outside_right=legend_outside_right,
        legend_source_df=legend_source_df,
    )
    zero_shot_column = (
        f"zero_shot_{metric_column.removeprefix('oracle_')}"
        if metric_column.startswith("oracle_")
        else None
    )
    zero_shot_rows = (
        oracle_df.loc[:, ["dataset", zero_shot_column]]
        .dropna(subset=[zero_shot_column])
        .drop_duplicates(subset=["dataset"], keep="first")
        if zero_shot_column is not None and zero_shot_column in oracle_df.columns
        else pd.DataFrame(columns=["dataset", "zero_shot_value"])
    )
    if not zero_shot_rows.empty:
        yaxis_transform = mtransforms.blended_transform_factory(
            ax.transAxes, ax.transData
        )
        _, y_max = ax.get_ylim()
        clipped_zero_shot_rows = zero_shot_rows.loc[
            zero_shot_rows[zero_shot_column].astype(float) > y_max
        ].reset_index(drop=True)
        clipped_x_positions = (
            np.linspace(-0.015, 0.015, num=len(clipped_zero_shot_rows))
            if len(clipped_zero_shot_rows) > 1
            else np.array([0.0]) if len(clipped_zero_shot_rows) == 1 else np.array([])
        )
        clipped_x_by_dataset = {
            str(row["dataset"]): float(x_pos)
            for (_, row), x_pos in zip(
                clipped_zero_shot_rows.iterrows(),
                clipped_x_positions,
                strict=True,
            )
        }
        for _, row in zero_shot_rows.iterrows():
            dataset = str(row["dataset"])
            zero_shot_value = float(row[zero_shot_column])
            if zero_shot_value > y_max:
                ax.text(
                    clipped_x_by_dataset[dataset],
                    y_max,
                    "↑",
                    transform=yaxis_transform,
                    ha="center",
                    va="bottom",
                    fontsize=_DEFAULT_TICK_FONTSIZE + 4,
                    color=_dataset_color(dataset),
                    clip_on=False,
                    zorder=7,
                )
            else:
                ax.scatter(
                    [0.0],
                    [zero_shot_value],
                    marker="s",
                    s=36,
                    color=_dataset_color(dataset),
                    edgecolors="black",
                    linewidths=0.6,
                    transform=yaxis_transform,
                    clip_on=False,
                    zorder=5,
                )
    if show_legend and not zero_shot_rows.empty:
        handles, labels = ax.get_legend_handles_labels()
        handles.append(
            Line2D(
                [],
                [],
                linestyle="None",
                marker="s",
                markerfacecolor="black",
                markeredgecolor="black",
                color="black",
                markersize=6,
            )
        )
        labels.append("Zero-shot")
        ax.legend_.remove()
        if legend_outside_right:
            ax.legend(
                handles,
                labels,
                fontsize=_DEFAULT_LEGEND_FONTSIZE,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
            )
        else:
            ax.legend(handles, labels, fontsize=_DEFAULT_LEGEND_FONTSIZE)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _dataset_label_by_tag_from_config() -> dict[str, str]:
    from oasis.config import get_config

    cfg = get_config()
    labels: dict[str, str] = {}
    for dataset_tag, profile in getattr(cfg, "datasets", {}).items():
        alias = getattr(profile, "alias", None)
        mlip_run_dirname = getattr(profile, "mlip_run_dirname", None)
        labels[str(dataset_tag)] = str(alias or mlip_run_dirname or dataset_tag)
    return labels


def _load_cached_policy_artifacts_for_screening_curve(
    plot_root: Path,
    suffix: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    dataset_labels = _dataset_label_by_tag_from_config()
    summary_paths = sorted(
        plot_root.glob(f"*/policy_selection_diagnostic_summary_{suffix}.csv")
    )
    artifact_paths = sorted(
        (plot_root.parent / "screening").glob("policy_selection_diagnostic_*.json")
    )
    if not summary_paths:
        raise ValueError("No cached policy summary CSVs found.")
    if not artifact_paths:
        raise ValueError("No cached policy-selection artifacts found.")

    artifact_index: list[dict[str, Any]] = []
    for artifact_path in artifact_paths:
        payload = json.loads(artifact_path.read_text())
        results_payload = payload.get("results")
        if not isinstance(results_payload, dict):
            continue
        summary_payload = results_payload.get("summary_df")
        if not isinstance(summary_payload, str):
            continue
        artifact_index.append(
            {
                "payload": payload,
                "summary_df": pd.read_json(StringIO(summary_payload), orient="table"),
            }
        )

    def _matches_summary(left_df: pd.DataFrame, right_df: pd.DataFrame) -> bool:
        if list(left_df.columns) != list(right_df.columns):
            return False
        if len(left_df) != len(right_df):
            return False
        for column in right_df.columns:
            left = left_df[column]
            right = right_df[column]
            if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(
                right
            ):
                if not np.allclose(
                    left.to_numpy(dtype=float),
                    right.to_numpy(dtype=float),
                    equal_nan=True,
                ):
                    return False
            elif left.astype(str).tolist() != right.astype(str).tolist():
                return False
        return True

    dataset_entries: list[dict[str, Any]] = []
    dataset_order: list[str] = []
    for summary_path in summary_paths:
        summary_df = pd.read_csv(summary_path)
        if summary_df.empty:
            continue
        artifact_match = next(
            (
                entry
                for entry in artifact_index
                if _matches_summary(entry["summary_df"], summary_df)
            ),
            None,
        )
        if artifact_match is None:
            raise ValueError(
                f"Could not match summary CSV to cached artifact: {summary_path}"
            )
        payload = artifact_match["payload"]
        metadata = payload.get("metadata", {})
        cache_signature = payload.get("cache_signature", {})
        learning_curve_signature = cache_signature.get("learning_curve", {})
        dataset_tag = str(metadata.get("dataset_tag") or summary_path.parent.name)
        dataset_order.append(dataset_tag)
        dataset_entries.append(
            {
                "dataset": dataset_tag,
                "dataset_label": dataset_labels.get(dataset_tag, summary_path.parent.name),
                "dataset_size": int(metadata["dataset_size"]),
                "summary_df": summary_df.copy(),
                "sweep_sizes": tuple(
                    int(value)
                    for value in learning_curve_signature.get("sweep_sizes", [])
                ),
                "sweep_fractions": tuple(
                    float(value)
                    for value in learning_curve_signature.get("sweep_fractions", [])
                ),
            }
        )
    if not dataset_entries:
        raise ValueError("No non-empty cached policy summaries found.")
    return dataset_entries, dataset_order


def _load_policy_artifact_for_screening_curve_plot_dir(
    plot_dir: Path,
    suffix: str,
) -> dict[str, Any]:
    summary_path = plot_dir / f"policy_selection_diagnostic_summary_{suffix}.csv"
    if not summary_path.is_file():
        raise ValueError(f"Policy summary CSV not found: {summary_path}")
    summary_df = pd.read_csv(summary_path)
    if summary_df.empty:
        raise ValueError(f"Policy summary CSV is empty: {summary_path}")
    plot_root = plot_dir.parent
    artifact_paths = sorted(
        (plot_root.parent / "screening").glob("policy_selection_diagnostic_*.json")
    )
    if not artifact_paths:
        raise ValueError("No cached policy-selection artifacts found.")

    def _matches_summary(left_df: pd.DataFrame, right_df: pd.DataFrame) -> bool:
        if list(left_df.columns) != list(right_df.columns):
            return False
        if len(left_df) != len(right_df):
            return False
        for column in right_df.columns:
            left = left_df[column]
            right = right_df[column]
            if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(
                right
            ):
                if not np.allclose(
                    left.to_numpy(dtype=float),
                    right.to_numpy(dtype=float),
                    equal_nan=True,
                ):
                    return False
            elif left.astype(str).tolist() != right.astype(str).tolist():
                return False
        return True

    for artifact_path in artifact_paths:
        payload = json.loads(artifact_path.read_text())
        results_payload = payload.get("results")
        if not isinstance(results_payload, dict):
            continue
        summary_payload = results_payload.get("summary_df")
        outer_metrics_payload = results_payload.get("outer_metrics_df")
        if not isinstance(summary_payload, str) or not isinstance(
            outer_metrics_payload, str
        ):
            continue
        artifact_summary_df = pd.read_json(StringIO(summary_payload), orient="table")
        if not _matches_summary(artifact_summary_df, summary_df):
            continue
        return {
            "summary_df": summary_df,
            "outer_metrics_df": pd.read_json(
                StringIO(outer_metrics_payload), orient="table"
            ),
            "metadata": payload.get("metadata", {}),
            "cache_signature": payload.get("cache_signature", {}),
        }
    raise ValueError(f"Could not match summary CSV to cached artifact: {summary_path}")


def _configured_policy_fixed_method_baselines_for_screening_curve() -> (
    tuple[tuple[str, str], ...]
):
    from oasis.config import get_config

    cfg = get_config()
    screening_cfg = getattr(getattr(cfg, "experiment", None), "screening", None)
    plot_baselines_cfg = getattr(screening_cfg, "plot_baselines", None)
    if plot_baselines_cfg is None:
        return ()
    baselines: list[tuple[str, str]] = []
    for field_name in ("low_data_domain", "high_data_domain"):
        baseline_cfg = getattr(plot_baselines_cfg, field_name, None)
        if baseline_cfg is None or not getattr(baseline_cfg, "enabled", True):
            continue
        method_name = str(getattr(baseline_cfg, "method_name", "")).strip()
        if not method_name:
            continue
        label = getattr(baseline_cfg, "label", None)
        baselines.append((method_name, method_name if label is None else str(label)))
    return tuple(baselines)


def _build_screening_curve_regret_frame(
    dataset_entries: list[dict[str, Any]],
    *,
    mode: str,
    excluded_datasets: set[str],
) -> pd.DataFrame:
    from oasis.experiment.splits import resolve_configured_sweep_sizes

    rows: list[pd.DataFrame] = []
    for entry in dataset_entries:
        dataset_tag = str(entry["dataset"])
        if _dataset_aliases(dataset_tag) & excluded_datasets:
            continue
        summary_df = entry["summary_df"].copy()
        summary_df.insert(0, "dataset_label", entry["dataset_label"])
        summary_df.insert(0, "dataset", dataset_tag)
        if mode == "absolute":
            include_x = list(entry["sweep_sizes"])
        elif mode == "fraction":
            include_x = list(
                resolve_configured_sweep_sizes(
                    int(entry["dataset_size"]),
                    min_train=None,
                    max_train=None,
                    sweep_sizes=(),
                    sweep_fractions=entry["sweep_fractions"],
                )
            )
        else:
            raise ValueError(f"Unsupported regret frame mode: {mode}")
        filtered = summary_df.loc[
            summary_df["budget"].astype(int).isin(include_x)
        ].reset_index(drop=True)
        if not filtered.empty:
            rows.append(filtered)
    if not rows:
        raise ValueError(f"No rows available for {mode} regret panel.")
    return pd.concat(rows, ignore_index=True)


def _render_screening_curve_regret_panel(
    summary_df: pd.DataFrame,
    *,
    output_path: Path,
    dataset_order: list[str],
    log_x: bool,
    show_legend: bool = True,
) -> Path:
    ordered_frame = summary_df.assign(
        dataset=pd.Categorical(
            summary_df["dataset"], categories=dataset_order, ordered=True
        )
    ).sort_values(["dataset", "budget"])
    return all_datasets_policy_regret_plot(
        ordered_frame,
        output_path=output_path,
        log_x=log_x,
        show_uncertainty=False,
        show_title=False,
        show_legend=show_legend,
    )


def compose_screening_curve_figure(
    plot_dir: str | Path,
    *,
    suffix: str = "anomalyaware_on",
    output_name: str = "figure_screening_curve.png",
    exclude_panel_d_datasets: list[str] | tuple[str, ...] = ("bio_mass",),
) -> Path:
    from oasis.figure import two_by_two_figure
    from oasis.experiment.policy_diagnostic import summarize_fixed_method_baseline_frame
    from oasis.experiment.splits import resolve_configured_sweep_sizes

    plot_dir = Path(plot_dir).resolve()
    excluded = {
        alias
        for value in exclude_panel_d_datasets
        for alias in _dataset_aliases(value)
    }
    policy_artifact = _load_policy_artifact_for_screening_curve_plot_dir(plot_dir, suffix)
    metadata = policy_artifact["metadata"]
    cache_signature = policy_artifact["cache_signature"]
    learning_curve_signature = cache_signature.get("learning_curve", {})
    dataset_size = int(metadata["dataset_size"])
    absolute_include_x = [
        int(value) for value in learning_curve_signature.get("sweep_sizes", [])
    ]
    fraction_include_x = list(
        resolve_configured_sweep_sizes(
            dataset_size,
            min_train=None,
            max_train=None,
            sweep_sizes=(),
            sweep_fractions=tuple(
                float(value)
                for value in learning_curve_signature.get("sweep_fractions", [])
            ),
        )
    )
    fixed_method_summary_df = summarize_fixed_method_baseline_frame(
        policy_artifact["outer_metrics_df"],
        baselines=_configured_policy_fixed_method_baselines_for_screening_curve(),
    )
    dataset_entries, dataset_order = _load_cached_policy_artifacts_for_screening_curve(
        plot_dir.parent, suffix
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        panel_a_path = policy_selected_vs_oracle_plot(
            policy_artifact["summary_df"],
            fixed_method_summary_df=fixed_method_summary_df,
            output_path=tmp_path / "panel_a_absolute.png",
            include_x=absolute_include_x,
            show_title=False,
            show_legend=False,
        )
        panel_b_path = policy_selected_vs_oracle_plot(
            policy_artifact["summary_df"],
            fixed_method_summary_df=fixed_method_summary_df,
            output_path=tmp_path / "panel_b_fraction.png",
            include_x=fraction_include_x,
            show_title=False,
            legend_labels={
                "oracle": "Oracle best",
                "min_screening_rmse": "Screening-selected method",
                "Residual": "Mean-residual method",
                "Kernel ridge": "Kernel ridge regression method",
            },
        )
        panel_c_path = _render_screening_curve_regret_panel(
            _build_screening_curve_regret_frame(
                dataset_entries,
                mode="absolute",
                excluded_datasets=set(),
            ),
            output_path=tmp_path / "panel_c_absolute.png",
            dataset_order=dataset_order,
            log_x=False,
            show_legend=False,
        )
        panel_d_path = _render_screening_curve_regret_panel(
            _build_screening_curve_regret_frame(
                dataset_entries,
                mode="fraction",
                excluded_datasets=excluded,
            ),
            output_path=tmp_path / "panel_d_fraction.png",
            dataset_order=dataset_order,
            log_x=True,
        )
        return two_by_two_figure(
            top_left_path=panel_a_path,
            top_right_path=panel_b_path,
            bottom_left_path=panel_c_path,
            bottom_right_path=panel_d_path,
            output_path=plot_dir / output_name,
            panel_labels=("a)", "b)", "c)", "d)"),
        )


def _draw_all_datasets_uq_oracle(
    ax: Any,
    oracle_df: pd.DataFrame,
    *,
    metric_column: str,
    ylabel: str,
    title: str,
    min_x: int | None = None,
    max_x: int | None = None,
    include_x: list[int] | tuple[int, ...] | None = None,
    fontsize: int = _DEFAULT_PLOT_FONTSIZE,
    log_x: bool = False,
    show_legend: bool = True,
    legend_outside_right: bool = False,
    legend_source_df: pd.DataFrame | None = None,
) -> None:
    required_columns = {"dataset", "dataset_label", "n_train", metric_column}
    missing_columns = required_columns.difference(oracle_df.columns)
    if missing_columns:
        raise ValueError(
            f"oracle_df is missing required columns: {sorted(missing_columns)}"
        )
    if oracle_df.empty:
        raise ValueError("oracle_df must contain at least one row.")

    filtered = _filter_curve_frame(
        oracle_df.sort_values(["dataset", "n_train"]).reset_index(drop=True),
        x_column="n_train",
        min_x=min_x,
        max_x=max_x,
        include_x=include_x,
    )
    if filtered is None or filtered.empty:
        raise ValueError("oracle_df does not contain any rows after x-axis filtering.")

    legend_source = oracle_df if legend_source_df is None else legend_source_df
    dataset_order = list(dict.fromkeys(legend_source["dataset"].tolist()))
    plot_dataset_order = list(dict.fromkeys(filtered["dataset"].tolist()))
    label_rows = legend_source.loc[:, ["dataset", "dataset_label"]].drop_duplicates(
        subset=["dataset"],
        keep="first",
    )
    dataset_labels = (
        label_rows.set_index("dataset")
        .reindex(dataset_order)["dataset_label"]
        .fillna(pd.Series(dataset_order, index=dataset_order))
        .to_dict()
    )

    for dataset in dataset_order:
        dataset_rows = filtered.loc[filtered["dataset"] == dataset]
        if dataset in plot_dataset_order:
            ax.plot(
                dataset_rows["n_train"],
                dataset_rows[metric_column],
                marker="o",
                color=_dataset_color(dataset),
                label=dataset_labels[dataset],
            )
        elif show_legend:
            ax.plot(
                [],
                [],
                marker="o",
                color=_dataset_color(dataset),
                label=dataset_labels[dataset],
            )

    ax.set_xlabel("Train size", fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    if log_x:
        ax.set_xscale("log")
    else:
        _set_integer_x_ticks(ax)
    ax.tick_params(axis="both", labelsize=_tick_fontsize())
    ax.grid(True, linestyle="--", alpha=0.3)
    if show_legend:
        if legend_outside_right:
            ax.legend(
                fontsize=_DEFAULT_LEGEND_FONTSIZE,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
            )
        else:
            ax.legend(fontsize=_DEFAULT_LEGEND_FONTSIZE)


def fixed_split_training_time_accuracy_plot(
    results: LearningCurveResults,
    generation_timing_by_mlip: dict[str, MlipGenerationTimingSummary],
    *,
    dataset_size: int,
    output_path: str | Path,
    train_fraction: float = 0.8,
    mlip_feature_names: tuple[str, ...] | list[str] | None = None,
    generation_timing_by_method: dict[str, GenerationTimingAggregate] | None = None,
    method_names: tuple[str, ...] | list[str] | None = None,
    fontsize: int = _DEFAULT_PLOT_FONTSIZE,
    show_legend: bool = True,
) -> Path:
    table = build_fixed_split_time_accuracy_table(
        results,
        generation_timing_by_mlip,
        dataset_size=dataset_size,
        train_fraction=train_fraction,
        mlip_feature_names=mlip_feature_names,
        generation_timing_by_method=generation_timing_by_method,
        method_names=method_names,
    )
    return _fixed_split_time_accuracy_plot(
        table,
        x_column="training_time_mean_s",
        xerr_column="training_time_std_s",
        output_path=output_path,
        fontsize=fontsize,
        title=(
            "Fixed-split training time vs RMSE "
            f"(train={_format_train_fraction_label(train_fraction)})"
        ),
        xlabel="Training time (s)",
        show_legend=show_legend,
    )


def fixed_split_total_time_accuracy_plot(
    results: LearningCurveResults,
    generation_timing_by_mlip: dict[str, MlipGenerationTimingSummary],
    *,
    dataset_size: int,
    output_path: str | Path,
    train_fraction: float = 0.8,
    mlip_feature_names: tuple[str, ...] | list[str] | None = None,
    generation_timing_by_method: dict[str, GenerationTimingAggregate] | None = None,
    method_names: tuple[str, ...] | list[str] | None = None,
    fontsize: int = _DEFAULT_PLOT_FONTSIZE,
    show_legend: bool = True,
) -> Path:
    table = build_fixed_split_time_accuracy_table(
        results,
        generation_timing_by_mlip,
        dataset_size=dataset_size,
        train_fraction=train_fraction,
        mlip_feature_names=mlip_feature_names,
        generation_timing_by_method=generation_timing_by_method,
        method_names=method_names,
    )
    return _fixed_split_time_accuracy_plot(
        table,
        x_column="total_time_mean_s",
        xerr_column="total_time_std_s",
        output_path=output_path,
        fontsize=fontsize,
        title=(
            "Fixed-split total time vs RMSE "
            f"(train={_format_train_fraction_label(train_fraction)})"
        ),
        xlabel="Total time (s)",
        show_legend=show_legend,
    )
