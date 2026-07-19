from __future__ import annotations

import os
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
from oasis.sweep import LearningCurveResults

_MLIP_DISPLAY_NAMES = {
    "7net-omni": "7Net-Omni",
    "mace-mh-1": "MACE-MH-1",
    "mattersim-v1-5m": "MatterSim-v1-5M",
    "orb-v3-conservative-inf-omat": "ORB-v3\nconservative",
    "uma-s-1p1": "UMA-s-1p1",
}
_MLIP_SWARM_MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">", "8", "h", "*")
_MLIP_SWARM_COLORS = (
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
)
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
)
_METHOD_RESULT_FIELDS = {
    method_name: result_field for method_name, result_field, *_ in _METHOD_PLOT_STYLES
}
_METHOD_UQ_FIELDS = {
    method_name: uq_field for method_name, _, uq_field, *_ in _METHOD_PLOT_STYLES
}


def _mlip_display_name(mlip: str) -> str:
    return _MLIP_DISPLAY_NAMES.get(mlip, mlip)


def _mlip_marker_map(mlips: list[str] | tuple[str, ...]) -> dict[str, str]:
    return {
        mlip: _MLIP_SWARM_MARKERS[index % len(_MLIP_SWARM_MARKERS)]
        for index, mlip in enumerate(sorted(dict.fromkeys(mlips)))
    }


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
    show_xlabel: bool = True,
    zero_shot_value: float | None = None,
    title_prefix: str,
    ylabel: str,
) -> Path:
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
    fig, ax = plt.subplots(figsize=(7, 4))
    for _, _, uq_field, display_name, marker, color in _METHOD_PLOT_STYLES:
        frame = getattr(results, uq_field)
        if frame is None or frame.empty or metric_column not in frame.columns:
            continue
        metric_std_column = f"{metric_column}_std"
        ax.plot(
            frame[x_column],
            frame[metric_column],
            marker=marker,
            color=color,
            label=display_name,
        )
        if metric_std_column in frame.columns:
            ax.fill_between(
                frame[x_column],
                frame[metric_column] - frame[metric_std_column],
                frame[metric_column] + frame[metric_std_column],
                color=color,
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
    ax.tick_params(axis="both", labelsize=_DEFAULT_TICK_FONTSIZE)
    ax.grid(True, linestyle="--", alpha=0.3)
    if show_legend:
        ax.legend(fontsize=_DEFAULT_LEGEND_FONTSIZE)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


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
) -> Path:
    """
    Create a parity plot comparing reference adsorption energies to each MLIP prediction.

    Returns the path to the saved PNG.
    """
    plot_data = prepare_parity_plot_data(df)
    ref = plot_data.reference

    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(len(plot_data.predictions))
    plotted_ref_values: list[np.ndarray] = []
    plotted_prediction_values: list[np.ndarray] = []
    parity_markers = _mlip_marker_map(list(plot_data.predictions))

    for idx, (label, preds) in enumerate(plot_data.predictions.items()):
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
            label=label,
            color=cmap(idx),
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
    min_val = min(plotted_ref.min(), mlip_vals.min())
    max_val = max(plotted_ref.max(), mlip_vals.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1, label="Parity")

    ax.set_xlabel("Reference adsorption energy (eV)", fontsize=_DEFAULT_PLOT_FONTSIZE)
    ax.set_ylabel("MLIP adsorption energy (eV)", fontsize=_DEFAULT_PLOT_FONTSIZE)
    ax.tick_params(axis="both", labelsize=_DEFAULT_TICK_FONTSIZE)
    ax.set_aspect("equal", adjustable="box")
    if show_legend:
        ax.legend(fontsize=legend_fontsize)
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
    stage_colors = {
        "Full / all MLIPs": "tab:blue",
        "Matched subset / all MLIPs": "tab:orange",
        "Matched subset / anomaly-aware selection": "tab:green",
    }
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
    offsets = np.array([-width / 2, width / 2, -width / 2], dtype=float)
    clipped_any = False

    fig, ax = plt.subplots(figsize=(max(4.4, 0.82 + 1.05 * len(dataset_order)), 4.8))
    anomaly_stage_rows = (
        bar_rows.loc[bar_rows["stage"] == "Matched subset / anomaly-aware selection"]
        .set_index("dataset")
        .reindex(dataset_order)
    )
    anomaly_overlay_offset = offsets[stage_order.index("Matched subset / all MLIPs")]
    for offset, stage_name in zip(offsets, stage_order, strict=True):
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
        if stage_name == "Matched subset / anomaly-aware selection":
            valid_mask = plotted_rmse.notna().to_numpy()
            line_x = x[valid_mask] + anomaly_overlay_offset
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
                    marker="D",
                    markerfacecolor="black",
                    markeredgecolor="black",
                    markersize=5.5,
                    markevery=[1],
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
                x + offset,
                plotted_rmse,
                width,
                label=stage_name,
                color=stage_colors[stage_name],
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
        swarm_offset = offsets[stage_order.index(swarm_stage)]
        swarm_mlips = sorted(swarm_rows["mlip"].dropna().unique().tolist())
        swarm_markers = _mlip_marker_map(swarm_mlips)
        swarm_colors = {
            mlip: _MLIP_SWARM_COLORS[index % len(_MLIP_SWARM_COLORS)]
            for index, mlip in enumerate(swarm_mlips)
        }
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
    ax.tick_params(axis="y", labelsize=_DEFAULT_TICK_FONTSIZE)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    stage_handles = [
        Line2D(
            [],
            [],
            color=stage_colors["Full / all MLIPs"],
            linewidth=8,
            solid_capstyle="butt",
            label="Full / all MLIPs",
        ),
        Line2D(
            [],
            [],
            color=stage_colors["Matched subset / all MLIPs"],
            linewidth=8,
            solid_capstyle="butt",
            label="Matched subset / all MLIPs",
        ),
        Line2D(
            [],
            [],
            color="black",
            linewidth=2.8,
            marker="D",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=6,
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
    show_legend: bool = True,
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

    dataset_order = list(dict.fromkeys(filtered["dataset"].tolist()))
    label_rows = oracle_df.loc[:, ["dataset", "dataset_label"]].drop_duplicates(
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
    cmap = plt.cm.get_cmap("tab10", len(dataset_order))
    for idx, dataset in enumerate(dataset_order):
        dataset_rows = filtered.loc[filtered["dataset"] == dataset]
        ax.plot(
            dataset_rows["n_train"],
            dataset_rows["oracle_rmse"],
            marker="o",
            color=cmap(idx),
            label=dataset_labels[dataset],
        )

    ax.set_xlabel("Train size", fontsize=fontsize)
    ax.set_ylabel("Oracle RMSE (eV)", fontsize=fontsize)
    ax.set_title("Oracle learning curve by dataset", fontsize=fontsize)
    if log_x:
        ax.set_xscale("log")
    else:
        _set_integer_x_ticks(ax)
    ax.tick_params(axis="both", labelsize=_DEFAULT_TICK_FONTSIZE)
    ax.grid(True, linestyle="--", alpha=0.3)
    if show_legend:
        ax.legend(fontsize=_DEFAULT_LEGEND_FONTSIZE)
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
    min_x: int | None = None,
    max_x: int | None = None,
    include_x: list[int] | tuple[int, ...] | None = None,
    zero_shot_rmse: float | None = None,
    show_legend: bool = True,
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
        ax.plot(
            results.ridge_df["n_train"],
            results.ridge_df["rmse_mean"],
            marker="o",
            color="tab:blue",
            label="Ridge mean",
        )
        ax.fill_between(
            results.ridge_df["n_train"],
            results.ridge_df["rmse_mean"] - results.ridge_df["rmse_std"],
            results.ridge_df["rmse_mean"] + results.ridge_df["rmse_std"],
            color="tab:blue",
            alpha=0.2,
            label="Ridge +/- 1sd",
        )
    if results.kernel_ridge_df is not None:
        ax.plot(
            results.kernel_ridge_df["n_train"],
            results.kernel_ridge_df["rmse_mean"],
            marker="X",
            color="tab:cyan",
            label="Kernel Ridge mean",
        )
        ax.fill_between(
            results.kernel_ridge_df["n_train"],
            results.kernel_ridge_df["rmse_mean"] - results.kernel_ridge_df["rmse_std"],
            results.kernel_ridge_df["rmse_mean"] + results.kernel_ridge_df["rmse_std"],
            color="tab:cyan",
            alpha=0.2,
            label="Kernel Ridge +/- 1sd",
        )
    if results.lasso_df is not None:
        ax.plot(
            results.lasso_df["n_train"],
            results.lasso_df["rmse_mean"],
            marker="s",
            color="tab:orange",
            label="Lasso mean",
        )
        ax.fill_between(
            results.lasso_df["n_train"],
            results.lasso_df["rmse_mean"] - results.lasso_df["rmse_std"],
            results.lasso_df["rmse_mean"] + results.lasso_df["rmse_std"],
            color="tab:orange",
            alpha=0.2,
            label="Lasso +/- 1sd",
        )
    if results.elastic_df is not None:
        ax.plot(
            results.elastic_df["n_train"],
            results.elastic_df["rmse_mean"],
            marker="D",
            color="tab:purple",
            label="Elastic Net mean",
        )
        ax.fill_between(
            results.elastic_df["n_train"],
            results.elastic_df["rmse_mean"] - results.elastic_df["rmse_std"],
            results.elastic_df["rmse_mean"] + results.elastic_df["rmse_std"],
            color="tab:purple",
            alpha=0.2,
            label="Elastic Net +/- 1sd",
        )
    if results.resid_df is not None:
        ax.plot(
            results.resid_df["n_train"],
            results.resid_df["rmse_mean"],
            marker="^",
            color="tab:green",
            label="Residual mean",
        )
        ax.fill_between(
            results.resid_df["n_train"],
            results.resid_df["rmse_mean"] - results.resid_df["rmse_std"],
            results.resid_df["rmse_mean"] + results.resid_df["rmse_std"],
            color="tab:green",
            alpha=0.2,
            label="Residual +/- 1sd",
        )
    if results.weighted_linear_df is not None:
        ax.plot(
            results.weighted_linear_df["n_train"],
            results.weighted_linear_df["rmse_mean"],
            marker="*",
            color="tab:gray",
            label="Weighted linear mean",
        )
        ax.fill_between(
            results.weighted_linear_df["n_train"],
            results.weighted_linear_df["rmse_mean"]
            - results.weighted_linear_df["rmse_std"],
            results.weighted_linear_df["rmse_mean"]
            + results.weighted_linear_df["rmse_std"],
            color="tab:gray",
            alpha=0.2,
            label="Weighted linear +/- 1sd",
        )
    if results.weighted_simplex_df is not None:
        ax.plot(
            results.weighted_simplex_df["n_train"],
            results.weighted_simplex_df["rmse_mean"],
            marker="8",
            color="teal",
            label="Weighted simplex mean",
        )
        ax.fill_between(
            results.weighted_simplex_df["n_train"],
            results.weighted_simplex_df["rmse_mean"]
            - results.weighted_simplex_df["rmse_std"],
            results.weighted_simplex_df["rmse_mean"]
            + results.weighted_simplex_df["rmse_std"],
            color="teal",
            alpha=0.2,
            label="Weighted simplex +/- 1sd",
        )
    if results.graph_mean_df is not None:
        ax.plot(
            results.graph_mean_df["n_train"],
            results.graph_mean_df["rmse_mean"],
            marker="P",
            color="tab:red",
            label="Graph mean mean",
        )
        ax.fill_between(
            results.graph_mean_df["n_train"],
            results.graph_mean_df["rmse_mean"] - results.graph_mean_df["rmse_std"],
            results.graph_mean_df["rmse_mean"] + results.graph_mean_df["rmse_std"],
            color="tab:red",
            alpha=0.2,
            label="Graph mean +/- 1sd",
        )
    if results.moe_df is not None:
        ax.plot(
            results.moe_df["n_train"],
            results.moe_df["rmse_mean"],
            marker="*",
            color="tab:purple",
            label="MoE mean",
        )
        ax.fill_between(
            results.moe_df["n_train"],
            results.moe_df["rmse_mean"] - results.moe_df["rmse_std"],
            results.moe_df["rmse_mean"] + results.moe_df["rmse_std"],
            color="tab:purple",
            alpha=0.2,
            label="MoE +/- 1sd",
        )
    if results.gnn_direct_df is not None:
        ax.plot(
            results.gnn_direct_df["n_train"],
            results.gnn_direct_df["rmse_mean"],
            marker="s",
            color="tab:cyan",
            label="GNN direct mean",
        )
        ax.fill_between(
            results.gnn_direct_df["n_train"],
            results.gnn_direct_df["rmse_mean"] - results.gnn_direct_df["rmse_std"],
            results.gnn_direct_df["rmse_mean"] + results.gnn_direct_df["rmse_std"],
            color="tab:cyan",
            alpha=0.2,
            label="GNN direct +/- 1sd",
        )
    if results.probe_gnn_df is not None:
        ax.plot(
            results.probe_gnn_df["n_train"],
            results.probe_gnn_df["rmse_mean"],
            marker="D",
            color="tab:olive",
            label="Probe GNN mean",
        )
        ax.fill_between(
            results.probe_gnn_df["n_train"],
            results.probe_gnn_df["rmse_mean"] - results.probe_gnn_df["rmse_std"],
            results.probe_gnn_df["rmse_mean"] + results.probe_gnn_df["rmse_std"],
            color="tab:olive",
            alpha=0.2,
            label="Probe GNN +/- 1sd",
        )
    if results.latent_df is not None:
        ax.plot(
            results.latent_df["n_train"],
            results.latent_df["rmse_mean"],
            marker="v",
            color="tab:brown",
            label="Latent mean",
        )
        ax.fill_between(
            results.latent_df["n_train"],
            results.latent_df["rmse_mean"] - results.latent_df["rmse_std"],
            results.latent_df["rmse_mean"] + results.latent_df["rmse_std"],
            color="tab:brown",
            alpha=0.2,
            label="Latent +/- 1sd",
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
    ax.set_title("Learning curve (ensemble vs sample size)", fontsize=fontsize)
    _set_integer_x_ticks(ax)
    ax.tick_params(axis="both", labelsize=_DEFAULT_TICK_FONTSIZE)
    ax.grid(True, linestyle="--", alpha=0.3)
    if show_legend:
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
        show_xlabel=show_xlabel,
        zero_shot_value=zero_shot_value,
        title_prefix="Dispersion",
        ylabel="Dispersion",
    )


def screening_budget_plot(
    results: LearningCurveResults,
    output_path: str | Path,
    fontsize: int = _DEFAULT_PLOT_FONTSIZE,
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
        ax.plot(
            results.ridge_df["n_budget"],
            results.ridge_df[mean_col],
            marker="o",
            color="tab:blue",
            label="Ridge mean",
        )
        ax.fill_between(
            results.ridge_df["n_budget"],
            results.ridge_df[mean_col] - results.ridge_df[std_col],
            results.ridge_df[mean_col] + results.ridge_df[std_col],
            color="tab:blue",
            alpha=0.2,
            label="Ridge +/- 1sd",
        )
    if results.kernel_ridge_df is not None:
        mean_col, std_col = _screening_metric_columns(results.kernel_ridge_df)
        ax.plot(
            results.kernel_ridge_df["n_budget"],
            results.kernel_ridge_df[mean_col],
            marker="X",
            color="tab:cyan",
            label="Kernel Ridge mean",
        )
        ax.fill_between(
            results.kernel_ridge_df["n_budget"],
            results.kernel_ridge_df[mean_col] - results.kernel_ridge_df[std_col],
            results.kernel_ridge_df[mean_col] + results.kernel_ridge_df[std_col],
            color="tab:cyan",
            alpha=0.2,
            label="Kernel Ridge +/- 1sd",
        )
    if results.lasso_df is not None:
        mean_col, std_col = _screening_metric_columns(results.lasso_df)
        ax.plot(
            results.lasso_df["n_budget"],
            results.lasso_df[mean_col],
            marker="s",
            color="tab:orange",
            label="Lasso mean",
        )
        ax.fill_between(
            results.lasso_df["n_budget"],
            results.lasso_df[mean_col] - results.lasso_df[std_col],
            results.lasso_df[mean_col] + results.lasso_df[std_col],
            color="tab:orange",
            alpha=0.2,
            label="Lasso +/- 1sd",
        )
    if results.elastic_df is not None:
        mean_col, std_col = _screening_metric_columns(results.elastic_df)
        ax.plot(
            results.elastic_df["n_budget"],
            results.elastic_df[mean_col],
            marker="D",
            color="tab:purple",
            label="Elastic Net mean",
        )
        ax.fill_between(
            results.elastic_df["n_budget"],
            results.elastic_df[mean_col] - results.elastic_df[std_col],
            results.elastic_df[mean_col] + results.elastic_df[std_col],
            color="tab:purple",
            alpha=0.2,
            label="Elastic Net +/- 1sd",
        )
    if results.resid_df is not None:
        mean_col, std_col = _screening_metric_columns(results.resid_df)
        ax.plot(
            results.resid_df["n_budget"],
            results.resid_df[mean_col],
            marker="^",
            color="tab:green",
            label="Residual mean",
        )
        ax.fill_between(
            results.resid_df["n_budget"],
            results.resid_df[mean_col] - results.resid_df[std_col],
            results.resid_df[mean_col] + results.resid_df[std_col],
            color="tab:green",
            alpha=0.2,
            label="Residual +/- 1sd",
        )
    if results.weighted_linear_df is not None:
        mean_col, std_col = _screening_metric_columns(results.weighted_linear_df)
        ax.plot(
            results.weighted_linear_df["n_budget"],
            results.weighted_linear_df[mean_col],
            marker="*",
            color="tab:gray",
            label="Weighted linear mean",
        )
        ax.fill_between(
            results.weighted_linear_df["n_budget"],
            results.weighted_linear_df[mean_col] - results.weighted_linear_df[std_col],
            results.weighted_linear_df[mean_col] + results.weighted_linear_df[std_col],
            color="tab:gray",
            alpha=0.2,
            label="Weighted linear +/- 1sd",
        )
    if results.weighted_simplex_df is not None:
        mean_col, std_col = _screening_metric_columns(results.weighted_simplex_df)
        ax.plot(
            results.weighted_simplex_df["n_budget"],
            results.weighted_simplex_df[mean_col],
            marker="8",
            color="teal",
            label="Weighted simplex mean",
        )
        ax.fill_between(
            results.weighted_simplex_df["n_budget"],
            results.weighted_simplex_df[mean_col]
            - results.weighted_simplex_df[std_col],
            results.weighted_simplex_df[mean_col]
            + results.weighted_simplex_df[std_col],
            color="teal",
            alpha=0.2,
            label="Weighted simplex +/- 1sd",
        )
    if results.graph_mean_df is not None:
        mean_col, std_col = _screening_metric_columns(results.graph_mean_df)
        ax.plot(
            results.graph_mean_df["n_budget"],
            results.graph_mean_df[mean_col],
            marker="P",
            color="tab:red",
            label="Graph mean mean",
        )
        ax.fill_between(
            results.graph_mean_df["n_budget"],
            results.graph_mean_df[mean_col] - results.graph_mean_df[std_col],
            results.graph_mean_df[mean_col] + results.graph_mean_df[std_col],
            color="tab:red",
            alpha=0.2,
            label="Graph mean +/- 1sd",
        )
    if results.moe_df is not None:
        mean_col, std_col = _screening_metric_columns(results.moe_df)
        ax.plot(
            results.moe_df["n_budget"],
            results.moe_df[mean_col],
            marker="*",
            color="tab:purple",
            label="MoE mean",
        )
        ax.fill_between(
            results.moe_df["n_budget"],
            results.moe_df[mean_col] - results.moe_df[std_col],
            results.moe_df[mean_col] + results.moe_df[std_col],
            color="tab:purple",
            alpha=0.2,
            label="MoE +/- 1sd",
        )
    if results.gnn_direct_df is not None:
        mean_col, std_col = _screening_metric_columns(results.gnn_direct_df)
        ax.plot(
            results.gnn_direct_df["n_budget"],
            results.gnn_direct_df[mean_col],
            marker="s",
            color="tab:cyan",
            label="GNN direct mean",
        )
        ax.fill_between(
            results.gnn_direct_df["n_budget"],
            results.gnn_direct_df[mean_col] - results.gnn_direct_df[std_col],
            results.gnn_direct_df[mean_col] + results.gnn_direct_df[std_col],
            color="tab:cyan",
            alpha=0.2,
            label="GNN direct +/- 1sd",
        )
    if results.probe_gnn_df is not None:
        mean_col, std_col = _screening_metric_columns(results.probe_gnn_df)
        ax.plot(
            results.probe_gnn_df["n_budget"],
            results.probe_gnn_df[mean_col],
            marker="D",
            color="tab:olive",
            label="Probe GNN mean",
        )
        ax.fill_between(
            results.probe_gnn_df["n_budget"],
            results.probe_gnn_df[mean_col] - results.probe_gnn_df[std_col],
            results.probe_gnn_df[mean_col] + results.probe_gnn_df[std_col],
            color="tab:olive",
            alpha=0.2,
            label="Probe GNN +/- 1sd",
        )
    if results.latent_df is not None:
        mean_col, std_col = _screening_metric_columns(results.latent_df)
        ax.plot(
            results.latent_df["n_budget"],
            results.latent_df[mean_col],
            marker="v",
            color="tab:brown",
            label="Latent mean",
        )
        ax.fill_between(
            results.latent_df["n_budget"],
            results.latent_df[mean_col] - results.latent_df[std_col],
            results.latent_df[mean_col] + results.latent_df[std_col],
            color="tab:brown",
            alpha=0.2,
            label="Latent +/- 1sd",
        )
    ax.set_xlabel("Sample budget", fontsize=fontsize)
    ax.set_ylabel("CV RMSE (eV)", fontsize=fontsize)
    ax.set_title("Screening curve (method performance vs budget)", fontsize=fontsize)
    _set_integer_x_ticks(ax)
    ax.tick_params(axis="both", labelsize=_DEFAULT_TICK_FONTSIZE)
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
            ax.scatter(
                ordered[x_column],
                ordered["rmse_mean"],
                marker=marker,
                color=color,
                label=display_name,
                s=55,
                alpha=0.9,
            )
            if len(ordered) > 1:
                ax.plot(
                    ordered[x_column],
                    ordered["rmse_mean"],
                    color=color,
                    alpha=0.35,
                    linewidth=1.0,
                )

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.tick_params(axis="both", labelsize=_DEFAULT_TICK_FONTSIZE)
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
            ax.errorbar(
                [row[x_column]],
                [row["rmse_mean"]],
                xerr=[row[xerr_column]],
                yerr=[row["rmse_std"]],
                fmt="none",
                ecolor=color,
                elinewidth=1.0,
                capsize=3,
                alpha=0.45,
            )
            ax.scatter(
                [row[x_column]],
                [row["rmse_mean"]],
                marker=marker,
                color=color,
                label=display_name,
                s=80,
                alpha=0.9,
            )

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.tick_params(axis="both", labelsize=_DEFAULT_TICK_FONTSIZE)
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
    oracle_frame = frame.sort_values("budget").drop_duplicates(
        subset=["budget"], keep="first"
    )
    ax.plot(
        oracle_frame["budget"],
        oracle_frame["oracle_outer_rmse_mean"],
        marker="o",
        color="tab:blue",
        label="Oracle best held-out RMSE",
    )
    if "policy_name" in frame.columns:
        colors = [
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
        ]
        for index, (policy_name, group) in enumerate(
            frame.groupby("policy_name", sort=True)
        ):
            ordered = group.sort_values("budget")
            ax.plot(
                ordered["budget"],
                ordered["screening_selected_outer_rmse_mean"],
                marker="s",
                color=colors[index % len(colors)],
                label=f"{policy_name} held-out RMSE",
            )
    else:
        ax.plot(
            frame["budget"],
            frame["screening_selected_outer_rmse_mean"],
            marker="s",
            color="tab:orange",
            label="Screening-selected held-out RMSE",
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
        baseline_styles = [
            ("tab:green", "^", ":"),
            ("tab:brown", "D", "-."),
        ]
        for index, (baseline_name, group) in enumerate(
            fixed_frame.groupby("baseline_name", sort=True)
        ):
            color, marker, linestyle = baseline_styles[index % len(baseline_styles)]
            ordered = group.sort_values("budget")
            ax.plot(
                ordered["budget"],
                ordered["outer_rmse_mean"],
                marker=marker,
                color=color,
                linestyle=linestyle,
                label=f"{baseline_name} held-out RMSE",
            )
    ax.set_xlabel("Sample budget", fontsize=fontsize)
    ax.set_ylabel("Held-out RMSE", fontsize=fontsize)
    ax.set_title("Oracle vs screening-selected held-out RMSE", fontsize=fontsize)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=_DEFAULT_LEGEND_FONTSIZE)
    _set_integer_x_ticks(ax)
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
        colors = [
            "tab:red",
            "tab:orange",
            "tab:green",
            "tab:purple",
            "tab:brown",
        ]
        for index, (policy_name, group) in enumerate(
            frame.groupby("policy_name", sort=True)
        ):
            ordered = group.sort_values("budget")
            color = colors[index % len(colors)]
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
        baseline_styles = [
            ("tab:green", "^", ":"),
            ("tab:brown", "D", "-."),
        ]
        for index, (baseline_name, group) in enumerate(
            fixed_frame.groupby("baseline_name", sort=True)
        ):
            color, marker, linestyle = baseline_styles[index % len(baseline_styles)]
            ordered = group.sort_values("budget")
            ax.plot(
                ordered["budget"],
                ordered["mean_regret"],
                marker=marker,
                color=color,
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
    cmap = plt.cm.get_cmap("tab10", max(1, len(dataset_order)))
    for idx, dataset in enumerate(dataset_order):
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
            color = cmap(idx)
            ax.plot(
                ordered["budget"],
                ordered["mean_regret"],
                marker="o",
                color=color,
                label=label,
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

    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Sample budget", fontsize=fontsize)
    ax.set_ylabel("Regret", fontsize=fontsize)
    ax.set_title("Screening policy regret by dataset", fontsize=fontsize)
    if log_x:
        ax.set_xscale("log")
    else:
        _set_integer_x_ticks(ax)
    ax.tick_params(axis="both", labelsize=_DEFAULT_TICK_FONTSIZE)
    ax.grid(True, alpha=0.3)
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
) -> Path:
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

    dataset_order = list(dict.fromkeys(filtered["dataset"].tolist()))
    label_rows = oracle_df.loc[:, ["dataset", "dataset_label"]].drop_duplicates(
        subset=["dataset"],
        keep="first",
    )
    dataset_labels = (
        label_rows.set_index("dataset")
        .reindex(dataset_order)["dataset_label"]
        .fillna(pd.Series(dataset_order, index=dataset_order))
        .to_dict()
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.cm.get_cmap("tab10", max(1, len(dataset_order)))
    for idx, dataset in enumerate(dataset_order):
        dataset_rows = filtered.loc[filtered["dataset"] == dataset]
        ax.plot(
            dataset_rows["n_train"],
            dataset_rows[metric_column],
            marker="o",
            color=cmap(idx),
            label=dataset_labels[dataset],
        )

    ax.set_xlabel("Train size", fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    if log_x:
        ax.set_xscale("log")
    else:
        _set_integer_x_ticks(ax)
    ax.tick_params(axis="both", labelsize=_DEFAULT_TICK_FONTSIZE)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=_DEFAULT_LEGEND_FONTSIZE)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


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
