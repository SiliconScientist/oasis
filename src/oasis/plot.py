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
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from oasis.exp import prepare_parity_plot_data
from oasis.learning_curve.time_accuracy import build_time_accuracy_table
from oasis.mlip.timing import MlipGenerationTimingSummary
from oasis.sweep import LearningCurveResults

_MLIP_DISPLAY_NAMES = {
    "7net-omni": "7Net-Omni",
    "mace-mh-1": "MACE-MH-1",
    "mattersim-v1-5m": "MatterSim-v1-5M",
    "orb-v3-conservative-inf-omat": "ORB-v3\nconservative",
    "uma-s-1p1": "UMA-s-1p1",
}
_DEFAULT_PLOT_FONTSIZE = 16
_DEFAULT_TICK_FONTSIZE = 8
_DEFAULT_LEGEND_FONTSIZE = 8
_METHOD_PLOT_STYLES = (
    ("ridge", "ridge_df", "ridge_uq_df", "Ridge", "o", "tab:blue"),
    ("kernel_ridge", "kernel_ridge_df", "kernel_ridge_uq_df", "Kernel Ridge", "X", "tab:cyan"),
    ("lasso", "lasso_df", "lasso_uq_df", "Lasso", "s", "tab:orange"),
    ("elastic", "elastic_df", "elastic_uq_df", "Elastic Net", "D", "tab:purple"),
    ("residual", "resid_df", "resid_uq_df", "Residual", "^", "tab:green"),
    ("weighted_linear", "weighted_linear_df", "weighted_linear_uq_df", "Weighted linear", "*", "tab:gray"),
    ("weighted_simplex", "weighted_simplex_df", "weighted_simplex_uq_df", "Weighted simplex", "8", "teal"),
    ("graph_mean", "graph_mean_df", "graph_mean_uq_df", "Graph mean", "P", "tab:red"),
    ("moe", "moe_df", "moe_uq_df", "MoE", "*", "tab:purple"),
    ("gnn_direct", "gnn_direct_df", "gnn_direct_uq_df", "GNN direct", "s", "tab:cyan"),
    ("probe_gnn", "probe_gnn_df", "probe_gnn_uq_df", "Probe GNN", "D", "tab:olive"),
    ("latent", "latent_df", "latent_uq_df", "Latent", "v", "tab:brown"),
)


def _ordered_learning_curve_frame(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    if frame is None or frame.empty:
        return frame
    if "n_train" not in frame.columns:
        raise ValueError("learning-curve result frames must contain an n_train column.")
    return frame.sort_values("n_train").reset_index(drop=True)


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
    raise ValueError(
        "screening result frames must contain cv_rmse_mean/cv_rmse_std."
    )


def _set_integer_x_ticks(ax: Any) -> None:
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


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
    title: str,
    ylabel: str,
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
    for _, _, uq_field, display_name, marker, color in _METHOD_PLOT_STYLES:
        frame = getattr(results, uq_field)
        if frame is None or frame.empty or metric_column not in frame.columns:
            continue
        metric_std_column = f"{metric_column}_std"
        ax.plot(
            frame["n_train"],
            frame[metric_column],
            marker=marker,
            color=color,
            label=display_name,
        )
        if metric_std_column in frame.columns:
            ax.fill_between(
                frame["n_train"],
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
        ax.set_xlabel("Train size", fontsize=fontsize)
    else:
        ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
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


def parity_plot(df: Any, output_path: str | Path) -> Path:
    """
    Create a parity plot comparing reference adsorption energies to each MLIP prediction.

    Returns the path to the saved PNG.
    """
    plot_data = prepare_parity_plot_data(df)
    ref = plot_data.reference

    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = plt.cm.get_cmap("tab10", len(plot_data.predictions))

    for idx, (label, preds) in enumerate(plot_data.predictions.items()):
        ax.scatter(
            ref,
            preds,
            s=35,
            alpha=0.85,
            label=label,
            color=cmap(idx),
            edgecolor="black",
            linewidth=0.5,
        )

    mlip_vals = np.concatenate(list(plot_data.predictions.values()))
    min_val = min(ref.min(), mlip_vals.min())
    max_val = max(ref.max(), mlip_vals.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1, label="Parity")

    ax.set_xlabel("Reference adsorption energy (eV)", fontsize=_DEFAULT_PLOT_FONTSIZE)
    ax.set_ylabel("MLIP adsorption energy (eV)", fontsize=_DEFAULT_PLOT_FONTSIZE)
    ax.set_title("Parity plot (all MLIPs)", fontsize=_DEFAULT_PLOT_FONTSIZE)
    ax.tick_params(axis="both", labelsize=_DEFAULT_TICK_FONTSIZE)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=_DEFAULT_LEGEND_FONTSIZE)
    ax.grid(True, linestyle="--", alpha=0.3)
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
        title="Miscalibration area vs train size",
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
        title="Sharpness vs train size",
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
        title="Dispersion vs train size",
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
            results.weighted_linear_df[mean_col]
            - results.weighted_linear_df[std_col],
            results.weighted_linear_df[mean_col]
            + results.weighted_linear_df[std_col],
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
            ordered = method_table.sort_values("n_train").reset_index(drop=True)
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
    method_names: tuple[str, ...] | list[str] | None = None,
    fontsize: int = _DEFAULT_PLOT_FONTSIZE,
    show_legend: bool = True,
) -> Path:
    table = build_time_accuracy_table(
        results,
        generation_timing_by_mlip,
        mlip_feature_names=mlip_feature_names,
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
    method_names: tuple[str, ...] | list[str] | None = None,
    fontsize: int = _DEFAULT_PLOT_FONTSIZE,
    show_legend: bool = True,
) -> Path:
    table = build_time_accuracy_table(
        results,
        generation_timing_by_mlip,
        mlip_feature_names=mlip_feature_names,
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
    method_names: tuple[str, ...] | list[str] | None = None,
    fontsize: int = _DEFAULT_PLOT_FONTSIZE,
    show_legend: bool = True,
) -> Path:
    table = build_time_accuracy_table(
        results,
        generation_timing_by_mlip,
        mlip_feature_names=mlip_feature_names,
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
