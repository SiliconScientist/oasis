from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from oasis.exp import LearningCurveResults, prepare_parity_plot_data

_MLIP_DISPLAY_NAMES = {
    "7net-omni": "7Net-Omni",
    "mace-mh-1": "MACE-MH-1",
    "mattersim-v1-5m": "MatterSim-v1-5M",
    "orb-v3-conservative-inf-omat": "ORB-v3\nconservative",
    "uma-s-1p1": "UMA-s-1p1",
}


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

    ax.set_xlabel("Reference adsorption energy (eV)")
    ax.set_ylabel("MLIP adsorption energy (eV)")
    ax.set_title("Parity plot (all MLIPs)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
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
    fontsize: int = 8,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    if results.ridge_df is not None:
        ax.plot(
            results.ridge_df["n_train"],
            results.ridge_df["rmse_mean"],
            marker="o",
            color="tab:blue",
            label="Ridge (alpha=0.1) mean",
        )
        ax.fill_between(
            results.ridge_df["n_train"],
            results.ridge_df["rmse_mean"] - results.ridge_df["rmse_std"],
            results.ridge_df["rmse_mean"] + results.ridge_df["rmse_std"],
            color="tab:blue",
            alpha=0.2,
            label="Ridge (alpha=0.1) +/- 1sd",
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
    if results.lasso_trimmed_df is not None:
        ax.plot(
            results.lasso_trimmed_df["n_train"],
            results.lasso_trimmed_df["rmse_mean"],
            marker="+",
            color="tab:orange",
            label="Lasso (trim residual) mean",
        )
        ax.fill_between(
            results.lasso_trimmed_df["n_train"],
            results.lasso_trimmed_df["rmse_mean"] - results.lasso_trimmed_df["rmse_std"],
            results.lasso_trimmed_df["rmse_mean"] + results.lasso_trimmed_df["rmse_std"],
            color="tab:orange",
            alpha=0.2,
            label="Lasso (trim residual) +/- 1sd",
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
    if results.elastic_trimmed_df is not None:
        ax.plot(
            results.elastic_trimmed_df["n_train"],
            results.elastic_trimmed_df["rmse_mean"],
            marker="x",
            color="tab:purple",
            label="Elastic Net (trim residual) mean",
        )
        ax.fill_between(
            results.elastic_trimmed_df["n_train"],
            results.elastic_trimmed_df["rmse_mean"] - results.elastic_trimmed_df["rmse_std"],
            results.elastic_trimmed_df["rmse_mean"] + results.elastic_trimmed_df["rmse_std"],
            color="tab:purple",
            alpha=0.2,
            label="Elastic Net (trim residual) +/- 1sd",
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
    if results.resid_trimmed_df is not None:
        ax.plot(
            results.resid_trimmed_df["n_train"],
            results.resid_trimmed_df["rmse_mean"],
            marker="P",
            color="tab:brown",
            label="Residual (trimmed) mean",
        )
        ax.fill_between(
            results.resid_trimmed_df["n_train"],
            results.resid_trimmed_df["rmse_mean"] - results.resid_trimmed_df["rmse_std"],
            results.resid_trimmed_df["rmse_mean"] + results.resid_trimmed_df["rmse_std"],
            color="tab:brown",
            alpha=0.2,
            label="Residual (trimmed) +/- 1sd",
        )
    if results.linear_df is not None:
        ax.plot(
            results.linear_df["n_train"],
            results.linear_df["rmse_mean"],
            marker="v",
            color="tab:red",
            label="Linearization mean",
        )
        ax.fill_between(
            results.linear_df["n_train"],
            results.linear_df["rmse_mean"] - results.linear_df["rmse_std"],
            results.linear_df["rmse_mean"] + results.linear_df["rmse_std"],
            color="tab:red",
            alpha=0.2,
            label="Linearization +/- 1sd",
        )
    if results.linear_trimmed_df is not None:
        ax.plot(
            results.linear_trimmed_df["n_train"],
            results.linear_trimmed_df["rmse_mean"],
            marker="<",
            color="tab:pink",
            label="Linearization (trimmed) mean",
        )
        ax.fill_between(
            results.linear_trimmed_df["n_train"],
            results.linear_trimmed_df["rmse_mean"] - results.linear_trimmed_df["rmse_std"],
            results.linear_trimmed_df["rmse_mean"] + results.linear_trimmed_df["rmse_std"],
            color="tab:pink",
            alpha=0.2,
            label="Linearization (trimmed) +/- 1sd",
        )
    if results.weighted_combiner_df is not None:
        ax.plot(
            results.weighted_combiner_df["n_train"],
            results.weighted_combiner_df["rmse_mean"],
            marker="*",
            color="tab:gray",
            label="Weighted combiner mean",
        )
        ax.fill_between(
            results.weighted_combiner_df["n_train"],
            results.weighted_combiner_df["rmse_mean"]
            - results.weighted_combiner_df["rmse_std"],
            results.weighted_combiner_df["rmse_mean"]
            + results.weighted_combiner_df["rmse_std"],
            color="tab:gray",
            alpha=0.2,
            label="Weighted combiner +/- 1sd",
        )
    if results.ridge_trimmed_df is not None:
        ax.plot(
            results.ridge_trimmed_df["n_train"],
            results.ridge_trimmed_df["rmse_mean"],
            marker="h",
            color="tab:olive",
            label="Ridge (trim z-score) mean",
        )
        ax.fill_between(
            results.ridge_trimmed_df["n_train"],
            results.ridge_trimmed_df["rmse_mean"] - results.ridge_trimmed_df["rmse_std"],
            results.ridge_trimmed_df["rmse_mean"] + results.ridge_trimmed_df["rmse_std"],
            color="tab:olive",
            alpha=0.2,
            label="Ridge (trim z-score) +/- 1sd",
        )
    ax.set_xlabel("Train size", fontsize=fontsize)
    ax.set_ylabel("RMSE (eV)", fontsize=fontsize)
    ax.set_title("Learning curve (ensemble vs sample size)", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=fontsize)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path
