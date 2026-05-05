from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from oasis.config import Config
from oasis.exp import build_learning_curve_sweeps

try:
    import polars as pl
except ModuleNotFoundError:  # optional for parity plotting from pandas inputs
    pl = None

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


def _mlip_columns(df: Any) -> list[str]:
    return [c for c in df.columns if c.endswith("_mlip_ads_eng_median")]


def _column_to_numpy(df: Any, col: str) -> np.ndarray:
    series = df[col]
    if hasattr(series, "to_numpy"):
        return series.to_numpy()
    return np.asarray(series)


def parity_plot(df: Any, output_path: str | Path) -> Path:
    """
    Create a parity plot comparing reference adsorption energies to each MLIP prediction.

    Returns the path to the saved PNG.
    """
    mlip_cols = _mlip_columns(df)
    if not mlip_cols:
        raise ValueError(
            "No MLIP prediction columns found (expected *_mlip_ads_eng_median)."
        )
    n_rows = len(df)
    if n_rows == 0:
        raise ValueError("No data available to plot.")

    ref = _column_to_numpy(df, "reference_ads_eng")

    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = plt.cm.get_cmap("tab10", len(mlip_cols))

    for idx, col in enumerate(mlip_cols):
        ax.scatter(
            ref,
            _column_to_numpy(df, col),
            s=35,
            alpha=0.85,
            label=col.removesuffix("_mlip_ads_eng_median"),
            color=cmap(idx),
            edgecolor="black",
            linewidth=0.5,
        )

    mlip_vals = np.concatenate([_column_to_numpy(df, c) for c in mlip_cols])
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


def _coerce_sweep_dataframe(
    sweep_data: Any,
) -> Any:
    if isinstance(sweep_data, pd.DataFrame):
        return sweep_data.copy()
    if isinstance(sweep_data, (str, Path)):
        path = Path(sweep_data)
        if path.suffix.lower() == ".csv":
            if pl is not None:
                return pl.read_csv(path)
            return pd.read_csv(path)
        raise ValueError(f"Unsupported sweep file format: {path.suffix}")
    if pl is not None and isinstance(sweep_data, pl.DataFrame):
        return sweep_data.clone()
    if isinstance(sweep_data, Sequence) and not isinstance(sweep_data, (str, bytes)):
        rows: list[Mapping[str, Any] | dict[str, Any]] = []
        for item in sweep_data:
            if hasattr(item, "__dict__"):
                rows.append(vars(item))
            else:
                rows.append(dict(item))
        if pl is not None:
            return pl.DataFrame(rows)
        return pd.DataFrame(rows)
    raise TypeError("sweep_data must be a DataFrame, CSV path, or row sequence")


def moe_learning_speed_plot(
    sweep_data: Any,
    output_path: str | Path,
    *,
    x_col: str = "train_size",
    metric: str = "val_rmse",
    title: str = "MOE learning speed",
) -> Path:
    df = _coerce_sweep_dataframe(sweep_data)
    required_columns = {x_col, metric}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Sweep data is missing required columns: {sorted(missing_columns)}"
        )

    if pl is not None and isinstance(df, pl.DataFrame):
        cleaned = (
            df.select(
                [
                    pl.col(x_col).cast(pl.Float64, strict=False).alias(x_col),
                    pl.col(metric).cast(pl.Float64, strict=False).alias(metric),
                ]
            )
            .drop_nulls()
            .sort(x_col)
        )
        if cleaned.height == 0:
            raise ValueError(f"No numeric values available for metric '{metric}'")
        x = cleaned[x_col].to_numpy()
        y = cleaned[metric].to_numpy()
    else:
        df = df.sort_values(x_col).reset_index(drop=True)
        x = pd.to_numeric(df[x_col], errors="coerce")
        y = pd.to_numeric(df[metric], errors="coerce")
        valid = x.notna() & y.notna()
        if not valid.any():
            raise ValueError(f"No numeric values available for metric '{metric}'")
        x = x[valid].to_numpy()
        y = y[valid].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y, marker="o", linewidth=2, markersize=6, color="#1f77b4")
    ax.set_xlabel("Training samples")
    ax.set_ylabel(metric.replace("_", " ").upper())
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def learning_curve_plot(
    df: pl.DataFrame,
    output_path: str | Path,
    min_train: int | None = None,
    max_train: int | None = None,
    n_repeats: int = 50,
    fontsize: int = 8,
    cfg: Config | None = None,
) -> Path:
    """
    Reproduce the ensemble RMSE sweeps from the notebook's final cell and plot the overlay.
    """
    use_trim = cfg.plot.trim if cfg else True
    use_ridge = cfg.plot.use_ridge if cfg else True
    use_kernel_ridge = cfg.plot.use_kernel_ridge if cfg else True
    use_lasso = cfg.plot.use_lasso if cfg else True
    use_elastic = cfg.plot.use_elastic_net if cfg else True
    use_residual = cfg.plot.use_residual if cfg else True
    use_linearization = cfg.plot.use_linearization if cfg else True
    cfg_min_train = cfg.plot.min_train if cfg else 5
    cfg_max_train = cfg.plot.max_train if cfg else 10
    min_train_val = min_train if min_train is not None else cfg_min_train
    max_train_val = max_train if max_train is not None else cfg_max_train

    sweep_results = build_learning_curve_sweeps(
        df,
        min_train=min_train_val,
        max_train=max_train_val,
        n_repeats=n_repeats,
        use_trim=use_trim,
        use_ridge=use_ridge,
        use_kernel_ridge=use_kernel_ridge,
        use_lasso=use_lasso,
        use_elastic=use_elastic,
        use_residual=use_residual,
        use_linearization=use_linearization,
    )
    ridge_df = sweep_results["ridge_df"]
    kernel_ridge_df = sweep_results["kernel_ridge_df"]
    ridge_trimmed_df = sweep_results["ridge_trimmed_df"]
    lasso_df = sweep_results["lasso_df"]
    lasso_trimmed_df = sweep_results["lasso_trimmed_df"]
    elastic_df = sweep_results["elastic_df"]
    elastic_trimmed_df = sweep_results["elastic_trimmed_df"]
    resid_df = sweep_results["resid_df"]
    resid_trimmed_df = sweep_results["resid_trimmed_df"]
    linear_df = sweep_results["linear_df"]
    linear_trimmed_df = sweep_results["linear_trimmed_df"]
    fig, ax = plt.subplots(figsize=(7, 4))
    if ridge_df is not None:
        ax.plot(
            ridge_df["n_train"],
            ridge_df["rmse_mean"],
            marker="o",
            color="tab:blue",
            label="Ridge (alpha=0.1) mean",
        )
        ax.fill_between(
            ridge_df["n_train"],
            ridge_df["rmse_mean"] - ridge_df["rmse_std"],
            ridge_df["rmse_mean"] + ridge_df["rmse_std"],
            color="tab:blue",
            alpha=0.2,
            label="Ridge (alpha=0.1) +/- 1sd",
        )
    if kernel_ridge_df is not None:
        ax.plot(
            kernel_ridge_df["n_train"],
            kernel_ridge_df["rmse_mean"],
            marker="X",
            color="tab:cyan",
            label="Kernel Ridge mean",
        )
        ax.fill_between(
            kernel_ridge_df["n_train"],
            kernel_ridge_df["rmse_mean"] - kernel_ridge_df["rmse_std"],
            kernel_ridge_df["rmse_mean"] + kernel_ridge_df["rmse_std"],
            color="tab:cyan",
            alpha=0.2,
            label="Kernel Ridge +/- 1sd",
        )
    if lasso_df is not None:
        ax.plot(
            lasso_df["n_train"],
            lasso_df["rmse_mean"],
            marker="s",
            color="tab:orange",
            label="Lasso mean",
        )
        ax.fill_between(
            lasso_df["n_train"],
            lasso_df["rmse_mean"] - lasso_df["rmse_std"],
            lasso_df["rmse_mean"] + lasso_df["rmse_std"],
            color="tab:orange",
            alpha=0.2,
            label="Lasso +/- 1sd",
        )
    if lasso_trimmed_df is not None:
        ax.plot(
            lasso_trimmed_df["n_train"],
            lasso_trimmed_df["rmse_mean"],
            marker="+",
            color="tab:orange",
            label="Lasso (trim residual) mean",
        )
        ax.fill_between(
            lasso_trimmed_df["n_train"],
            lasso_trimmed_df["rmse_mean"] - lasso_trimmed_df["rmse_std"],
            lasso_trimmed_df["rmse_mean"] + lasso_trimmed_df["rmse_std"],
            color="tab:orange",
            alpha=0.2,
            label="Lasso (trim residual) +/- 1sd",
        )
    if elastic_df is not None:
        ax.plot(
            elastic_df["n_train"],
            elastic_df["rmse_mean"],
            marker="D",
            color="tab:purple",
            label="Elastic Net mean",
        )
        ax.fill_between(
            elastic_df["n_train"],
            elastic_df["rmse_mean"] - elastic_df["rmse_std"],
            elastic_df["rmse_mean"] + elastic_df["rmse_std"],
            color="tab:purple",
            alpha=0.2,
            label="Elastic Net +/- 1sd",
        )
    if elastic_trimmed_df is not None:
        ax.plot(
            elastic_trimmed_df["n_train"],
            elastic_trimmed_df["rmse_mean"],
            marker="x",
            color="tab:purple",
            label="Elastic Net (trim residual) mean",
        )
        ax.fill_between(
            elastic_trimmed_df["n_train"],
            elastic_trimmed_df["rmse_mean"] - elastic_trimmed_df["rmse_std"],
            elastic_trimmed_df["rmse_mean"] + elastic_trimmed_df["rmse_std"],
            color="tab:purple",
            alpha=0.2,
            label="Elastic Net (trim residual) +/- 1sd",
        )
    if resid_df is not None:
        ax.plot(
            resid_df["n_holdout"],
            resid_df["rmse_mean"],
            marker="^",
            color="tab:green",
            label="Residual mean",
        )
        ax.fill_between(
            resid_df["n_holdout"],
            resid_df["rmse_mean"] - resid_df["rmse_std"],
            resid_df["rmse_mean"] + resid_df["rmse_std"],
            color="tab:green",
            alpha=0.2,
            label="Residual +/- 1sd",
        )
    if resid_trimmed_df is not None:
        ax.plot(
            resid_trimmed_df["n_holdout"],
            resid_trimmed_df["rmse_mean"],
            marker="P",
            color="tab:brown",
            label="Residual (trimmed) mean",
        )
        ax.fill_between(
            resid_trimmed_df["n_holdout"],
            resid_trimmed_df["rmse_mean"] - resid_trimmed_df["rmse_std"],
            resid_trimmed_df["rmse_mean"] + resid_trimmed_df["rmse_std"],
            color="tab:brown",
            alpha=0.2,
            label="Residual (trimmed) +/- 1sd",
        )
    if linear_df is not None:
        ax.plot(
            linear_df["n_holdout"],
            linear_df["rmse_mean"],
            marker="v",
            color="tab:red",
            label="Linearization mean",
        )
        ax.fill_between(
            linear_df["n_holdout"],
            linear_df["rmse_mean"] - linear_df["rmse_std"],
            linear_df["rmse_mean"] + linear_df["rmse_std"],
            color="tab:red",
            alpha=0.2,
            label="Linearization +/- 1sd",
        )
    if linear_trimmed_df is not None:
        ax.plot(
            linear_trimmed_df["n_holdout"],
            linear_trimmed_df["rmse_mean"],
            marker="<",
            color="tab:pink",
            label="Linearization (trimmed) mean",
        )
        ax.fill_between(
            linear_trimmed_df["n_holdout"],
            linear_trimmed_df["rmse_mean"] - linear_trimmed_df["rmse_std"],
            linear_trimmed_df["rmse_mean"] + linear_trimmed_df["rmse_std"],
            color="tab:pink",
            alpha=0.2,
            label="Linearization (trimmed) +/- 1sd",
        )
    if ridge_trimmed_df is not None:
        ax.plot(
            ridge_trimmed_df["n_train"],
            ridge_trimmed_df["rmse_mean"],
            marker="h",
            color="tab:olive",
            label="Ridge (trim z-score) mean",
        )
        ax.fill_between(
            ridge_trimmed_df["n_train"],
            ridge_trimmed_df["rmse_mean"] - ridge_trimmed_df["rmse_std"],
            ridge_trimmed_df["rmse_mean"] + ridge_trimmed_df["rmse_std"],
            color="tab:olive",
            alpha=0.2,
            label="Ridge (trim z-score) +/- 1sd",
        )
    ax.set_xlabel("Training / holdout size", fontsize=fontsize)
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
