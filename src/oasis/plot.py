from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from oasis.config import Config
from oasis.exp import SweepSplit, generate_sweep_splits

try:
    import polars as pl
except ModuleNotFoundError:  # optional for parity plotting from pandas inputs
    pl = None

from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error


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


def _trimmed_mean_predictions(X_corrected: np.ndarray) -> np.ndarray:
    """
    Compute per-sample mean after dropping MLIP predictions outside 1 std from row mean.
    """
    row_means = X_corrected.mean(axis=1)
    row_stds = X_corrected.std(axis=1)
    # Broadcast masks; keep all if std is zero or mask would be empty.
    mask = np.abs(X_corrected - row_means[:, None]) <= row_stds[:, None]
    empty_mask = mask.sum(axis=1) == 0
    if empty_mask.any():
        mask[empty_mask] = True
    return (X_corrected * mask).sum(axis=1) / mask.sum(axis=1)


def _sweep_model(
    model_factory,
    X: np.ndarray,
    y: np.ndarray,
    splits: Sequence[SweepSplit],
) -> pd.DataFrame:
    rmses_by_size: dict[int, list[float]] = {}
    for split in splits:
        model = model_factory()
        model.fit(X[split.train_idx], y[split.train_idx])
        X_test = X[split.test_idx]
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y[split.test_idx], preds))
        rmses_by_size.setdefault(split.sweep_size, []).append(rmse)

    results = []
    for n_train, rmses in rmses_by_size.items():
        results.append(
            {
                "n_train": n_train,
                "rmse_mean": float(np.mean(rmses)),
                "rmse_std": float(np.std(rmses)),
            }
        )
    return pd.DataFrame(results)


def _sweep_model_trimmed(
    model_factory,
    X: np.ndarray,
    y: np.ndarray,
    splits: Sequence[SweepSplit],
    z_thresh: float = 1.0,
) -> pd.DataFrame:
    """
    Fit the model, drop test samples with large contribution z-scores, and refit.
    """
    rmses_by_size: dict[int, list[float]] = {}
    for split in splits:
        model = model_factory()
        model.fit(X[split.train_idx], y[split.train_idx])

        X_test = X[split.test_idx]
        preds = model.predict(X_test)

        # Compute contribution z-scores per sample
        w = model.coef_
        contrib = X_test * w
        mu = contrib.mean()
        sigma = contrib.std() if contrib.std() > 0 else 1.0
        z = (contrib - mu) / sigma
        keep_mask = (np.abs(z) <= z_thresh).all(axis=1)
        if keep_mask.sum() == 0:
            keep_mask = np.ones(len(X_test), dtype=bool)

        preds_eval = preds[keep_mask]
        y_eval = y[split.test_idx][keep_mask]
        rmse = np.sqrt(mean_squared_error(y_eval, preds_eval))
        rmses_by_size.setdefault(split.sweep_size, []).append(rmse)

    results = []
    for n_train, rmses in rmses_by_size.items():
        results.append(
            {
                "n_train": n_train,
                "rmse_mean": float(np.mean(rmses)),
                "rmse_std": float(np.std(rmses)),
            }
        )
    return pd.DataFrame(results)


def _residual_sweep(
    X: np.ndarray,
    y: np.ndarray,
    splits: Sequence[SweepSplit],
) -> pd.DataFrame:
    rmses_by_size: dict[int, list[float]] = {}
    for split in splits:
        X_train = X[split.train_idx]
        y_train = y[split.train_idx]

        residuals = y_train[:, None] - X_train
        mean_residuals = residuals.mean(axis=0)

        X_corrected = X[split.test_idx] + mean_residuals
        preds = X_corrected.mean(axis=1)
        rmse = np.sqrt(mean_squared_error(y[split.test_idx], preds))
        rmses_by_size.setdefault(split.sweep_size, []).append(rmse)

    results = []
    for n_train, rmses in rmses_by_size.items():
        results.append(
            {
                "n_train": n_train,
                "rmse_mean": float(np.mean(rmses)),
                "rmse_std": float(np.std(rmses)),
            }
        )
    return pd.DataFrame(results)


def _residual_sweep_trimmed(
    X: np.ndarray,
    y: np.ndarray,
    splits: Sequence[SweepSplit],
) -> pd.DataFrame:
    """
    Residual correction with per-sample outlier MLIP removal before averaging.
    """
    rmses_by_size: dict[int, list[float]] = {}
    for split in splits:
        X_train = X[split.train_idx]
        y_train = y[split.train_idx]

        residuals = y_train[:, None] - X_train
        mean_residuals = residuals.mean(axis=0)

        X_corrected = X[split.test_idx] + mean_residuals
        preds = _trimmed_mean_predictions(X_corrected)
        rmse = np.sqrt(mean_squared_error(y[split.test_idx], preds))
        rmses_by_size.setdefault(split.sweep_size, []).append(rmse)

    results = []
    for n_train, rmses in rmses_by_size.items():
        results.append(
            {
                "n_train": n_train,
                "rmse_mean": float(np.mean(rmses)),
                "rmse_std": float(np.std(rmses)),
            }
        )
    return pd.DataFrame(results)


def _linearization_sweep(
    X: np.ndarray,
    y: np.ndarray,
    splits: Sequence[SweepSplit],
) -> pd.DataFrame:
    rmses_by_size: dict[int, list[float]] = {}
    for split in splits:
        X_train = X[split.train_idx]
        y_train = y[split.train_idx]

        Xh = np.asarray(X_train)
        yh = np.asarray(y_train).reshape(-1, 1)

        if Xh.ndim == 1:
            mu_h = Xh.reshape(-1, 1)
        else:
            mu_h = Xh.mean(axis=1, keepdims=True)

        lr = LinearRegression().fit(mu_h, yh)
        a = float(lr.coef_.ravel()[0])
        b = float(lr.intercept_.ravel()[0])

        X_linearized = a * X + b
        preds = X_linearized[split.test_idx].mean(axis=1)
        rmse = np.sqrt(mean_squared_error(y[split.test_idx], preds))
        rmses_by_size.setdefault(split.sweep_size, []).append(rmse)

    results = []
    for n_train, rmses in rmses_by_size.items():
        results.append(
            {
                "n_train": n_train,
                "rmse_mean": float(np.mean(rmses)),
                "rmse_std": float(np.std(rmses)),
            }
        )
    return pd.DataFrame(results)


def _linearization_sweep_trimmed(
    X: np.ndarray,
    y: np.ndarray,
    splits: Sequence[SweepSplit],
) -> pd.DataFrame:
    """
    Linearize against trimmed train means, then trim ensemble averaging.
    """
    rmses_by_size: dict[int, list[float]] = {}
    for split in splits:
        X_train = X[split.train_idx]
        y_train = y[split.train_idx]

        Xh = np.asarray(X_train)
        yh = np.asarray(y_train).reshape(-1, 1)

        if Xh.ndim == 1:
            mu_h = Xh.reshape(-1, 1)
        else:
            mu_h = _trimmed_mean_predictions(Xh).reshape(-1, 1)

        lr = LinearRegression().fit(mu_h, yh)
        a = float(lr.coef_.ravel()[0])
        b = float(lr.intercept_.ravel()[0])

        X_linearized = a * X + b
        preds = _trimmed_mean_predictions(X_linearized[split.test_idx])
        rmse = np.sqrt(mean_squared_error(y[split.test_idx], preds))
        rmses_by_size.setdefault(split.sweep_size, []).append(rmse)

    results = []
    for n_train, rmses in rmses_by_size.items():
        results.append(
            {
                "n_train": n_train,
                "rmse_mean": float(np.mean(rmses)),
                "rmse_std": float(np.std(rmses)),
            }
        )
    return pd.DataFrame(results)


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
    feature_cols = _mlip_columns(df)
    target_col = "reference_ads_eng"

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

    if not feature_cols:
        raise ValueError(
            "No MLIP prediction columns found (expected *_mlip_ads_eng_median)."
        )
    if df.height <= 5:
        raise ValueError("Not enough data to evaluate (need >5 samples).")

    X = df.select(feature_cols).to_numpy()
    y = df[target_col].to_numpy()

    max_train_val = min(max_train_val, len(X) - 1)
    seed = cfg.seed if cfg and cfg.seed is not None else 42
    shared_splits = list(
        generate_sweep_splits(
            len(X),
            min_train_val,
            max_train_val,
            n_repeats,
            np.random.default_rng(seed),
        )
    )

    ridge_df = (
        _sweep_model(lambda: Ridge(alpha=0.1), X, y, shared_splits)
        if use_ridge
        else None
    )
    kernel_ridge_df = (
        _sweep_model(
            lambda: KernelRidge(alpha=1.0, kernel="rbf"),
            X,
            y,
            shared_splits,
        )
        if use_kernel_ridge
        else None
    )
    ridge_trimmed_df = (
        _sweep_model_trimmed(
            lambda: Ridge(alpha=0.1),
            X,
            y,
            shared_splits,
            z_thresh=1.0,
        )
        if use_trim and use_ridge
        else None
    )
    lasso_df = (
        _sweep_model(lambda: Lasso(alpha=0.1, max_iter=10000), X, y, shared_splits)
        if use_lasso
        else None
    )
    lasso_trimmed_df = (
        _sweep_model_trimmed(
            lambda: Lasso(alpha=0.1, max_iter=10000),
            X,
            y,
            shared_splits,
            z_thresh=1.0,
        )
        if use_trim and use_lasso
        else None
    )
    elastic_df = (
        _sweep_model(
            lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=20000),
            X,
            y,
            shared_splits,
        )
        if use_elastic
        else None
    )
    elastic_trimmed_df = (
        _sweep_model_trimmed(
            lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=20000),
            X,
            y,
            shared_splits,
            z_thresh=1.0,
        )
        if use_trim and use_elastic
        else None
    )
    resid_df = (
        _residual_sweep(X, y, shared_splits)
        if use_residual
        else None
    )
    resid_trimmed_df = (
        _residual_sweep_trimmed(X, y, shared_splits)
        if use_trim and use_residual
        else None
    )
    linear_df = (
        _linearization_sweep(X, y, shared_splits)
        if use_linearization
        else None
    )
    linear_trimmed_df = (
        _linearization_sweep_trimmed(X, y, shared_splits)
        if use_trim and use_linearization
        else None
    )
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
            resid_df["n_train"],
            resid_df["rmse_mean"],
            marker="^",
            color="tab:green",
            label="Residual mean",
        )
        ax.fill_between(
            resid_df["n_train"],
            resid_df["rmse_mean"] - resid_df["rmse_std"],
            resid_df["rmse_mean"] + resid_df["rmse_std"],
            color="tab:green",
            alpha=0.2,
            label="Residual +/- 1sd",
        )
    if resid_trimmed_df is not None:
        ax.plot(
            resid_trimmed_df["n_train"],
            resid_trimmed_df["rmse_mean"],
            marker="P",
            color="tab:brown",
            label="Residual (trimmed) mean",
        )
        ax.fill_between(
            resid_trimmed_df["n_train"],
            resid_trimmed_df["rmse_mean"] - resid_trimmed_df["rmse_std"],
            resid_trimmed_df["rmse_mean"] + resid_trimmed_df["rmse_std"],
            color="tab:brown",
            alpha=0.2,
            label="Residual (trimmed) +/- 1sd",
        )
    if linear_df is not None:
        ax.plot(
            linear_df["n_train"],
            linear_df["rmse_mean"],
            marker="v",
            color="tab:red",
            label="Linearization mean",
        )
        ax.fill_between(
            linear_df["n_train"],
            linear_df["rmse_mean"] - linear_df["rmse_std"],
            linear_df["rmse_mean"] + linear_df["rmse_std"],
            color="tab:red",
            alpha=0.2,
            label="Linearization +/- 1sd",
        )
    if linear_trimmed_df is not None:
        ax.plot(
            linear_trimmed_df["n_train"],
            linear_trimmed_df["rmse_mean"],
            marker="<",
            color="tab:pink",
            label="Linearization (trimmed) mean",
        )
        ax.fill_between(
            linear_trimmed_df["n_train"],
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
