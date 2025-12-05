from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error


def _mlip_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith("_mlip_ads_eng_median")]


def parity_plot(df: pd.DataFrame, output_path: str | Path) -> Path:
    """
    Create a parity plot comparing reference adsorption energies to each MLIP prediction.

    Returns the path to the saved PNG.
    """
    mlip_cols = _mlip_columns(df)
    if not mlip_cols:
        raise ValueError("No MLIP prediction columns found (expected *_mlip_ads_eng_median).")
    if df.empty:
        raise ValueError("No data available to plot.")

    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = plt.cm.get_cmap("tab10", len(mlip_cols))

    for idx, col in enumerate(mlip_cols):
        ax.scatter(
            df["reference_ads_eng"],
            df[col],
            s=35,
            alpha=0.85,
            label=col.removesuffix("_mlip_ads_eng_median"),
            color=cmap(idx),
            edgecolor="black",
            linewidth=0.5,
        )

    min_val = min(df["reference_ads_eng"].min(), df[mlip_cols].min().min())
    max_val = max(df["reference_ads_eng"].max(), df[mlip_cols].max().max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1, label="Parity")

    ax.set_xlabel("Reference adsorption energy (eV)")
    ax.set_ylabel("MLIP adsorption energy (eV)")
    ax.set_title("Normal predictions parity plot (all MLIPs)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def _sweep_model(model_factory, X: np.ndarray, y: np.ndarray, min_train: int, max_train: int, n_repeats: int, rng: np.random.Generator) -> pd.DataFrame:
    results = []
    max_train = min(max_train, len(X) - 1)
    for n_train in range(min_train, max_train + 1):
        rmses = []
        for _ in range(n_repeats):
            idx = np.arange(len(X))
            train_idx = rng.choice(idx, size=n_train, replace=False)
            test_idx = np.setdiff1d(idx, train_idx, assume_unique=False)
            model = model_factory()
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[test_idx])
            rmses.append(np.sqrt(mean_squared_error(y[test_idx], preds)))
        results.append({
            "n_train": n_train,
            "rmse_mean": float(np.mean(rmses)),
            "rmse_std": float(np.std(rmses)),
        })
    return pd.DataFrame(results)


def _residual_sweep(X: np.ndarray, y: np.ndarray, min_hold: int, max_hold: int, n_repeats: int, rng: np.random.Generator) -> pd.DataFrame:
    results = []
    max_hold = min(max_hold, len(X) - 1)
    for n_hold in range(min_hold, max_hold + 1):
        rmses = []
        for _ in range(n_repeats):
            idx = np.arange(len(X))
            hold_idx = rng.choice(idx, size=n_hold, replace=False)
            keep_idx = np.setdiff1d(idx, hold_idx, assume_unique=False)

            X_hold = X[hold_idx]
            y_hold = y[hold_idx]

            residuals = y_hold[:, None] - X_hold
            mean_residuals = residuals.mean(axis=0)

            X_corrected = X[keep_idx] + mean_residuals
            preds = X_corrected.mean(axis=1)

            rmses.append(np.sqrt(mean_squared_error(y[keep_idx], preds)))

        results.append({
            "n_holdout": n_hold,
            "rmse_mean": float(np.mean(rmses)),
            "rmse_std": float(np.std(rmses)),
        })
    return pd.DataFrame(results)


def _linearization_sweep(X: np.ndarray, y: np.ndarray, min_hold: int, max_hold: int, n_repeats: int, rng: np.random.Generator) -> pd.DataFrame:
    results = []
    max_hold = min(max_hold, len(X) - 1)
    for n_hold in range(min_hold, max_hold + 1):
        rmses = []
        for _ in range(n_repeats):
            idx = np.arange(len(X))
            hold_idx = rng.choice(idx, size=n_hold, replace=False)
            keep_idx = np.setdiff1d(idx, hold_idx, assume_unique=False)

            X_hold = X[hold_idx]
            y_hold = y[hold_idx]

            Xh = np.asarray(X_hold)
            yh = np.asarray(y_hold).reshape(-1, 1)

            if Xh.ndim == 1:
                mu_h = Xh.reshape(-1, 1)
            else:
                mu_h = Xh.mean(axis=1, keepdims=True)

            lr = LinearRegression().fit(mu_h, yh)
            a = float(lr.coef_.ravel()[0])
            b = float(lr.intercept_.ravel()[0])

            X_linearized = a * X + b
            preds = X_linearized[keep_idx].mean(axis=1)

            rmses.append(np.sqrt(mean_squared_error(y[keep_idx], preds)))

        results.append({
            "n_holdout": n_hold,
            "rmse_mean": float(np.mean(rmses)),
            "rmse_std": float(np.std(rmses)),
        })
    return pd.DataFrame(results)


def ensemble_rmse_plot(
    df: pd.DataFrame,
    output_path: str | Path,
    min_train: int = 10,
    max_train: int = 20,
    n_repeats: int = 50,
) -> Path:
    """
    Reproduce the ensemble RMSE sweeps from the notebook's final cell and plot the overlay.
    """
    feature_cols = _mlip_columns(df)
    target_col = "reference_ads_eng"

    if not feature_cols:
        raise ValueError("No MLIP prediction columns found (expected *_mlip_ads_eng_median).")
    if len(df) <= 5:
        raise ValueError("Not enough data to evaluate (need >5 samples).")

    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()

    rng_ridge = np.random.default_rng(42)
    rng_lasso = np.random.default_rng(123)
    rng_elastic = np.random.default_rng(321)
    rng_resid = np.random.default_rng(999)
    rng_linear = np.random.default_rng(2024)

    ridge_df = _sweep_model(lambda: Ridge(alpha=1.0), X, y, min_train, max_train, n_repeats, rng_ridge)
    lasso_df = _sweep_model(lambda: Lasso(alpha=0.1, max_iter=10000), X, y, min_train, max_train, n_repeats, rng_lasso)
    elastic_df = _sweep_model(lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=20000), X, y, min_train, max_train, n_repeats, rng_elastic)
    resid_df = _residual_sweep(X, y, min_train, max_train, n_repeats, rng_resid)
    linear_df = _linearization_sweep(X, y, min_train, max_train, n_repeats, rng_linear)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ridge_df["n_train"], ridge_df["rmse_mean"], marker="o", color="tab:blue", label="Ridge mean")
    ax.fill_between(
        ridge_df["n_train"],
        ridge_df["rmse_mean"] - ridge_df["rmse_std"],
        ridge_df["rmse_mean"] + ridge_df["rmse_std"],
        color="tab:blue",
        alpha=0.2,
        label="Ridge +/- 1sd",
    )
    ax.plot(lasso_df["n_train"], lasso_df["rmse_mean"], marker="s", color="tab:orange", label="Lasso mean")
    ax.fill_between(
        lasso_df["n_train"],
        lasso_df["rmse_mean"] - lasso_df["rmse_std"],
        lasso_df["rmse_mean"] + lasso_df["rmse_std"],
        color="tab:orange",
        alpha=0.2,
        label="Lasso +/- 1sd",
    )
    ax.plot(elastic_df["n_train"], elastic_df["rmse_mean"], marker="D", color="tab:purple", label="Elastic Net mean")
    ax.fill_between(
        elastic_df["n_train"],
        elastic_df["rmse_mean"] - elastic_df["rmse_std"],
        elastic_df["rmse_mean"] + elastic_df["rmse_std"],
        color="tab:purple",
        alpha=0.2,
        label="Elastic Net +/- 1sd",
    )
    ax.plot(resid_df["n_holdout"], resid_df["rmse_mean"], marker="^", color="tab:green", label="Residual mean")
    ax.fill_between(
        resid_df["n_holdout"],
        resid_df["rmse_mean"] - resid_df["rmse_std"],
        resid_df["rmse_mean"] + resid_df["rmse_std"],
        color="tab:green",
        alpha=0.2,
        label="Residual +/- 1sd",
    )
    ax.plot(linear_df["n_holdout"], linear_df["rmse_mean"], marker="v", color="tab:red", label="Linearization mean")
    ax.fill_between(
        linear_df["n_holdout"],
        linear_df["rmse_mean"] - linear_df["rmse_std"],
        linear_df["rmse_mean"] + linear_df["rmse_std"],
        color="tab:red",
        alpha=0.2,
        label="Linearization +/- 1sd",
    )
    ax.set_xlabel("Training / holdout size")
    ax.set_ylabel("RMSE (eV)")
    ax.set_title("Ensemble RMSE vs sample size")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path
