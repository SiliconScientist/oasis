from __future__ import annotations

from collections.abc import Iterator
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from oasis.config import Config


@dataclass(frozen=True, slots=True)
class SweepSplit:
    """One train/test split for a sweep size."""

    sweep_size: int
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass(frozen=True, slots=True)
class LearningCurveResults:
    ridge_df: pd.DataFrame | None
    kernel_ridge_df: pd.DataFrame | None
    ridge_trimmed_df: pd.DataFrame | None
    lasso_df: pd.DataFrame | None
    lasso_trimmed_df: pd.DataFrame | None
    elastic_df: pd.DataFrame | None
    elastic_trimmed_df: pd.DataFrame | None
    resid_df: pd.DataFrame | None
    resid_trimmed_df: pd.DataFrame | None
    linear_df: pd.DataFrame | None
    linear_trimmed_df: pd.DataFrame | None


@dataclass(frozen=True, slots=True)
class ParityPlotData:
    reference: np.ndarray
    predictions: Mapping[str, np.ndarray]


def generate_sweep_splits(
    n_samples: int,
    min_train: int,
    max_train: int,
    n_repeats: int,
    rng: np.random.Generator,
) -> Iterator[SweepSplit]:
    """Yield repeated train/test splits for each sweep size in the range."""

    idx = np.arange(n_samples)
    max_train = min(max_train, n_samples - 1)
    for n_train in range(min_train, max_train + 1):
        for _ in range(n_repeats):
            train_idx = rng.choice(idx, size=n_train, replace=False)
            test_idx = np.setdiff1d(idx, train_idx, assume_unique=False)
            yield SweepSplit(
                sweep_size=n_train,
                train_idx=train_idx,
                test_idx=test_idx,
            )


def mlip_columns(df: Any) -> list[str]:
    return [c for c in df.columns if c.endswith("_mlip_ads_eng_median")]


def column_to_numpy(df: Any, col: str) -> np.ndarray:
    series = df[col]
    if hasattr(series, "to_numpy"):
        return series.to_numpy()
    return np.asarray(series)


def prepare_parity_plot_data(df: Any) -> ParityPlotData:
    mlip_cols = mlip_columns(df)
    if not mlip_cols:
        raise ValueError(
            "No MLIP prediction columns found (expected *_mlip_ads_eng_median)."
        )
    if len(df) == 0:
        raise ValueError("No data available to plot.")

    return ParityPlotData(
        reference=column_to_numpy(df, "reference_ads_eng"),
        predictions={
            col.removesuffix("_mlip_ads_eng_median"): column_to_numpy(df, col)
            for col in mlip_cols
        },
    )


def run_learning_curve_experiments_from_frame(
    df: Any,
    *,
    min_train: int,
    max_train: int,
    n_repeats: int,
    seed: int = 42,
    use_trim: bool = True,
    use_ridge: bool = True,
    use_kernel_ridge: bool = True,
    use_lasso: bool = True,
    use_elastic: bool = True,
    use_residual: bool = True,
    use_linearization: bool = True,
) -> LearningCurveResults:
    feature_cols = mlip_columns(df)
    if not feature_cols:
        raise ValueError(
            "No MLIP prediction columns found (expected *_mlip_ads_eng_median)."
        )

    n_rows = getattr(df, "height", len(df))
    if n_rows <= 5:
        raise ValueError("Not enough data to evaluate (need >5 samples).")

    if hasattr(df, "select"):
        X = df.select(feature_cols).to_numpy()
    else:
        X = np.column_stack([column_to_numpy(df, col) for col in feature_cols])
    y = column_to_numpy(df, "reference_ads_eng")

    return run_learning_curve_experiments(
        X,
        y,
        min_train=min_train,
        max_train=max_train,
        n_repeats=n_repeats,
        seed=seed,
        use_trim=use_trim,
        use_ridge=use_ridge,
        use_kernel_ridge=use_kernel_ridge,
        use_lasso=use_lasso,
        use_elastic=use_elastic,
        use_residual=use_residual,
        use_linearization=use_linearization,
    )


def run_learning_curve_experiments_from_config(
    df: Any,
    cfg: Config | None,
) -> LearningCurveResults:
    plot_cfg = cfg.plot if cfg else None
    return run_learning_curve_experiments_from_frame(
        df,
        min_train=plot_cfg.min_train if plot_cfg else 5,
        max_train=plot_cfg.max_train if plot_cfg else 10,
        n_repeats=plot_cfg.n_repeats if plot_cfg else 50,
        seed=cfg.seed if cfg and cfg.seed is not None else 42,
        use_trim=plot_cfg.trim if plot_cfg else True,
        use_ridge=plot_cfg.use_ridge if plot_cfg else True,
        use_kernel_ridge=plot_cfg.use_kernel_ridge if plot_cfg else True,
        use_lasso=plot_cfg.use_lasso if plot_cfg else True,
        use_elastic=plot_cfg.use_elastic_net if plot_cfg else True,
        use_residual=plot_cfg.use_residual if plot_cfg else True,
        use_linearization=plot_cfg.use_linearization if plot_cfg else True,
    )


def run_learning_curve_experiments(
    X: np.ndarray,
    y: np.ndarray,
    *,
    min_train: int,
    max_train: int,
    n_repeats: int,
    seed: int = 42,
    use_trim: bool = True,
    use_ridge: bool = True,
    use_kernel_ridge: bool = True,
    use_lasso: bool = True,
    use_elastic: bool = True,
    use_residual: bool = True,
    use_linearization: bool = True,
) -> LearningCurveResults:
    from oasis.method import (
        linearization_sweep,
        linearization_sweep_trimmed,
        residual_sweep,
        residual_sweep_trimmed,
        sweep_model,
        sweep_model_trimmed,
    )
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.linear_model import ElasticNet, Lasso, Ridge

    max_train = min(max_train, len(X) - 1)
    shared_splits = list(
        generate_sweep_splits(
            len(X),
            min_train,
            max_train,
            n_repeats,
            np.random.default_rng(seed),
        )
    )

    return LearningCurveResults(
        ridge_df=(
            sweep_model(lambda: Ridge(alpha=0.1), X, y, shared_splits)
            if use_ridge
            else None
        ),
        kernel_ridge_df=(
            sweep_model(
                lambda: KernelRidge(alpha=1.0, kernel="rbf"),
                X,
                y,
                shared_splits,
            )
            if use_kernel_ridge
            else None
        ),
        ridge_trimmed_df=(
            sweep_model_trimmed(
                lambda: Ridge(alpha=0.1),
                X,
                y,
                shared_splits,
                z_thresh=1.0,
            )
            if use_trim and use_ridge
            else None
        ),
        lasso_df=(
            sweep_model(
                lambda: Lasso(alpha=0.1, max_iter=10000),
                X,
                y,
                shared_splits,
            )
            if use_lasso
            else None
        ),
        lasso_trimmed_df=(
            sweep_model_trimmed(
                lambda: Lasso(alpha=0.1, max_iter=10000),
                X,
                y,
                shared_splits,
                z_thresh=1.0,
            )
            if use_trim and use_lasso
            else None
        ),
        elastic_df=(
            sweep_model(
                lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=20000),
                X,
                y,
                shared_splits,
            )
            if use_elastic
            else None
        ),
        elastic_trimmed_df=(
            sweep_model_trimmed(
                lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=20000),
                X,
                y,
                shared_splits,
                z_thresh=1.0,
            )
            if use_trim and use_elastic
            else None
        ),
        resid_df=(
            residual_sweep(X, y, shared_splits) if use_residual else None
        ),
        resid_trimmed_df=(
            residual_sweep_trimmed(X, y, shared_splits)
            if use_trim and use_residual
            else None
        ),
        linear_df=(
            linearization_sweep(X, y, shared_splits)
            if use_linearization
            else None
        ),
        linear_trimmed_df=(
            linearization_sweep_trimmed(X, y, shared_splits)
            if use_trim and use_linearization
            else None
        ),
    )
