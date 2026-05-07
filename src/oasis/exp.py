from __future__ import annotations

from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
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

    @classmethod
    def from_mapping(
        cls,
        frames: Mapping[str, pd.DataFrame | None],
    ) -> LearningCurveResults:
        return cls(
            ridge_df=frames.get("ridge_df"),
            kernel_ridge_df=frames.get("kernel_ridge_df"),
            ridge_trimmed_df=frames.get("ridge_trimmed_df"),
            lasso_df=frames.get("lasso_df"),
            lasso_trimmed_df=frames.get("lasso_trimmed_df"),
            elastic_df=frames.get("elastic_df"),
            elastic_trimmed_df=frames.get("elastic_trimmed_df"),
            resid_df=frames.get("resid_df"),
            resid_trimmed_df=frames.get("resid_trimmed_df"),
            linear_df=frames.get("linear_df"),
            linear_trimmed_df=frames.get("linear_trimmed_df"),
        )


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
    model_families: Sequence[Any] | None = None,
) -> LearningCurveResults:
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

    families = model_families
    if families is None:
        from oasis.method import default_sweep_model_families

        families = default_sweep_model_families(
            use_ridge=use_ridge,
            use_kernel_ridge=use_kernel_ridge,
            use_lasso=use_lasso,
            use_elastic=use_elastic,
            use_residual=use_residual,
            use_linearization=use_linearization,
        )

    frames: dict[str, pd.DataFrame | None] = {}
    for family in families:
        frames.update(
            family.run(
                X,
                y,
                shared_splits,
                use_trim=use_trim,
            )
        )
    return LearningCurveResults.from_mapping(frames)
