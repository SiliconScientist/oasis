from __future__ import annotations

from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
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
class SweepDataset:
    X: np.ndarray
    y: np.ndarray


@dataclass(frozen=True, slots=True)
class SweepFamilyRequirements:
    min_train_size: int = 0
    requires_inner_validation: bool = False


@dataclass(frozen=True, slots=True)
class SweepSplitCollection:
    splits: tuple[SweepSplit, ...]
    planning_requirements: SweepFamilyRequirements = field(
        default_factory=SweepFamilyRequirements
    )


@dataclass(frozen=True, slots=True)
class SweepRunPayload:
    dataset: SweepDataset
    split_collection: SweepSplitCollection
    use_trim: bool


@dataclass(frozen=True, slots=True)
class LearningCurveResults:
    ridge_df: pd.DataFrame | None = None
    kernel_ridge_df: pd.DataFrame | None = None
    ridge_trimmed_df: pd.DataFrame | None = None
    lasso_df: pd.DataFrame | None = None
    lasso_trimmed_df: pd.DataFrame | None = None
    elastic_df: pd.DataFrame | None = None
    elastic_trimmed_df: pd.DataFrame | None = None
    resid_df: pd.DataFrame | None = None
    resid_trimmed_df: pd.DataFrame | None = None
    linear_df: pd.DataFrame | None = None
    linear_trimmed_df: pd.DataFrame | None = None

    @classmethod
    def empty(cls) -> LearningCurveResults:
        return cls()

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

    def merge(self, other: LearningCurveResults) -> LearningCurveResults:
        return LearningCurveResults(
            ridge_df=other.ridge_df if other.ridge_df is not None else self.ridge_df,
            kernel_ridge_df=(
                other.kernel_ridge_df
                if other.kernel_ridge_df is not None
                else self.kernel_ridge_df
            ),
            ridge_trimmed_df=(
                other.ridge_trimmed_df
                if other.ridge_trimmed_df is not None
                else self.ridge_trimmed_df
            ),
            lasso_df=other.lasso_df if other.lasso_df is not None else self.lasso_df,
            lasso_trimmed_df=(
                other.lasso_trimmed_df
                if other.lasso_trimmed_df is not None
                else self.lasso_trimmed_df
            ),
            elastic_df=(
                other.elastic_df if other.elastic_df is not None else self.elastic_df
            ),
            elastic_trimmed_df=(
                other.elastic_trimmed_df
                if other.elastic_trimmed_df is not None
                else self.elastic_trimmed_df
            ),
            resid_df=other.resid_df if other.resid_df is not None else self.resid_df,
            resid_trimmed_df=(
                other.resid_trimmed_df
                if other.resid_trimmed_df is not None
                else self.resid_trimmed_df
            ),
            linear_df=(
                other.linear_df if other.linear_df is not None else self.linear_df
            ),
            linear_trimmed_df=(
                other.linear_trimmed_df
                if other.linear_trimmed_df is not None
                else self.linear_trimmed_df
            ),
        )


@dataclass(frozen=True, slots=True)
class ParityPlotData:
    reference: np.ndarray
    predictions: Mapping[str, np.ndarray]


def combine_sweep_family_requirements(
    families: Sequence[Any],
) -> SweepFamilyRequirements:
    requirements = [
        family.requirements()
        for family in families
        if hasattr(family, "requirements")
    ]
    if not requirements:
        return SweepFamilyRequirements()
    return SweepFamilyRequirements(
        min_train_size=max(req.min_train_size for req in requirements),
        requires_inner_validation=any(
            req.requires_inner_validation for req in requirements
        ),
    )


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
    enabled_model_names: Sequence[str] | None = None,
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
        SweepDataset(X=X, y=y),
        min_train=min_train,
        max_train=max_train,
        n_repeats=n_repeats,
        seed=seed,
        use_trim=use_trim,
        enabled_model_names=enabled_model_names,
    )


def run_learning_curve_experiments_from_config(
    df: Any,
    cfg: Config | None,
) -> LearningCurveResults:
    from oasis.method import enabled_learning_curve_model_names_from_config

    plot_cfg = cfg.plot if cfg else None
    return run_learning_curve_experiments_from_frame(
        df,
        min_train=plot_cfg.min_train if plot_cfg else 5,
        max_train=plot_cfg.max_train if plot_cfg else 10,
        n_repeats=plot_cfg.n_repeats if plot_cfg else 50,
        seed=cfg.seed if cfg and cfg.seed is not None else 42,
        use_trim=plot_cfg.trim if plot_cfg else True,
        enabled_model_names=enabled_learning_curve_model_names_from_config(plot_cfg),
    )


def run_learning_curve_experiments(
    dataset: SweepDataset,
    *,
    min_train: int,
    max_train: int,
    n_repeats: int,
    seed: int = 42,
    use_trim: bool = True,
    enabled_model_names: Sequence[str] | None = None,
    model_families: Sequence[Any] | None = None,
) -> LearningCurveResults:
    families = model_families
    if families is None:
        from oasis.method import default_sweep_model_families

        families = default_sweep_model_families(enabled_model_names)
    family_requirements = combine_sweep_family_requirements(families)

    max_train = min(max_train, len(dataset.X) - 1)
    split_collection = SweepSplitCollection(
        splits=tuple(
            generate_sweep_splits(
                len(dataset.X),
                max(min_train, family_requirements.min_train_size),
                max_train,
                n_repeats,
                np.random.default_rng(seed),
            )
        ),
        planning_requirements=family_requirements,
    )
    payload = SweepRunPayload(
        dataset=dataset,
        split_collection=split_collection,
        use_trim=use_trim,
    )

    results = LearningCurveResults.empty()
    for family in families:
        results = results.merge(family.run(payload))
    return results
