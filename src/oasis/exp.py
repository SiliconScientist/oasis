from __future__ import annotations

from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from oasis.config import Config


_INNER_VALIDATION_SIZE = 1


@dataclass(frozen=True, slots=True)
class SweepSplit:
    """One sweep split for a sweep size."""

    sweep_size: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    val_idx: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class SweepDataset:
    X: np.ndarray
    y: np.ndarray


@dataclass(frozen=True, slots=True)
class SweepFamilyRequirements:
    min_train_size: int = 0
    requires_inner_validation: bool = False

    @property
    def requires_validation(self) -> bool:
        return self.requires_inner_validation


@dataclass(frozen=True, slots=True)
class SweepModelCapabilities:
    min_train_size: int = 0
    requires_validation: bool = False

    def to_requirements(self) -> SweepFamilyRequirements:
        return SweepFamilyRequirements(
            min_train_size=self.min_train_size,
            requires_inner_validation=self.requires_validation,
        )


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

    def to_runner_payload(self) -> SweepRunnerPayload:
        return SweepRunnerPayload(
            splits=tuple(
                split_to_runner_input(self.dataset, split)
                for split in self.split_collection.splits
            ),
            planning_requirements=self.split_collection.planning_requirements,
            use_trim=self.use_trim,
        )


@dataclass(frozen=True, slots=True)
class TrainTestSweepRunnerInput:
    dataset: SweepDataset
    sweep_size: int
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass(frozen=True, slots=True)
class TrainValTestSweepRunnerInput:
    dataset: SweepDataset
    sweep_size: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


SweepRunnerInput: TypeAlias = (
    TrainTestSweepRunnerInput | TrainValTestSweepRunnerInput
)


@dataclass(frozen=True, slots=True)
class SweepRunnerPayload:
    splits: tuple[SweepRunnerInput, ...]
    planning_requirements: SweepFamilyRequirements = field(
        default_factory=SweepFamilyRequirements
    )
    use_trim: bool = False


def split_to_runner_input(
    dataset: SweepDataset,
    split: SweepSplit,
) -> SweepRunnerInput:
    if split.val_idx is None:
        return TrainTestSweepRunnerInput(
            dataset=dataset,
            sweep_size=split.sweep_size,
            train_idx=split.train_idx,
            test_idx=split.test_idx,
        )
    return TrainValTestSweepRunnerInput(
        dataset=dataset,
        sweep_size=split.sweep_size,
        train_idx=split.train_idx,
        val_idx=split.val_idx,
        test_idx=split.test_idx,
    )


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
    weighted_linear_df: pd.DataFrame | None = None
    weighted_simplex_df: pd.DataFrame | None = None

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
            weighted_linear_df=frames.get("weighted_linear_df"),
            weighted_simplex_df=frames.get("weighted_simplex_df"),
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
            weighted_linear_df=(
                other.weighted_linear_df
                if other.weighted_linear_df is not None
                else self.weighted_linear_df
            ),
            weighted_simplex_df=(
                other.weighted_simplex_df
                if other.weighted_simplex_df is not None
                else self.weighted_simplex_df
            ),
        )


@dataclass(frozen=True, slots=True)
class ParityPlotData:
    reference: np.ndarray
    predictions: Mapping[str, np.ndarray]


def combine_sweep_family_requirements(
    families: Sequence[Any],
) -> SweepFamilyRequirements:
    capabilities = []
    for family in families:
        if hasattr(family, "capabilities"):
            capabilities.append(family.capabilities())
            continue
        if hasattr(family, "requirements"):
            capabilities.append(family.requirements())
    if not capabilities:
        return SweepFamilyRequirements()

    def requires_validation(capability: Any) -> bool:
        if hasattr(capability, "requires_validation"):
            return capability.requires_validation
        return capability.requires_inner_validation

    return SweepFamilyRequirements(
        min_train_size=max(cap.min_train_size for cap in capabilities),
        requires_inner_validation=any(
            requires_validation(cap)
            for cap in capabilities
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


def generate_sweep_splits_with_validation(
    n_samples: int,
    min_train: int,
    max_train: int,
    n_val: int,
    n_repeats: int,
    rng: np.random.Generator,
) -> Iterator[SweepSplit]:
    """Yield repeated outer train/test splits with inner train/val partitions."""

    if n_val <= 0:
        raise ValueError("n_val must be positive.")
    if n_val >= n_samples:
        raise ValueError("n_val must be smaller than n_samples.")

    idx = np.arange(n_samples)
    max_train = min(max_train, n_samples - 1)
    for n_train in range(max(min_train, n_val + 1), max_train + 1):
        for _ in range(n_repeats):
            outer_train_idx = rng.choice(idx, size=n_train, replace=False)
            test_idx = np.setdiff1d(idx, outer_train_idx, assume_unique=False)
            val_idx = rng.choice(outer_train_idx, size=n_val, replace=False)
            train_idx = np.setdiff1d(outer_train_idx, val_idx, assume_unique=False)
            yield SweepSplit(
                sweep_size=n_train,
                train_idx=train_idx,
                test_idx=test_idx,
                val_idx=val_idx,
            )


def generate_inner_validation_sweep_splits(
    n_samples: int,
    min_train: int,
    max_train: int,
    n_repeats: int,
    rng: np.random.Generator,
    *,
    n_val: int = _INNER_VALIDATION_SIZE,
) -> Iterator[SweepSplit]:
    """Yield sweep splits with one inner validation holdout per outer train split."""

    yield from generate_sweep_splits_with_validation(
        n_samples=n_samples,
        min_train=min_train,
        max_train=max_train,
        n_val=n_val,
        n_repeats=n_repeats,
        rng=rng,
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
    split_generator = generate_sweep_splits
    if family_requirements.requires_inner_validation:
        split_generator = generate_inner_validation_sweep_splits
    split_collection = SweepSplitCollection(
        splits=tuple(
            split_generator(
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
