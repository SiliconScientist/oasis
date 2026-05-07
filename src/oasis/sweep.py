from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class SweepSplit:
    """One sweep split for a sweep size.

    `test_idx` is always the outer evaluation holdout. When `val_idx` is present,
    it is carved out of the training budget and is intended only for model
    selection inside that outer train split.
    """

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
    """Split-planning requirements for a model family.

    `min_train_size` refers to the outer training budget for a sweep point.
    `requires_inner_validation=True` means that budget must be partitioned into
    inner-train and validation subsets, while `test_idx` remains reserved for
    outer evaluation only.
    """

    min_train_size: int = 0
    requires_inner_validation: bool = False

    @property
    def requires_validation(self) -> bool:
        return self.requires_inner_validation


@dataclass(frozen=True, slots=True)
class SweepModelCapabilities:
    """Model-facing declaration of split needs for learning-curve sweeps."""

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
    """Runner input for methods that train on the full outer train split."""

    dataset: SweepDataset
    sweep_size: int
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass(frozen=True, slots=True)
class TrainValTestSweepRunnerInput:
    """Runner input for methods that use validation inside the outer train split."""

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
    """Convert a stored split into the runner-facing train/test or train/val/test form."""

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
