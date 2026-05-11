from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias, runtime_checkable

import numpy as np
import pandas as pd


SampleId: TypeAlias = Hashable


def _format_sample_id_list(sample_ids: Sequence[SampleId], *, limit: int = 5) -> str:
    preview = ", ".join(repr(sample_id) for sample_id in sample_ids[:limit])
    if len(sample_ids) > limit:
        preview = f"{preview}, ..."
    return preview


def _duplicate_sample_ids(sample_ids: Sequence[SampleId]) -> tuple[SampleId, ...]:
    seen: set[SampleId] = set()
    duplicates: list[SampleId] = []
    duplicate_set: set[SampleId] = set()
    for sample_id in sample_ids:
        if sample_id in seen and sample_id not in duplicate_set:
            duplicates.append(sample_id)
            duplicate_set.add(sample_id)
        seen.add(sample_id)
    return tuple(duplicates)


@dataclass(frozen=True, slots=True)
class GraphRecord:
    sample_id: SampleId
    node_features: np.ndarray
    edge_index: np.ndarray
    node_positions: np.ndarray | None = None
    edge_features: np.ndarray | None = None
    graph_features: np.ndarray | None = None

    @property
    def n_nodes(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def __post_init__(self) -> None:
        if self.sample_id is None:
            raise ValueError("sample_id is required.")

        if self.node_features.ndim != 2:
            raise ValueError("node_features must be a 2D array.")

        node_positions = self.node_positions
        if node_positions is not None:
            if node_positions.ndim != 2 or node_positions.shape[1] != 3:
                raise ValueError("node_positions must have shape (n_nodes, 3).")
            if len(node_positions) != self.n_nodes:
                raise ValueError(
                    "node_positions must have the same number of rows as node_features."
                )

        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, n_edges).")
        if not np.issubdtype(self.edge_index.dtype, np.integer):
            raise ValueError("edge_index must contain integer node indices.")
        if np.any(self.edge_index < 0):
            raise ValueError("edge_index cannot contain negative node indices.")
        if self.edge_index.size and np.any(self.edge_index >= self.n_nodes):
            raise ValueError("edge_index cannot reference nodes outside node_features.")

        edge_features = self.edge_features
        if edge_features is not None:
            if edge_features.ndim not in (1, 2):
                raise ValueError("edge_features must be a 1D or 2D array.")
            if len(edge_features) != self.n_edges:
                raise ValueError(
                    "edge_features must have the same number of rows as edge_index columns."
                )

        graph_features = self.graph_features
        if graph_features is not None and graph_features.ndim != 1:
            raise ValueError("graph_features must be a 1D array.")


@dataclass(frozen=True, slots=True)
class GraphDatasetView:
    records_by_sample_id: Mapping[SampleId, GraphRecord]

    def __post_init__(self) -> None:
        normalized_records = dict(self.records_by_sample_id)
        for sample_id, record in normalized_records.items():
            if sample_id != record.sample_id:
                raise ValueError(
                    "records_by_sample_id keys must match each record's sample_id: "
                    f"key={sample_id!r}, record.sample_id={record.sample_id!r}."
                )
        object.__setattr__(self, "records_by_sample_id", normalized_records)

    @classmethod
    def from_records(
        cls,
        records: Sequence[GraphRecord],
    ) -> GraphDatasetView:
        records_by_sample_id: dict[SampleId, GraphRecord] = {}
        for record in records:
            if record.sample_id in records_by_sample_id:
                raise ValueError(
                    f"duplicate graph record for sample_id={record.sample_id!r}."
                )
            records_by_sample_id[record.sample_id] = record
        return cls(records_by_sample_id=records_by_sample_id)

    def __len__(self) -> int:
        return len(self.records_by_sample_id)

    def __getitem__(self, sample_id: SampleId) -> GraphRecord:
        return self.records_by_sample_id[sample_id]

    def get(self, sample_id: SampleId) -> GraphRecord | None:
        return self.records_by_sample_id.get(sample_id)

    @property
    def sample_ids(self) -> tuple[SampleId, ...]:
        return tuple(self.records_by_sample_id)


@dataclass(frozen=True, slots=True)
class SweepSplit:
    """One sweep split for a sweep size.

    `test_idx` is always the outer evaluation holdout. When `val_idx` is present,
    it is carved out of the training budget and is intended only for model
    selection inside that outer train split. In other words, selection-aware
    methods do not receive any extra samples beyond `sweep_size`: they must
    partition that budget into inner-train and validation while leaving
    `test_idx` untouched for final outer evaluation only.
    """

    sweep_size: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    val_idx: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class SweepDataset:
    mlip_features: np.ndarray
    targets: np.ndarray
    sample_ids: np.ndarray | None = None
    graph_view: GraphDatasetView | None = None
    auxiliary_views: Mapping[str, Any] | None = None

    @property
    def n_samples(self) -> int:
        return len(self.mlip_features)

    def __post_init__(self) -> None:
        if len(self.targets) != self.n_samples:
            raise ValueError(
                "targets must have the same length as mlip_features."
            )

        sample_ids = self.sample_ids
        if sample_ids is None:
            object.__setattr__(self, "sample_ids", np.arange(self.n_samples))
        elif len(sample_ids) != self.n_samples:
            raise ValueError(
                "sample_ids must have the same length as mlip_features."
            )

        if self.auxiliary_views is None:
            auxiliary_views = None
        else:
            auxiliary_views = dict(self.auxiliary_views)
            object.__setattr__(self, "auxiliary_views", auxiliary_views)

        for view_name, view in (auxiliary_views or {}).items():
            if len(view) != self.n_samples:
                raise ValueError(
                    f"auxiliary view '{view_name}' must have the same length as mlip_features."
                )

        graph_view = self.graph_view
        if graph_view is None:
            return

        dataset_sample_ids = tuple(self.sample_ids.tolist())
        duplicate_dataset_ids = _duplicate_sample_ids(dataset_sample_ids)
        if duplicate_dataset_ids:
            raise ValueError(
                "dataset sample_ids must be unique when graph_view is provided; "
                f"duplicates: {_format_sample_id_list(duplicate_dataset_ids)}."
            )

        graph_sample_ids = graph_view.sample_ids
        missing_graph_ids = tuple(
            sample_id
            for sample_id in dataset_sample_ids
            if sample_id not in graph_view.records_by_sample_id
        )
        extra_graph_ids = tuple(
            sample_id
            for sample_id in graph_sample_ids
            if sample_id not in set(dataset_sample_ids)
        )
        if missing_graph_ids or extra_graph_ids:
            details: list[str] = []
            if missing_graph_ids:
                details.append(
                    "missing graph sample_ids: "
                    f"{_format_sample_id_list(missing_graph_ids)}"
                )
            if extra_graph_ids:
                details.append(
                    "extra graph sample_ids: "
                    f"{_format_sample_id_list(extra_graph_ids)}"
                )
            raise ValueError(
                "graph_view sample_ids must match dataset sample_ids; "
                + "; ".join(details)
                + "."
            )

    @property
    def has_graphs(self) -> bool:
        return self.graph_view is not None

    @property
    def graphs(self) -> GraphDatasetView:
        if self.graph_view is None:
            raise ValueError("graph_view is not available for this dataset.")
        return self.graph_view

    @property
    def X(self) -> np.ndarray:
        return self.mlip_features

    @property
    def y(self) -> np.ndarray:
        return self.targets

    def __len__(self) -> int:
        return self.n_samples

    def sample(self, index: int) -> SweepSample:
        row_index = _normalize_sample_index(index, self.n_samples)
        sample_id = self.sample_ids[row_index]
        graph = None if self.graph_view is None else self.graph_view[sample_id]
        auxiliary = (
            None
            if self.auxiliary_views is None
            else {
                view_name: _index_view(view, row_index)
                for view_name, view in self.auxiliary_views.items()
            }
        )
        return SweepSample(
            index=row_index,
            mlip_features=self.mlip_features[row_index],
            target=self.targets[row_index],
            sample_id=sample_id,
            graph=graph,
            auxiliary=auxiliary,
        )

    def subset(self, indices: slice | Sequence[int] | np.ndarray) -> SweepDataset:
        row_index = _normalize_row_indices(indices, self.n_samples)
        sample_ids = self.sample_ids[row_index]
        graph_view = None
        if self.graph_view is not None:
            graph_view = GraphDatasetView.from_records(
                tuple(self.graph_view[sample_id] for sample_id in sample_ids.tolist())
            )
        auxiliary_views = None
        if self.auxiliary_views is not None:
            auxiliary_views = {
                view_name: _index_view(view, row_index)
                for view_name, view in self.auxiliary_views.items()
            }
        return SweepDataset(
            mlip_features=self.mlip_features[row_index],
            targets=self.targets[row_index],
            sample_ids=sample_ids,
            graph_view=graph_view,
            auxiliary_views=auxiliary_views,
        )


@dataclass(frozen=True, slots=True)
class SweepSample:
    index: int
    mlip_features: np.ndarray
    target: Any
    sample_id: SampleId
    graph: GraphRecord | None = None
    auxiliary: Mapping[str, Any] | None = None


def _normalize_sample_index(index: int, n_samples: int) -> int:
    row_index = int(index)
    if row_index < 0:
        row_index += n_samples
    if row_index < 0 or row_index >= n_samples:
        raise IndexError(
            f"sample index {index} is out of bounds for dataset with {n_samples} rows."
        )
    return row_index


def _normalize_row_indices(
    indices: slice | Sequence[int] | np.ndarray,
    n_samples: int,
) -> np.ndarray:
    if isinstance(indices, slice):
        return np.arange(n_samples)[indices]

    row_index = np.asarray(indices)
    if row_index.ndim != 1:
        raise ValueError("row indices must be a 1D slice, mask, or integer index array.")

    if np.issubdtype(row_index.dtype, np.bool_):
        if len(row_index) != n_samples:
            raise ValueError(
                "boolean row mask must have the same length as the dataset."
            )
        return np.flatnonzero(row_index)

    normalized = np.asarray(row_index, dtype=np.int64).copy()
    normalized[normalized < 0] += n_samples
    if np.any(normalized < 0) or np.any(normalized >= n_samples):
        raise IndexError(
            f"row indices are out of bounds for dataset with {n_samples} rows."
        )
    return normalized


def _index_view(view: Any, row_index: int | np.ndarray) -> Any:
    try:
        return view[row_index]
    except (TypeError, KeyError):
        if isinstance(row_index, np.ndarray):
            return [view[int(i)] for i in row_index.tolist()]
        return view[int(row_index)]


@dataclass(frozen=True, slots=True)
class SweepFamilyRequirements:
    """Split-planning requirements for a model family.

    `min_train_size` refers to the outer training budget for a sweep point. The
    planner takes the maximum of the caller's requested `min_train`, this
    family-level minimum, and any additional feasibility guards implied by the
    split policy.

    `requires_inner_validation=True` means that outer-train budget must be
    partitioned into inner-train and validation subsets, while `test_idx`
    remains reserved for outer evaluation only. Selection-based families
    therefore skip sweep sizes that cannot support the configured minimum
    validation budget and at least one inner-train sample.
    """

    min_train_size: int = 0
    requires_inner_validation: bool = False

    @property
    def requires_validation(self) -> bool:
        return self.requires_inner_validation


@dataclass(frozen=True, slots=True)
class SweepModelCapabilities:
    """Model-facing declaration of split needs for learning-curve sweeps.

    `requires_validation=True` means the family expects train/val/test splits
    rather than plain train/test splits. The outer test split stays evaluation
    only; validation is always carved out of the requested outer training
    budget.
    """

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

    def to_runner_payload(self) -> SweepRunnerPayload:
        return SweepRunnerPayload(
            splits=tuple(
                split_to_runner_input(self.dataset, split)
                for split in self.split_collection.splits
            ),
            planning_requirements=self.split_collection.planning_requirements,
        )


@dataclass(frozen=True, slots=True)
class TrainTestSweepRunnerInput:
    """Runner input for methods that train on the full outer train split."""

    dataset: SweepDataset
    sweep_size: int
    train_idx: np.ndarray
    test_idx: np.ndarray

    def dataset_subsets(self) -> TrainTestSplitDatasets:
        return TrainTestSplitDatasets(
            train=self.dataset.subset(self.train_idx),
            test=self.dataset.subset(self.test_idx),
        )

    def loaders(
        self,
        loader_factory: DatasetLoaderFactory,
    ) -> TrainTestSplitLoaders:
        subsets = self.dataset_subsets()
        return TrainTestSplitLoaders(
            train=loader_factory(subsets.train, split_name="train"),
            test=loader_factory(subsets.test, split_name="test"),
        )


@dataclass(frozen=True, slots=True)
class TrainValTestSweepRunnerInput:
    """Runner input for methods that use validation inside the outer train split.

    `train_idx` and `val_idx` together make up the full outer training budget
    for that sweep point. `test_idx` remains a held-out outer evaluation split
    and must not be touched during candidate selection.

    Developer contract for learned families:
    - `dataset_subsets()` materializes aligned `SweepDataset` views for
      train/val/test, including sample IDs, graphs, and auxiliary views.
    - `loaders(...)` is the framework seam: build PyTorch Geometric,
      DGL, or other framework loaders from those subset datasets outside the
      core sweep planner.
    - model selection may use only train/val data.
    - outer-test data must remain held out until one final evaluation after
      selection and any optional refit.
    """

    dataset: SweepDataset
    sweep_size: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray

    def dataset_subsets(self) -> TrainValTestSplitDatasets:
        return TrainValTestSplitDatasets(
            train=self.dataset.subset(self.train_idx),
            val=self.dataset.subset(self.val_idx),
            test=self.dataset.subset(self.test_idx),
        )

    def loaders(
        self,
        loader_factory: DatasetLoaderFactory,
    ) -> TrainValTestSplitLoaders:
        subsets = self.dataset_subsets()
        return TrainValTestSplitLoaders(
            train=loader_factory(subsets.train, split_name="train"),
            val=loader_factory(subsets.val, split_name="val"),
            test=loader_factory(subsets.test, split_name="test"),
        )


SweepRunnerInput: TypeAlias = (
    TrainTestSweepRunnerInput | TrainValTestSweepRunnerInput
)


@dataclass(frozen=True, slots=True)
class SweepRunnerPayload:
    splits: tuple[SweepRunnerInput, ...]
    planning_requirements: SweepFamilyRequirements = field(
        default_factory=SweepFamilyRequirements
    )


@dataclass(frozen=True, slots=True)
class TrainTestSplitDatasets:
    train: SweepDataset
    test: SweepDataset


@dataclass(frozen=True, slots=True)
class TrainValTestSplitDatasets:
    train: SweepDataset
    val: SweepDataset
    test: SweepDataset


SplitDatasets: TypeAlias = TrainTestSplitDatasets | TrainValTestSplitDatasets


@runtime_checkable
class DatasetLoaderFactory(Protocol):
    def __call__(self, dataset: SweepDataset, *, split_name: str) -> Any: ...


@dataclass(frozen=True, slots=True)
class TrainTestSplitLoaders:
    train: Any
    test: Any


@dataclass(frozen=True, slots=True)
class TrainValTestSplitLoaders:
    train: Any
    val: Any
    test: Any


SplitLoaders: TypeAlias = TrainTestSplitLoaders | TrainValTestSplitLoaders


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


def split_to_dataset_subsets(split: SweepRunnerInput) -> SplitDatasets:
    return split.dataset_subsets()


def split_to_loaders(
    split: SweepRunnerInput,
    loader_factory: DatasetLoaderFactory,
) -> SplitLoaders:
    return split.loaders(loader_factory)


@dataclass(frozen=True, slots=True)
class LearningCurveResults:
    """Learning-curve outputs.

    The `*_df` fields keep the historical RMSE result shape unchanged:
    `n_train`, `rmse_mean`, and `rmse_std`.

    Selection-aware families may also populate `*_selection_df` metadata frames.
    Those are optional companion outputs keyed by `n_train` that surface the
    chosen hyperparameters for each sweep size without changing the RMSE frames.
    """

    ridge_df: pd.DataFrame | None = None
    kernel_ridge_df: pd.DataFrame | None = None
    ridge_selection_df: pd.DataFrame | None = None
    lasso_df: pd.DataFrame | None = None
    lasso_selection_df: pd.DataFrame | None = None
    elastic_df: pd.DataFrame | None = None
    elastic_selection_df: pd.DataFrame | None = None
    resid_df: pd.DataFrame | None = None
    weighted_linear_df: pd.DataFrame | None = None
    weighted_simplex_df: pd.DataFrame | None = None
    graph_mean_df: pd.DataFrame | None = None
    graph_mean_selection_df: pd.DataFrame | None = None
    kernel_ridge_selection_df: pd.DataFrame | None = None

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
            ridge_selection_df=frames.get("ridge_selection_df"),
            lasso_df=frames.get("lasso_df"),
            lasso_selection_df=frames.get("lasso_selection_df"),
            elastic_df=frames.get("elastic_df"),
            elastic_selection_df=frames.get("elastic_selection_df"),
            resid_df=frames.get("resid_df"),
            weighted_linear_df=frames.get("weighted_linear_df"),
            weighted_simplex_df=frames.get("weighted_simplex_df"),
            graph_mean_df=frames.get("graph_mean_df"),
            graph_mean_selection_df=frames.get("graph_mean_selection_df"),
            kernel_ridge_selection_df=frames.get("kernel_ridge_selection_df"),
        )

    def merge(self, other: LearningCurveResults) -> LearningCurveResults:
        return LearningCurveResults(
            ridge_df=other.ridge_df if other.ridge_df is not None else self.ridge_df,
            kernel_ridge_df=(
                other.kernel_ridge_df
                if other.kernel_ridge_df is not None
                else self.kernel_ridge_df
            ),
            ridge_selection_df=(
                other.ridge_selection_df
                if other.ridge_selection_df is not None
                else self.ridge_selection_df
            ),
            lasso_df=other.lasso_df if other.lasso_df is not None else self.lasso_df,
            lasso_selection_df=(
                other.lasso_selection_df
                if other.lasso_selection_df is not None
                else self.lasso_selection_df
            ),
            elastic_df=(
                other.elastic_df if other.elastic_df is not None else self.elastic_df
            ),
            elastic_selection_df=(
                other.elastic_selection_df
                if other.elastic_selection_df is not None
                else self.elastic_selection_df
            ),
            resid_df=other.resid_df if other.resid_df is not None else self.resid_df,
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
            graph_mean_df=(
                other.graph_mean_df
                if other.graph_mean_df is not None
                else self.graph_mean_df
            ),
            graph_mean_selection_df=(
                other.graph_mean_selection_df
                if other.graph_mean_selection_df is not None
                else self.graph_mean_selection_df
            ),
            kernel_ridge_selection_df=(
                other.kernel_ridge_selection_df
                if other.kernel_ridge_selection_df is not None
                else self.kernel_ridge_selection_df
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
