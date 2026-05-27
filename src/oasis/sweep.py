from __future__ import annotations

from collections.abc import Callable, Collection, Hashable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

import numpy as np
import pandas as pd


SampleId: TypeAlias = Hashable
SplitName: TypeAlias = Literal["train", "val", "test"]
LoaderCollateFn: TypeAlias = Callable[[Sequence[Any]], Any]


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

    @classmethod
    def from_inputs(
        cls,
        *,
        inputs: SweepDatasetInputs | SweepDatasetModalities,
        targets: np.ndarray,
        sample_ids: np.ndarray | None = None,
        auxiliary_views: Mapping[str, Any] | None = None,
    ) -> SweepDataset:
        return cls(
            mlip_features=inputs.mlip_features,
            targets=targets,
            sample_ids=sample_ids,
            graph_view=inputs.graph_view,
            auxiliary_views=auxiliary_views,
        )

    @property
    def n_samples(self) -> int:
        return len(self.mlip_features)

    def __post_init__(self) -> None:
        if self.targets.ndim != 1:
            raise ValueError("targets must be a 1D array aligned to mlip_features rows.")

        if len(self.targets) != self.n_samples:
            raise ValueError(
                "targets must have the same length as mlip_features."
            )

        sample_ids = self.sample_ids
        if sample_ids is None:
            object.__setattr__(self, "sample_ids", np.arange(self.n_samples))
        elif sample_ids.ndim != 1:
            raise ValueError("sample_ids must be a 1D array aligned to mlip_features rows.")
        elif len(sample_ids) != self.n_samples:
            raise ValueError(
                "sample_ids must have the same length as mlip_features."
            )

        if not _has_mlip_feature_modality(self.mlip_features) and self.graph_view is None:
            raise ValueError(
                "dataset must provide at least one modality: non-empty mlip_features or graph_view."
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
        duplicate_graph_ids = _duplicate_sample_ids(graph_sample_ids)
        if duplicate_graph_ids:
            raise ValueError(
                "graph_view sample_ids must be unique; "
                f"duplicates: {_format_sample_id_list(duplicate_graph_ids)}."
            )

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
    def modalities(self) -> SweepDatasetModalities:
        return SweepDatasetModalities(
            mlip_features=self.mlip_features,
            graph_view=self.graph_view,
        )

    @property
    def inputs(self) -> SweepDatasetInputs:
        return SweepDatasetInputs(
            mlip_features=self.mlip_features,
            graph_view=self.graph_view,
        )

    @property
    def graphs(self) -> GraphDatasetView:
        if self.graph_view is None:
            raise ValueError("graph_view is not available for this dataset.")
        return self.graph_view

    def mlip_view(self) -> np.ndarray:
        return self.inputs.mlip_features

    def graph_view_required(self) -> GraphDatasetView:
        return self.inputs.graph_view_required()

    @property
    def X(self) -> np.ndarray:
        return self.mlip_features

    @property
    def y(self) -> np.ndarray:
        return self.targets

    def __len__(self) -> int:
        return self.n_samples

    def _sample_bundle(self, row_index: int | np.ndarray) -> _SweepDatasetRowBundle:
        sample_ids = self.sample_ids[row_index]
        graph = None
        graph_view = None
        if self.graph_view is not None:
            if isinstance(row_index, np.ndarray):
                ordered_sample_ids = sample_ids.tolist()
                graph_view = GraphDatasetView.from_records(
                    tuple(self.graph_view[sample_id] for sample_id in ordered_sample_ids)
                )
            else:
                graph = self.graph_view[sample_ids]
        auxiliary = (
            None
            if self.auxiliary_views is None
            else {
                view_name: _index_view(view, row_index)
                for view_name, view in self.auxiliary_views.items()
            }
        )
        return _SweepDatasetRowBundle(
            mlip_features=self.mlip_features[row_index],
            targets=self.targets[row_index],
            sample_ids=sample_ids,
            graph=graph,
            graph_view=graph_view,
            auxiliary=auxiliary,
            auxiliary_views=auxiliary,
        )

    def sample(self, index: int) -> SweepSample:
        row_index = _normalize_sample_index(index, self.n_samples)
        bundle = self._sample_bundle(row_index)
        return SweepSample(
            index=row_index,
            mlip_features=bundle.mlip_features,
            target=bundle.targets,
            sample_id=bundle.sample_ids,
            graph=bundle.graph,
            auxiliary=bundle.auxiliary,
        )

    def subset(self, indices: slice | Sequence[int] | np.ndarray) -> SweepDataset:
        row_index = _normalize_row_indices(indices, self.n_samples)
        return self._sample_bundle(row_index).to_dataset()


@dataclass(frozen=True, slots=True)
class _SweepDatasetRowBundle:
    mlip_features: Any
    targets: Any
    sample_ids: Any
    graph: GraphRecord | None = None
    graph_view: GraphDatasetView | None = None
    auxiliary: Mapping[str, Any] | None = None
    auxiliary_views: Mapping[str, Any] | None = None

    def to_dataset(self) -> SweepDataset:
        return SweepDataset(
            mlip_features=self.mlip_features,
            targets=self.targets,
            sample_ids=self.sample_ids,
            graph_view=self.graph_view,
            auxiliary_views=self.auxiliary_views,
        )


@dataclass(frozen=True, slots=True)
class SweepDatasetModalities:
    mlip_features: np.ndarray
    graph_view: GraphDatasetView | None = None

    @property
    def has_graphs(self) -> bool:
        return self.graph_view is not None

    @property
    def graphs(self) -> GraphDatasetView:
        if self.graph_view is None:
            raise ValueError("graph_view is not available for this modality bundle.")
        return self.graph_view


@dataclass(frozen=True, slots=True)
class SweepDatasetInputs:
    mlip_features: np.ndarray
    graph_view: GraphDatasetView | None = None

    @property
    def has_graphs(self) -> bool:
        return self.graph_view is not None

    def graph_view_required(self) -> GraphDatasetView:
        if self.graph_view is None:
            raise ValueError("graph_view is required for these dataset inputs.")
        return self.graph_view


@dataclass(frozen=True, slots=True)
class SweepSampleModalities:
    mlip_features: np.ndarray
    graph: GraphRecord | None = None

    @property
    def has_graph(self) -> bool:
        return self.graph is not None


@dataclass(frozen=True, slots=True)
class SweepSampleInputs:
    mlip_features: np.ndarray
    graph: GraphRecord | None = None

    @property
    def has_graph(self) -> bool:
        return self.graph is not None

    def graph_required(self) -> GraphRecord:
        if self.graph is None:
            raise ValueError("graph is required for these sample inputs.")
        return self.graph


@dataclass(frozen=True, slots=True)
class SweepSample:
    index: int
    mlip_features: np.ndarray
    target: Any
    sample_id: SampleId
    graph: GraphRecord | None = None
    auxiliary: Mapping[str, Any] | None = None

    @property
    def modalities(self) -> SweepSampleModalities:
        return SweepSampleModalities(
            mlip_features=self.mlip_features,
            graph=self.graph,
        )

    @property
    def inputs(self) -> SweepSampleInputs:
        return SweepSampleInputs(
            mlip_features=self.mlip_features,
            graph=self.graph,
        )

    def mlip_view(self) -> np.ndarray:
        return self.inputs.mlip_features

    def graph_required(self) -> GraphRecord:
        return self.inputs.graph_required()


def _has_mlip_feature_modality(mlip_features: np.ndarray) -> bool:
    if mlip_features.ndim == 0:
        return False
    if mlip_features.ndim == 1:
        return True
    return bool(mlip_features.shape[1])


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
        return _build_train_test_split_datasets(
            self.dataset,
            train_idx=self.train_idx,
            test_idx=self.test_idx,
        )

    def loader_inputs(
        self,
        loader_adapter: DatasetLoaderAdapter | DatasetLoaderFactory,
    ) -> TrainTestSplitLoaderInputs:
        adapter = _as_loader_adapter(loader_adapter)
        subsets = self.dataset_subsets()
        return TrainTestSplitLoaderInputs(
            train=LoaderAdapterInput(
                dataset=subsets.train,
                split_name="train",
                batching=adapter.batching_for_split(split_name="train"),
            ),
            test=LoaderAdapterInput(
                dataset=subsets.test,
                split_name="test",
                batching=adapter.batching_for_split(split_name="test"),
            ),
        )

    def loaders(
        self,
        loader_adapter: DatasetLoaderAdapter | DatasetLoaderFactory,
    ) -> TrainTestSplitLoaders:
        adapter = _as_loader_adapter(loader_adapter)
        inputs = self.loader_inputs(adapter)
        return TrainTestSplitLoaders(
            train=adapter.build_loader(inputs.train),
            test=adapter.build_loader(inputs.test),
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
    - `loader_inputs(...)` separates split membership from batching/collation.
    - `loaders(...)` is the framework seam: build PyTorch Geometric, DGL,
      or other framework loaders from those subset datasets outside the core
      sweep planner.
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
        return _build_train_val_test_split_datasets(
            self.dataset,
            train_idx=self.train_idx,
            val_idx=self.val_idx,
            test_idx=self.test_idx,
        )

    def loader_inputs(
        self,
        loader_adapter: DatasetLoaderAdapter | DatasetLoaderFactory,
    ) -> TrainValTestSplitLoaderInputs:
        adapter = _as_loader_adapter(loader_adapter)
        subsets = self.dataset_subsets()
        return TrainValTestSplitLoaderInputs(
            train=LoaderAdapterInput(
                dataset=subsets.train,
                split_name="train",
                batching=adapter.batching_for_split(split_name="train"),
            ),
            val=LoaderAdapterInput(
                dataset=subsets.val,
                split_name="val",
                batching=adapter.batching_for_split(split_name="val"),
            ),
            test=LoaderAdapterInput(
                dataset=subsets.test,
                split_name="test",
                batching=adapter.batching_for_split(split_name="test"),
            ),
        )

    def loaders(
        self,
        loader_adapter: DatasetLoaderAdapter | DatasetLoaderFactory,
    ) -> TrainValTestSplitLoaders:
        adapter = _as_loader_adapter(loader_adapter)
        inputs = self.loader_inputs(adapter)
        return TrainValTestSplitLoaders(
            train=adapter.build_loader(inputs.train),
            val=adapter.build_loader(inputs.val),
            test=adapter.build_loader(inputs.test),
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

    @property
    def inputs(self) -> TrainTestSplitDatasetInputs:
        return TrainTestSplitDatasetInputs(
            train=self.train.inputs,
            test=self.test.inputs,
        )


@dataclass(frozen=True, slots=True)
class TrainValTestSplitDatasets:
    train: SweepDataset
    val: SweepDataset
    test: SweepDataset

    @property
    def inputs(self) -> TrainValTestSplitDatasetInputs:
        return TrainValTestSplitDatasetInputs(
            train=self.train.inputs,
            val=self.val.inputs,
            test=self.test.inputs,
        )


@dataclass(frozen=True, slots=True)
class TrainTestSplitDatasetInputs:
    train: SweepDatasetInputs
    test: SweepDatasetInputs


@dataclass(frozen=True, slots=True)
class TrainValTestSplitDatasetInputs:
    train: SweepDatasetInputs
    val: SweepDatasetInputs
    test: SweepDatasetInputs


SplitDatasets: TypeAlias = TrainTestSplitDatasets | TrainValTestSplitDatasets


@dataclass(frozen=True, slots=True)
class LoaderBatching:
    batch_size: int | None = None
    shuffle: bool = False
    collate_fn: LoaderCollateFn | None = None


@dataclass(frozen=True, slots=True)
class TrainEvalLoaderPolicy:
    batch_size: int | None = None
    eval_batch_size: int | None = None
    train_shuffle: bool = True
    eval_shuffle: bool = False
    train_collate_fn: LoaderCollateFn | None = None
    eval_collate_fn: LoaderCollateFn | None = None

    def batching_for_split(self, *, split_name: SplitName) -> LoaderBatching:
        if split_name == "train":
            return LoaderBatching(
                batch_size=self.batch_size,
                shuffle=self.train_shuffle,
                collate_fn=self.train_collate_fn,
            )
        return LoaderBatching(
            batch_size=(
                self.eval_batch_size
                if self.eval_batch_size is not None
                else self.batch_size
            ),
            shuffle=self.eval_shuffle,
            collate_fn=self.eval_collate_fn,
        )


@dataclass(frozen=True, slots=True)
class LoaderAdapterInput:
    dataset: SweepDataset
    split_name: SplitName
    batching: LoaderBatching = field(default_factory=LoaderBatching)


@dataclass(frozen=True, slots=True)
class SweepBatch:
    split_name: SplitName
    batch_index: int
    sample_ids: tuple[SampleId, ...]
    targets: np.ndarray
    mlip_features: np.ndarray
    graphs: tuple[GraphRecord | None, ...]
    auxiliary: tuple[Mapping[str, Any] | None, ...]

    @property
    def has_graphs(self) -> bool:
        return any(graph is not None for graph in self.graphs)


@runtime_checkable
class DatasetLoaderFactory(Protocol):
    def __call__(self, dataset: SweepDataset, *, split_name: str) -> Any: ...


@runtime_checkable
class DatasetLoaderAdapter(Protocol):
    def batching_for_split(self, *, split_name: SplitName) -> LoaderBatching: ...

    def build_loader(self, loader_input: LoaderAdapterInput) -> Any: ...


@dataclass(frozen=True, slots=True)
class DatasetLoaderFactoryAdapter:
    loader_factory: DatasetLoaderFactory

    def batching_for_split(self, *, split_name: SplitName) -> LoaderBatching:
        del split_name
        return LoaderBatching()

    def build_loader(self, loader_input: LoaderAdapterInput) -> Any:
        return self.loader_factory(
            loader_input.dataset,
            split_name=loader_input.split_name,
        )


@dataclass(frozen=True, slots=True)
class SweepDatasetBatchLoaderAdapter:
    policy: TrainEvalLoaderPolicy = field(default_factory=TrainEvalLoaderPolicy)

    def batching_for_split(self, *, split_name: SplitName) -> LoaderBatching:
        return self.policy.batching_for_split(split_name=split_name)

    def build_loader(self, loader_input: LoaderAdapterInput) -> tuple[Any, ...]:
        return build_sweep_batches(loader_input)


@dataclass(frozen=True, slots=True)
class TrainTestSplitLoaderInputs:
    train: LoaderAdapterInput
    test: LoaderAdapterInput


@dataclass(frozen=True, slots=True)
class TrainValTestSplitLoaderInputs:
    train: LoaderAdapterInput
    val: LoaderAdapterInput
    test: LoaderAdapterInput


SplitLoaderInputs: TypeAlias = (
    TrainTestSplitLoaderInputs | TrainValTestSplitLoaderInputs
)


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


def _as_loader_adapter(
    loader_adapter: DatasetLoaderAdapter | DatasetLoaderFactory,
) -> DatasetLoaderAdapter:
    if isinstance(loader_adapter, DatasetLoaderAdapter):
        return loader_adapter
    return DatasetLoaderFactoryAdapter(loader_adapter)


def _normalized_batch_size(loader_input: LoaderAdapterInput) -> int:
    batch_size = loader_input.batching.batch_size
    if batch_size is None:
        return len(loader_input.dataset)
    normalized = int(batch_size)
    if normalized <= 0:
        raise ValueError("batch_size must be positive when batching is enabled.")
    return normalized


def _deterministic_row_indices(loader_input: LoaderAdapterInput) -> np.ndarray:
    row_indices = np.arange(len(loader_input.dataset), dtype=np.int64)
    if not loader_input.batching.shuffle or len(row_indices) <= 1:
        return row_indices
    seed_by_split = {"train": 0, "val": 1, "test": 2}
    rng = np.random.default_rng(seed_by_split[loader_input.split_name])
    return row_indices[rng.permutation(len(row_indices))]


def dataset_batch_slices(loader_input: LoaderAdapterInput) -> tuple[slice, ...]:
    batch_size = _normalized_batch_size(loader_input)
    n_samples = len(loader_input.dataset)
    return tuple(
        slice(start, min(start + batch_size, n_samples))
        for start in range(0, n_samples, batch_size)
    )


def collate_sweep_samples(
    samples: Sequence[SweepSample],
    *,
    split_name: SplitName,
    batch_index: int,
) -> SweepBatch:
    sample_seq = tuple(samples)
    if not sample_seq:
        raise ValueError("cannot collate an empty batch.")
    return SweepBatch(
        split_name=split_name,
        batch_index=batch_index,
        sample_ids=tuple(sample.sample_id for sample in sample_seq),
        targets=np.asarray([sample.target for sample in sample_seq]),
        mlip_features=np.stack([sample.mlip_features for sample in sample_seq]),
        graphs=tuple(sample.graph for sample in sample_seq),
        auxiliary=tuple(sample.auxiliary for sample in sample_seq),
    )


def build_sweep_batches(loader_input: LoaderAdapterInput) -> tuple[Any, ...]:
    collate_fn = loader_input.batching.collate_fn
    batch_slices = dataset_batch_slices(loader_input)
    row_indices = _deterministic_row_indices(loader_input)
    batches: list[Any] = []
    for batch_index, batch_slice in enumerate(batch_slices):
        batch_row_indices = row_indices[batch_slice]
        samples = tuple(
            loader_input.dataset.sample(int(row_index)) for row_index in batch_row_indices
        )
        if collate_fn is None:
            batch = collate_sweep_samples(
                samples,
                split_name=loader_input.split_name,
                batch_index=batch_index,
            )
        else:
            batch = collate_fn(samples)
        batches.append(batch)
    return tuple(batches)


def _build_train_test_split_datasets(
    dataset: SweepDataset,
    *,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> TrainTestSplitDatasets:
    return TrainTestSplitDatasets(
        train=dataset.subset(train_idx),
        test=dataset.subset(test_idx),
    )


def _build_train_val_test_split_datasets(
    dataset: SweepDataset,
    *,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> TrainValTestSplitDatasets:
    return TrainValTestSplitDatasets(
        train=dataset.subset(train_idx),
        val=dataset.subset(val_idx),
        test=dataset.subset(test_idx),
    )


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


def split_to_loader_inputs(
    split: SweepRunnerInput,
    loader_adapter: DatasetLoaderAdapter | DatasetLoaderFactory,
) -> SplitLoaderInputs:
    return split.loader_inputs(loader_adapter)


def split_to_loaders(
    split: SweepRunnerInput,
    loader_adapter: DatasetLoaderAdapter | DatasetLoaderFactory,
) -> SplitLoaders:
    return split.loaders(loader_adapter)


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
    moe_df: pd.DataFrame | None = None
    moe_selection_df: pd.DataFrame | None = None
    probe_gnn_df: pd.DataFrame | None = None
    probe_gnn_selection_df: pd.DataFrame | None = None
    gnn_direct_df: pd.DataFrame | None = None
    gnn_direct_selection_df: pd.DataFrame | None = None
    latent_df: pd.DataFrame | None = None

    @classmethod
    def empty(cls) -> LearningCurveResults:
        return cls()

    def to_mapping(self) -> dict[str, pd.DataFrame | None]:
        return {
            field_def.name: getattr(self, field_def.name)
            for field_def in fields(self)
        }

    @classmethod
    def from_mapping(
        cls,
        frames: Mapping[str, pd.DataFrame | None],
    ) -> LearningCurveResults:
        return cls(**{field_def.name: frames.get(field_def.name) for field_def in fields(cls)})

    def merge(
        self,
        other: LearningCurveResults,
        *,
        overwrite_fields: Collection[str] = (),
        overwrite_train_sizes_by_field: Mapping[str, Collection[int]] | None = None,
    ) -> LearningCurveResults:
        overwrite_field_names = set(overwrite_fields)
        overwrite_sizes_mapping = {
            field_name: {int(value) for value in sweep_sizes}
            for field_name, sweep_sizes in (overwrite_train_sizes_by_field or {}).items()
        }
        return LearningCurveResults.from_mapping(
            {
                field_def.name: _merge_learning_curve_frame(
                    field_def.name,
                    getattr(self, field_def.name),
                    getattr(other, field_def.name),
                    allow_overlap=field_def.name in overwrite_field_names,
                    allowed_overlap_train_sizes=overwrite_sizes_mapping.get(
                        field_def.name,
                        set(),
                    ),
                )
                for field_def in fields(self)
            }
        )


def _merge_learning_curve_frame(
    field_name: str,
    left: pd.DataFrame | None,
    right: pd.DataFrame | None,
    *,
    allow_overlap: bool = False,
    allowed_overlap_train_sizes: Collection[int] = (),
) -> pd.DataFrame | None:
    if left is None:
        return right
    if right is None:
        return left
    if "n_train" not in left.columns or "n_train" not in right.columns:
        raise ValueError("learning-curve result frames must contain an n_train column.")
    overlapping_train_sizes = sorted(
        set(left["n_train"].tolist()).intersection(right["n_train"].tolist())
    )
    allowed_overlap_sizes = {int(value) for value in allowed_overlap_train_sizes}
    disallowed_overlap_sizes = [
        sweep_size
        for sweep_size in overlapping_train_sizes
        if sweep_size not in allowed_overlap_sizes
    ]
    if disallowed_overlap_sizes and not allow_overlap:
        raise ValueError(
            f"{field_name} contains duplicate n_train rows: {disallowed_overlap_sizes!r}."
        )
    merged = pd.concat([left, right], ignore_index=True)
    merged = merged.drop_duplicates(subset=["n_train"], keep="last")
    return merged.sort_values("n_train").reset_index(drop=True)


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
