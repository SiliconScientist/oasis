from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from oasis.mlip.artifacts import INFERENCE_DETAIL_COLUMNS
from oasis.sweep import GraphDatasetView, SweepDataset, SweepDatasetInputs

if TYPE_CHECKING:
    from oasis.config import Config


@dataclass(frozen=True, slots=True)
class ParityPlotData:
    reference: np.ndarray
    predictions: Mapping[str, np.ndarray]


def mlip_columns(df: Any) -> list[str]:
    return [c for c in df.columns if c.endswith("_mlip_ads_eng_median")]


def mlip_feature_names(df: Any) -> tuple[str, ...]:
    return tuple(
        column.removesuffix("_mlip_ads_eng_median")
        for column in mlip_columns(df)
    )


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


def _validate_learning_curve_frame(df: Any) -> None:
    feature_cols = mlip_columns(df)
    if not feature_cols:
        raise ValueError(
            "No MLIP prediction columns found (expected *_mlip_ads_eng_median)."
        )

    n_rows = getattr(df, "height", len(df))
    if n_rows <= 5:
        raise ValueError("Not enough data to evaluate (need >5 samples).")


def assemble_learning_curve_dataset_from_frame(
    df: Any,
    *,
    graph_view: GraphDatasetView | None = None,
    graph_join_key: str = "reaction",
    auxiliary_views: dict[str, Any] | None = None,
) -> SweepDataset:
    return build_sweep_dataset_from_frame(
        df,
        graph_view=graph_view,
        graph_join_key=graph_join_key,
        auxiliary_views=auxiliary_views,
    )


def build_sweep_dataset_from_frame(
    df: Any,
    *,
    graph_view: GraphDatasetView | None = None,
    graph_join_key: str = "reaction",
    auxiliary_views: dict[str, Any] | None = None,
) -> SweepDataset:
    if graph_view is not None:
        from oasis.graphs import build_graph_sweep_dataset

        return build_graph_sweep_dataset(
            df, graph_view, join_key=graph_join_key, auxiliary_views=auxiliary_views
        )

    feature_cols = mlip_columns(df)
    if hasattr(df, "select"):
        X = df.select(feature_cols).to_numpy()
    else:
        X = np.column_stack([column_to_numpy(df, col) for col in feature_cols])
    y = column_to_numpy(df, "reference_ads_eng")
    sample_ids = (
        column_to_numpy(df, "reaction")
        if "reaction" in getattr(df, "columns", ())
        else None
    )
    return SweepDataset.from_inputs(
        inputs=SweepDatasetInputs(
            mlip_features=X,
        ),
        targets=y,
        sample_ids=sample_ids,
        auxiliary_views=auxiliary_views,
    )


def _strict_inference_masking_enabled(cfg: Config | None) -> bool:
    experiment_cfg = cfg.experiment.learning_curve if cfg and cfg.experiment else None
    mlip_selection_cfg = (
        getattr(experiment_cfg, "mlip_selection", None)
        if experiment_cfg is not None
        else None
    )
    return bool(
        getattr(mlip_selection_cfg, "exclude_anomalous", False)
        and getattr(mlip_selection_cfg, "strict_inference_anomaly", False)
    )


def _strict_validity_mask_from_frame(df: Any) -> np.ndarray:
    feature_names = mlip_feature_names(df)
    if not feature_names:
        raise ValueError("No MLIP feature columns found for strict inference masking.")
    validity_columns: list[np.ndarray] = []
    available_columns = set(getattr(df, "columns", ()))
    for feature_name in feature_names:
        detail_column_names = [
            f"{feature_name}_{detail_name}"
            for detail_name in INFERENCE_DETAIL_COLUMNS
            if f"{feature_name}_{detail_name}" in available_columns
        ]
        if not detail_column_names:
            raise ValueError(
                "Strict inference masking requires detail columns for "
                f"{feature_name!r}."
            )
        detail_matrix = np.column_stack(
            [column_to_numpy(df, column_name) for column_name in detail_column_names]
        )
        validity_columns.append(np.all(detail_matrix == 0, axis=1))
    return np.column_stack(validity_columns)


def _impute_invalid_mlip_entries_with_row_mean(
    X: np.ndarray,
    validity_mask: np.ndarray,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.shape != validity_mask.shape:
        raise ValueError("Feature matrix and validity mask must have matching shapes.")
    valid_counts = validity_mask.sum(axis=1)
    if np.any(valid_counts <= 0):
        raise ValueError("Strict inference masking requires at least one valid MLIP per row.")
    valid_sum = np.where(validity_mask, X, 0.0).sum(axis=1)
    row_means = valid_sum / valid_counts
    return np.where(validity_mask, X, row_means[:, None])


def _apply_strict_inference_masking(
    dataset: SweepDataset,
    df: Any,
) -> SweepDataset:
    validity_mask = _strict_validity_mask_from_frame(df)
    masked_features = _impute_invalid_mlip_entries_with_row_mean(
        dataset.mlip_features,
        validity_mask,
    )
    auxiliary_views = dict(dataset.auxiliary_views or {})
    auxiliary_views["mlip_validity_mask"] = validity_mask
    return SweepDataset(
        mlip_features=masked_features,
        targets=dataset.targets,
        sample_ids=dataset.sample_ids,
        graph_view=dataset.graph_view,
        auxiliary_views=auxiliary_views,
    )


def build_sweep_dataset_from_config(
    df: Any,
    cfg: Config | None,
    *,
    graph_view: GraphDatasetView | None = None,
    auxiliary_views: dict[str, Any] | None = None,
) -> SweepDataset:
    experiment_cfg = cfg.experiment.learning_curve if cfg and cfg.experiment else None
    graph_dataset_cfg = getattr(experiment_cfg, "graph_dataset", None)
    graph_join_key = (
        graph_dataset_cfg.join_key if graph_dataset_cfg is not None else "reaction"
    )
    if (
        graph_view is None
        and graph_dataset_cfg is not None
        and Path(graph_dataset_cfg.path).is_file()
    ):
        from oasis.graphs import load_sweep_dataset_from_graph_artifact

        dataset = load_sweep_dataset_from_graph_artifact(
            graph_dataset_cfg.path,
            join_key=graph_join_key,
            auxiliary_views=auxiliary_views,
            filter_df=df,
        )
        if _strict_inference_masking_enabled(cfg):
            return _apply_strict_inference_masking(dataset, df)
        return dataset
    if graph_view is None and experiment_cfg:
        from oasis.graphs import load_configured_graph_dataset_view

        graph_view = load_configured_graph_dataset_view(graph_dataset_cfg)

    dataset = assemble_learning_curve_dataset_from_frame(
        df,
        graph_view=graph_view,
        graph_join_key=graph_join_key,
        auxiliary_views=auxiliary_views,
    )
    if _strict_inference_masking_enabled(cfg):
        return _apply_strict_inference_masking(dataset, df)
    return dataset
