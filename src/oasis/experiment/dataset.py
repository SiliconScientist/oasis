from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

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

        return load_sweep_dataset_from_graph_artifact(
            graph_dataset_cfg.path,
            join_key=graph_join_key,
            auxiliary_views=auxiliary_views,
            filter_df=df,
        )
    if graph_view is None and experiment_cfg:
        from oasis.graphs import load_configured_graph_dataset_view

        graph_view = load_configured_graph_dataset_view(graph_dataset_cfg)

    return assemble_learning_curve_dataset_from_frame(
        df,
        graph_view=graph_view,
        graph_join_key=graph_join_key,
        auxiliary_views=auxiliary_views,
    )
