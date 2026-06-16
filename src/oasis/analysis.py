from __future__ import annotations

import polars as pl

from oasis.mlip.artifacts import INFERENCE_DETAIL_COLUMNS


def filter_anomalous_mlip_columns(
    wide_df: pl.DataFrame,
    *,
    enabled: bool = False,
    strict_inference_anomaly: bool = False,
) -> pl.DataFrame:
    """Return a wide frame with anomalous MLIP columns removed.

    When ``enabled`` is false, the input frame is returned unchanged. Future
    commits will implement anomaly-aware MLIP selection using the per-MLIP
    ``*_label`` and inference detail columns already present in the wide frame.
    """
    del strict_inference_anomaly
    return wide_df.clone() if enabled else wide_df


def filter_wide_predictions(
    wide_df: pl.DataFrame,
    adsorbate_filter: str | None = None,
    anomaly_filter: str | None = None,
    reaction_contains_filter: list[str] | None = None,
) -> pl.DataFrame:
    filtered_df = wide_df.clone()

    if adsorbate_filter is not None:
        if "adsorbate" not in filtered_df.columns:
            raise ValueError(
                f"Configured plot.filters.adsorbate='{adsorbate_filter}', but no "
                "'adsorbate' column exists in the combined dataframe"
            )
        filtered_df = filtered_df.filter(pl.col("adsorbate") == adsorbate_filter)
        if filtered_df.height == 0:
            raise ValueError(
                f"No rows left after adsorbate filter '{adsorbate_filter}'"
            )

    if anomaly_filter is not None:
        label_cols = [col for col in filtered_df.columns if col.endswith("_label")]
        detail_cols = [
            col
            for col in filtered_df.columns
            if any(
                col.endswith(f"_{detail_name}")
                for detail_name in INFERENCE_DETAIL_COLUMNS
            )
        ]
        if not label_cols and not detail_cols:
            raise ValueError(
                f"Configured plot.filters.anomaly_label='{anomaly_filter}', but no "
                "label/detail columns exist in the combined dataframe"
            )
        exclude_mode = anomaly_filter.startswith(("!", "not:"))
        anomaly_value = anomaly_filter[1:] if anomaly_filter.startswith("!") else (
            anomaly_filter[4:] if anomaly_filter.startswith("not:") else anomaly_filter
        )
        if not anomaly_value:
            raise ValueError(
                "plot.filters.anomaly_label exclusion must specify a label, e.g. "
                "'!adsorbate_migration' or 'not:adsorbate_migration'"
            )
        if anomaly_value == "inference_anomaly":
            if not detail_cols:
                raise ValueError(
                    "Configured plot.filters.anomaly_label for inference anomaly "
                    "filtering, but no inference detail columns exist in the combined dataframe"
                )
            anomaly_expr = pl.any_horizontal([pl.col(col) > 0 for col in detail_cols])
            label_expr = ~anomaly_expr if exclude_mode else anomaly_expr
        else:
            if exclude_mode:
                label_expr = pl.all_horizontal(
                    [pl.col(col) != anomaly_value for col in label_cols]
                )
            else:
                label_expr = pl.all_horizontal(
                    [pl.col(col) == anomaly_value for col in label_cols]
                )
        filtered_df = filtered_df.filter(label_expr)
        if filtered_df.height == 0:
            raise ValueError(
                f"No rows left after anomaly_label filter '{anomaly_filter}'"
            )

    if reaction_contains_filter is not None:
        mask_expr = None
        for substring in reaction_contains_filter:
            token = f"_{substring}_"
            expr = (
                pl.lit("_") + pl.col("reaction").cast(pl.String) + pl.lit("_")
            ).str.contains(token, literal=True)
            mask_expr = expr if mask_expr is None else (mask_expr | expr)
        if mask_expr is not None:
            filtered_df = filtered_df.filter(mask_expr)
        if filtered_df.height == 0:
            raise ValueError(
                f"No rows left after reaction_contains filter "
                f"'{reaction_contains_filter}'"
            )

    return filtered_df
