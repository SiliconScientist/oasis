from __future__ import annotations

import polars as pl

from oasis.mlip.artifacts import INFERENCE_DETAIL_COLUMNS

MLIP_ENERGY_SUFFIX = "_mlip_ads_eng_median"
MLIP_LABEL_SUFFIX = "_label"


def _mlip_prefixes(wide_df: pl.DataFrame) -> list[str]:
    return [
        column.removesuffix(MLIP_ENERGY_SUFFIX)
        for column in wide_df.columns
        if column.endswith(MLIP_ENERGY_SUFFIX)
    ]


def _mlip_detail_columns(wide_df: pl.DataFrame, prefix: str) -> list[str]:
    return [
        f"{prefix}_{detail_name}"
        for detail_name in INFERENCE_DETAIL_COLUMNS
        if f"{prefix}_{detail_name}" in wide_df.columns
    ]


def filter_structures_with_insufficient_valid_mlips(
    wide_df: pl.DataFrame,
    *,
    enabled: bool = False,
    label_allowlist: list[str] | None = None,
    strict_inference_anomaly: bool = False,
    min_valid_mlips: int = 2,
) -> pl.DataFrame:
    if not enabled:
        return wide_df

    allowlist = set(label_allowlist or ["normal"])
    prefixes = _mlip_prefixes(wide_df)
    if not prefixes:
        raise ValueError(
            "No MLIP prediction columns found for anomaly-aware structure filtering "
            f"(expected *{MLIP_ENERGY_SUFFIX})."
        )

    validity_exprs: list[pl.Expr] = []
    for prefix in prefixes:
        label_col = f"{prefix}{MLIP_LABEL_SUFFIX}"
        detail_cols = _mlip_detail_columns(wide_df, prefix)

        if strict_inference_anomaly:
            if not detail_cols:
                raise ValueError(
                    "Strict anomaly-aware structure filtering requires inference detail "
                    f"columns for {prefix!r}."
                )
            validity_exprs.append(
                pl.all_horizontal([pl.col(detail_col) == 0 for detail_col in detail_cols])
                .cast(pl.Int64)
            )
        else:
            if label_col not in wide_df.columns:
                raise ValueError(
                    "Anomaly-aware structure filtering requires label columns; "
                    f"missing {label_col!r}."
                )
            validity_exprs.append(pl.col(label_col).is_in(sorted(allowlist)).cast(pl.Int64))

    valid_count_expr = sum(validity_exprs[1:], validity_exprs[0]).alias(
        "__valid_mlip_count"
    )
    counted_df = wide_df.with_columns(valid_count_expr)
    filtered_df = counted_df.filter(pl.col("__valid_mlip_count") >= min_valid_mlips).drop(
        "__valid_mlip_count"
    )

    dropped_rows = wide_df.height - filtered_df.height
    mode = "details" if strict_inference_anomaly else "labels"
    print(
        "Applied anomaly-aware structure filtering"
        f" mode={mode}"
        f" allowlist={sorted(allowlist)!r}"
        f" min_valid_mlips={min_valid_mlips}"
        f": {wide_df.height} -> {filtered_df.height} rows"
        f" (dropped {dropped_rows})"
    )
    return filtered_df


def filter_anomalous_mlip_columns(
    wide_df: pl.DataFrame,
    *,
    enabled: bool = False,
    label_allowlist: list[str] | None = None,
    strict_inference_anomaly: bool = False,
) -> pl.DataFrame:
    """Return a wide frame with anomalous MLIP columns removed."""
    if not enabled:
        return wide_df

    allowlist = set(label_allowlist or ["normal"])
    prefixes = _mlip_prefixes(wide_df)
    if not prefixes:
        raise ValueError(
            "No MLIP prediction columns found for anomaly-aware selection "
            f"(expected *{MLIP_ENERGY_SUFFIX})."
        )

    if strict_inference_anomaly:
        print(
            "Skipped global anomaly-aware MLIP selection in strict inference mode;"
            " relying on per-structure filtering to remove invalid rows"
        )
        return wide_df

    kept_prefixes: list[str] = []
    removed_prefixes: list[str] = []
    for prefix in prefixes:
        label_col = f"{prefix}{MLIP_LABEL_SUFFIX}"
        detail_cols = _mlip_detail_columns(wide_df, prefix)

        if label_col not in wide_df.columns:
            raise ValueError(
                "Anomaly-aware MLIP selection requires label columns; "
                f"missing {label_col!r}."
            )
        has_anomaly = any(
            label not in allowlist
            for label in wide_df.get_column(label_col).to_list()
        )

        if has_anomaly:
            removed_prefixes.append(prefix)
        else:
            kept_prefixes.append(prefix)

    if not kept_prefixes:
        raise ValueError("No MLIP prediction columns remain after anomaly-aware selection.")

    drop_columns: list[str] = []
    for prefix in removed_prefixes:
        drop_columns.extend(
            [
                f"{prefix}{MLIP_ENERGY_SUFFIX}",
                f"{prefix}{MLIP_LABEL_SUFFIX}",
                *_mlip_detail_columns(wide_df, prefix),
            ]
        )
    filtered_df = wide_df.drop([column for column in drop_columns if column in wide_df.columns])

    mode = "details" if strict_inference_anomaly else "labels"
    print(
        "Applied anomaly-aware MLIP selection"
        f" mode={mode}"
        f" allowlist={sorted(allowlist)!r}"
        f": kept {len(kept_prefixes)}/{len(prefixes)} MLIPs {kept_prefixes!r};"
        f" removed {len(removed_prefixes)} {removed_prefixes!r}"
    )
    return filtered_df

