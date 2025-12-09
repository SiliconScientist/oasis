from __future__ import annotations

from pathlib import Path
from typing import Sequence

import polars as pl

from oasis.config import Config


def _collect_parquet_files(root: Path) -> list[Path]:
    """Return sorted list of result parquet files below ``root``."""
    parquet_files = sorted(root.glob("*/*_normal_predictions.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No *_normal_predictions.parquet files found under {root.resolve()}"
        )
    return parquet_files


def _load_wide_parts(
    parquet_files: Sequence[Path],
) -> tuple[pl.DataFrame, list[pl.DataFrame], list[str]]:
    """
    Build reference frame and per-MLIP wide parts from the result parquet files.

    Returns (reference_df, wide_parts, mlip_names).
    """
    reference_df: pl.DataFrame | None = None
    wide_parts: list[pl.DataFrame] = []
    mlip_names: list[str] = []

    required_cols = {
        "reaction",
        "adsorbate",
        "reference_ads_eng",
        "mlip_ads_eng_median",
    }

    for path in parquet_files:
        mlip = path.stem.replace("_normal_predictions", "")
        mlip_names.append(mlip)

        df = pl.read_parquet(path)
        missing = required_cols.difference(df.columns)
        if missing:
            missing_cols = ", ".join(sorted(missing))
            raise ValueError(f"Missing required columns in {path}: {missing_cols}")

        df = df.select(list(required_cols))

        part = df.select(["reaction", "mlip_ads_eng_median"]).rename(
            {"mlip_ads_eng_median": f"{mlip}_mlip_ads_eng_median"}
        )
        wide_parts.append(part)

        ref_part = df.select(["reaction", "reference_ads_eng"])
        if reference_df is None:
            reference_df = ref_part
        else:
            overlap = reference_df.join(
                ref_part, on="reaction", how="inner", suffix="_new"
            )
            mismatch = overlap.filter(
                pl.col("reference_ads_eng") != pl.col("reference_ads_eng_new")
            )
            if mismatch.height > 0:
                raise ValueError(f"Reference adsorption energies differ in {path}")
            reference_df = (
                pl.concat([reference_df, ref_part])
                .unique(subset="reaction", keep="first")
                .sort("reaction")
            )

    if reference_df is None:
        raise RuntimeError("No reference data loaded from parquet files.")

    return reference_df, wide_parts, mlip_names


def _build_wide(
    reference_df: pl.DataFrame, wide_parts: list[pl.DataFrame], mlip_names: list[str]
) -> pl.DataFrame:
    """Combine reference adsorption energies with MLIP medians into a wide DataFrame."""
    wide_df = reference_df.clone()
    for part in wide_parts:
        wide_df = wide_df.join(part, on="reaction", how="outer")
        if "reaction_right" in wide_df.columns:
            wide_df = (
                wide_df.with_columns(
                    pl.coalesce(pl.col("reaction"), pl.col("reaction_right")).alias(
                        "reaction"
                    )
                )
                .drop("reaction_right")
                .rechunk()
            )

    required_columns = ["reference_ads_eng"] + [
        f"{mlip}_mlip_ads_eng_median" for mlip in mlip_names
    ]
    wide_df = wide_df.drop_nulls(subset=required_columns)
    return wide_df.sort("reaction")


def get_data(cfg: Config) -> pl.DataFrame:
    """
    Load and align MLIP adsorption energies into a single wide Polars DataFrame.

    The function searches for ``*_normal_predictions.parquet`` under the configured
    ``processing.root`` directory and returns the filtered wide-format DataFrame, where
    each row is a reaction present for all MLIPs. The returned frame includes the
    ``reaction`` and ``reference_ads_eng`` columns plus one column per MLIP prediction.
    """
    results_root = Path(cfg.processing.root)
    parquet_files = _collect_parquet_files(results_root)

    if cfg.dev_run:
        parquet_files = parquet_files[:2]  # keep runtime short in dev mode

    reference_df, wide_parts, mlip_names = _load_wide_parts(parquet_files)
    wide_df = _build_wide(reference_df, wide_parts, mlip_names)
    wide_df.write_parquet(file=cfg.processing.root / "combined_mlips.parquet")
    return wide_df
