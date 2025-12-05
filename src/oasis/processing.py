from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

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
) -> tuple[pd.DataFrame, list[pd.DataFrame], list[str]]:
    """
    Build reference frame and per-MLIP wide parts from the result parquet files.

    Returns (reference_df, wide_parts, mlip_names).
    """
    reference_df: pd.DataFrame | None = None
    wide_parts: list[pd.DataFrame] = []
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

        df = pd.read_parquet(path)
        missing = required_cols.difference(df.columns)
        if missing:
            missing_cols = ", ".join(sorted(missing))
            raise ValueError(f"Missing required columns in {path}: {missing_cols}")

        df = df[list(required_cols)].copy()

        part = df.set_index("reaction")[["mlip_ads_eng_median"]].rename(
            columns={"mlip_ads_eng_median": f"{mlip}_mlip_ads_eng_median"}
        )
        wide_parts.append(part)

        ref_part = df.set_index("reaction")[["reference_ads_eng"]]
        if reference_df is None:
            reference_df = ref_part
        else:
            overlap = reference_df.index.intersection(ref_part.index)
            if not overlap.empty:
                mismatch = reference_df.loc[overlap, "reference_ads_eng"].compare(
                    ref_part.loc[overlap, "reference_ads_eng"]
                )
                if not mismatch.empty:
                    raise ValueError(f"Reference adsorption energies differ in {path}")
            reference_df = reference_df.combine_first(ref_part)

    if reference_df is None:
        raise RuntimeError("No reference data loaded from parquet files.")

    return reference_df, wide_parts, mlip_names


def _build_wide(
    reference_df: pd.DataFrame, wide_parts: list[pd.DataFrame], mlip_names: list[str]
) -> pd.DataFrame:
    """Combine reference adsorption energies with MLIP medians into a wide DataFrame."""
    wide_df = reference_df.copy()
    for part in wide_parts:
        wide_df = wide_df.join(part, how="outer")

    required_columns = ["reference_ads_eng"] + [
        f"{mlip}_mlip_ads_eng_median" for mlip in mlip_names
    ]
    wide_df = wide_df.dropna(subset=required_columns)
    return wide_df.sort_index()


def get_data(cfg: Config) -> pd.DataFrame:
    """
    Load and align MLIP adsorption energies into a single wide DataFrame.

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

    return wide_df.reset_index()
