from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from oasis.config import get_config
from oasis.plot import parity_plot


def _find_processed_result_files() -> list[Path]:
    """
    Find processed MLIP result JSON files under common project layouts.
    """
    candidates = sorted(Path("data/mlips").glob("*/*_processed_result.json"))
    if candidates:
        return candidates

    # Fallbacks for user-typed variants.
    candidates = sorted(Path("data/mlip").glob("*/*processed_results.json"))
    if candidates:
        return candidates

    candidates = sorted(Path("data/mlip").glob("*/*_processed_result.json"))
    if candidates:
        return candidates

    raise FileNotFoundError(
        "No processed result files found under data/mlips/*/*_processed_result.json "
        "or data/mlip/*/*processed_results.json"
    )


def _load_wide_predictions(
    processed_files: list[Path],
    adsorbate_filter: str | None = None,
    anomaly_filter: str | None = None,
) -> pd.DataFrame:
    """
    Build a wide table with:
      reaction, reference_ads_eng, <mlip>_mlip_ads_eng_median, ...
    """
    reference_df: pd.DataFrame | None = None
    wide_parts: list[pd.DataFrame] = []
    mlip_cols: list[str] = []

    for path in processed_files:
        model_name = path.stem.replace("_processed_result", "")

        with path.open("r") as f:
            rows = json.load(f)
        df = pd.DataFrame(rows)

        if adsorbate_filter is not None:
            if "adsorbate" not in df.columns:
                raise ValueError(
                    f"Configured plot.adsorbate='{adsorbate_filter}', but no "
                    f"'adsorbate' column exists in {path}"
                )
            df = df[df["adsorbate"] == adsorbate_filter]
            if df.empty:
                raise ValueError(
                    f"No rows left in {path} after adsorbate filter '{adsorbate_filter}'"
                )

        if anomaly_filter is not None:
            if "label" not in df.columns:
                raise ValueError(
                    f"Configured plot.anomaly_label='{anomaly_filter}', but no "
                    f"'label' column exists in {path}"
                )
            df = df[df["label"] == anomaly_filter]
            if df.empty:
                raise ValueError(
                    f"No rows left in {path} after anomaly_label filter '{anomaly_filter}'"
                )

        reaction_col = "id" if "id" in df.columns else "reaction"
        required = {reaction_col, "dft_ads_eng", "mlip_ads_eng_median"}
        missing = required.difference(set(df.columns))
        if missing:
            missing_cols = ", ".join(sorted(missing))
            raise ValueError(f"Missing required columns in {path}: {missing_cols}")

        part = df[[reaction_col, "dft_ads_eng", "mlip_ads_eng_median"]].rename(
            columns={
                reaction_col: "reaction",
                "dft_ads_eng": "reference_ads_eng",
                "mlip_ads_eng_median": f"{model_name}_mlip_ads_eng_median",
            }
        )
        part = part.dropna(
            subset=["reaction", "reference_ads_eng", f"{model_name}_mlip_ads_eng_median"]
        )

        if reference_df is None:
            reference_df = part[["reaction", "reference_ads_eng"]].copy()
        else:
            ref_part = part[["reaction", "reference_ads_eng"]].copy()
            overlap = reference_df.merge(
                ref_part,
                on="reaction",
                how="inner",
                suffixes=("", "_new"),
            )
            mismatch = overlap[overlap["reference_ads_eng"] != overlap["reference_ads_eng_new"]]
            if not mismatch.empty:
                raise ValueError(f"Reference energies differ for overlapping reactions in {path}")
            reference_df = (
                pd.concat([reference_df, ref_part], ignore_index=True)
                .drop_duplicates(subset="reaction", keep="first")
                .sort_values("reaction")
            )

        mlip_col = f"{model_name}_mlip_ads_eng_median"
        mlip_cols.append(mlip_col)
        wide_parts.append(part[["reaction", mlip_col]].copy())

    if reference_df is None:
        raise RuntimeError("No processed MLIP rows were loaded.")

    wide_df = reference_df.copy()
    for part in wide_parts:
        wide_df = wide_df.merge(part, on="reaction", how="inner")

    wide_df = wide_df.dropna(subset=["reference_ads_eng", *mlip_cols]).sort_values(
        "reaction"
    )
    return wide_df


def main() -> None:
    cfg = get_config()
    processed_files = _find_processed_result_files()
    adsorbate_filter = cfg.plot.adsorbate if cfg.plot else None
    anomaly_filter = cfg.plot.anomaly_label if cfg.plot else None
    wide_df = _load_wide_predictions(
        processed_files,
        adsorbate_filter=adsorbate_filter,
        anomaly_filter=anomaly_filter,
    )

    output_dir = cfg.plot.output_dir if cfg.plot else Path("data/results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix_parts: list[str] = []
    if adsorbate_filter:
        suffix_parts.append(f"adsorbate_{adsorbate_filter}")
    if anomaly_filter:
        suffix_parts.append(f"anomaly_{anomaly_filter}")
    suffix = f"_{'_'.join(suffix_parts)}" if suffix_parts else ""
    output_path = output_dir / f"mlips_vs_dft_parity{suffix}.png"
    saved_path = parity_plot(wide_df, output_path=output_path)
    print(
        f"Processed {len(processed_files)} MLIP files"
        f"{f' with adsorbate={adsorbate_filter}' if adsorbate_filter else ''}"
        f"{f' with anomaly_label={anomaly_filter}' if anomaly_filter else ''}"
        f" -> parity plot: {saved_path}"
    )
    print(f"Rows in combined parity dataset: {len(wide_df)}")


if __name__ == "__main__":
    main()
