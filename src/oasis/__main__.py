from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

from oasis.analysis import write_processed_result_files
from oasis.config import get_config
from oasis.plot import parity_plot
from oasis.mlip.cli import main as mlip_main


def _find_processed_result_files(base_dir: Path) -> list[Path]:
    """
    Find processed MLIP result JSON files under a directory containing
    per-model subdirectories.
    """
    candidates = sorted(base_dir.glob("*/*_processed_result.json"))
    if candidates:
        return candidates

    # Fallback for filename variant.
    candidates = sorted(base_dir.glob("*/*processed_results.json"))
    if candidates:
        return candidates

    raise FileNotFoundError(
        f"No processed result files found under {base_dir} "
        "(expected patterns: */*_processed_result.json or */*processed_results.json)"
    )


def _load_wide_predictions(
    processed_files: list[Path],
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
            subset=[
                "reaction",
                "reference_ads_eng",
                f"{model_name}_mlip_ads_eng_median",
            ]
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
            mismatch = overlap[
                overlap["reference_ads_eng"] != overlap["reference_ads_eng_new"]
            ]
            if not mismatch.empty:
                raise ValueError(
                    f"Reference energies differ for overlapping reactions in {path}"
                )
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
    if len(sys.argv) >= 2 and sys.argv[1] == "mlip":
        # Forward remaining args to mlip CLI
        mlip_main(sys.argv[2:])
        return

    cfg = get_config()
    base_dir = cfg.analysis.base_dir if cfg.analysis else Path("data/mlips")
    write_processed_result_files(base_dir)
    processed_files = _find_processed_result_files(base_dir)
    wide_df = _load_wide_predictions(processed_files)

    output_dir = cfg.plot.output_dir if cfg.plot else Path("data/results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mlips_vs_dft_parity.png"
    saved_path = parity_plot(wide_df, output_path=output_path)
    print(
        f"Processed {len(processed_files)} MLIP files"
        f" -> parity plot: {saved_path}"
    )
    print(f"Rows in combined parity dataset: {len(wide_df)}")


if __name__ == "__main__":
    main()
