from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

from oasis.analysis import detect_anomalies_from_result_json
from oasis.config import get_config
from oasis.plot import parity_plot
from oasis.mlip.cli import main as mlip_main


def _find_result_files(base_dir: Path) -> list[Path]:
    """
    Find raw MLIP result JSON files under a directory containing
    per-model subdirectories.
    """
    candidates = sorted(
        path
        for path in base_dir.glob("*/*_result.json")
        if not path.name.endswith("_processed_result.json")
    )
    if candidates:
        return candidates

    raise FileNotFoundError(
        f"No result files found under {base_dir} "
        "(expected pattern: */*_result.json)"
    )


def _extract_adsorbate(reaction: str) -> str | None:
    if "->" not in reaction:
        return None
    product = reaction.split("->", 1)[1].strip()
    return product.removesuffix("*") or None


def _load_wide_predictions(
    result_files: list[Path],
) -> pd.DataFrame:
    """
    Build a wide table with:
      reaction, adsorbate, reference_ads_eng, <mlip>_mlip_ads_eng_median, <mlip>_label, ...
    """
    reference_df: pd.DataFrame | None = None
    wide_parts: list[pd.DataFrame] = []
    mlip_cols: list[str] = []
    label_cols: list[str] = []

    for path in result_files:
        model_name = path.stem.replace("_result", "")
        per_reaction = detect_anomalies_from_result_json(path)
        rows = [
            {
                "reaction": reaction,
                "adsorbate": _extract_adsorbate(reaction),
                **payload,
            }
            for reaction, payload in per_reaction.items()
        ]
        df = pd.DataFrame(rows)

        reaction_col = "id" if "id" in df.columns else "reaction"
        required = {reaction_col, "adsorbate", "dft_ads_eng", "mlip_ads_eng_median", "label"}
        missing = required.difference(set(df.columns))
        if missing:
            missing_cols = ", ".join(sorted(missing))
            raise ValueError(f"Missing required columns in {path}: {missing_cols}")

        part = df[
            [reaction_col, "adsorbate", "dft_ads_eng", "mlip_ads_eng_median", "label"]
        ].rename(
            columns={
                reaction_col: "reaction",
                "dft_ads_eng": "reference_ads_eng",
                "mlip_ads_eng_median": f"{model_name}_mlip_ads_eng_median",
                "label": f"{model_name}_label",
            }
        )
        part = part.dropna(
            subset=[
                "reaction",
                f"{model_name}_mlip_ads_eng_median",
            ]
        )

        if reference_df is None:
            reference_df = part[["reaction", "adsorbate", "reference_ads_eng"]].copy()
        else:
            ref_part = part[["reaction", "adsorbate", "reference_ads_eng"]].copy()
            overlap = reference_df.merge(
                ref_part,
                on="reaction",
                how="inner",
                suffixes=("", "_incoming"),
            )
            energy_mismatch = overlap[
                overlap["reference_ads_eng"] != overlap["reference_ads_eng_incoming"]
            ]
            if not energy_mismatch.empty:
                raise ValueError(
                    f"Reference energies differ for overlapping reactions in {path}"
                )
            adsorbate_mismatch = overlap[
                overlap["adsorbate"].fillna("")
                != overlap["adsorbate_incoming"].fillna("")
            ]
            if not adsorbate_mismatch.empty:
                raise ValueError(
                    f"Adsorbates differ for overlapping reactions in {path}"
                )
            reference_df = (
                pd.concat([reference_df, ref_part], ignore_index=True)
                .drop_duplicates(subset="reaction", keep="first")
                .sort_values("reaction")
            )

        mlip_col = f"{model_name}_mlip_ads_eng_median"
        label_col = f"{model_name}_label"
        mlip_cols.append(mlip_col)
        label_cols.append(label_col)
        wide_parts.append(part[["reaction", mlip_col, label_col]].copy())

    if reference_df is None:
        raise RuntimeError("No MLIP result rows were loaded.")

    wide_df = reference_df.copy()
    for part in wide_parts:
        wide_df = wide_df.merge(part, on="reaction", how="inner")

    wide_df = wide_df.dropna(
        subset=["reference_ads_eng", *mlip_cols, *label_cols]
    ).sort_values("reaction")
    return wide_df


def _filter_wide_predictions(
    wide_df: pd.DataFrame,
    adsorbate_filter: str | None = None,
    anomaly_filter: str | None = None,
    reaction_contains_filter: list[str] | None = None,
) -> pd.DataFrame:
    filtered_df = wide_df.copy()

    if adsorbate_filter is not None:
        if "adsorbate" not in filtered_df.columns:
            raise ValueError(
                f"Configured plot.adsorbate='{adsorbate_filter}', but no "
                "'adsorbate' column exists in the combined dataframe"
            )
        filtered_df = filtered_df[filtered_df["adsorbate"] == adsorbate_filter]
        if filtered_df.empty:
            raise ValueError(
                f"No rows left after adsorbate filter '{adsorbate_filter}'"
            )

    if anomaly_filter is not None:
        label_cols = [col for col in filtered_df.columns if col.endswith("_label")]
        if not label_cols:
            raise ValueError(
                f"Configured plot.anomaly_label='{anomaly_filter}', but no "
                "label columns exist in the combined dataframe"
            )
        mask = filtered_df[label_cols].eq(anomaly_filter).all(axis=1)
        filtered_df = filtered_df[mask]
        if filtered_df.empty:
            raise ValueError(
                f"No rows left after anomaly_label filter '{anomaly_filter}'"
            )

    if reaction_contains_filter is not None:
        mask = pd.Series(False, index=filtered_df.index)
        for substring in reaction_contains_filter:
            token = f"_{substring}_"
            reaction_with_edges = "_" + filtered_df["reaction"].astype(str) + "_"
            mask = mask | reaction_with_edges.str.contains(
                token,
                regex=False,
                na=False,
            )
        filtered_df = filtered_df[mask]
        if filtered_df.empty:
            raise ValueError(
                f"No rows left after reaction_contains filter "
                f"'{reaction_contains_filter}'"
            )

    return filtered_df


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "mlip":
        # Forward remaining args to mlip CLI
        mlip_main(sys.argv[2:])
        return

    cfg = get_config()
    base_dir = cfg.analysis.base_dir if cfg.analysis else Path("data/mlips")
    result_files = _find_result_files(base_dir)
    wide_df = _load_wide_predictions(result_files)
    adsorbate_filter = cfg.plot.adsorbate if cfg.plot else None
    anomaly_filter = cfg.plot.anomaly_label if cfg.plot else None
    reaction_contains_filter = cfg.plot.reaction_contains if cfg.plot else None
    if reaction_contains_filter is not None:
        reaction_contains_filter = [s for s in reaction_contains_filter if s]
        if not reaction_contains_filter:
            reaction_contains_filter = None
    wide_df = _filter_wide_predictions(
        wide_df,
        adsorbate_filter=adsorbate_filter,
        anomaly_filter=anomaly_filter,
        reaction_contains_filter=reaction_contains_filter,
    )

    output_dir = cfg.plot.output_dir if cfg.plot else Path("data/results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix_parts: list[str] = []
    if adsorbate_filter:
        suffix_parts.append(f"adsorbate_{adsorbate_filter}")
    if anomaly_filter:
        suffix_parts.append(f"anomaly_{anomaly_filter}")
    if reaction_contains_filter:
        joined = "-".join(reaction_contains_filter)
        suffix_parts.append(f"reaction_contains_{joined}")
    suffix = f"_{'_'.join(suffix_parts)}" if suffix_parts else ""
    output_path = output_dir / f"mlips_vs_dft_parity{suffix}.png"
    saved_path = parity_plot(wide_df, output_path=output_path)
    print(
        f"Processed {len(result_files)} MLIP files"
        f"{f' with adsorbate={adsorbate_filter}' if adsorbate_filter else ''}"
        f"{f' with anomaly_label={anomaly_filter}' if anomaly_filter else ''}"
        f"{f' with reaction_contains={reaction_contains_filter}' if reaction_contains_filter else ''}"
        f" -> parity plot: {saved_path}"
    )
    print(f"Rows in combined parity dataset: {len(wide_df)}")


if __name__ == "__main__":
    main()
