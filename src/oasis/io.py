from __future__ import annotations

from pathlib import Path

import pandas as pd

from oasis.analysis import detect_anomalies_from_result_json, extract_adsorbate


def find_result_files(base_dir: Path) -> list[Path]:
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


def load_wide_predictions(result_files: list[Path]) -> pd.DataFrame:
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
                "adsorbate": extract_adsorbate(reaction),
                **payload,
            }
            for reaction, payload in per_reaction.items()
        ]
        df = pd.DataFrame(rows)

        reaction_col = "id" if "id" in df.columns else "reaction"
        required = {
            reaction_col,
            "adsorbate",
            "dft_ads_eng",
            "mlip_ads_eng_median",
            "label",
        }
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
