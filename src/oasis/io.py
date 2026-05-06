from __future__ import annotations

from pathlib import Path

import polars as pl

from oasis.analysis import detect_anomalies_from_result_json, extract_adsorbate

_INFERENCE_DETAIL_COLUMNS = (
    "slab_conv",
    "ads_conv",
    "slab_move",
    "ads_move",
    "slab_seed",
    "ads_seed",
    "ads_eng_seed",
    "adsorbate_migration",
)


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


def load_wide_predictions(result_files: list[Path]) -> pl.DataFrame:
    """
    Build a wide table with:
      reaction, adsorbate, reference_ads_eng, <mlip>_mlip_ads_eng_median,
      <mlip>_label, <mlip>_<detail_flag>, ...
    """
    reference_df: pl.DataFrame | None = None
    wide_parts: list[pl.DataFrame] = []
    mlip_cols: list[str] = []
    label_cols: list[str] = []
    detail_cols: list[str] = []

    for path in result_files:
        model_name = path.stem.replace("_result", "")
        per_reaction = detect_anomalies_from_result_json(path)
        rows = [
            {
                "reaction": reaction,
                "adsorbate": extract_adsorbate(reaction),
                **{
                    detail_name: int(payload.get("details", {}).get(detail_name, 0))
                    for detail_name in _INFERENCE_DETAIL_COLUMNS
                },
                **payload,
            }
            for reaction, payload in per_reaction.items()
        ]
        df = pl.from_dicts(rows)

        reaction_col = "id" if "id" in df.columns else "reaction"
        required = {
            reaction_col,
            "adsorbate",
            "dft_ads_eng",
            "mlip_ads_eng_median",
            "label",
            *_INFERENCE_DETAIL_COLUMNS,
        }
        missing = required.difference(set(df.columns))
        if missing:
            missing_cols = ", ".join(sorted(missing))
            raise ValueError(f"Missing required columns in {path}: {missing_cols}")

        part = df.select(
            [
                reaction_col,
                "adsorbate",
                "dft_ads_eng",
                "mlip_ads_eng_median",
                "label",
                *_INFERENCE_DETAIL_COLUMNS,
            ]
        ).rename(
            {
                reaction_col: "reaction",
                "dft_ads_eng": "reference_ads_eng",
                "mlip_ads_eng_median": f"{model_name}_mlip_ads_eng_median",
                "label": f"{model_name}_label",
                **{
                    detail_name: f"{model_name}_{detail_name}"
                    for detail_name in _INFERENCE_DETAIL_COLUMNS
                },
            }
        )
        part = part.drop_nulls(
            subset=["reaction", f"{model_name}_mlip_ads_eng_median"]
        )

        if reference_df is None:
            reference_df = part.select(["reaction", "adsorbate", "reference_ads_eng"])
        else:
            ref_part = part.select(["reaction", "adsorbate", "reference_ads_eng"])
            overlap = reference_df.join(
                ref_part, on="reaction", how="inner", suffix="_incoming"
            )
            energy_mismatch = overlap.filter(
                pl.col("reference_ads_eng") != pl.col("reference_ads_eng_incoming")
            )
            if energy_mismatch.height > 0:
                raise ValueError(
                    f"Reference energies differ for overlapping reactions in {path}"
                )
            adsorbate_mismatch = overlap.filter(
                pl.col("adsorbate").fill_null("")
                != pl.col("adsorbate_incoming").fill_null("")
            )
            if adsorbate_mismatch.height > 0:
                raise ValueError(
                    f"Adsorbates differ for overlapping reactions in {path}"
                )
            reference_df = (
                pl.concat([reference_df, ref_part])
                .unique(subset="reaction", keep="first")
                .sort("reaction")
            )

        mlip_col = f"{model_name}_mlip_ads_eng_median"
        label_col = f"{model_name}_label"
        mlip_cols.append(mlip_col)
        label_cols.append(label_col)
        model_detail_cols = [
            f"{model_name}_{detail_name}" for detail_name in _INFERENCE_DETAIL_COLUMNS
        ]
        detail_cols.extend(model_detail_cols)
        wide_parts.append(part.select(["reaction", mlip_col, label_col, *model_detail_cols]))

    if reference_df is None:
        raise RuntimeError("No MLIP result rows were loaded.")

    wide_df = reference_df.clone()
    for part in wide_parts:
        wide_df = wide_df.join(part, on="reaction", how="inner")

    wide_df = wide_df.drop_nulls(
        subset=["reference_ads_eng", *mlip_cols, *label_cols, *detail_cols]
    ).sort("reaction")
    return wide_df
