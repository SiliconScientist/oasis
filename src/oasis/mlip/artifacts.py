from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from oasis.mlip.result_parsing import (
    detect_anomalies_from_result_json,
    extract_adsorbate,
)

INFERENCE_DETAIL_COLUMNS = (
    "slab_conv",
    "ads_conv",
    "slab_move",
    "ads_move",
    "slab_seed",
    "ads_seed",
    "ads_eng_seed",
    "adsorbate_migration",
)

_REQUESTED_MODEL_ALIASES = {
    "sevennet": ("7net",),
}


def _normalize_model_token(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _matches_requested_model(requested_model: str, actual_model: str) -> bool:
    requested_variants = [requested_model, *_REQUESTED_MODEL_ALIASES.get(requested_model, ())]
    actual_normalized = _normalize_model_token(actual_model)
    for variant in requested_variants:
        normalized = _normalize_model_token(variant)
        if actual_normalized == normalized or actual_normalized.startswith(normalized):
            return True
    return False


def find_result_files(
    base_dir: Path,
    *,
    pattern: str = "*/*_result.json",
    exclude_processed: bool = True,
    enabled_models: list[str] | None = None,
) -> list[Path]:
    candidates = sorted(base_dir.glob(pattern))
    if pattern == "*/*_result.json":
        # Accept both the standard per-model subdirectory layout and legacy
        # flat dataset roots such as OC20-Dense-NHx.
        candidates = sorted(
            dict.fromkeys([*candidates, *base_dir.glob("*_result.json")])
        )
    if exclude_processed:
        candidates = [
            path
            for path in candidates
            if not path.name.endswith("_processed_result.json")
        ]
    if enabled_models:
        matched_candidates: list[Path] = []
        missing_models: list[str] = []
        for requested_model in enabled_models:
            matched_paths = [
                path
                for path in candidates
                if _matches_requested_model(
                    requested_model,
                    model_name_from_result_path(path),
                )
            ]
            if not matched_paths:
                missing_models.append(requested_model)
                continue
            matched_candidates.extend(matched_paths)
        candidates = sorted(dict.fromkeys(matched_candidates))
        if missing_models:
            missing = ", ".join(missing_models)
            raise FileNotFoundError(
                f"Requested MLIP result files were not found under {base_dir}: {missing}"
            )
    if candidates:
        return candidates

    if enabled_models:
        enabled_str = ", ".join(enabled_models)
        raise FileNotFoundError(
            "No enabled MLIP result files found under "
            f"{base_dir} (requested: {enabled_str}; expected pattern: {pattern})"
        )

    raise FileNotFoundError(
        f"No result files found under {base_dir} (expected pattern: {pattern})"
    )


def load_result_json(result_path: str | Path) -> dict[str, Any]:
    path = Path(result_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected result JSON top-level to be an object/dict, got {type(payload).__name__}"
        )
    return payload


def result_file_name(model_name: str) -> str:
    return f"{model_name}_result.json"


def model_name_from_result_path(result_path: str | Path) -> str:
    return Path(result_path).name.removesuffix("_result.json")


def mlip_energy_column_name(model_name: str) -> str:
    return f"{model_name}_mlip_ads_eng_median"


def mlip_label_column_name(model_name: str) -> str:
    return f"{model_name}_label"


def mlip_detail_column_name(model_name: str, detail_name: str) -> str:
    return f"{model_name}_{detail_name}"


def load_wide_predictions(result_files: list[Path]) -> pl.DataFrame:
    reference_df: pl.DataFrame | None = None
    wide_parts: list[pl.DataFrame] = []
    mlip_cols: list[str] = []
    label_cols: list[str] = []
    detail_cols: list[str] = []

    for path in result_files:
        model_name = model_name_from_result_path(path)
        per_reaction = detect_anomalies_from_result_json(path)
        if not per_reaction:
            raise ValueError(
                "MLIP result file contains no prediction rows after parsing: "
                f"{path}. Expected CatBench reaction entries, but found only "
                "metadata or empty results."
            )
        rows = [
            {
                "reaction": reaction,
                "adsorbate": extract_adsorbate(reaction),
                **{
                    detail_name: int(payload.get("details", {}).get(detail_name, 0))
                    for detail_name in INFERENCE_DETAIL_COLUMNS
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
            *INFERENCE_DETAIL_COLUMNS,
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
                *INFERENCE_DETAIL_COLUMNS,
            ]
        ).rename(
            {
                reaction_col: "reaction",
                "dft_ads_eng": "reference_ads_eng",
                "mlip_ads_eng_median": mlip_energy_column_name(model_name),
                "label": mlip_label_column_name(model_name),
                **{
                    detail_name: mlip_detail_column_name(model_name, detail_name)
                    for detail_name in INFERENCE_DETAIL_COLUMNS
                },
            }
        )
        part = part.drop_nulls(
            subset=["reaction", mlip_energy_column_name(model_name)]
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

        mlip_col = mlip_energy_column_name(model_name)
        label_col = mlip_label_column_name(model_name)
        mlip_cols.append(mlip_col)
        label_cols.append(label_col)
        model_detail_cols = [
            mlip_detail_column_name(model_name, detail_name)
            for detail_name in INFERENCE_DETAIL_COLUMNS
        ]
        detail_cols.extend(model_detail_cols)
        wide_parts.append(
            part.select(["reaction", mlip_col, label_col, *model_detail_cols])
        )

    if reference_df is None:
        raise RuntimeError("No MLIP result rows were loaded.")

    wide_df = reference_df.clone()
    for part in wide_parts:
        wide_df = wide_df.join(part, on="reaction", how="inner")

    return wide_df.drop_nulls(
        subset=["reference_ads_eng", *mlip_cols, *label_cols, *detail_cols]
    ).sort("reaction")
