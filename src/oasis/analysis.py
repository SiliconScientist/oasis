from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
from catbench.adsorption import AdsorptionAnalysis

from oasis.config import Config, get_config
from oasis.mlip.result_parsing import (
    detect_anomalies_from_result_dict,
    detect_anomalies_from_result_json,
    extract_adsorbate,
)
from oasis.plot import mae_comparison_plot

_INFERENCE_ANOMALY_DETAILS = (
    "slab_conv",
    "ads_conv",
    "slab_move",
    "ads_move",
    "slab_seed",
    "ads_seed",
    "ads_eng_seed",
    "adsorbate_migration",
)


def _load_summary_dataframe(workbook_path: str | Path) -> pd.DataFrame:
    try:
        excel_file = pd.ExcelFile(workbook_path)
    except ImportError as exc:
        raise ImportError(
            "Reading Excel summary files requires openpyxl. Install it first."
        ) from exc

    sheet_name = "Summary" if "Summary" in excel_file.sheet_names else 0
    return pd.read_excel(workbook_path, sheet_name=sheet_name)


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
                for detail_name in _INFERENCE_ANOMALY_DETAILS
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


def run_summary_analysis(cfg: Config | None = None) -> Path | None:
    cfg = cfg or get_config()
    if cfg.analysis is None:
        return None

    if cfg.analysis.summary_workbook_path is None:
        raise ValueError("mlip.toml must define analysis.summary_workbook_path")
    if cfg.analysis.comparison_workbook_path is None:
        raise ValueError("mlip.toml must define analysis.comparison_workbook_path")
    if cfg.analysis.comparison_plot_path is None:
        raise ValueError("mlip.toml must define analysis.comparison_plot_path")
    if cfg.analysis.run_adsorption_analysis and cfg.analysis.calculating_path is None:
        raise ValueError(
            "mlip.toml must define analysis.calculating_path when "
            "run_adsorption_analysis is true"
        )

    summary_df = _load_summary_dataframe(cfg.analysis.summary_workbook_path)
    comparison_df = _load_summary_dataframe(cfg.analysis.comparison_workbook_path)
    comparison_plot_path = mae_comparison_plot(
        comparison_df=comparison_df,
        summary_df=summary_df,
        output_path=cfg.analysis.comparison_plot_path,
    )

    if cfg.analysis.run_adsorption_analysis:
        analysis = AdsorptionAnalysis(calculating_path=str(cfg.analysis.calculating_path))
        analysis.analysis()
        analysis.threshold_sensitivity_analysis()

    return comparison_plot_path
