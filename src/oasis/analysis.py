from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from catbench.adsorption import AdsorptionAnalysis
from catbench.utils.analysis_utils import (
    get_calculator_keys,
    get_median_calculator_key,
)

from oasis.config import Config, get_config
from oasis.plot import mae_comparison_plot


def _load_summary_dataframe(workbook_path: str | Path) -> pd.DataFrame:
    try:
        excel_file = pd.ExcelFile(workbook_path)
    except ImportError as exc:
        raise ImportError(
            "Reading Excel summary files requires openpyxl. Install it first."
        ) from exc

    sheet_name = "Summary" if "Summary" in excel_file.sheet_names else 0
    return pd.read_excel(workbook_path, sheet_name=sheet_name)


def detect_anomalies_from_result_dict(
    mlip_result: dict[str, Any],
    *,
    disp_thrs: float = 1.0,
    energy_thrs: float = 2.0,
    reproduction_thrs: float = 0.2,
    bond_length_change_threshold: float = 0.2,
    energy_cutoff: float | None = None,
) -> dict[str, dict[str, Any]]:
    calc_settings = mlip_result.get("calculation_settings", {})
    if "chemical_bond_cutoff" not in calc_settings:
        raise ValueError(
            "chemical_bond_cutoff not found in calculation_settings. "
            "This detector expects CatBench result.json produced with that setting."
        )

    n_crit_relax = calc_settings.get("n_crit_relax", 999)
    bond_threshold_pct = bond_length_change_threshold * 100.0

    has_slab_calculations: bool | None = None
    for reaction, reaction_data in mlip_result.items():
        if reaction == "calculation_settings":
            continue
        calculator_keys = get_calculator_keys(reaction_data)
        if calculator_keys:
            has_slab_calculations = (
                "slab_max_disp" in reaction_data[calculator_keys[0]]
            )
        break
    if has_slab_calculations is None:
        has_slab_calculations = False

    per_reaction: dict[str, dict[str, Any]] = {}

    for reaction, reaction_data in mlip_result.items():
        if reaction == "calculation_settings":
            continue

        if energy_cutoff is not None:
            try:
                ref_energy = reaction_data["reference"]["ads_eng"]
                if ref_energy > energy_cutoff:
                    continue
            except (KeyError, TypeError):
                continue

        calculator_keys = get_calculator_keys(reaction_data)
        if not calculator_keys:
            continue

        median_calc_key = get_median_calculator_key(reaction_data)
        median_data = reaction_data.get(median_calc_key, {})
        bond_change_pct = float(median_data.get("max_bond_change", 0.0))
        substrate_disp_median = float(median_data.get("substrate_displacement", 0.0))

        anomalies = {
            "slab_conv": 0,
            "ads_conv": 0,
            "slab_move": 0,
            "ads_move": 0,
            "slab_seed": 0,
            "ads_seed": 0,
            "ads_eng_seed": 0,
            "adsorbate_migration": 0,
            "energy_anomaly": 0,
        }

        for calc_key in calculator_keys:
            calc_data = reaction_data.get(calc_key, {})

            if has_slab_calculations and "slab_steps" in calc_data:
                if calc_data["slab_steps"] == n_crit_relax:
                    anomalies["slab_conv"] += 1
            if "adslab_steps" in calc_data:
                if calc_data["adslab_steps"] == n_crit_relax:
                    anomalies["ads_conv"] += 1

            if has_slab_calculations and "slab_max_disp" in calc_data:
                if calc_data["slab_max_disp"] > disp_thrs:
                    anomalies["slab_move"] += 1

            if calc_key == median_calc_key and substrate_disp_median > disp_thrs:
                anomalies["ads_move"] += 1

        final_data = reaction_data.get("final", {})
        if final_data:
            if (
                has_slab_calculations
                and final_data.get("slab_seed_range", 0.0) > reproduction_thrs
            ):
                anomalies["slab_seed"] = 1
            if final_data.get("ads_seed_range", 0.0) > reproduction_thrs:
                anomalies["ads_seed"] = 1
            if final_data.get("ads_eng_seed_range", 0.0) > reproduction_thrs:
                anomalies["ads_eng_seed"] = 1

            try:
                median_energy = float(final_data["ads_eng_median"])
                ref_energy = float(reaction_data["reference"]["ads_eng"])
                if abs(median_energy - ref_energy) > energy_thrs:
                    anomalies["energy_anomaly"] = 1
            except (KeyError, TypeError, ValueError):
                pass

        migration_check_needed = (
            anomalies["slab_conv"] == 0
            and anomalies["ads_conv"] == 0
            and anomalies["slab_move"] == 0
            and anomalies["ads_move"] == 0
        )
        if migration_check_needed and bond_change_pct > bond_threshold_pct:
            anomalies["adsorbate_migration"] = 1

        if not has_slab_calculations:
            anomalies["slab_conv"] = 0
            anomalies["slab_move"] = 0
            anomalies["slab_seed"] = 0

        labels: list[str] = []
        if anomalies["slab_seed"] or anomalies["ads_seed"] or anomalies["ads_eng_seed"]:
            label = "reproduction_failure"
            if anomalies["slab_seed"]:
                labels.append("slab_seed")
            if anomalies["ads_seed"]:
                labels.append("ads_seed")
            if anomalies["ads_eng_seed"]:
                labels.append("ads_eng_seed")
        elif (
            anomalies["slab_conv"]
            or anomalies["ads_conv"]
            or anomalies["slab_move"]
            or anomalies["ads_move"]
        ):
            label = "unphysical_relaxation"
            if anomalies["slab_conv"]:
                labels.append("slab_conv")
            if anomalies["ads_conv"]:
                labels.append("ads_conv")
            if anomalies["slab_move"]:
                labels.append("slab_move")
            if anomalies["ads_move"]:
                labels.append("ads_move")
        elif anomalies["adsorbate_migration"]:
            label = "adsorbate_migration"
            labels.append("adsorbate_migration")
        elif anomalies["energy_anomaly"]:
            label = "energy_anomaly"
            labels.append("energy_anomaly")
        else:
            label = "normal"

        per_reaction[reaction] = {
            "dft_ads_eng": reaction_data.get("reference", {}).get("ads_eng"),
            "mlip_ads_eng_median": (
                final_data.get("ads_eng_median") if final_data else None
            ),
            "mlip_ads_eng_single": reaction_data.get("single_calculation", {}).get(
                "ads_eng"
            ),
            "label": label,
            "labels": labels,
            "details": anomalies,
        }

    return per_reaction


def detect_anomalies_from_result_json(
    result_json_path: str | Path,
    **kwargs,
) -> dict[str, dict[str, Any]]:
    result_json_path = Path(result_json_path)
    with result_json_path.open("r") as f:
        mlip_result = json.load(f)
    return detect_anomalies_from_result_dict(mlip_result, **kwargs)


def extract_adsorbate(reaction: str) -> str | None:
    if "->" not in reaction:
        return None
    product = reaction.split("->", 1)[1].strip()
    return product.removesuffix("*") or None


def filter_wide_predictions(
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


def run_summary_analysis(cfg: Config | None = None) -> Path | None:
    cfg = cfg or get_config()
    if cfg.analysis is None:
        return None

    if cfg.analysis.summary_workbook_path is None:
        raise ValueError("config.toml must define analysis.summary_workbook_path")
    if cfg.analysis.comparison_workbook_path is None:
        raise ValueError("config.toml must define analysis.comparison_workbook_path")
    if cfg.analysis.comparison_plot_path is None:
        raise ValueError("config.toml must define analysis.comparison_plot_path")
    if cfg.analysis.run_adsorption_analysis and cfg.analysis.calculating_path is None:
        raise ValueError(
            "config.toml must define analysis.calculating_path when "
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
