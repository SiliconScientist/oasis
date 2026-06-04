from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from catbench.utils.analysis_utils import (
    get_calculator_keys,
    get_median_calculator_key,
)


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
    with result_json_path.open("r", encoding="utf-8") as f:
        mlip_result = json.load(f)
    return detect_anomalies_from_result_dict(mlip_result, **kwargs)


def extract_adsorbate(reaction: str) -> str | None:
    if "->" not in reaction:
        return None
    product = reaction.split("->", 1)[1].strip()
    return product.removesuffix("*") or None
