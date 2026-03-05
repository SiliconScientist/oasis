# anomaly_only.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

# Reuse CatBench helpers (so you don't have to copy their internals)
from catbench.utils.analysis_utils import get_calculator_keys, get_median_calculator_key


def detect_anomalies_from_result_dict(
    mlip_result: Dict[str, Any],
    *,
    disp_thrs: float = 1.0,
    energy_thrs: float = 2.0,
    reproduction_thrs: float = 0.2,
    bond_length_change_threshold: float = 0.2,  # fraction (0.2 => 20%)
    energy_cutoff: Optional[float] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Returns:
      per_reaction[reaction] = {
        "dft_ads_eng": float|None,
        "mlip_ads_eng_median": float|None,
        "mlip_ads_eng_single": float|None,
        "label": str,  # normal | energy_anomaly | adsorbate_migration | unphysical_relaxation | reproduction_failure
        "labels": list[str],  # extra detail; for unphysical includes issues; for repro includes which seed issue(s)
        "details": dict[str, int],  # raw anomaly flags/counters copied from analysis.py logic
      }
    """

    # --- required by analysis.py logic ---
    calc_settings = mlip_result.get("calculation_settings", {})
    if "chemical_bond_cutoff" not in calc_settings:
        raise ValueError(
            "chemical_bond_cutoff not found in calculation_settings. "
            "This detector expects CatBench result.json produced with that setting."
        )

    n_crit_relax = calc_settings.get("n_crit_relax", 999)
    bond_threshold_pct = (
        bond_length_change_threshold * 100.0
    )  # match analysis.py behavior

    # Determine whether this MLIP has slab calculations (absolute-energy mode)
    has_slab_calculations: Optional[bool] = None
    for rxn, rxn_data in mlip_result.items():
        if rxn == "calculation_settings":
            continue
        calculator_keys = get_calculator_keys(rxn_data)
        if calculator_keys:
            has_slab_calculations = "slab_max_disp" in rxn_data[calculator_keys[0]]
        break
    if has_slab_calculations is None:
        has_slab_calculations = False

    per_reaction: Dict[str, Dict[str, Any]] = {}

    for reaction, rxn_data in mlip_result.items():
        if reaction == "calculation_settings":
            continue

        # Optional energy cutoff filter (copied behavior)
        if energy_cutoff is not None:
            try:
                ref_e = rxn_data["reference"]["ads_eng"]
                if ref_e > energy_cutoff:
                    continue
            except (KeyError, TypeError):
                continue

        calculator_keys = get_calculator_keys(rxn_data)
        if not calculator_keys:
            continue

        # Median calculator key & median-derived quantities
        median_calc_key = get_median_calculator_key(rxn_data)
        median_data = rxn_data.get(median_calc_key, {})
        bond_change_pct = float(median_data.get("max_bond_change", 0.0))
        substrate_disp_median = float(median_data.get("substrate_displacement", 0.0))

        # Initialize anomaly counters/flags (same keys as analysis.py)
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

        # Convergence + displacement checks over calculators
        for calc_key in calculator_keys:
            calc_data = rxn_data.get(calc_key, {})

            # Convergence anomalies
            if has_slab_calculations and "slab_steps" in calc_data:
                if calc_data["slab_steps"] == n_crit_relax:
                    anomalies["slab_conv"] += 1
            if "adslab_steps" in calc_data:
                if calc_data["adslab_steps"] == n_crit_relax:
                    anomalies["ads_conv"] += 1

            # Displacement anomalies
            if has_slab_calculations and "slab_max_disp" in calc_data:
                if calc_data["slab_max_disp"] > disp_thrs:
                    anomalies["slab_move"] += 1

            # Substrate displacement only for median calculator
            if calc_key == median_calc_key:
                if substrate_disp_median > disp_thrs:
                    anomalies["ads_move"] += 1

        # Seed range checks + energy anomaly check
        final_data = rxn_data.get("final", {})
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

            # Energy anomaly
            try:
                median_energy = float(final_data["ads_eng_median"])
                ref_energy = float(rxn_data["reference"]["ads_eng"])
                if abs(median_energy - ref_energy) > energy_thrs:
                    anomalies["energy_anomaly"] = 1
            except (KeyError, TypeError, ValueError):
                pass

        # Adsorbate migration check only if not already "unphysical"
        migration_check_needed = (
            anomalies["slab_conv"] == 0
            and anomalies["ads_conv"] == 0
            and anomalies["slab_move"] == 0
            and anomalies["ads_move"] == 0
        )
        if migration_check_needed and bond_change_pct > bond_threshold_pct:
            anomalies["adsorbate_migration"] = 1

        # OC20-style mode: strip slab-related anomalies if no slab calcs
        if not has_slab_calculations:
            anomalies["slab_conv"] = 0
            anomalies["slab_move"] = 0
            anomalies["slab_seed"] = 0

        # Priority classification (identical ordering)
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

        # Output values you said you want
        dft_ads_eng = rxn_data.get("reference", {}).get("ads_eng")
        mlip_ads_eng_median = final_data.get("ads_eng_median") if final_data else None
        mlip_ads_eng_single = rxn_data.get("single_calculation", {}).get("ads_eng")

        per_reaction[reaction] = {
            "dft_ads_eng": dft_ads_eng,
            "mlip_ads_eng_median": mlip_ads_eng_median,
            "mlip_ads_eng_single": mlip_ads_eng_single,
            "label": label,
            "labels": labels,
            "details": anomalies,
        }

    return per_reaction


def detect_anomalies_from_result_json(
    result_json_path: str | Path,
    **kwargs,
) -> Dict[str, Dict[str, Any]]:
    result_json_path = Path(result_json_path)
    with result_json_path.open("r") as f:
        mlip_result = json.load(f)
    return detect_anomalies_from_result_dict(mlip_result, **kwargs)


def main():
    mlip_results_dir = Path("data/mlips/khlohc")
    processed_count = 0

    for model_dir in mlip_results_dir.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        result_path = model_dir / f"{model_name}_result.json"

        # Fallback if result file naming differs from directory name.
        if not result_path.exists():
            result_candidates = sorted(
                p
                for p in model_dir.glob("*_result.json")
                if not p.name.endswith("_processed_result.json")
            )
            if not result_candidates:
                print(f"Skipping {model_name}: no result json found")
                continue
            result_path = result_candidates[0]

        output_path = result_path.with_name(
            result_path.name.replace("_result.json", "_processed_result.json")
        )

        per_rxn = detect_anomalies_from_result_json(result_path)
        rows = [
            {"reaction": reaction, **payload} for reaction, payload in per_rxn.items()
        ]
        output_path.write_text(json.dumps(rows, indent=2))
        print(f"Wrote {len(rows)} labeled reactions -> {output_path}")
        processed_count += 1

    print(f"Processed {processed_count} MLIP result directories in {mlip_results_dir}")


if __name__ == "__main__":
    main()
