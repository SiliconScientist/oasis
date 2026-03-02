# anomaly_only.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Reuse CatBench helpers (so you don't have to copy their internals)
from catbench.utils.analysis_utils import get_calculator_keys, get_median_calculator_key

# Minimal periodic table symbols list (1-118).
# This is stable and doesn't need the web.
ELEMENT_SYMBOLS = {
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
}


def _split_alloy_token(token: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Split an alloy token like 'ptni' or 'ircd' or 'runi' into ('Pt','Ni') etc.
    Strategy: try all splits and validate both sides as element symbols.
    """
    t = token.strip()
    if not t:
        return None, None

    # Try every split point
    for i in range(1, len(t)):
        a_raw, b_raw = t[:i], t[i:]
        a = a_raw[0].upper() + a_raw[1:].lower()
        b = b_raw[0].upper() + b_raw[1:].lower()
        if a in ELEMENT_SYMBOLS and b in ELEMENT_SYMBOLS:
            return a, b

    return None, None


def parse_reaction_id(rxn_id: str) -> dict:
    """
    Parse CatBench-style keys like:
      'ol-runi_O'
      'sa-ircd_O'
    into metadata fields.
    """
    left, _, ads = rxn_id.partition("_")
    nsa, _, alloy = left.partition("-")

    host, dopant = _split_alloy_token(alloy) if alloy else (None, None)

    return {
        "id": rxn_id,
        "nsa_type": nsa or None,  # e.g. 'ol', 'ss', 'sa'
        "alloy": alloy or None,  # e.g. 'runi'
        "host": host,  # e.g. 'Ru'
        "dopant": dopant,  # e.g. 'Ni'
        "adsorbate": ads or None,  # e.g. 'O'
    }


def with_metadata(per_rxn: dict[str, dict]) -> list[dict]:
    rows = []
    for rxn_id, payload in per_rxn.items():
        meta = parse_reaction_id(rxn_id)
        rows.append({**meta, **payload})
    return rows


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
    mlip_json_path = "data/mlips/7net-omni/7net-omni_result.json"
    output_path = Path("data/mlips/7net-omni/7net-omni_processed_result.json")
    per_rxn = detect_anomalies_from_result_json(mlip_json_path)
    rows = with_metadata(per_rxn)
    output_path.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {len(rows)} labeled reactions -> {output_path}")


if __name__ == "__main__":
    main()
