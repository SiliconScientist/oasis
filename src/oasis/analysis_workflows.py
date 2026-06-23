from __future__ import annotations

from pathlib import Path

import pandas as pd
from catbench.adsorption import AdsorptionAnalysis

from oasis.config import Config, get_config
from oasis.plot import mae_comparison_plot


def load_summary_dataframe(workbook_path: str | Path) -> pd.DataFrame:
    try:
        excel_file = pd.ExcelFile(workbook_path)
    except ImportError as exc:
        raise ImportError(
            "Reading Excel summary files requires openpyxl. Install it first."
        ) from exc

    sheet_name = "Summary" if "Summary" in excel_file.sheet_names else 0
    return pd.read_excel(workbook_path, sheet_name=sheet_name)


def run_summary_analysis(cfg: Config | None = None) -> Path | None:
    cfg = cfg or get_config()
    if cfg.analysis is None:
        return None

    if cfg.analysis.summary_workbook_path is None:
        raise ValueError("Config must define analysis.summary_workbook_path")
    if cfg.analysis.comparison_workbook_path is None:
        raise ValueError("Config must define analysis.comparison_workbook_path")
    if cfg.analysis.comparison_plot_path is None:
        raise ValueError("Config must define analysis.comparison_plot_path")
    if cfg.analysis.run_adsorption_analysis and cfg.analysis.calculating_path is None:
        raise ValueError(
            "Config must define analysis.calculating_path when "
            "run_adsorption_analysis is true"
        )

    summary_df = load_summary_dataframe(cfg.analysis.summary_workbook_path)
    comparison_df = load_summary_dataframe(cfg.analysis.comparison_workbook_path)
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
