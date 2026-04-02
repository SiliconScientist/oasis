import pandas as pd
from catbench.adsorption import AdsorptionAnalysis

from oasis.config import get_config
from oasis.plot import mae_comparison_plot


def _load_summary_dataframe(workbook_path) -> pd.DataFrame:
    try:
        excel_file = pd.ExcelFile(workbook_path)
    except ImportError as exc:
        raise ImportError(
            "Reading Excel summary files requires openpyxl. Install it first."
        ) from exc

    sheet_name = "Summary" if "Summary" in excel_file.sheet_names else 0
    return pd.read_excel(workbook_path, sheet_name=sheet_name)


cfg = get_config()
if cfg.analysis is None:
    raise ValueError("config.toml must define an [analysis] section")
if cfg.analysis.summary_workbook_path is None:
    raise ValueError("config.toml must define analysis.summary_workbook_path")
if cfg.analysis.comparison_workbook_path is None:
    raise ValueError("config.toml must define analysis.comparison_workbook_path")
if cfg.analysis.comparison_plot_path is None:
    raise ValueError("config.toml must define analysis.comparison_plot_path")
if cfg.analysis.run_adsorption_analysis and cfg.analysis.calculating_path is None:
    raise ValueError(
        "config.toml must define analysis.calculating_path when run_adsorption_analysis is true"
    )

summary_workbook_path = cfg.analysis.summary_workbook_path
summary_df = _load_summary_dataframe(summary_workbook_path)
comparison_workbook_path = cfg.analysis.comparison_workbook_path
comparison_df = _load_summary_dataframe(comparison_workbook_path)
comparison_plot_path = mae_comparison_plot(
    comparison_df=comparison_df,
    summary_df=summary_df,
    output_path=cfg.analysis.comparison_plot_path,
)

if cfg.analysis.run_adsorption_analysis:
    analysis = AdsorptionAnalysis(calculating_path=str(cfg.analysis.calculating_path))
    analysis.analysis()
    analysis.threshold_sensitivity_analysis()
