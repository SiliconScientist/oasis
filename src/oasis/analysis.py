from catbench.adsorption import AdsorptionAnalysis

from oasis.config import get_config

cfg = get_config()
if cfg.analysis is None or cfg.analysis.calculating_path is None:
    raise ValueError("config.toml must define analysis.calculating_path")

analysis = AdsorptionAnalysis(calculating_path=str(cfg.analysis.calculating_path))

analysis.analysis()
analysis.threshold_sensitivity_analysis()
