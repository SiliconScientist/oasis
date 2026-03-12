from catbench.adsorption import AdsorptionAnalysis

analysis = AdsorptionAnalysis(calculating_path="data/mlips/mamun")

analysis.analysis()
analysis.threshold_sensitivity_analysis()
