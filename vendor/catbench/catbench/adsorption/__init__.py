"""
Adsorption energy benchmarking module for CatBench.

Keep imports lightweight so calculation-only workflows do not require
analysis or preprocessing dependencies at import time.
"""

from catbench.adsorption.calculation.calculation import AdsorptionCalculation

__all__ = ["AdsorptionCalculation", "AdsorptionAnalysis", "cathub_preprocessing", "vasp_preprocessing", "process_output"]


def __getattr__(name):
    if name == "AdsorptionAnalysis":
        from catbench.adsorption.analysis.analysis import AdsorptionAnalysis

        return AdsorptionAnalysis
    if name == "cathub_preprocessing":
        from catbench.adsorption.data.cathub import cathub_preprocessing

        return cathub_preprocessing
    if name in {"vasp_preprocessing", "process_output"}:
        from catbench.adsorption.data.vasp import process_output, vasp_preprocessing

        return {"vasp_preprocessing": vasp_preprocessing, "process_output": process_output}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
