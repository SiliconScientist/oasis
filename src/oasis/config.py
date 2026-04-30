from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib


class StoichConfig(BaseModel):
    elements: list[str]
    basis_species: list[str]
    basis_composition: Dict[str, Dict[str, int]]


class IngestConfig(BaseModel):
    source: Path
    dataset_name: str
    catbench_folder: Optional[Path] = None
    stoich: StoichConfig


class MLIPModelsConfig(BaseModel):
    enabled: List[str]


class RootstockModelConfig(BaseModel):
    model: str
    mlip_name: str
    checkpoint: Optional[str] = None
    output_model: Optional[str] = None
    model_version: Optional[str] = None
    metadata: Dict[str, str] = Field(default_factory=dict)


class RootstockConfig(BaseModel):
    root: Path
    python: Optional[Path] = None
    models: Dict[str, RootstockModelConfig]


class MLIPConfig(BaseModel):
    dev_n: int
    dev_run: bool
    dataset: Optional[str] = None
    optimizer: str = "LBFGS"
    models: MLIPModelsConfig
    rootstock: RootstockConfig


class AnalysisConfig(BaseModel):
    calculating_path: Optional[Path] = None
    summary_workbook_path: Optional[Path] = None
    comparison_workbook_path: Optional[Path] = None
    comparison_plot_path: Optional[Path] = None
    run_adsorption_analysis: bool = False
    base_dir: Path
    out_dir: Path
    prefixes: List[str]


class PlotConfig(BaseModel):
    output_dir: Path
    min_train: int
    max_train: int
    n_repeats: int
    trim: bool
    use_ridge: bool
    use_kernel_ridge: bool
    use_lasso: bool
    use_elastic_net: bool
    use_residual: bool
    use_linearization: bool
    use_gnn: bool
    adsorbate: Optional[str] = None
    anomaly_label: Optional[str] = None
    reaction_contains: Optional[List[str]] = None


class Config(BaseModel):
    seed: Optional[int] = None
    dev_run: Optional[bool] = None
    train: Optional[bool] = None
    evaluate: Optional[bool] = None
    ingest: IngestConfig
    mlip: MLIPConfig
    analysis: Optional[AnalysisConfig] = None
    plot: Optional[PlotConfig] = None

    def init_paths(self):
        catbench_folder = (
            self.ingest.source.parent / f"{self.ingest.source.name}_catbench"
        )
        self.ingest.catbench_folder = catbench_folder


def get_config():
    with open("config.toml", "rb") as f:
        cfg_data = tomllib.load(f)
        cfg = Config(**cfg_data)
        cfg.init_paths()
    return cfg
