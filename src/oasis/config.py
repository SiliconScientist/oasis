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
    use_ridge: bool
    use_residual: bool
    plot_moe_baseline: bool = False
    plot_moe_schnet: bool = False
    adsorbate: Optional[str] = None
    anomaly_label: Optional[str] = None
    reaction_contains: Optional[List[str]] = None


class MoeGraphConfig(BaseModel):
    cutoff: float = 6.0
    max_neighbors: Optional[int] = None
    n_rbf: int = 32


class MoeBaselineGateConfig(BaseModel):
    hidden_dims: List[int] = Field(default_factory=lambda: [32, 32])
    dropout: float = 0.0


class MoeSchNetGateConfig(BaseModel):
    structure_hidden_dim: int = 64
    n_interactions: int = 3
    gate_hidden_dims: List[int] = Field(default_factory=lambda: [64, 32])
    dropout: float = 0.0


class MoeTrainConfig(BaseModel):
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 10
    val_fraction: float = 0.2
    checkpoint_dir: Optional[Path] = None
    device: str = "cpu"


class MoeConfig(BaseModel):
    use_structure_features: bool = True
    use_mlip_features: bool = True
    graph: MoeGraphConfig = Field(default_factory=MoeGraphConfig)
    baseline_gate: MoeBaselineGateConfig = Field(
        default_factory=MoeBaselineGateConfig
    )
    schnet_gate: MoeSchNetGateConfig = Field(default_factory=MoeSchNetGateConfig)
    train: MoeTrainConfig = Field(default_factory=MoeTrainConfig)


class Config(BaseModel):
    seed: Optional[int] = None
    dev_run: Optional[bool] = None
    train: Optional[bool] = None
    evaluate: Optional[bool] = None
    ingest: IngestConfig
    mlip: MLIPConfig
    analysis: Optional[AnalysisConfig] = None
    plot: Optional[PlotConfig] = None
    moe: Optional[MoeConfig] = None

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
