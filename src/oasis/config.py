from tomllib import load
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path


class ProcessingConfig(BaseModel):
    root: Path


class MLIPInterpretersConfig(BaseModel):
    mace: Path
    mattersim: Path
    orb_v3: Path
    sevennet: Path
    uma_s1p1: Path


class MLIPModelsConfig(BaseModel):
    enabled: List[str]


class MLIPConfig(BaseModel):
    run_tag: Optional[str] = None
    interpreters: MLIPInterpretersConfig
    models: MLIPModelsConfig


class PlotConfig(BaseModel):
    output_dir: Path
    min_train: int = 5
    max_train: int = 10
    trim: bool = True
    use_ridge: bool = True
    use_kernel_ridge: bool = True
    use_lasso: bool = True
    use_elastic_net: bool = True
    use_residual: bool = True
    use_linearization: bool = True
    use_gnn: bool = True


class Config(BaseModel):
    seed: int
    dev_run: bool
    processing: ProcessingConfig
    mlip: MLIPConfig
    plot: PlotConfig


def get_config():
    with open("config.toml", "rb") as f:
        cfg_data = load(f)
    return Config(**cfg_data)
