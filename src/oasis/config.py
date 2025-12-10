from tomllib import load
from pydantic import BaseModel
from pathlib import Path


class ProcessingConfig(BaseModel):
    root: Path


class PlotConfig(BaseModel):
    output_dir: Path
    use_ridge: bool = True
    use_kernel_ridge: bool = True
    use_lasso: bool = True
    use_elastic_net: bool = True
    use_residual: bool = True
    use_residual_trimmed: bool = True
    use_linearization: bool = True
    use_gnn: bool = True


class Config(BaseModel):
    seed: int
    dev_run: bool
    processing: ProcessingConfig
    plot: PlotConfig


def get_config():
    with open("config.toml", "rb") as f:
        cfg_data = load(f)
    return Config(**cfg_data)
