from pathlib import Path
from tomllib import load

from pydantic import BaseModel, ConfigDict


class ProcessingConfig(BaseModel):
    root: Path


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


class TuningConfig(BaseModel):
    enabled: bool
    p_max: float
    N0: float
    n_max: int
    n_trials: int


class Config(BaseModel):
    model_config = ConfigDict(extra="ignore")
    seed: int
    dev_run: bool
    processing: ProcessingConfig
    plot: PlotConfig
    tuning: TuningConfig


def get_config():
    with open("config.toml", "rb") as f:
        cfg_data = load(f)
    return Config(**cfg_data)
