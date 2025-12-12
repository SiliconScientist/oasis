from pathlib import Path
from tomllib import load

from pydantic import BaseModel, ConfigDict


class ProcessingConfig(BaseModel):
    root: Path


class PlotConfig(BaseModel):
    output_dir: Path
    min_train: int = 5
    max_train: int = 10
    n_repeats: int = 50
    trim: bool = True
    use_ridge: bool = True
    use_kernel_ridge: bool = True
    use_lasso: bool = True
    use_elastic_net: bool = True
    use_residual: bool = True
    use_linearization: bool = True
    use_gnn: bool = True


class TuningConfig(BaseModel):
    enabled: bool = False
    p_max: float = 0.25
    N0: float = 50.0
    n_max: int = 200
    n_trials: int = 30


class Config(BaseModel):
    model_config = ConfigDict(extra="ignore")
    seed: int
    dev_run: bool
    processing: ProcessingConfig
    plot: PlotConfig
    tuning: TuningConfig = TuningConfig()


def get_config():
    with open("config.toml", "rb") as f:
        cfg_data = load(f)
    return Config(**cfg_data)
