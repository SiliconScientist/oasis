from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


class DataGenerationConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    data_filepath: str
    surfep_path: str
    pure_element_path: str
    mathematica_path: str
    unrelaxed_models_path: str
    power_term: int | float
    dataset_module: str
    apply_ideal_lattice: bool = True
    data_slice: list[int] | None = None
    skip_indices: list[int] | None = None


class ModelFittingConfig(BaseModel):
    train_frac: float
    random_state: int
    cobyla_initial_guess: float
    cobyla_max_iter: int


class XGBoostSearchOptions(BaseModel):
    cv: int = 3
    n_iter: int = 30
    random_state: int = 4
    n_jobs: int = 1


class XGBoostConfig(BaseModel):
    base_params: dict[str, Any] = Field(default_factory=dict)
    search_space: dict[str, list[Any]] = Field(default_factory=dict)
    search_options: XGBoostSearchOptions = Field(
        default_factory=XGBoostSearchOptions
    )


class ExperimentConfig(BaseModel):
    low_theory_features: list[str]
    high_theory_features: list[str]
    adsorbate_map: list[list[str]]
    guest_list: list[str]
    site_list: list[str]
    host_elements: list[str]
    model_key: str
    target_feature: str
    params_output_dir: str = ""
    xgboost: XGBoostConfig = Field(default_factory=XGBoostConfig)


class Config(BaseModel):
    experiment_path: str
    data_generation: DataGenerationConfig
    model_fitting: ModelFittingConfig


def get_config(path: str = "config.toml") -> Config:
    with open(path, "rb") as f:
        cfg_data = tomllib.load(f)
    return Config(**cfg_data)


def get_experiment_config(path: str) -> ExperimentConfig:
    with open(path, "rb") as f:
        cfg_data = tomllib.load(f)
    return ExperimentConfig(**cfg_data)
