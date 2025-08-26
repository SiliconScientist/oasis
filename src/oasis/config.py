from tomllib import load
from pydantic import BaseModel


class DataConfig(BaseModel):
    systems: str
    surfaces: str


class DataLoaderConfig(BaseModel):
    batch_size: int
    num_workers: int


class ModelConfig(BaseModel):
    checkpoint: str


class TrainerConfig(BaseModel):
    device: str


class Config(BaseModel):
    seed: int
    data: DataConfig
    dataloader: DataLoaderConfig
    model: ModelConfig
    trainer: TrainerConfig


def get_config():
    with open("config.toml", "rb") as f:
        cfg_data = load(f)
    return Config(**cfg_data)
