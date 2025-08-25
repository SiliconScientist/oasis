from tomllib import load
from pydantic import BaseModel


class DataConfig(BaseModel):
    raw: str


class DataLoaderConfig(BaseModel):
    batch_size: int
    num_workers: int


class Config(BaseModel):
    seed: int
    data: DataConfig
    dataloader: DataLoaderConfig


def get_config():
    with open("config.toml", "rb") as f:
        cfg_data = load(f)
    return Config(**cfg_data)
