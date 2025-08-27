from tomllib import load
from pydantic import BaseModel


class DatasetConfig(BaseModel):
    slabs: str
    ads_slabs: str


class DataConfig(BaseModel):
    raw: DatasetConfig
    raw_relaxed: DatasetConfig
    processed: str


class ProcessingConfig(BaseModel):
    constrained_tags: list[int]


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
