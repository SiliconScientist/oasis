from tomllib import load
from pydantic import BaseModel
from pathlib import Path


class DatasetConfig(BaseModel):
    slabs: Path
    ads_slabs: Path


class DataConfig(BaseModel):
    raw: DatasetConfig
    raw_relaxed: DatasetConfig
    holdout: Path
    test: Path


class ProcessingConfig(BaseModel):
    constrained_tags: list[int]
    adsorbate_tag: int


class DataLoaderConfig(BaseModel):
    batch_size: int
    num_workers: int


class ModelConfig(BaseModel):
    checkpoint: Path


class TrainerConfig(BaseModel):
    device: str


class Config(BaseModel):
    seed: int
    processing: ProcessingConfig
    data: DataConfig
    dataloader: DataLoaderConfig
    model: ModelConfig
    trainer: TrainerConfig


def get_config():
    with open("config.toml", "rb") as f:
        cfg_data = load(f)
    return Config(**cfg_data)
