from tomllib import load
from pydantic import BaseModel
from pathlib import Path


class DataConfig(BaseModel):
    ideal_systems: Path
    relaxed_systems: Path
    relaxed_slabs: Path


class ProcessingConfig(BaseModel):
    constrained_tags: list[int]
    adsorbate_tag: int


class DataLoaderConfig(BaseModel):
    batch_size: int
    num_workers: int


class MaceConfig(BaseModel):
    checkpoint: Path
    head: str


class OrbConfig(BaseModel):
    precision: str  # e.g. "float32-high"


class MatterSimConfig(BaseModel):
    checkpoint: Path


class UmaConfig(BaseModel):
    checkpoint: Path
    task: str


class ModelsConfig(BaseModel):
    device: str
    dtype: str
    mace: MaceConfig
    orb: OrbConfig
    mattersim: MatterSimConfig
    uma: UmaConfig


class TrainerConfig(BaseModel):
    device: str


class Config(BaseModel):
    seed: int
    processing: ProcessingConfig
    data: DataConfig
    dataloader: DataLoaderConfig
    models: ModelsConfig
    trainer: TrainerConfig


def get_config():
    with open("config.toml", "rb") as f:
        cfg_data = load(f)
    return Config(**cfg_data)
