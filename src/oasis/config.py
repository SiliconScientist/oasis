from tomllib import load
from pydantic import BaseModel
from typing import List
from pathlib import Path


class IngestConfig(BaseModel):
    source: Path


class MLIPInterpretersConfig(BaseModel):
    mace: Path
    mattersim: Path


class MLIPModelPathConfig(BaseModel):
    mace: Path
    mattersim: Path


class MLIPModelsConfig(BaseModel):
    enabled: List[str]


class MLIPConfig(BaseModel):
    dev_n: int
    dev_run: bool
    interpreters: MLIPInterpretersConfig
    model_paths: MLIPModelPathConfig
    models: MLIPModelsConfig


class Config(BaseModel):
    ingest: IngestConfig
    mlip: MLIPConfig


def get_config():
    with open("config.toml", "rb") as f:
        cfg_data = load(f)
    return Config(**cfg_data)
