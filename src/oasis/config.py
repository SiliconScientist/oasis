from tomllib import load
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path


class IngestConfig(BaseModel):
    source: Path
    catbench_folder: Optional[Path] = None


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

    def init_paths(self):
        catbench_folder = (
            self.ingest.source.parent / f"{self.ingest.source.name}_catbench"
        )
        self.ingest.catbench_folder = catbench_folder


def get_config():
    with open("config.toml", "rb") as f:
        cfg_data = load(f)
        cfg = Config(**cfg_data)
        cfg.init_paths()
    return cfg
