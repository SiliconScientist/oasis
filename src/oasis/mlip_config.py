from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class StoichConfig(BaseModel):
    elements: list[str]
    basis_species: list[str]
    basis_composition: Dict[str, Dict[str, int]]


class IngestConfig(BaseModel):
    source: Path
    dataset_name: str
    catbench_folder: Optional[Path] = None
    stoich: StoichConfig


class MLIPModelsConfig(BaseModel):
    enabled: List[str]


class RootstockModelConfig(BaseModel):
    model: str
    mlip_name: str
    checkpoint: Optional[str] = None
    output_model: Optional[str] = None
    model_version: Optional[str] = None
    metadata: Dict[str, str] = Field(default_factory=dict)


class RootstockConfig(BaseModel):
    root: Path
    python: Optional[Path] = None
    models: Dict[str, RootstockModelConfig]


class MLIPConfig(BaseModel):
    dev_n: int
    dev_run: bool
    dataset: Optional[str] = None
    optimizer: str = "LBFGS"
    models: MLIPModelsConfig
    rootstock: RootstockConfig


def raw_dataset_path(raw_dataset_filename: str) -> Path:
    return Path("data/raw_data") / raw_dataset_filename


def mlip_results_dir(mlip_run_dirname: str) -> Path:
    return Path("data/mlips") / mlip_run_dirname


def default_catbench_folder(source: Path) -> Path:
    return source.parent / f"{source.name}_catbench"


def fill_mlip_dataset_path(
    mlip_config: MLIPConfig,
    *,
    dataset_path: Path | None,
) -> None:
    if mlip_config.dataset is None and dataset_path is not None:
        mlip_config.dataset = str(dataset_path)
