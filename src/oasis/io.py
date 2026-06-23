from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from oasis.ingest.site_constraints import extract_adsorbed_atom

if TYPE_CHECKING:
    from ase import Atoms
    from oasis.config import Config

def load_sample_atoms_for_wide_df(
    wide_df: pl.DataFrame,
    cfg: Config,
) -> list[Atoms]:
    """
    Load adsorbed ASE Atoms objects aligned to the rows of ``wide_df``.
    """
    dataset_path = cfg.resolved_dataset_path
    if not dataset_path:
        raise ValueError("No dataset path is configured.")

    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"MLIP dataset JSON not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not isinstance(dataset, dict):
        raise TypeError(
            f"Expected dataset JSON top-level to be an object/dict, got {type(dataset).__name__}"
        )

    reactions = wide_df.get_column("reaction").to_list()
    missing = [reaction for reaction in reactions if reaction not in dataset]
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(
            f"{len(missing)} reactions from wide_df were not found in {path}: {preview}"
        )

    return [extract_adsorbed_atom(dataset[reaction], reaction) for reaction in reactions]
