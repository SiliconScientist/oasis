from __future__ import annotations

import numpy as np
import pandas as pd

from oasis.config import LatentModelConfig


def _align_df_to_sample_ids(
    df: pd.DataFrame,
    sample_ids: np.ndarray,
    reaction_column: str = "reaction",
) -> pd.DataFrame:
    """Filter and reorder df rows to match the given sample_ids."""
    if reaction_column not in df.columns:
        return df
    return df.set_index(reaction_column).loc[sample_ids].reset_index()


def load_latent_df(
    latent_cfg: LatentModelConfig,
    sample_ids: np.ndarray,
    *,
    reaction_column: str = "reaction",
) -> pd.DataFrame:
    """Load the vendor latent DataFrame and align it to the oasis sample_ids."""
    from latent.config import get_config, get_experiment_config
    from latent.data import make_data

    cfg = get_config(str(latent_cfg.experiment_config_path))
    exp_cfg = get_experiment_config(cfg.experiment_path)
    df = make_data(cfg, exp_cfg)
    return _align_df_to_sample_ids(df, sample_ids, reaction_column)
