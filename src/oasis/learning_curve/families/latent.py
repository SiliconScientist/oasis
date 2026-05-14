from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from oasis.config import LatentModelConfig
from oasis.learning_curve.execution import _assert_train_test_payload, sweep_results_frame
from oasis.sweep import SweepRunnerPayload

if TYPE_CHECKING:
    from latent.config import ExperimentConfig


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


@dataclass(frozen=True, slots=True)
class LatentSweepRunner:
    exp_cfg: ExperimentConfig
    cobyla_initial_guess: float = 0.1
    cobyla_max_iter: int = 100

    def run(self, payload: SweepRunnerPayload) -> pd.DataFrame:
        from latent.model import LatentVariableModel
        from latent.train import train_model
        from sklearn.metrics import mean_squared_error

        rmses_by_size: dict[int, list[float]] = {}
        for split in _assert_train_test_payload(payload):
            latent_df: pd.DataFrame = split.dataset.auxiliary_views["latent"]
            train_df = latent_df.iloc[split.train_idx]
            test_df = latent_df.iloc[split.test_idx]
            untrained = LatentVariableModel.from_config(self.exp_cfg, train_df)
            trained = train_model(
                train_df,
                untrained,
                cobyla_initial_guess=self.cobyla_initial_guess,
                cobyla_max_iter=self.cobyla_max_iter,
            )
            x_test, y_test = trained.architecture.linearize_data(
                test_df, trained.train_params
            )
            preds = trained.estimator.predict(x_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            rmses_by_size.setdefault(split.sweep_size, []).append(rmse)
        return sweep_results_frame(rmses_by_size)
