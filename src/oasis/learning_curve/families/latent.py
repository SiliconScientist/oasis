from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from oasis.config import LatentModelConfig
from oasis.learning_curve.execution import (
    _assert_train_test_payload,
    _measure_duration_s,
    timed_sweep_results_frame,
)
from oasis.sweep import SweepRunnerPayload


def _align_df_to_sample_ids(
    df: pd.DataFrame,
    sample_ids: np.ndarray,
    reaction_column: str = "reaction",
) -> pd.DataFrame:
    if reaction_column not in df.columns:
        return df
    return df.set_index(reaction_column).loc[sample_ids].reset_index()


def load_latent_df(
    latent_cfg: LatentModelConfig,
    sample_ids: np.ndarray,
) -> pd.DataFrame:
    df = pd.read_csv(latent_cfg.csv_path)
    return _align_df_to_sample_ids(df, sample_ids, reaction_column="equation")


def _load_latent_vendor_modules(vendor_dir: Path) -> tuple[Any, Any]:
    """Load LatentVariableModel and train_model from vendor files via sys.modules injection."""
    # latent.config — load via importlib so it resolves as 'latent.config' in sys.modules
    if "latent.config" not in sys.modules:
        latent_pkg = types.ModuleType("latent")
        latent_pkg.__path__ = [str(vendor_dir)]
        latent_pkg.__package__ = "latent"
        sys.modules.setdefault("latent", latent_pkg)

        config_spec = importlib.util.spec_from_file_location(
            "latent.config", vendor_dir / "config.py"
        )
        config_mod = importlib.util.module_from_spec(config_spec)
        sys.modules["latent.config"] = config_mod
        config_spec.loader.exec_module(config_mod)

        ns = vars(config_mod)
        for model in (
            config_mod.XGBoostSearchOptions,
            config_mod.XGBoostConfig,
            config_mod.ExperimentConfig,
        ):
            model.model_rebuild(_types_namespace=ns)

    # latent.data — stub with the one pure function model.py needs
    if "latent.data" not in sys.modules:
        def get_zippable_arrays(array1, array2):
            a1, a2 = [], []
            for x in array1:
                for y in array2:
                    a1.append(x)
                    a2.append(y)
            return a1, a2

        data_stub = types.ModuleType("latent.data")
        data_stub.get_zippable_arrays = get_zippable_arrays
        sys.modules["latent.data"] = data_stub

    if "latent.model" not in sys.modules:
        model_spec = importlib.util.spec_from_file_location(
            "latent.model", vendor_dir / "model.py"
        )
        model_mod = importlib.util.module_from_spec(model_spec)
        sys.modules["latent.model"] = model_mod
        model_spec.loader.exec_module(model_mod)

    if "latent.train" not in sys.modules:
        train_spec = importlib.util.spec_from_file_location(
            "latent.train", vendor_dir / "train.py"
        )
        train_mod = importlib.util.module_from_spec(train_spec)
        sys.modules["latent.train"] = train_mod
        train_spec.loader.exec_module(train_mod)

    LatentVariableModel = sys.modules["latent.model"].LatentVariableModel
    train_model = sys.modules["latent.train"].train_model
    return LatentVariableModel, train_model


@dataclass(frozen=True, slots=True)
class LatentSweepRunner:
    exp_cfg: Any
    vendor_dir: Path
    cobyla_initial_guess: float = 0.1
    cobyla_max_iter: int = 100

    def run(self, payload: SweepRunnerPayload) -> pd.DataFrame:
        from sklearn.metrics import mean_squared_error

        LatentVariableModel, train_model = _load_latent_vendor_modules(self.vendor_dir)

        rmses_by_size: dict[int, list[float]] = {}
        fit_times_by_size: dict[int, list[float]] = {}
        for split in _assert_train_test_payload(payload):
            latent_df: pd.DataFrame = split.dataset.auxiliary_views["latent"]
            train_df = latent_df.iloc[split.train_idx]
            test_df = latent_df.iloc[split.test_idx]
            untrained = LatentVariableModel.from_config(self.exp_cfg, train_df)
            trained = None

            def fit_latent_model() -> None:
                nonlocal trained
                trained = train_model(
                    train_df,
                    untrained,
                    cobyla_initial_guess=self.cobyla_initial_guess,
                    cobyla_max_iter=self.cobyla_max_iter,
                )

            fit_time_s = _measure_duration_s(fit_latent_model)
            assert trained is not None
            x_test, y_test = trained.architecture.linearize_data(
                test_df, trained.train_params
            )
            preds = trained.estimator.predict(x_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            rmses_by_size.setdefault(split.sweep_size, []).append(rmse)
            fit_times_by_size.setdefault(split.sweep_size, []).append(fit_time_s)
        return timed_sweep_results_frame(rmses_by_size, fit_times_by_size)
