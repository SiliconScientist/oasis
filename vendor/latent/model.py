import copy
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from latent.config import ExperimentConfig
from latent.data import get_zippable_arrays


@dataclass
class LatentVariableModel:
    model_type: object
    features: list[str]
    block_keys: list[tuple[str, str]]
    host_elements: list[str]
    target_column: str
    block_columns: tuple[str, str] = ("guest_name", "site_name")

    @classmethod
    def from_config(cls, cfg: ExperimentConfig, df: pd.DataFrame | None = None) -> "LatentVariableModel":
        if df is not None:
            host_elements = [el for el in cfg.host_elements if f"constant_{el}" in df.columns]
            dropped = set(cfg.host_elements) - set(host_elements)
            if dropped:
                print(f"Warning: {dropped} in host_elements but not present as binding atoms in data; excluding from model.")
        else:
            host_elements = cfg.host_elements
        return cls(
            model_type=_build_model_type(cfg),
            features=cfg.high_theory_features,
            block_keys=_build_block_keys(cfg.guest_list, cfg.site_list),
            host_elements=host_elements,
            target_column=cfg.target_feature,
        )

    def get_fitting_features(self) -> list[str]:
        host_columns = [
            column
            for host_element in self.host_elements
            for column in (
                [f"{feature}_{host_element}" for feature in self.features]
                + [f"constant_{host_element}"]
            )
        ]
        guest_columns = [
            f"guest_constant_{guest}_{site}" for guest, site in self.block_keys
        ]
        return host_columns + guest_columns

    def get_block_mask(self, data, block_key: tuple[str, str]):
        guest_column, site_column = self.block_columns
        guest, site = block_key
        return (data[guest_column] == guest) & (data[site_column] == site)

    def get_parameter_matrix(self, guest_fitting_parameters):
        num_block_types = len(self.block_keys)
        parameter_matrix = np.reshape(
            guest_fitting_parameters,
            (num_block_types, len(self.features)),
        )
        return np.concatenate(
            (parameter_matrix, np.ones((num_block_types, 1))),
            axis=1,
        )

    def build_scale_vector(self, block_parameters):
        host_scale = np.tile(block_parameters, len(self.host_elements))
        block_scale = np.ones(len(self.block_keys))
        return np.concatenate((host_scale, block_scale))

    def linearize_data(
        self,
        data,
        guest_fitting_parameters,
    ):
        fitting_features = self.get_fitting_features()
        guest_parameter_matrix = self.get_parameter_matrix(guest_fitting_parameters)
        x_df_list = []
        y_df_list = []
        for guest_parameters, block_key in zip(
            guest_parameter_matrix,
            self.block_keys,
        ):
            df_subset = data.loc[self.get_block_mask(data, block_key)]
            scale_vector = self.build_scale_vector(guest_parameters)
            guest_data = df_subset[fitting_features]
            linearized_guest_data = guest_data.mul(scale_vector, axis=1)
            guest_target = df_subset[self.target_column]
            x_df_list.append(linearized_guest_data)
            y_df_list.append(guest_target)
        x = pd.concat(x_df_list)
        y = pd.concat(y_df_list)
        return x, y


@dataclass
class TrainedModel:
    estimator: object
    architecture: LatentVariableModel
    train_params: np.ndarray

    def predict(self, x):
        return self.estimator.predict(x)


def _build_model_type(cfg: ExperimentConfig):
    if cfg.model_key == "linear":
        return LinearRegression(fit_intercept=False)
    if cfg.model_key == "xgboost":
        xgb_config = cfg.xgboost
        return [
            XGBRegressor(**xgb_config.base_params),
            xgb_config.search_space,
            xgb_config.search_options.model_dump(),
        ]
    raise ValueError(f"Unsupported model_key: {cfg.model_key}")


def _build_block_keys(
    guest_list: list[str], site_list: list[str]
) -> list[tuple[str, str]]:
    zip_guest, zip_site = get_zippable_arrays(array1=guest_list, array2=site_list)
    return list(zip(zip_guest, zip_site))


def fit_model(model_type, x, y):
    if isinstance(model_type, list):
        model = random_search(model_type, x=x, y=y)
    else:
        model = model_type
        model.fit(X=x, y=y)
    return copy.deepcopy(model)


def get_residuals(model, x, y):
    y_pred = model.predict(x)
    return y_pred - y


def random_search(model_type, x, y):
    scaler = StandardScaler()
    regressor = model_type[0]
    params = model_type[1]
    search_options = model_type[2] if len(model_type) > 2 else {}
    pipeline = Pipeline([("scaler", scaler), ("regressor", regressor)])
    rand = RandomizedSearchCV(
        pipeline,
        params,
        cv=search_options.get("cv", 3),
        n_iter=search_options.get("n_iter", 30),
        random_state=search_options.get("random_state", 4),
        n_jobs=search_options.get("n_jobs", 1),
    )
    rand.fit(X=x, y=y)
    return rand.best_estimator_


def write_json_params(
    trained_model: TrainedModel,
    output_dir,
):
    os.makedirs(output_dir, exist_ok=True)
    coef = trained_model.estimator.coef_
    features = trained_model.architecture.features
    host_elements = trained_model.architecture.host_elements
    block_keys = trained_model.architecture.block_keys
    num_features = len(features)
    num_guest_types = len(block_keys)
    num_host_elements = len(host_elements)

    for i, el in enumerate(host_elements):
        base = i * (num_features + 1)
        host_dict = {feat: coef[base + j] for j, feat in enumerate(features)}
        host_dict["constant"] = coef[base + num_features]
        filepath = os.path.join(output_dir, f"hostParams_{el}.txt")
        with open(filepath, "w") as f:
            json.dump(host_dict, f, indent=4)

    guest_params_matrix = np.reshape(
        trained_model.train_params, [num_guest_types, num_features]
    )
    guest_offset = num_host_elements * (num_features + 1)
    for k, (guest, site) in enumerate(block_keys):
        guest_dict = {
            "coefficients": list(guest_params_matrix[k]),
            "intercept": coef[guest_offset + k],
        }
        filepath = os.path.join(output_dir, f"guestParams__{guest}__{site}")
        with open(filepath, "w") as f:
            json.dump(guest_dict, f, indent=4)
