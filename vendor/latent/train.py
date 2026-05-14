import numpy as np
from scipy.optimize import minimize

from latent.model import (
    TrainedModel,
    LatentVariableModel,
    fit_model,
    get_residuals,
    write_json_params,
)


def cost_function(
    guest_fitting_parameters,
    df,
    untrained_model: LatentVariableModel,
):
    x, y = untrained_model.linearize_data(df, guest_fitting_parameters)
    model = fit_model(model_type=untrained_model.model_type, x=x, y=y)
    residuals = get_residuals(model, x, y)
    return np.sqrt(np.mean(np.abs(residuals) ** 2))


def train_model(
    train_df,
    untrained_model: LatentVariableModel,
    cobyla_initial_guess: float = 0.1,
    cobyla_max_iter: int = 100,
):
    num_features = len(untrained_model.features)
    num_guest_types = len(untrained_model.block_keys)
    guest_fitting_parameters = cobyla_initial_guess * np.ones(
        num_features * num_guest_types
    )

    min_result = minimize(
        cost_function,
        guest_fitting_parameters,
        args=(train_df, untrained_model),
        method="COBYLA",
        options={"maxiter": cobyla_max_iter},
    )
    train_params = min_result.x
    x_train, y_train = untrained_model.linearize_data(train_df, train_params)
    model_train = fit_model(model_type=untrained_model.model_type, x=x_train, y=y_train)
    trained_model = TrainedModel(
        estimator=model_train,
        architecture=untrained_model,
        train_params=train_params,
    )
    return trained_model



def write_params(trained_model: TrainedModel, params_output_dir: str):
    if not params_output_dir:
        raise ValueError("params_output_dir must be set when writing parameters.")

    write_json_params(
        trained_model=trained_model,
        output_dir=params_output_dir,
    )
    print(f"Wrote fitting parameters to {params_output_dir}")
