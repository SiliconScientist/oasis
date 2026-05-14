import pandas as pd

from latent.config import get_config, get_experiment_config
from latent.data import output_data, split_data
from latent.model import LatentVariableModel
from latent.train import train_model, write_params
from latent.evaluate import test_model, predict
from latent.plot import plot_parity


def main():
    cfg = get_config()
    exp_cfg = get_experiment_config(cfg.experiment_path)
    df = output_data(cfg=cfg, exp_cfg=exp_cfg)
    train_df, test_df = split_data(df, cfg)
    model = LatentVariableModel.from_config(exp_cfg, df=df)
    model = train_model(
        train_df=train_df,
        untrained_model=model,
    )
    write_params(model, exp_cfg.params_output_dir)
    x_test, y_test, rmse, mae = test_model(
        model_train=model,
        test_df=test_df,
    )
    y_pred = predict(model, x_test)
    pd.DataFrame({"y_test": y_test, "y_pred": y_pred}).to_csv("predictions.csv", index=False)

    plot_parity(y_test, y_pred, rmse, mae)


if __name__ == "__main__":
    main()
