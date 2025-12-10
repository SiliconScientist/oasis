from oasis.config import get_config
from oasis.processing import get_data
from oasis.plot import learning_curve_plot, parity_plot


def main():
    cfg = get_config()
    df = get_data(cfg)
    plot_dir = cfg.plot.output_dir
    plot_path = parity_plot(df, output_path=plot_dir / "parity_plot.png")
    ensemble_path = learning_curve_plot(
        df,
        output_path=plot_dir / "learning_curve.png",
        cfg=cfg,
        gnn_train_fracs=[],
    )
    print(f"Saved parity plot to {plot_path}")
    print(f"Saved learning curve plot to {ensemble_path}")


if __name__ == "__main__":
    main()
