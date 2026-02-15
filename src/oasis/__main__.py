import sys
from oasis.ingest.vasp_to_catbench import main as ingest_main
from oasis.mlip.cli import main as mlip_main


def main():
    ingest_main()
    # if len(sys.argv) < 2:
    #     print("Usage: python -m oasis <command> [args...]")
    #     print("Available commands: mlip")
    #     sys.exit(1)

    # command = sys.argv[1]

    # if command == "mlip":
    #     # Forward remaining args to mlip CLI
    #     mlip_main(sys.argv[2:])
    # else:
    #     print(f"Unknown command: {command}")
    #     sys.exit(1)


if __name__ == "__main__":
    main()


# def main():
#     cfg = get_config()
#     df = get_data(cfg)
#     plot_dir = cfg.plot.output_dir
#     gnn_fracs = [0.95, 0.9, 0.85, 0.8] if cfg.plot.use_gnn else []
#     plot_path = parity_plot(df, output_path=plot_dir / "parity_plot.png")
#     ensemble_path = learning_curve_plot(
#         df,
#         output_path=plot_dir / "learning_curve.png",
#         cfg=cfg,
#         gnn_train_fracs=gnn_fracs,
#     )
#     print(f"Saved parity plot to {plot_path}")
#     print(f"Saved learning curve plot to {ensemble_path}")


# if __name__ == "__main__":
#     main()
