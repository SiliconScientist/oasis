from __future__ import annotations

import sys

def _dispatch_mlip_cli(argv: list[str]) -> None:
    from oasis.mlip.cli import main as mlip_main

    mlip_main(argv)


def _dispatch_experiment_cli(argv: list[str]) -> None:
    from oasis.experiment_runner import run_experiment_from_config

    run_experiment_from_config(argv or None)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "mlip":
        _dispatch_mlip_cli(argv[1:])
        return

    _dispatch_experiment_cli(argv)


if __name__ == "__main__":
    main()
