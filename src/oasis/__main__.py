from __future__ import annotations

import sys

_EXPERIMENT_CLI_ERROR = (
    "Experiment orchestration no longer runs via `python -m oasis`.\n"
    "Use Moirai for MLIP commands: https://github.com/SiliconScientist/Moirai\n"
    "Temporary compatibility path: `python -m oasis.mlip ...`.\n"
    "Run experiment workflows through config-driven entrypoints elsewhere."
)

def _dispatch_mlip_cli(argv: list[str]) -> None:
    from oasis.mlip.cli import main as mlip_main

    mlip_main(argv)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "mlip":
        _dispatch_mlip_cli(argv[1:])
        return

    raise SystemExit(_EXPERIMENT_CLI_ERROR)


if __name__ == "__main__":
    main()
