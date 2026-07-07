from __future__ import annotations

import argparse
import shlex
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_CONFIG_PATH = Path("experiment.toml")


@dataclass(frozen=True)
class SubmissionPlan:
    run_dir: Path
    argv_path: Path
    argv: tuple[str, ...]


def create_submission_run_dir(*, run_tag: str, root_dir: str | Path | None = None) -> Path:
    base_dir = Path.cwd() if root_dir is None else Path(root_dir)
    runs_dir = (base_dir / "slurm_output" / "runs").resolve()
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    run_dir = runs_dir / f"{run_tag}.{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def freeze_config_snapshot(config_path: str | Path, *, run_dir: str | Path) -> Path:
    source_path = Path(config_path).resolve()
    snapshot_path = Path(run_dir).resolve() / source_path.name
    if snapshot_path.exists() and snapshot_path != source_path:
        stem = source_path.stem
        suffix = source_path.suffix
        for index in range(1, 1000):
            candidate = snapshot_path.with_name(f"{stem}.{index}{suffix}")
            if not candidate.exists():
                snapshot_path = candidate
                break
        else:
            raise RuntimeError("Could not allocate a unique config snapshot path.")
    shutil.copy2(source_path, snapshot_path)
    return snapshot_path


def snapshot_cli_argv(argv: list[str] | tuple[str, ...], *, run_dir: str | Path) -> list[str]:
    raw_argv = list(argv)
    if not raw_argv:
        return [str(freeze_config_snapshot(DEFAULT_CONFIG_PATH, run_dir=run_dir))]
    if raw_argv[0] == "rank-candidates":
        config_args = raw_argv[1:] or [str(DEFAULT_CONFIG_PATH)]
        return [
            "rank-candidates",
            *[
                str(freeze_config_snapshot(config_arg, run_dir=run_dir))
                for config_arg in config_args
            ],
        ]
    return [
        str(freeze_config_snapshot(config_arg, run_dir=run_dir))
        for config_arg in raw_argv
    ]


def write_argv_file(argv: list[str] | tuple[str, ...], *, run_dir: str | Path) -> Path:
    argv_path = Path(run_dir).resolve() / "oasis_argv.sh"
    quoted_args = " ".join(shlex.quote(arg) for arg in argv)
    argv_path.write_text(
        f"declare -a OASIS_ARGV=({quoted_args})\n",
        encoding="utf-8",
    )
    return argv_path


def prepare_submission(
    argv: list[str] | tuple[str, ...],
    *,
    run_tag: str = "oasis",
    root_dir: str | Path | None = None,
) -> SubmissionPlan:
    run_dir = create_submission_run_dir(run_tag=run_tag, root_dir=root_dir)
    resolved_argv = tuple(snapshot_cli_argv(argv, run_dir=run_dir))
    argv_path = write_argv_file(list(resolved_argv), run_dir=run_dir)
    return SubmissionPlan(run_dir=run_dir, argv_path=argv_path, argv=resolved_argv)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m oasis.submission",
        description="Prepare a frozen config snapshot for Slurm submission.",
    )
    parser.add_argument("--run-tag", default="oasis")
    parser.add_argument("argv", nargs="*")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    plan = prepare_submission(args.argv, run_tag=args.run_tag)
    print(f"RUN_DIR={shlex.quote(str(plan.run_dir))}")
    print(f"OASIS_ARGV_FILE={shlex.quote(str(plan.argv_path))}")


if __name__ == "__main__":
    main()
