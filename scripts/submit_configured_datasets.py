#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oasis.dataset_overrides import configured_dataset_tags, write_dataset_overrides


def _default_output_dir(base_config: Path) -> Path:
    return Path("slurm_output") / "dataset_overrides" / base_config.stem


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one dataset-profile override TOML per configured dataset and "
            "optionally submit each via submit.sh."
        )
    )
    parser.add_argument(
        "base_config",
        nargs="?",
        default="experiment.toml",
        help="Base experiment config containing the [datasets] mapping.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory where override TOMLs will be written.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit one Oasis job per generated override via submit.sh.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional subset of dataset tags to generate and submit.",
    )
    parser.add_argument(
        "--submit-script",
        default="./submit.sh",
        help="Submit script to invoke when --submit is set.",
    )
    parser.add_argument(
        "--run-tag-prefix",
        default="",
        help=(
            "Optional prefix inserted before the dataset tag when printing or "
            "running submit commands."
        ),
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    base_config = Path(args.base_config)
    tags = configured_dataset_tags(base_config)
    if args.datasets:
        requested = set(args.datasets)
        unknown = sorted(requested.difference(tags))
        if unknown:
            parser.error(f"Unknown dataset tag(s): {', '.join(unknown)}")
        tags = [tag for tag in tags if tag in requested]
    output_dir = (
        Path(args.out_dir)
        if args.out_dir is not None
        else _default_output_dir(base_config)
    )
    override_paths = write_dataset_overrides(tags, output_dir=output_dir)

    print(f"Wrote {len(override_paths)} override config(s) to {output_dir}")
    for tag, override_path in zip(tags, override_paths, strict=True):
        run_tag = f"{args.run_tag_prefix}{tag}"
        cmd = [args.submit_script, str(base_config), str(override_path)]
        if run_tag:
            cmd.append(run_tag)
        print(" ".join(cmd))
        if args.submit:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
