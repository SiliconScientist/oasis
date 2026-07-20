from __future__ import annotations

from pathlib import Path
from typing import Iterable

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def configured_dataset_tags(config_path: str | Path) -> list[str]:
    path = Path(config_path)
    with path.open("rb") as handle:
        raw_cfg = tomllib.load(handle)
    datasets = raw_cfg.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError(f"No configured datasets found in {path}")
    return list(datasets.keys())


def render_dataset_override(tag: str) -> str:
    return f'[dataset_profile]\ntag = "{tag}"\n'


def write_dataset_override(tag: str, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_dataset_override(tag), encoding="utf-8")
    return path


def write_dataset_overrides(
    tags: Iterable[str],
    *,
    output_dir: str | Path,
    suffix: str = ".override.toml",
) -> list[Path]:
    out_dir = Path(output_dir)
    written: list[Path] = []
    for tag in tags:
        written.append(write_dataset_override(tag, out_dir / f"{tag}{suffix}"))
    return written
