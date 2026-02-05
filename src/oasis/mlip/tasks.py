# src/oasis/mlip/tasks.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Any

from oasis.mlip.registry import get_model_specs, load_config


def default_run_tag(cfg: dict) -> str:
    # allow config override; else caller provides; else fallback handled in CLI
    mlip = cfg.get("mlip", {})
    tag = mlip.get("run_tag", None)
    return str(tag) if tag else "run"


def resolve_datasets(datasets: list[str] | None) -> list[Path]:
    if datasets and len(datasets) > 0:
        return [Path(d) for d in datasets]

    # default: all json in data/datasets/
    base = Path("data/datasets")
    if not base.exists():
        raise FileNotFoundError(
            "No datasets provided and default folder data/datasets/ does not exist."
        )

    found = sorted(base.glob("*.json"))
    if not found:
        raise FileNotFoundError(f"No datasets found in {base}/ (expected *.json)")
    return found


def dataset_name_from_path(p: Path) -> str:
    return p.stem


def output_path(run_tag: str, dataset_name: str, model: str) -> Path:
    # data/results/mlips/<run_tag>/<dataset>/<model>.json
    return Path("data/results/mlips") / run_tag / dataset_name / f"{model}.json"


def _slice_json_obj(obj: Any, n: int) -> Any:
    # Case 1: list at top-level
    if isinstance(obj, list):
        return obj[:n]

    # Case 2: dict at top-level (your case)
    if isinstance(obj, dict):
        # If dict values look like records, take first n items.
        # JSON load preserves order in modern Python.
        items = list(obj.items())[:n]
        return dict(items)

    raise TypeError(f"Dev slicing expects a JSON list or dict; got {type(obj)}")


def maybe_make_dev_dataset(dpath: Path, cfg: dict) -> Path:
    mlip = cfg.get("mlip", {})
    dev_run = bool(mlip.get("dev_run", False))
    dev_n = int(mlip.get("dev_n", 2))
    if not dev_run:
        return dpath
    dev_dir = Path("data/datasets/_dev")
    dev_dir.mkdir(parents=True, exist_ok=True)
    out = dev_dir / f"{dpath.stem}__dev{dev_n}{dpath.suffix}"
    # Reuse existing dev dataset if present (keeps sbatch deterministic)
    if out.exists():
        return out
    with dpath.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    sliced = _slice_json_obj(obj, dev_n)
    with out.open("w", encoding="utf-8") as f:
        json.dump(sliced, f, indent=2)
        f.write("\n")
    return out


def make_task_lines(
    *,
    config_path: str | Path,
    run_tag: str,
    datasets: list[str] | None = None,
) -> list[str]:
    cfg = load_config(config_path)
    specs = get_model_specs(config_path)
    dataset_paths = resolve_datasets(datasets)
    lines: list[str] = []
    for dpath in dataset_paths:
        dpath_for_run = maybe_make_dev_dataset(dpath, cfg)
        dname = dataset_name_from_path(dpath)
        for model in specs:
            out = output_path(run_tag, dname, model)
            lines.append(f"{model} {dpath_for_run.as_posix()} {out.as_posix()}")
    return lines


def write_tasks(out_path: str | Path, lines: Iterable[str]) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")


def make_tasks(
    *,
    config_path: str | Path,
    run_tag: str,
    out_path: str | Path,
    datasets: list[str] | None = None,
) -> None:
    # Optional: if caller passes run_tag="auto", use config default
    cfg = load_config(config_path)
    if run_tag == "auto":
        run_tag = default_run_tag(cfg)

    lines = make_task_lines(config_path=config_path, run_tag=run_tag, datasets=datasets)
    write_tasks(out_path, lines)
    print(f"Wrote {len(lines)} tasks to {out_path}")
