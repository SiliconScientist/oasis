# src/oasis/mlip/registry.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib


@dataclass(frozen=True)
class ModelSpec:
    name: str
    python: str | None
    adapter_module: str


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("rb") as f:
        return tomllib.load(f)


def get_enabled_models(cfg: dict[str, Any]) -> list[str]:
    mlip = cfg.get("mlip", {})
    models = mlip.get("models", {})
    enabled = models.get("enabled", None)
    if not enabled:
        raise ValueError(
            "No enabled MLIP models found. Expected config like:\n"
            "[mlip.models]\n"
            'enabled = ["mace", "mattersim", ...]'
        )
    if not isinstance(enabled, list):
        raise TypeError("mlip.models.enabled must be a list of strings")
    return [str(x) for x in enabled]


def get_rootstock_python(config_path: str | Path) -> str | None:
    cfg = load_config(config_path)
    python = cfg.get("mlip", {}).get("rootstock", {}).get("python")
    return str(python) if python is not None else None


def get_model_specs(config_path: str | Path) -> dict[str, ModelSpec]:
    cfg = load_config(config_path)
    enabled = get_enabled_models(cfg)

    specs: dict[str, ModelSpec] = {}
    for name in enabled:
        specs[name] = ModelSpec(
            name=name,
            python=get_rootstock_python(config_path),
            adapter_module="oasis.adapters.rootstock_adapter",
        )
    return specs


def get_model_python(model: str, config_path: str | Path) -> str | None:
    specs = get_model_specs(config_path)
    if model not in specs:
        raise KeyError(f"Unknown model '{model}'. Known: {', '.join(sorted(specs))}")
    return specs[model].python


def get_model_adapter_module(model: str, config_path: str | Path) -> str:
    specs = get_model_specs(config_path)
    if model not in specs:
        raise KeyError(f"Unknown model '{model}'. Known: {', '.join(sorted(specs))}")
    return specs[model].adapter_module


def get_catbench_source_path(config_path: str | Path) -> Path | None:
    config_path = Path(config_path).resolve()
    cfg = load_config(config_path)
    raw_path = cfg.get("mlip", {}).get("catbench_source")
    if raw_path is None:
        default_path = config_path.parent / "vendor" / "catbench"
        return default_path if default_path.exists() else None

    catbench_path = Path(raw_path)
    if not catbench_path.is_absolute():
        catbench_path = (config_path.parent / catbench_path).resolve()
    return catbench_path
