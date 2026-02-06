# src/oasis/mlip/registry.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib


@dataclass(frozen=True)
class ModelSpec:
    name: str
    python: str
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


def get_interpreters(cfg: dict[str, Any]) -> dict[str, str]:
    mlip = cfg.get("mlip", {})
    interps = mlip.get("interpreters", {})
    if not isinstance(interps, dict) or not interps:
        raise ValueError(
            "No interpreters found. Expected config like:\n"
            "[mlip.interpreters]\n"
            'mace = "envs/mace/.venv/bin/python"'
        )
    return {str(k): str(v) for k, v in interps.items()}


def get_model_specs(config_path: str | Path) -> dict[str, ModelSpec]:
    cfg = load_config(config_path)
    enabled = get_enabled_models(cfg)
    interps = get_interpreters(cfg)

    specs: dict[str, ModelSpec] = {}
    for name in enabled:
        if name not in interps:
            raise KeyError(
                f"Model '{name}' is enabled but has no interpreter in [mlip.interpreters]."
            )
        specs[name] = ModelSpec(
            name=name,
            python=interps[name],
            adapter_module=f"oasis.adapters.{name}_adapter",
        )
    return specs


def get_model_python(model: str, config_path: str | Path) -> str:
    specs = get_model_specs(config_path)
    if model not in specs:
        raise KeyError(f"Unknown model '{model}'. Known: {', '.join(sorted(specs))}")
    return specs[model].python


def get_model_adapter_module(model: str, config_path: str | Path) -> str:
    specs = get_model_specs(config_path)
    if model not in specs:
        raise KeyError(f"Unknown model '{model}'. Known: {', '.join(sorted(specs))}")
    return specs[model].adapter_module


def get_model_path(model: str, config_path: str | Path) -> str | None:
    cfg = load_config(config_path)
    paths = cfg.get("mlip", {}).get("model_paths", {})
    return paths.get(model)  # returns None if absent
