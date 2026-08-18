from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from oasis.config_base import load_toml_file

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATASET_PALETTE = (
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
)
_DEFAULT_POLICY_PALETTE = (
    "tab:red",
    "tab:orange",
    "tab:green",
    "tab:purple",
    "tab:brown",
)
_DEFAULT_BASELINE_PALETTE = (
    "tab:green",
    "tab:brown",
    "tab:olive",
    "tab:cyan",
)
_DEFAULT_MLIP_PALETTE = (
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
)
_DEFAULT_METHOD_COLORS = {
    "ridge": "tab:blue",
    "kernel_ridge": "tab:cyan",
    "lasso": "tab:orange",
    "elastic": "tab:purple",
    "residual": "tab:green",
    "weighted_linear": "tab:gray",
    "weighted_simplex": "teal",
    "graph_mean": "tab:red",
    "moe": "tab:purple",
    "gnn_direct": "tab:cyan",
    "probe_gnn": "tab:olive",
    "latent": "tab:brown",
}
_DEFAULT_STAGE_COLORS = {
    "Full / all MLIPs": "tab:blue",
    "Matched subset / all MLIPs": "tab:orange",
    "Matched subset / anomaly-aware selection": "tab:orange",
}
_DEFAULT_TICK_FONTSIZE = 8


def _stable_palette_color(key: str, palette: tuple[str, ...]) -> str:
    digest = hashlib.sha256(str(key).encode("utf-8")).digest()
    return palette[int.from_bytes(digest[:4], "big") % len(palette)]


def _stable_hex_color(key: str) -> str:
    digest = hashlib.sha256(str(key).encode("utf-8")).digest()
    channels = [64 + (value % 160) for value in digest[:3]]
    return "#" + "".join(f"{channel:02x}" for channel in channels)


def _validated_color_map(raw_section: Any, section_name: str) -> dict[str, str]:
    if raw_section is None:
        return {}
    if not isinstance(raw_section, dict):
        raise ValueError(f"[{section_name}] must be a table of name = color entries.")
    validated: dict[str, str] = {}
    for key, value in raw_section.items():
        if not isinstance(value, str):
            raise ValueError(
                f"[{section_name}] entry {key!r} must map to a string color value."
            )
        validated[str(key)] = value
    return validated


def _validated_mlip_profiles(raw_section: Any) -> dict[str, dict[str, str]]:
    if raw_section is None:
        return {}
    if not isinstance(raw_section, dict):
        raise ValueError("[mlips] must be a table of MLIP profile subtables.")
    validated: dict[str, dict[str, str]] = {}
    for mlip_name, profile in raw_section.items():
        if isinstance(profile, str):
            validated[str(mlip_name)] = {"color": profile}
            continue
        if not isinstance(profile, dict):
            raise ValueError(
                f"[mlips.{mlip_name}] must be a table with alias/color fields."
            )
        normalized: dict[str, str] = {}
        alias = profile.get("alias")
        color = profile.get("color")
        if alias is not None:
            if not isinstance(alias, str):
                raise ValueError(f"[mlips.{mlip_name}].alias must be a string.")
            normalized["alias"] = alias
        if color is not None:
            if not isinstance(color, str):
                raise ValueError(f"[mlips.{mlip_name}].color must be a string.")
            normalized["color"] = color
        unknown_fields = sorted(
            str(field_name)
            for field_name in profile.keys()
            if field_name not in {"alias", "color"}
        )
        if unknown_fields:
            raise ValueError(
                f"[mlips.{mlip_name}] contains unsupported fields: {unknown_fields}"
            )
        validated[str(mlip_name)] = normalized
    return validated


def _validated_fontsize(raw_value: Any, field_name: str, *, default: int) -> int:
    if raw_value is None:
        return default
    if not isinstance(raw_value, int) or isinstance(raw_value, bool):
        raise ValueError(f"{field_name} must be an integer.")
    if raw_value <= 0:
        raise ValueError(f"{field_name} must be positive.")
    return raw_value


class PlotStyle:
    def __init__(self, raw_config: dict[str, Any] | None = None) -> None:
        config = {} if raw_config is None else raw_config
        self.dataset_colors = _validated_color_map(config.get("datasets"), "datasets")
        self.method_colors = {
            **_DEFAULT_METHOD_COLORS,
            **_validated_color_map(config.get("methods"), "methods"),
        }
        self.mlip_profiles = _validated_mlip_profiles(config.get("mlips"))
        self.policy_colors = _validated_color_map(config.get("policies"), "policies")
        self.baseline_colors = _validated_color_map(
            config.get("baselines"), "baselines"
        )
        self.stage_colors = {
            **_DEFAULT_STAGE_COLORS,
            **_validated_color_map(config.get("stages"), "stages"),
        }
        self.tick_fontsize = _validated_fontsize(
            config.get("tick_fontsize"),
            "tick_fontsize",
            default=_DEFAULT_TICK_FONTSIZE,
        )

    def dataset_color(self, dataset: str) -> str:
        return self.dataset_colors.get(dataset, _stable_hex_color(dataset))

    def method_color(self, method: str, default: str | None = None) -> str:
        if method in self.method_colors:
            return self.method_colors[method]
        if default is not None:
            return default
        return _stable_palette_color(method, _DEFAULT_DATASET_PALETTE)

    def mlip_color(self, mlip: str) -> str:
        profile = self.mlip_profiles.get(mlip, {})
        return profile.get("color", _stable_hex_color(mlip))

    def mlip_alias(self, mlip: str, default: str | None = None) -> str:
        profile = self.mlip_profiles.get(mlip, {})
        if "alias" in profile:
            return profile["alias"]
        if default is not None:
            return default
        return mlip

    def policy_color(self, policy_name: str) -> str:
        return self.policy_colors.get(policy_name, _stable_hex_color(policy_name))

    def baseline_color(self, baseline_name: str) -> str:
        return self.baseline_colors.get(baseline_name, _stable_hex_color(baseline_name))

    def stage_color(self, stage_name: str) -> str:
        return self.stage_colors.get(stage_name, "tab:gray")


def resolve_plot_style_path() -> Path:
    configured = os.environ.get("OASIS_PLOT_STYLE_PATH")
    if configured:
        return Path(configured).expanduser()
    return _REPO_ROOT / "plot_style.toml"


@lru_cache(maxsize=1)
def get_plot_style() -> PlotStyle:
    path = resolve_plot_style_path()
    if not path.is_file():
        return PlotStyle()
    return PlotStyle(load_toml_file(path))


def reset_plot_style_cache() -> None:
    get_plot_style.cache_clear()
