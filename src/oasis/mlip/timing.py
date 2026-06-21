from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class MlipGenerationTimingSummary:
    model_name: str
    reaction_count: int
    generation_time_total_s: float
    generation_time_slab_s: float
    generation_time_adslab_s: float
    generation_steps_total: int
    generation_steps_slab: int
    generation_steps_adslab: int
    time_per_step_s: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "reaction_count": self.reaction_count,
            "generation_time_total_s": self.generation_time_total_s,
            "generation_time_slab_s": self.generation_time_slab_s,
            "generation_time_adslab_s": self.generation_time_adslab_s,
            "generation_steps_total": self.generation_steps_total,
            "generation_steps_slab": self.generation_steps_slab,
            "generation_steps_adslab": self.generation_steps_adslab,
            "time_per_step_s": self.time_per_step_s,
        }


def model_name_from_result_path(result_path: str | Path) -> str:
    return Path(result_path).name.removesuffix("_result.json")


def load_result_json(result_path: str | Path) -> dict[str, Any]:
    path = Path(result_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected result JSON top-level to be an object/dict, got {type(payload).__name__}"
        )
    return payload


def generation_timing_summary_from_result_dict(
    mlip_result: dict[str, Any],
    *,
    model_name: str,
) -> MlipGenerationTimingSummary:
    reaction_count = 0
    total_slab_time = 0.0
    total_adslab_time = 0.0
    total_slab_steps = 0
    total_adslab_steps = 0

    for reaction, reaction_data in mlip_result.items():
        if reaction == "calculation_settings" or not isinstance(reaction_data, dict):
            continue
        final_data = reaction_data.get("final", {})
        if not isinstance(final_data, dict):
            continue

        reaction_count += 1
        total_slab_time += float(final_data.get("time_total_slab", 0.0) or 0.0)
        total_adslab_time += float(final_data.get("time_total_adslab", 0.0) or 0.0)
        total_slab_steps += int(final_data.get("steps_total_slab", 0) or 0)
        total_adslab_steps += int(final_data.get("steps_total_adslab", 0) or 0)

    generation_time_total_s = total_slab_time + total_adslab_time
    generation_steps_total = total_slab_steps + total_adslab_steps

    return MlipGenerationTimingSummary(
        model_name=model_name,
        reaction_count=reaction_count,
        generation_time_total_s=generation_time_total_s,
        generation_time_slab_s=total_slab_time,
        generation_time_adslab_s=total_adslab_time,
        generation_steps_total=generation_steps_total,
        generation_steps_slab=total_slab_steps,
        generation_steps_adslab=total_adslab_steps,
        time_per_step_s=(
            None
            if generation_steps_total <= 0
            else generation_time_total_s / generation_steps_total
        ),
    )


def load_generation_timing_summary(
    result_path: str | Path,
) -> MlipGenerationTimingSummary:
    path = Path(result_path)
    return generation_timing_summary_from_result_dict(
        load_result_json(path),
        model_name=model_name_from_result_path(path),
    )


def load_generation_timing_summaries(
    result_files: list[Path],
) -> dict[str, MlipGenerationTimingSummary]:
    return {
        summary.model_name: summary
        for summary in (
            load_generation_timing_summary(result_path) for result_path in result_files
        )
    }
