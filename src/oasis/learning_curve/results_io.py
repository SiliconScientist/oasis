from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from oasis.sweep import LearningCurveResults


_RESULTS_ARTIFACT_VERSION = 1


def dump_learning_curve_results(
    results: LearningCurveResults,
) -> dict[str, Any]:
    return {
        "version": _RESULTS_ARTIFACT_VERSION,
        "frames": {
            field_name: None if frame is None else frame.to_json(orient="table")
            for field_name, frame in results.to_mapping().items()
        },
    }


def load_learning_curve_results_mapping(
    payload: dict[str, Any],
) -> LearningCurveResults:
    version = payload.get("version")
    if version != _RESULTS_ARTIFACT_VERSION:
        raise ValueError(
            "unsupported learning-curve results artifact version: "
            f"{version!r}."
        )

    frames_payload = payload.get("frames")
    if not isinstance(frames_payload, dict):
        raise TypeError("learning-curve results artifact must contain a frames mapping.")

    frames: dict[str, pd.DataFrame | None] = {}
    for field_name, frame_payload in frames_payload.items():
        if frame_payload is None:
            frames[field_name] = None
            continue
        if not isinstance(frame_payload, str):
            raise TypeError(
                "learning-curve frame payloads must be JSON strings or null."
            )
        frames[field_name] = pd.read_json(StringIO(frame_payload), orient="table")
    return LearningCurveResults.from_mapping(frames)


def save_learning_curve_results(
    results: LearningCurveResults,
    path: str | Path,
) -> Path:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(
        json.dumps(dump_learning_curve_results(results), indent=2),
        encoding="utf-8",
    )
    return resolved_path


def load_learning_curve_results(
    path: str | Path,
) -> LearningCurveResults:
    resolved_path = Path(path)
    return load_learning_curve_results_mapping(
        json.loads(resolved_path.read_text(encoding="utf-8"))
    )
