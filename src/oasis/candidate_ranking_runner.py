from __future__ import annotations

from pathlib import Path

import pandas as pd

from oasis.candidate_ranking import (
    rank_candidates_from_results_dir,
    rank_candidates_from_results_dir_and_references,
)
from oasis.config import get_config
from oasis.plot import (
    dispersion_plot,
    learning_curve_plot,
    miscalibration_area_plot,
    sharpness_plot,
)
from oasis.sweep import LearningCurveResults


def _candidate_ranking_dataset_metadata(cfg: object) -> dict[str, object]:
    profile = getattr(cfg, "dataset_profile", None)
    return {
        "dataset_tag": getattr(profile, "tag", None),
    }


def _candidate_ranking_diagnostics_output_dir(cfg: object) -> Path:
    ranking_cfg = getattr(cfg, "candidate_ranking")
    if ranking_cfg.diagnostics_output_dir is not None:
        return Path(ranking_cfg.diagnostics_output_dir)
    dataset_tag = _candidate_ranking_dataset_metadata(cfg).get("dataset_tag")
    suffix = str(dataset_tag) if dataset_tag else "default"
    return Path("data/results/candidate_ranking") / suffix


def _candidate_ranking_results_field(predictor_name: str) -> tuple[str, str] | None:
    return {
        "residual": ("resid_df", "resid_uq_df"),
        "weighted_simplex": ("weighted_simplex_df", "weighted_simplex_uq_df"),
        "ridge": ("ridge_df", "ridge_uq_df"),
    }.get(predictor_name)


def _learning_curve_results_from_predictor_diagnostics(
    diagnostics: dict[str, dict[str, object]],
) -> LearningCurveResults:
    empty_results = LearningCurveResults.empty()
    frames: dict[str, pd.DataFrame | None] = {
        field_name: None for field_name in empty_results.to_mapping()
    }
    for predictor_name, metrics in diagnostics.items():
        field_names = _candidate_ranking_results_field(predictor_name)
        if field_names is None:
            continue
        rmse_field, uq_field = field_names
        frames[rmse_field] = pd.DataFrame(
            [
                {
                    "n_train": int(metrics["n_train"]),
                    "rmse_mean": float(metrics["cv_rmse_mean"]),
                    "rmse_std": float(metrics["cv_rmse_std"]),
                }
            ]
        )
        frames[uq_field] = pd.DataFrame(
            [
                {
                    "n_train": int(metrics["n_train"]),
                    "miscalibration_area": float(metrics["miscalibration_area"]),
                    "miscalibration_area_std": float(
                        metrics["miscalibration_area_std"]
                    ),
                    "sharpness": float(metrics["sharpness"]),
                    "sharpness_std": float(metrics["sharpness_std"]),
                    "dispersion": float(metrics["dispersion"]),
                    "dispersion_std": float(metrics["dispersion_std"]),
                    "uncertainty_kind": str(metrics["uncertainty_kind"]),
                }
            ]
        )
    return LearningCurveResults.from_mapping(frames)


def _write_candidate_ranking_diagnostics(
    *,
    diagnostics: dict[str, dict[str, object]],
    output_dir: Path,
) -> dict[str, str]:
    results = _learning_curve_results_from_predictor_diagnostics(diagnostics)
    output_dir.mkdir(parents=True, exist_ok=True)
    rmse_path = learning_curve_plot(
        results=results,
        output_path=output_dir / "predictor_cv_rmse.png",
    )
    miscalibration_path = miscalibration_area_plot(
        results,
        output_path=output_dir / "predictor_cv_miscalibration_area.png",
    )
    sharpness_path = sharpness_plot(
        results,
        output_path=output_dir / "predictor_cv_sharpness.png",
    )
    dispersion_path = dispersion_plot(
        results,
        output_path=output_dir / "predictor_cv_dispersion.png",
    )
    return {
        "cv_rmse": str(rmse_path),
        "miscalibration_area": str(miscalibration_path),
        "sharpness": str(sharpness_path),
        "dispersion": str(dispersion_path),
    }


def run_candidate_ranking_from_config(
    argv: list[str] | None = None,
):
    cfg = get_config(argv or None)
    ranking_cfg = getattr(cfg, "candidate_ranking", None)
    if ranking_cfg is None:
        raise ValueError(
            "candidate_ranking config section is required for `oasis rank-candidates`."
        )

    rank_kwargs = {
        "predictor_name": ranking_cfg.selected_predictor,
        "predictor_names": ranking_cfg.predictors,
        "target_binding_energy": ranking_cfg.target_binding_energy,
        "dataset_metadata": _candidate_ranking_dataset_metadata(cfg),
        "method_config": ranking_cfg.resolved_predictor_config(),
        "predictor_configs": ranking_cfg.predictor_configs,
    }
    if ranking_cfg.validated_references_path is None:
        result = rank_candidates_from_results_dir(
            Path(ranking_cfg.results_dir),
            **rank_kwargs,
        )
    else:
        result = rank_candidates_from_results_dir_and_references(
            Path(ranking_cfg.results_dir),
            validated_references_path=Path(ranking_cfg.validated_references_path),
            **rank_kwargs,
        )

    top_k = max(1, int(ranking_cfg.top_k))
    print(
        "Candidate ranking"
        f" predictors={list(ranking_cfg.predictors)}"
        f" selected_predictor={ranking_cfg.selected_predictor}"
        f" validated_references_path={ranking_cfg.validated_references_path}"
        f" target_binding_energy={ranking_cfg.target_binding_energy}"
        f" top_k={top_k}"
    )
    diagnostics = getattr(result, "metadata", {}).get("predictor_diagnostics")
    if isinstance(diagnostics, dict) and diagnostics:
        diagnostics_paths = _write_candidate_ranking_diagnostics(
            diagnostics=diagnostics,
            output_dir=_candidate_ranking_diagnostics_output_dir(cfg),
        )
        print(f"Predictor diagnostics plots={diagnostics_paths}")
    for rank, candidate in enumerate(result.ranked_candidates[:top_k], start=1):
        print(
            f"{rank}. parent={candidate.parent_slab_id}"
            f" adslab={candidate.selected_adslab_id}"
            f" score={candidate.score}"
            f" predicted_binding_energy={candidate.predicted_binding_energy}"
            f" uncertainty="
            f"{candidate.uncertainty.value if candidate.uncertainty is not None else None}"
        )
    return result
