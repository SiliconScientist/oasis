from __future__ import annotations

from pathlib import Path

from oasis.candidate_ranking import rank_candidates_from_results_dir
from oasis.config import get_config


def _candidate_ranking_dataset_metadata(cfg: object) -> dict[str, object]:
    profile = getattr(cfg, "dataset_profile", None)
    return {
        "dataset_tag": getattr(profile, "tag", None),
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

    result = rank_candidates_from_results_dir(
        Path(ranking_cfg.results_dir),
        method_name=ranking_cfg.method,
        shot_count=ranking_cfg.shot_count,
        target_binding_energy=ranking_cfg.target_binding_energy,
        dataset_metadata=_candidate_ranking_dataset_metadata(cfg),
        method_config=ranking_cfg.resolved_method_config(),
    )

    top_k = max(1, int(ranking_cfg.top_k))
    print(
        "Candidate ranking"
        f" method={ranking_cfg.method}"
        f" shot_count={ranking_cfg.shot_count}"
        f" target_binding_energy={ranking_cfg.target_binding_energy}"
        f" top_k={top_k}"
    )
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
