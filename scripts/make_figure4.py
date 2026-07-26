#!/usr/bin/env python3
from __future__ import annotations

import argparse
from io import StringIO
import json
from pathlib import Path
import re
import sys
import tempfile


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assemble Figure 4 from policy-selection and regret plots.",
    )
    parser.add_argument(
        "plot_dir",
        type=Path,
        help="Dataset plot directory containing the source panel PNGs.",
    )
    parser.add_argument(
        "--suffix",
        default="anomalyaware_on",
        help="Shared filename suffix for the source panel PNGs.",
    )
    parser.add_argument(
        "--output-name",
        default="figure4.png",
        help="Output filename written inside plot_dir.",
    )
    parser.add_argument(
        "--exclude-panel-d-dataset",
        action="append",
        default=[],
        help="Dataset tag to exclude from panel d. Repeat for multiple datasets.",
    )
    return parser


def _normalize_dataset_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _dataset_aliases(value: str) -> set[str]:
    raw = str(value).strip().lower()
    return {raw, _normalize_dataset_token(raw)}


def _dataset_label_by_tag() -> dict[str, str]:
    from oasis.config import get_config

    cfg = get_config()
    labels: dict[str, str] = {}
    for dataset_tag, profile in getattr(cfg, "datasets", {}).items():
        alias = getattr(profile, "alias", None)
        mlip_run_dirname = getattr(profile, "mlip_run_dirname", None)
        labels[str(dataset_tag)] = str(alias or mlip_run_dirname or dataset_tag)
    return labels


def _load_cached_policy_artifacts(plot_root: Path, suffix: str) -> tuple[list[dict], list[str]]:
    import pandas as pd
    import numpy as np

    dataset_labels = _dataset_label_by_tag()
    summary_paths = sorted(
        plot_root.glob(f"*/policy_selection_diagnostic_summary_{suffix}.csv")
    )
    artifact_paths = sorted(
        (plot_root.parent / "screening").glob("policy_selection_diagnostic_*.json")
    )
    if not summary_paths:
        raise ValueError("No cached policy summary CSVs found.")
    if not artifact_paths:
        raise ValueError("No cached policy-selection artifacts found.")

    artifact_index: list[dict[str, object]] = []
    for artifact_path in artifact_paths:
        payload = json.loads(artifact_path.read_text())
        results_payload = payload.get("results")
        if not isinstance(results_payload, dict):
            continue
        summary_payload = results_payload.get("summary_df")
        if not isinstance(summary_payload, str):
            continue
        summary_df = pd.read_json(StringIO(summary_payload), orient="table")
        artifact_index.append(
            {
                "path": artifact_path,
                "payload": payload,
                "summary_df": summary_df,
            }
        )

    dataset_entries: list[dict] = []
    dataset_order: list[str] = []
    for summary_path in summary_paths:
        summary_df = pd.read_csv(summary_path)
        if summary_df.empty:
            continue
        def _matches_summary(candidate_df: pd.DataFrame) -> bool:
            if list(candidate_df.columns) != list(summary_df.columns):
                return False
            if len(candidate_df) != len(summary_df):
                return False
            for column in summary_df.columns:
                left = candidate_df[column]
                right = summary_df[column]
                if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
                    if not np.allclose(
                        left.to_numpy(dtype=float),
                        right.to_numpy(dtype=float),
                        equal_nan=True,
                    ):
                        return False
                else:
                    if left.astype(str).tolist() != right.astype(str).tolist():
                        return False
            return True
        artifact_match = next(
            (
                entry
                for entry in artifact_index
                if _matches_summary(entry["summary_df"])
            ),
            None,
        )
        if artifact_match is None:
            raise ValueError(
                f"Could not match summary CSV to cached artifact: {summary_path}"
            )
        payload = artifact_match["payload"]
        metadata = payload.get("metadata", {})
        cache_signature = payload.get("cache_signature", {})
        learning_curve_signature = cache_signature.get("learning_curve", {})
        dataset_tag = str(metadata.get("dataset_tag") or summary_path.parent.name)
        dataset_order.append(dataset_tag)
        dataset_entries.append(
            {
                "dataset": dataset_tag,
                "dataset_label": dataset_labels.get(dataset_tag, summary_path.parent.name),
                "dataset_size": int(metadata["dataset_size"]),
                "summary_df": summary_df.copy(),
                "sweep_sizes": tuple(
                    int(value)
                    for value in learning_curve_signature.get("sweep_sizes", [])
                ),
                "sweep_fractions": tuple(
                    float(value)
                    for value in learning_curve_signature.get("sweep_fractions", [])
                ),
            }
        )
    if not dataset_entries:
        raise ValueError("No non-empty cached policy summaries found.")
    return dataset_entries, dataset_order


def _build_regret_frame(
    dataset_entries: list[dict],
    *,
    mode: str,
    excluded_datasets: set[str],
):
    import pandas as pd

    from oasis.experiment.splits import resolve_configured_sweep_sizes

    rows: list[pd.DataFrame] = []
    for entry in dataset_entries:
        dataset_tag = str(entry["dataset"])
        if _dataset_aliases(dataset_tag) & excluded_datasets:
            continue
        summary_df = entry["summary_df"].copy()
        summary_df.insert(0, "dataset_label", entry["dataset_label"])
        summary_df.insert(0, "dataset", dataset_tag)
        if mode == "absolute":
            include_x = list(entry["sweep_sizes"])
        elif mode == "fraction":
            include_x = list(
                resolve_configured_sweep_sizes(
                    int(entry["dataset_size"]),
                    min_train=None,
                    max_train=None,
                    sweep_sizes=(),
                    sweep_fractions=entry["sweep_fractions"],
                )
            )
        else:
            raise ValueError(f"Unsupported regret frame mode: {mode}")
        filtered = summary_df.loc[
            summary_df["budget"].astype(int).isin(include_x)
        ].reset_index(drop=True)
        if not filtered.empty:
            rows.append(filtered)
    if not rows:
        raise ValueError(f"No rows available for {mode} regret panel.")
    return pd.concat(rows, ignore_index=True)


def _render_regret_panel(
    summary_df,
    *,
    output_path: Path,
    dataset_order: list[str],
    log_x: bool,
):
    import pandas as pd

    from oasis.plot import (
        _DEFAULT_LEGEND_FONTSIZE,
        _DEFAULT_PLOT_FONTSIZE,
        _DEFAULT_TICK_FONTSIZE,
        _set_integer_x_ticks,
        plt,
    )

    filtered = summary_df.sort_values(["dataset", "budget"]).reset_index(drop=True)
    label_rows = filtered.loc[:, ["dataset", "dataset_label"]].drop_duplicates(
        subset=["dataset"],
        keep="first",
    )
    dataset_labels = (
        label_rows.set_index("dataset")
        .reindex(dataset_order)["dataset_label"]
        .fillna(pd.Series(dataset_order, index=dataset_order))
        .to_dict()
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.cm.get_cmap("tab10", max(1, len(dataset_order)))
    for idx, dataset in enumerate(dataset_order):
        dataset_rows = filtered.loc[filtered["dataset"].astype(str) == dataset]
        if dataset_rows.empty:
            continue
        ordered = dataset_rows.sort_values("budget")
        color = cmap(idx)
        ax.plot(
            ordered["budget"],
            ordered["mean_regret"],
            marker="o",
            color=color,
            label=dataset_labels[dataset],
        )
        if {"ci95_low", "ci95_high"}.issubset(ordered.columns):
            ax.fill_between(
                ordered["budget"],
                ordered["ci95_low"],
                ordered["ci95_high"],
                color=color,
                alpha=0.2,
            )
        elif "std_regret" in ordered.columns:
            ax.fill_between(
                ordered["budget"],
                ordered["mean_regret"] - ordered["std_regret"],
                ordered["mean_regret"] + ordered["std_regret"],
                color=color,
                alpha=0.2,
            )

    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Sample budget", fontsize=_DEFAULT_PLOT_FONTSIZE)
    ax.set_ylabel("Regret", fontsize=_DEFAULT_PLOT_FONTSIZE)
    ax.set_title("Screening policy regret by dataset", fontsize=_DEFAULT_PLOT_FONTSIZE)
    if log_x:
        ax.set_xscale("log")
    else:
        _set_integer_x_ticks(ax)
    ax.tick_params(axis="both", labelsize=_DEFAULT_TICK_FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=_DEFAULT_LEGEND_FONTSIZE)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from oasis.figure import two_by_two_figure

    plot_dir = args.plot_dir.resolve()
    suffix = args.suffix
    excluded = {
        alias
        for value in args.exclude_panel_d_dataset
        for alias in _dataset_aliases(value)
    }

    panel_a_path = plot_dir / f"policy_selected_vs_oracle_{suffix}_absolute.png"
    panel_b_path = plot_dir / f"policy_selected_vs_oracle_{suffix}_fraction.png"

    dataset_entries, dataset_order = _load_cached_policy_artifacts(plot_dir.parent, suffix)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        panel_c_df = _build_regret_frame(
            dataset_entries,
            mode="absolute",
            excluded_datasets=set(),
        )
        panel_d_df = _build_regret_frame(
            dataset_entries,
            mode="fraction",
            excluded_datasets=excluded,
        )
        panel_c_path = _render_regret_panel(
            panel_c_df,
            output_path=tmp_path / "panel_c_absolute.png",
            dataset_order=dataset_order,
            log_x=False,
        )
        panel_d_path = _render_regret_panel(
            panel_d_df,
            output_path=tmp_path / "panel_d_fraction.png",
            dataset_order=dataset_order,
            log_x=True,
        )
        output_path = two_by_two_figure(
            top_left_path=panel_a_path,
            top_right_path=panel_b_path,
            bottom_left_path=panel_c_path,
            bottom_right_path=panel_d_path,
            output_path=plot_dir / args.output_name,
            panel_labels=("a)", "b)", "c)", "d)"),
        )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
