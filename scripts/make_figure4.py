#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
        help="Dataset plot directory containing the four panel PNGs.",
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
        "--config",
        default="experiment.toml",
        help="Unused compatibility flag reserved for future use.",
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


def _load_summary_frame_grid(plot_root: Path, suffix: str):
    import pandas as pd

    summary_paths = sorted(
        plot_root.glob(f"*/policy_selection_diagnostic_summary_{suffix}.csv")
    )
    summary_frames = []
    dataset_order: list[str] = []
    for summary_path in summary_paths:
        frame = pd.read_csv(summary_path)
        if frame.empty:
            continue
        dataset_token = _normalize_dataset_token(summary_path.parent.name)
        dataset_order.append(dataset_token)
        frame = frame.copy()
        frame.insert(0, "dataset_label", summary_path.parent.name)
        frame.insert(0, "dataset", dataset_token)
        summary_frames.append(frame)
    if not summary_frames:
        raise ValueError("No cached policy summary CSVs found.")
    summary_df = pd.concat(summary_frames, ignore_index=True)
    summary_df["dataset"] = pd.Categorical(
        summary_df["dataset"],
        categories=dataset_order,
        ordered=True,
    )
    return summary_df, dataset_order


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from oasis.figure import two_by_two_figure
    from oasis.plot import all_datasets_policy_regret_plot
    import pandas as pd

    plot_dir = args.plot_dir.resolve()
    suffix = args.suffix
    panel_c_path = plot_dir / f"policy_regret_all_datasets_{suffix}_absolute.png"
    panel_d_path = plot_dir / f"policy_regret_all_datasets_{suffix}_fraction.png"

    if args.exclude_panel_d_dataset:
        summary_df, dataset_order = _load_summary_frame_grid(plot_dir.parent, suffix)
        excluded = {_normalize_dataset_token(value) for value in args.exclude_panel_d_dataset}
        panel_d_df = summary_df.loc[
            ~summary_df["dataset"].astype(str).isin(excluded)
        ].reset_index(drop=True)
        if panel_d_df.empty:
            raise ValueError("Panel d has no remaining rows after dataset exclusion.")
        panel_d_df["dataset"] = pd.Categorical(
            panel_d_df["dataset"],
            categories=[dataset for dataset in dataset_order if dataset not in excluded],
            ordered=True,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            panel_c_path = all_datasets_policy_regret_plot(
                summary_df,
                output_path=Path(tmpdir) / "panel_c_absolute.png",
                log_x=False,
            )
            panel_d_path = all_datasets_policy_regret_plot(
                panel_d_df,
                output_path=Path(tmpdir) / "panel_d_fraction.png",
                log_x=True,
            )
            output_path = two_by_two_figure(
                top_left_path=plot_dir / f"policy_selected_vs_oracle_{suffix}_absolute.png",
                top_right_path=plot_dir / f"policy_selected_vs_oracle_{suffix}_fraction.png",
                bottom_left_path=panel_c_path,
                bottom_right_path=panel_d_path,
                output_path=plot_dir / args.output_name,
                panel_labels=("a)", "b)", "c)", "d)"),
            )
            print(output_path)
            return 0

    output_path = two_by_two_figure(
        top_left_path=plot_dir / f"policy_selected_vs_oracle_{suffix}_absolute.png",
        top_right_path=plot_dir / f"policy_selected_vs_oracle_{suffix}_fraction.png",
        bottom_left_path=plot_dir / f"policy_regret_all_datasets_{suffix}_absolute.png",
        bottom_right_path=panel_d_path,
        output_path=plot_dir / args.output_name,
        panel_labels=("a)", "b)", "c)", "d)"),
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
