from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from oasis.plot import parity_plot, plt, zero_shot_rmse_stage_plot


def vertical_panel_figure(
    panel_paths: Sequence[str | Path],
    *,
    output_path: str | Path,
    panel_labels: Sequence[str] | None = None,
    panel_label_positions: Sequence[tuple[float, float]] | None = None,
    label_fontsize: int = 16,
) -> Path:
    if not panel_paths:
        raise ValueError("panel_paths must contain at least one image path.")

    resolved_paths = [Path(path) for path in panel_paths]
    for panel_path in resolved_paths:
        if not panel_path.is_file():
            raise FileNotFoundError(f"panel image not found: {panel_path}")

    if panel_labels is None:
        panel_labels = tuple(f"{chr(ord('a') + idx)})" for idx in range(len(resolved_paths)))
    if len(panel_labels) != len(resolved_paths):
        raise ValueError("panel_labels must match the number of panel_paths.")
    if panel_label_positions is None:
        panel_label_positions = tuple((0.02, 0.98) for _ in resolved_paths)
    if len(panel_label_positions) != len(resolved_paths):
        raise ValueError("panel_label_positions must match the number of panel_paths.")

    images = [plt.imread(path) for path in resolved_paths]
    height_ratios = [image.shape[0] / max(image.shape[1], 1) for image in images]
    fig_height = max(4.0 * sum(height_ratios), 4.0 * len(images))
    fig, axes = plt.subplots(
        len(images),
        1,
        figsize=(8, fig_height),
        gridspec_kw={"height_ratios": height_ratios},
    )
    if not isinstance(axes, (list, tuple)):
        axes = [axes] if not hasattr(axes, "__len__") else list(axes)

    for ax, image, label, (label_x, label_y) in zip(
        axes,
        images,
        panel_labels,
        panel_label_positions,
        strict=True,
    ):
        ax.imshow(image)
        ax.axis("off")
        ax.text(
            label_x,
            label_y,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=label_fontsize,
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 2},
            clip_on=False,
        )

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def learning_screening_figure(
    *,
    learning_curve_path: str | Path,
    screening_curve_path: str | Path,
    output_path: str | Path,
    panel_labels: Sequence[str] = ("a)", "b)"),
) -> Path:
    return vertical_panel_figure(
        [learning_curve_path, screening_curve_path],
        output_path=output_path,
        panel_labels=panel_labels,
        panel_label_positions=((0.02, 0.98), (0.02, 1.01)),
    )


def two_top_one_bottom_figure(
    *,
    top_left_path: str | Path,
    top_right_path: str | Path,
    bottom_path: str | Path,
    output_path: str | Path,
    panel_labels: Sequence[str] = ("a)", "b)", "c)"),
    panel_label_positions: Sequence[tuple[float, float]] = (
        (0.02, 0.98),
        (0.02, 0.98),
        (0.02, 1.055),
    ),
    label_fontsize: int = 16,
) -> Path:
    panel_paths = [Path(top_left_path), Path(top_right_path), Path(bottom_path)]
    for panel_path in panel_paths:
        if not panel_path.is_file():
            raise FileNotFoundError(f"panel image not found: {panel_path}")

    if len(panel_labels) != 3:
        raise ValueError("panel_labels must match the number of panels.")
    if len(panel_label_positions) != 3:
        raise ValueError("panel_label_positions must match the number of panels.")

    top_left_image, top_right_image, bottom_image = [plt.imread(path) for path in panel_paths]
    top_aspect = max(
        top_left_image.shape[0] / max(top_left_image.shape[1], 1),
        top_right_image.shape[0] / max(top_right_image.shape[1], 1),
    )
    bottom_aspect = bottom_image.shape[0] / max(bottom_image.shape[1], 1)
    fig_width = 12
    fig_height = max(fig_width * (top_aspect / 2.0 + bottom_aspect), 6.0)

    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=(top_aspect / 2.0, bottom_aspect),
    )
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, :]),
    ]
    images = [top_left_image, top_right_image, bottom_image]

    for ax, image, label, (label_x, label_y) in zip(
        axes,
        images,
        panel_labels,
        panel_label_positions,
        strict=True,
    ):
        ax.imshow(image)
        ax.axis("off")
        ax.text(
            label_x,
            label_y,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=label_fontsize,
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 2},
            clip_on=False,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def two_by_two_figure(
    *,
    top_left_path: str | Path,
    top_right_path: str | Path,
    bottom_left_path: str | Path,
    bottom_right_path: str | Path,
    output_path: str | Path,
    panel_labels: Sequence[str] = ("a)", "b)", "c)", "d)"),
    panel_label_positions: Sequence[tuple[float, float]] = (
        (0.02, 0.98),
        (0.02, 0.98),
        (0.02, 0.98),
        (0.02, 0.98),
    ),
    label_fontsize: int = 16,
) -> Path:
    panel_paths = [
        Path(top_left_path),
        Path(top_right_path),
        Path(bottom_left_path),
        Path(bottom_right_path),
    ]
    for panel_path in panel_paths:
        if not panel_path.is_file():
            raise FileNotFoundError(f"panel image not found: {panel_path}")

    if len(panel_labels) != 4:
        raise ValueError("panel_labels must match the number of panels.")
    if len(panel_label_positions) != 4:
        raise ValueError("panel_label_positions must match the number of panels.")

    images = [plt.imread(path) for path in panel_paths]
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]

    for ax, image, label, (label_x, label_y) in zip(
        axes,
        images,
        panel_labels,
        panel_label_positions,
        strict=True,
    ):
        ax.imshow(image)
        ax.axis("off")
        ax.text(
            label_x,
            label_y,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=label_fontsize,
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 2},
            clip_on=False,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def zero_shot_overview_figure(
    *,
    all_mlips_df: Any,
    matched_subset_df: Any,
    all_datasets_stage_df: pd.DataFrame,
    output_path: str | Path,
    anomaly_aware_validity_mask_by_prediction: dict[str, object] | None = None,
    show_lone_mlip_swarm: bool = True,
    max_rmse: float | None = None,
    panel_labels: Sequence[str] = ("a)", "b)", "c)"),
    panel_label_positions: Sequence[tuple[float, float]] = (
        (0.02, 0.98),
        (0.02, 0.98),
        (0.02, 0.98),
    ),
    label_fontsize: int = 16,
) -> Path:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        top_left_path = parity_plot(
            all_mlips_df,
            output_path=tmp_path / "panel_a.png",
            show_legend=False,
            metrics_position=(0.5, 0.91),
            metrics_fontsize=18,
            y_label_fontsize=24,
        )
        top_right_path = parity_plot(
            matched_subset_df,
            output_path=tmp_path / "panel_b.png",
            validity_mask_by_prediction=anomaly_aware_validity_mask_by_prediction,
            show_legend=True,
            legend_fontsize=14,
            metrics_position=(0.5, 0.91),
            metrics_fontsize=18,
            y_label="",
        )
        bottom_path = zero_shot_rmse_stage_plot(
            all_datasets_stage_df,
            output_path=tmp_path / "panel_c.png",
            show_lone_mlip_swarm=show_lone_mlip_swarm,
            show_lone_mlip_legend=False,
            stage_legend_loc="upper right",
            max_rmse=max_rmse,
        )
        return two_top_one_bottom_figure(
            top_left_path=top_left_path,
            top_right_path=top_right_path,
            bottom_path=bottom_path,
            output_path=output_path,
            panel_labels=panel_labels,
            panel_label_positions=panel_label_positions,
            label_fontsize=label_fontsize,
        )


def uq_summary_figure(
    *,
    miscalibration_area_path: str | Path,
    sharpness_path: str | Path,
    dispersion_path: str | Path,
    output_path: str | Path,
    panel_labels: Sequence[str] = ("a)", "b)", "c)"),
) -> Path:
    return vertical_panel_figure(
        [
            miscalibration_area_path,
            sharpness_path,
            dispersion_path,
        ],
        output_path=output_path,
        panel_labels=panel_labels,
    )
