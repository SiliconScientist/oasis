from __future__ import annotations

from pathlib import Path
from typing import Sequence

from oasis.plot import plt


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
