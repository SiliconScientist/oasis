from __future__ import annotations

from pathlib import Path
import sys

from oasis.analysis import filter_wide_predictions
from oasis.config import get_config
from oasis.graph import batch_adsorption_graphs, build_adsorption_graphs
from oasis.io import find_result_files, load_corresponding_atoms, load_wide_predictions
from oasis.plot import learning_curve_plot, parity_plot
from oasis.mlip.cli import main as mlip_main


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "mlip":
        # Forward remaining args to mlip CLI
        mlip_main(sys.argv[2:])
        return

    cfg = get_config()
    base_dir = cfg.analysis.base_dir if cfg.analysis else Path("data/mlips")
    result_files = find_result_files(base_dir)
    wide_df = load_wide_predictions(result_files)
    atoms_list = load_corresponding_atoms(wide_df, cfg.mlip.dataset)
    graphs = build_adsorption_graphs(wide_df, atoms_list)
    graph_batch = batch_adsorption_graphs(graphs)

    adsorbate_filter = cfg.plot.adsorbate if cfg.plot else None
    anomaly_filter = cfg.plot.anomaly_label if cfg.plot else None
    reaction_contains_filter = cfg.plot.reaction_contains if cfg.plot else None
    if reaction_contains_filter is not None:
        reaction_contains_filter = [s for s in reaction_contains_filter if s]
        if not reaction_contains_filter:
            reaction_contains_filter = None
    wide_df = filter_wide_predictions(
        wide_df,
        adsorbate_filter=adsorbate_filter,
        anomaly_filter=anomaly_filter,
        reaction_contains_filter=reaction_contains_filter,
    )

    output_dir = cfg.plot.output_dir if cfg.plot else Path("data/results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix_parts: list[str] = []
    if adsorbate_filter:
        suffix_parts.append(f"adsorbate_{adsorbate_filter}")
    if anomaly_filter:
        suffix_parts.append(f"anomaly_{anomaly_filter}")
    if reaction_contains_filter:
        joined = "-".join(reaction_contains_filter)
        suffix_parts.append(f"reaction_contains_{joined}")
    suffix = f"_{'_'.join(suffix_parts)}" if suffix_parts else ""
    output_path = output_dir / f"mlips_vs_dft_parity{suffix}.png"
    saved_path = parity_plot(wide_df, output_path=output_path)
    print(
        f"Processed {len(result_files)} MLIP files"
        f"{f' with adsorbate={adsorbate_filter}' if adsorbate_filter else ''}"
        f"{f' with anomaly_label={anomaly_filter}' if anomaly_filter else ''}"
        f"{f' with reaction_contains={reaction_contains_filter}' if reaction_contains_filter else ''}"
        f" -> parity plot: {saved_path}"
    )
    print(f"Rows in combined parity dataset: {len(wide_df)}")
    print(f"Loaded {len(atoms_list)} corresponding adsorbed Atoms objects")
    print(
        f"Built {len(graphs)} adsorption graphs"
        f" with {graph_batch.z.shape[0]} total nodes"
        f" and {graph_batch.edge_index.shape[1]} total edges"
    )
    print(
        f"Graph targets shape: {tuple(graph_batch.y.shape)}, "
        f"MLIP feature shape: {tuple(graph_batch.mlip_energies.shape)}"
    )

    learning_curve_plot(
        cfg=cfg,
        df=wide_df,
        output_path=output_dir / "learning_curve.png",
    )


if __name__ == "__main__":
    main()
