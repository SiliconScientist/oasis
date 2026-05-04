from __future__ import annotations

import unittest

from ase import Atoms
import polars as pl
import torch

from oasis.graph import (
    atoms_to_graph,
    batch_adsorption_graphs,
    build_adsorption_graphs,
)


class GraphConversionTests(unittest.TestCase):
    def test_atoms_to_graph_shapes_and_neighbor_distances(self) -> None:
        atoms = Atoms(
            "H2",
            positions=[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.74],
            ],
            cell=[10.0, 10.0, 10.0],
            pbc=[False, False, False],
        )

        graph = atoms_to_graph(
            atoms,
            reaction="H2(g) + * -> H2*",
            adsorbate="H2",
            reference_ads_eng=-0.1,
            mlip_energies=torch.tensor([-0.2, -0.15], dtype=torch.float32),
            mlip_names=["a_mlip_ads_eng_median", "b_mlip_ads_eng_median"],
            cutoff=1.0,
        )

        self.assertEqual(tuple(graph.z.shape), (2,))
        self.assertEqual(tuple(graph.pos.shape), (2, 3))
        self.assertEqual(tuple(graph.edge_index.shape), (2, 2))
        self.assertEqual(tuple(graph.edge_weight.shape), (2,))
        self.assertEqual(tuple(graph.batch.shape), (2,))
        self.assertEqual(tuple(graph.y.shape), (1,))
        self.assertEqual(tuple(graph.mlip_energies.shape), (2,))
        self.assertEqual(graph.z.tolist(), [1, 1])
        self.assertEqual(graph.batch.tolist(), [0, 0])
        self.assertEqual(
            sorted(round(x, 2) for x in graph.edge_weight.tolist()),
            [0.74, 0.74],
        )

    def test_build_and_batch_adsorption_graphs(self) -> None:
        wide_df = pl.DataFrame(
            {
                "reaction": ["CuOH_* -> OH*", "CuH_* -> H*"],
                "adsorbate": ["OH", "H"],
                "reference_ads_eng": [-1.25, -0.45],
                "mace_mlip_ads_eng_median": [-1.10, -0.40],
                "mace_label": ["normal", "normal"],
                "orb_mlip_ads_eng_median": [-1.30, -0.50],
                "orb_label": ["normal", "energy_anomaly"],
            }
        )
        atoms_list = [
            Atoms(
                "Cu2OH",
                positions=[
                    [0.0, 0.0, 0.0],
                    [2.5, 0.0, 0.0],
                    [1.2, 0.1, 1.3],
                    [1.2, 0.1, 2.2],
                ],
                cell=[12.0, 12.0, 12.0],
                pbc=[False, False, False],
            ),
            Atoms(
                "Cu2H",
                positions=[
                    [0.0, 0.0, 0.0],
                    [2.5, 0.0, 0.0],
                    [1.3, 0.0, 1.1],
                ],
                cell=[12.0, 12.0, 12.0],
                pbc=[False, False, False],
            ),
        ]

        graphs = build_adsorption_graphs(wide_df, atoms_list, cutoff=3.0)
        self.assertEqual(len(graphs), 2)
        self.assertEqual(graphs[0].reaction, "CuOH_* -> OH*")
        self.assertEqual(graphs[0].adsorbate, "OH")
        self.assertEqual(
            graphs[0].mlip_names,
            ("mace_mlip_ads_eng_median", "orb_mlip_ads_eng_median"),
        )
        self.assertEqual(tuple(graphs[0].mlip_energies.shape), (2,))
        self.assertGreater(graphs[0].edge_index.shape[1], 0)
        self.assertGreater(graphs[1].edge_index.shape[1], 0)

        batch = batch_adsorption_graphs(graphs)
        self.assertEqual(tuple(batch.z.shape), (7,))
        self.assertEqual(tuple(batch.pos.shape), (7, 3))
        self.assertEqual(tuple(batch.batch.shape), (7,))
        self.assertEqual(tuple(batch.ptr.shape), (3,))
        self.assertEqual(tuple(batch.y.shape), (2,))
        self.assertEqual(tuple(batch.mlip_energies.shape), (2, 2))
        self.assertEqual(batch.ptr.tolist(), [0, 4, 7])
        self.assertEqual(batch.reactions, ["CuOH_* -> OH*", "CuH_* -> H*"])
        self.assertEqual(batch.adsorbates, ["OH", "H"])


if __name__ == "__main__":
    unittest.main()
