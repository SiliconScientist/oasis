from __future__ import annotations

import unittest

from ase import Atoms
import polars as pl

from oasis.dataset import GatingDataset, collate_gating_samples
from oasis.graph import build_adsorption_graphs


class GatingDatasetTests(unittest.TestCase):
    def test_dataset_samples_and_collation(self) -> None:
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
        dataset = GatingDataset(graphs, wide_df)

        self.assertEqual(len(dataset), 2)

        sample0 = dataset[0]
        self.assertEqual(sample0.graph.reaction, "CuOH_* -> OH*")
        self.assertEqual(sample0.metadata["adsorbate"], "OH")
        self.assertEqual(sample0.metadata["n_atoms"], 4)
        self.assertEqual(tuple(sample0.mlip_energies.shape), (2,))
        self.assertEqual(tuple(sample0.target_ads_eng.shape), (1,))
        self.assertEqual(
            sample0.expert_labels,
            {"mace": "normal", "orb": "normal"},
        )

        batch = collate_gating_samples([dataset[0], dataset[1]])
        self.assertEqual(tuple(batch.mlip_energies.shape), (2, 2))
        self.assertEqual(tuple(batch.target_ads_eng.shape), (2,))
        self.assertEqual(batch.metadata[1]["reaction"], "CuH_* -> H*")
        self.assertEqual(
            batch.expert_labels[1],
            {"mace": "normal", "orb": "energy_anomaly"},
        )
        self.assertEqual(batch.graph_batch.ptr.tolist(), [0, 4, 7])


if __name__ == "__main__":
    unittest.main()
