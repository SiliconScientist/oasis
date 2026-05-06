from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oasis.exp import generate_sweep_splits


class GenerateSweepSplitsTests(unittest.TestCase):
    def test_generate_sweep_splits_yields_disjoint_full_partitions(self) -> None:
        rng = np.random.default_rng(123)

        splits = list(
            generate_sweep_splits(
                n_samples=6,
                min_train=2,
                max_train=3,
                n_repeats=2,
                rng=rng,
            )
        )

        self.assertEqual(len(splits), 4)

        expected_sizes = [2, 2, 3, 3]
        observed_sizes = [n_train for n_train, _, _ in splits]
        self.assertEqual(observed_sizes, expected_sizes)

        full_idx = np.arange(6)
        for n_train, train_idx, test_idx in splits:
            self.assertEqual(len(train_idx), n_train)
            self.assertEqual(len(test_idx), 6 - n_train)
            self.assertEqual(len(np.intersect1d(train_idx, test_idx)), 0)
            np.testing.assert_array_equal(
                np.sort(np.concatenate([train_idx, test_idx])),
                full_idx,
            )


if __name__ == "__main__":
    unittest.main()
