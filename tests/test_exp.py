from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oasis.exp import SweepSplit, generate_sweep_splits
from oasis.method import (
    linearization_sweep,
    linearization_sweep_trimmed,
    residual_sweep,
    residual_sweep_trimmed,
    sweep_model,
    sweep_model_trimmed,
)


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
        self.assertTrue(all(isinstance(split, SweepSplit) for split in splits))

        expected_sizes = [2, 2, 3, 3]
        observed_sizes = [split.sweep_size for split in splits]
        self.assertEqual(observed_sizes, expected_sizes)

        full_idx = np.arange(6)
        for split in splits:
            self.assertEqual(len(split.train_idx), split.sweep_size)
            self.assertEqual(len(split.test_idx), 6 - split.sweep_size)
            self.assertEqual(
                len(np.intersect1d(split.train_idx, split.test_idx)),
                0,
            )
            np.testing.assert_array_equal(
                np.sort(np.concatenate([split.train_idx, split.test_idx])),
                full_idx,
            )

    def test_same_seed_gives_same_splits(self) -> None:
        splits_a = list(
            generate_sweep_splits(
                n_samples=8,
                min_train=2,
                max_train=4,
                n_repeats=3,
                rng=np.random.default_rng(42),
            )
        )
        splits_b = list(
            generate_sweep_splits(
                n_samples=8,
                min_train=2,
                max_train=4,
                n_repeats=3,
                rng=np.random.default_rng(42),
            )
        )

        self.assertEqual(len(splits_a), len(splits_b))
        for split_a, split_b in zip(splits_a, splits_b, strict=True):
            self.assertEqual(split_a.sweep_size, split_b.sweep_size)
            np.testing.assert_array_equal(split_a.train_idx, split_b.train_idx)
            np.testing.assert_array_equal(split_a.test_idx, split_b.test_idx)


class SweepOutputRegressionTests(unittest.TestCase):
    def test_all_methods_consume_same_split_counts_and_keep_result_shape(self) -> None:
        X = np.array(
            [
                [1.0, 1.1, 0.9],
                [2.0, 2.1, 1.9],
                [3.0, 3.1, 2.9],
                [4.0, 4.1, 3.9],
                [5.0, 5.1, 4.9],
                [6.0, 6.1, 5.9],
            ]
        )
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        splits = list(
            generate_sweep_splits(
                n_samples=len(X),
                min_train=2,
                max_train=4,
                n_repeats=2,
                rng=np.random.default_rng(7),
            )
        )
        expected_counts = [2, 3, 4]
        expected_columns = ["n_train", "rmse_mean", "rmse_std"]

        class DummyModel:
            def fit(self, X, y):
                X = np.asarray(X)
                self.coef_ = np.ones(X.shape[1], dtype=float)
                return self

            def predict(self, X):
                return np.asarray(X).mean(axis=1)

        results = [
            sweep_model(lambda: DummyModel(), X, y, splits),
            sweep_model_trimmed(lambda: DummyModel(), X, y, splits),
            residual_sweep(X, y, splits),
            residual_sweep_trimmed(X, y, splits),
            linearization_sweep(X, y, splits),
            linearization_sweep_trimmed(X, y, splits),
        ]

        for df in results:
            self.assertEqual(df["n_train"].tolist(), expected_counts)
            self.assertEqual(df.columns.tolist(), expected_columns)


if __name__ == "__main__":
    unittest.main()
