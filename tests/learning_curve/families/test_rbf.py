from __future__ import annotations

import unittest

import numpy as np

from oasis.learning_curve.families.rbf import GaussianRBF


class GaussianRBFShapeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rbf = GaussianRBF(n_rbf=16, r_min=0.0, r_max=6.0)

    def test_n_distances_maps_to_n_rows(self) -> None:
        out = self.rbf(np.array([1.0, 2.0, 3.0]))
        self.assertEqual(out.shape, (3, 16))

    def test_single_distance_gives_one_row(self) -> None:
        out = self.rbf(np.array([1.5]))
        self.assertEqual(out.shape, (1, 16))

    def test_empty_input_gives_empty_output(self) -> None:
        out = self.rbf(np.array([]))
        self.assertEqual(out.shape, (0, 16))


class GaussianRBFValueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rbf = GaussianRBF(n_rbf=8, r_min=0.0, r_max=7.0)

    def test_all_values_in_unit_interval(self) -> None:
        distances = np.linspace(0.0, 8.0, 50)
        out = self.rbf(distances)
        self.assertTrue(np.all(out > 0.0))
        self.assertTrue(np.all(out <= 1.0 + 1e-12))

    def test_peak_at_each_center_is_one(self) -> None:
        centers = self.rbf.centers
        for k, mu in enumerate(centers):
            val = self.rbf(np.array([mu]))[0, k]
            self.assertAlmostEqual(val, 1.0, places=12)

    def test_other_bases_below_peak_at_center(self) -> None:
        centers = self.rbf.centers
        # At center k, all other basis functions must be strictly less than 1.
        for k, mu in enumerate(centers):
            row = self.rbf(np.array([mu]))[0]
            for j, v in enumerate(row):
                if j != k:
                    self.assertLess(v, 1.0)

    def test_symmetry_around_center(self) -> None:
        centers = self.rbf.centers
        delta = self.rbf.width * 0.3
        mu = centers[3]
        left = self.rbf(np.array([mu - delta]))[0, 3]
        right = self.rbf(np.array([mu + delta]))[0, 3]
        self.assertAlmostEqual(left, right, places=12)

    def test_decay_far_beyond_r_max(self) -> None:
        r_far = self.rbf.r_max + 10 * self.rbf.width
        out = self.rbf(np.array([r_far]))[0]
        self.assertTrue(np.all(out < 1e-6))

    def test_decay_far_below_r_min(self) -> None:
        r_far = self.rbf.r_min - 10 * self.rbf.width
        out = self.rbf(np.array([r_far]))[0]
        self.assertTrue(np.all(out < 1e-6))

    def test_monotone_decrease_moving_away_from_center(self) -> None:
        centers = self.rbf.centers
        mu = centers[4]
        deltas = np.array([0.0, 0.5, 1.0, 1.5]) * self.rbf.width
        vals = [self.rbf(np.array([mu + d]))[0, 4] for d in deltas]
        for i in range(len(vals) - 1):
            self.assertGreaterEqual(vals[i], vals[i + 1])


class GaussianRBFCentersAndWidthTests(unittest.TestCase):
    def test_centers_are_linspace(self) -> None:
        rbf = GaussianRBF(n_rbf=5, r_min=1.0, r_max=5.0)
        np.testing.assert_allclose(rbf.centers, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_first_center_at_r_min(self) -> None:
        rbf = GaussianRBF(n_rbf=10, r_min=0.5, r_max=6.0)
        self.assertAlmostEqual(rbf.centers[0], 0.5)

    def test_last_center_at_r_max(self) -> None:
        rbf = GaussianRBF(n_rbf=10, r_min=0.5, r_max=6.0)
        self.assertAlmostEqual(rbf.centers[-1], 6.0)

    def test_width_equals_center_spacing(self) -> None:
        rbf = GaussianRBF(n_rbf=5, r_min=0.0, r_max=4.0)
        expected_spacing = 1.0  # (4 - 0) / (5 - 1)
        self.assertAlmostEqual(rbf.width, expected_spacing)

    def test_width_single_basis_function(self) -> None:
        rbf = GaussianRBF(n_rbf=1, r_min=0.0, r_max=6.0)
        self.assertAlmostEqual(rbf.width, 6.0)

    def test_adjacent_bases_cross_at_exp_neg_half(self) -> None:
        # At exactly one width from a center, the basis value should be exp(-0.5).
        rbf = GaussianRBF(n_rbf=8, r_min=0.0, r_max=7.0)
        mu = rbf.centers[3]
        val = rbf(np.array([mu + rbf.width]))[0, 3]
        self.assertAlmostEqual(val, np.exp(-0.5), places=12)


class GaussianRBFValidationTests(unittest.TestCase):
    def test_n_rbf_zero_raises(self) -> None:
        with self.assertRaises(ValueError):
            GaussianRBF(n_rbf=0)

    def test_n_rbf_negative_raises(self) -> None:
        with self.assertRaises(ValueError):
            GaussianRBF(n_rbf=-1)

    def test_r_max_equals_r_min_raises(self) -> None:
        with self.assertRaises(ValueError):
            GaussianRBF(n_rbf=8, r_min=3.0, r_max=3.0)

    def test_r_max_less_than_r_min_raises(self) -> None:
        with self.assertRaises(ValueError):
            GaussianRBF(n_rbf=8, r_min=5.0, r_max=2.0)


if __name__ == "__main__":
    unittest.main()
