from __future__ import annotations

import unittest

import numpy as np

from oasis.learning_curve.families.gating_policy import (
    DenseGatingPolicy,
    GatingPolicy,
    TopKGatingPolicy,
)


class DenseGatingPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = DenseGatingPolicy()

    def test_satisfies_protocol(self) -> None:
        self.assertIsInstance(self.policy, GatingPolicy)

    # --- 1-D (single sample) ---

    def test_1d_sums_to_one(self) -> None:
        weights = self.policy.apply(np.array([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(float(weights.sum()), 1.0)

    def test_1d_all_positive(self) -> None:
        weights = self.policy.apply(np.array([1.0, 2.0, 3.0]))
        self.assertTrue(np.all(weights > 0))

    def test_1d_monotone_wrt_logits(self) -> None:
        weights = self.policy.apply(np.array([3.0, 1.0, 2.0]))
        # Logit order 3 > 2 > 1 → weight order must match.
        self.assertGreater(weights[0], weights[2])
        self.assertGreater(weights[2], weights[1])

    def test_1d_uniform_logits_give_uniform_weights(self) -> None:
        weights = self.policy.apply(np.zeros(4))
        np.testing.assert_allclose(weights, 0.25)

    # --- 2-D (batch of samples) ---

    def test_2d_each_row_sums_to_one(self) -> None:
        logits = np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, 1.0]])
        weights = self.policy.apply(logits)
        np.testing.assert_allclose(weights.sum(axis=-1), 1.0)

    def test_2d_all_positive(self) -> None:
        logits = np.array([[0.0, 1.0, -1.0], [2.0, -2.0, 0.0]])
        weights = self.policy.apply(logits)
        self.assertTrue(np.all(weights > 0))

    def test_2d_output_shape_preserved(self) -> None:
        logits = np.zeros((5, 3))
        self.assertEqual(self.policy.apply(logits).shape, (5, 3))

    # --- Numerical stability ---

    def test_large_logit_spread_does_not_overflow(self) -> None:
        logits = np.array([1000.0, 0.0, -1000.0])
        weights = self.policy.apply(logits)
        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertAlmostEqual(float(weights.sum()), 1.0)

    # --- Regularization loss ---

    def test_regularization_loss_always_zero_1d(self) -> None:
        self.assertEqual(self.policy.regularization_loss(np.array([1.0, 2.0])), 0.0)

    def test_regularization_loss_always_zero_2d(self) -> None:
        logits = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(self.policy.regularization_loss(logits), 0.0)


class TopKGatingPolicyTests(unittest.TestCase):
    # --- apply: correct count of non-zero weights ---

    def test_1d_exactly_k_nonzero(self) -> None:
        policy = TopKGatingPolicy(k=2)
        weights = policy.apply(np.array([3.0, 1.0, 4.0, 1.0, 5.0]))
        self.assertEqual(int((weights > 0).sum()), 2)

    def test_2d_each_row_exactly_k_nonzero(self) -> None:
        policy = TopKGatingPolicy(k=2)
        logits = np.array([[1.0, 3.0, 2.0], [4.0, 1.0, 3.0]])
        weights = policy.apply(logits)
        np.testing.assert_array_equal((weights > 0).sum(axis=-1), [2, 2])

    # --- apply: survivors sum to 1 ---

    def test_1d_survivors_sum_to_one(self) -> None:
        policy = TopKGatingPolicy(k=2)
        weights = policy.apply(np.array([1.0, 5.0, 2.0]))
        self.assertAlmostEqual(float(weights.sum()), 1.0)

    def test_2d_each_row_sums_to_one(self) -> None:
        policy = TopKGatingPolicy(k=2)
        logits = np.array([[1.0, 3.0, 2.0], [4.0, 1.0, 3.0]])
        weights = policy.apply(logits)
        np.testing.assert_allclose(weights.sum(axis=-1), 1.0)

    # --- apply: correct experts selected ---

    def test_top_experts_are_selected(self) -> None:
        policy = TopKGatingPolicy(k=2)
        # Experts 0 and 2 have the two largest logits.
        weights = policy.apply(np.array([5.0, 0.0, 3.0, 1.0]))
        self.assertGreater(weights[0], 0.0)
        self.assertEqual(weights[1], 0.0)
        self.assertGreater(weights[2], 0.0)
        self.assertEqual(weights[3], 0.0)

    def test_2d_correct_experts_selected_per_row(self) -> None:
        policy = TopKGatingPolicy(k=1)
        # Row 0: expert 1 wins. Row 1: expert 0 wins.
        logits = np.array([[0.0, 10.0], [10.0, 0.0]])
        weights = policy.apply(logits)
        self.assertEqual(weights[0, 1], 1.0)
        self.assertEqual(weights[0, 0], 0.0)
        self.assertEqual(weights[1, 0], 1.0)
        self.assertEqual(weights[1, 1], 0.0)

    # --- apply: selected survivors have monotone relative ordering ---

    def test_higher_logit_among_survivors_gets_higher_weight(self) -> None:
        policy = TopKGatingPolicy(k=2)
        # Experts 0 (logit=5) and 2 (logit=3) survive; expert 0 should weigh more.
        weights = policy.apply(np.array([5.0, 0.0, 3.0]))
        self.assertGreater(weights[0], weights[2])

    # --- edge cases ---

    def test_k_equals_one_active_weight_is_one(self) -> None:
        policy = TopKGatingPolicy(k=1)
        weights = policy.apply(np.array([1.0, 3.0, 2.0]))
        self.assertAlmostEqual(float(weights.max()), 1.0)
        self.assertAlmostEqual(float(weights.sum()), 1.0)

    def test_k_equals_n_experts_matches_dense(self) -> None:
        n = 4
        logits = np.array([1.0, -1.0, 2.0, 0.5])
        dense_weights = DenseGatingPolicy().apply(logits)
        topk_weights = TopKGatingPolicy(k=n).apply(logits)
        np.testing.assert_allclose(topk_weights, dense_weights, atol=1e-12)

    def test_2d_k_equals_n_experts_matches_dense(self) -> None:
        logits = np.array([[1.0, 2.0, 0.0], [3.0, -1.0, 0.5]])
        dense_weights = DenseGatingPolicy().apply(logits)
        topk_weights = TopKGatingPolicy(k=3).apply(logits)
        np.testing.assert_allclose(topk_weights, dense_weights, atol=1e-12)

    def test_output_shape_preserved(self) -> None:
        policy = TopKGatingPolicy(k=2)
        self.assertEqual(policy.apply(np.zeros((7, 4))).shape, (7, 4))

    def test_large_logit_spread_does_not_overflow(self) -> None:
        policy = TopKGatingPolicy(k=2)
        logits = np.array([1000.0, -1000.0, 999.0])
        weights = policy.apply(logits)
        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertAlmostEqual(float(weights.sum()), 1.0)

    # --- regularization_loss ---

    def test_reg_loss_returns_zero_for_1d(self) -> None:
        policy = TopKGatingPolicy(k=1)
        self.assertEqual(policy.regularization_loss(np.array([1.0, 2.0])), 0.0)

    def test_reg_loss_balanced_equals_k(self) -> None:
        # k=1, n_experts=2: alternating samples route one to each expert → perfect balance.
        # f=[0.5,0.5], P=[0.5,0.5] → loss = 2*(0.5*0.5+0.5*0.5) = 1.0 = k.
        policy = TopKGatingPolicy(k=1)
        logits = np.array([[10.0, 0.0], [0.0, 10.0]])
        self.assertAlmostEqual(policy.regularization_loss(logits), 1.0)

    def test_reg_loss_imbalanced_greater_than_balanced(self) -> None:
        policy = TopKGatingPolicy(k=1)
        balanced = np.array([[10.0, 0.0], [0.0, 10.0]])
        imbalanced = np.array([[10.0, 0.0], [10.0, 0.0]])
        self.assertLess(
            policy.regularization_loss(balanced),
            policy.regularization_loss(imbalanced),
        )

    def test_reg_loss_all_to_one_expert(self) -> None:
        # k=1, all samples routed to expert 0: f=[1,0], P=[1,0] → loss = 2*(1*1+0*0) = 2.
        policy = TopKGatingPolicy(k=1)
        logits = np.array([[10.0, 0.0], [10.0, 0.0]])
        self.assertAlmostEqual(policy.regularization_loss(logits), 2.0)

    def test_reg_loss_returns_float(self) -> None:
        policy = TopKGatingPolicy(k=1)
        result = policy.regularization_loss(np.array([[1.0, 0.0], [0.0, 1.0]]))
        self.assertIsInstance(result, float)


if __name__ == "__main__":
    unittest.main()
