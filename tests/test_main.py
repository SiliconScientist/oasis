from __future__ import annotations

import unittest

from oasis.exp import select_moe_train_sizes


class MainTests(unittest.TestCase):
    def test_select_moe_train_sizes_clamps_below_two(self) -> None:
        self.assertEqual(
            select_moe_train_sizes([1, 2, 3, 6], available_train=4),
            [2, 3],
        )


if __name__ == "__main__":
    unittest.main()
