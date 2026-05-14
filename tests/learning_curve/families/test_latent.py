from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from oasis.learning_curve.families.latent import _align_df_to_sample_ids, load_latent_df


class TestAlignDfToSampleIds(unittest.TestCase):
    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "reaction": ["A", "B", "C", "D"],
                "feature_x": [1.0, 2.0, 3.0, 4.0],
            }
        )

    def test_filters_and_reorders(self) -> None:
        df = self._make_df()
        result = _align_df_to_sample_ids(df, np.array(["C", "A"]))
        self.assertEqual(list(result["reaction"]), ["C", "A"])
        self.assertEqual(list(result["feature_x"]), [3.0, 1.0])

    def test_subset_only(self) -> None:
        df = self._make_df()
        result = _align_df_to_sample_ids(df, np.array(["B", "D"]))
        self.assertEqual(len(result), 2)
        self.assertEqual(list(result["reaction"]), ["B", "D"])

    def test_no_reaction_column_returns_df_unchanged(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0]})
        result = _align_df_to_sample_ids(df, np.array(["A"]))
        pd.testing.assert_frame_equal(result, df)

    def test_custom_reaction_column(self) -> None:
        df = pd.DataFrame({"rxn": ["X", "Y"], "val": [10.0, 20.0]})
        result = _align_df_to_sample_ids(df, np.array(["Y"]), reaction_column="rxn")
        self.assertEqual(list(result["rxn"]), ["Y"])
        self.assertEqual(list(result["val"]), [20.0])

    def test_missing_sample_id_raises(self) -> None:
        df = self._make_df()
        with self.assertRaises(KeyError):
            _align_df_to_sample_ids(df, np.array(["Z"]))


class TestLoadLatentDfImport(unittest.TestCase):
    def test_importable(self) -> None:
        self.assertTrue(callable(load_latent_df))


if __name__ == "__main__":
    unittest.main()
