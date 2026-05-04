from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import polars as pl

from oasis.plot import moe_learning_speed_plot


class PlotTests(unittest.TestCase):
    def test_moe_learning_speed_plot_from_csv(self) -> None:
        sweep_df = pl.DataFrame(
            {
                "fraction": [0.25, 0.5, 1.0],
                "train_size": [10, 20, 40],
                "val_size": [8, 8, 8],
                "train_loss_final": [0.30, 0.20, 0.10],
                "val_loss_final": [0.35, 0.24, 0.12],
                "val_mae": [0.50, 0.35, 0.20],
                "val_rmse": [0.60, 0.40, 0.22],
            }
        )
        with TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "sweep.csv"
            png_path = Path(tmpdir) / "learning_speed.png"
            sweep_df.write_csv(csv_path)

            saved_path = moe_learning_speed_plot(csv_path, png_path)

            self.assertEqual(saved_path, png_path)
            self.assertTrue(saved_path.is_file())


if __name__ == "__main__":
    unittest.main()
