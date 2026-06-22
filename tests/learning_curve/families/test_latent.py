from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pandas as pd

from oasis.learning_curve.learned_specs import _latent_config_family_factory
from oasis.learning_curve.families.latent import (
    LatentSweepRunner,
    _align_df_to_sample_ids,
    load_latent_df,
)
from oasis.sweep import SweepDataset, SweepRunnerPayload, TrainTestSweepRunnerInput


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


def _make_latent_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reaction": ["A", "B", "C", "D"],
            "feature": [1.0, 2.0, 3.0, 4.0],
        }
    )


def _make_payload(latent_df: pd.DataFrame) -> SweepRunnerPayload:
    n = len(latent_df)
    dataset = SweepDataset(
        mlip_features=np.ones((n, 1)),
        targets=np.zeros(n),
        auxiliary_views={"latent": latent_df},
    )
    split = TrainTestSweepRunnerInput(
        dataset=dataset,
        sweep_size=2,
        train_idx=np.array([0, 1]),
        test_idx=np.array([2, 3]),
    )
    return SweepRunnerPayload(splits=(split,))


class TestLatentSweepRunner(unittest.TestCase):
    def setUp(self) -> None:
        self._mock_model_module = MagicMock()
        self._mock_train_module = MagicMock()
        sys.modules.setdefault("latent", MagicMock())
        sys.modules["latent.config"] = MagicMock()
        sys.modules["latent.data"] = MagicMock()
        sys.modules["latent.model"] = self._mock_model_module
        sys.modules["latent.train"] = self._mock_train_module

    def tearDown(self) -> None:
        for key in ["latent.config", "latent.data", "latent.model", "latent.train"]:
            sys.modules.pop(key, None)

    def _setup_trained_mock(
        self, x_test: np.ndarray, y_test: np.ndarray, preds: np.ndarray
    ) -> MagicMock:
        mock_trained = MagicMock()
        mock_trained.train_params = np.array([0.5])
        mock_trained.architecture.linearize_data.return_value = (x_test, y_test)
        mock_trained.estimator.predict.return_value = preds
        self._mock_train_module.train_model.return_value = mock_trained
        return mock_trained

    def test_importable(self) -> None:
        self.assertTrue(callable(LatentSweepRunner))

    def test_run_returns_sweep_results_frame(self) -> None:
        x_test = np.array([[3.0], [4.0]])
        y_test = np.array([0.1, 0.2])
        preds = np.array([0.1, 0.2])
        self._setup_trained_mock(x_test, y_test, preds)

        runner = LatentSweepRunner(exp_cfg=MagicMock(), vendor_dir=Path("."))
        result = runner.run(_make_payload(_make_latent_df()))

        self.assertIsInstance(result, pd.DataFrame)
        self.assertListEqual(
            list(result.columns),
            [
                "n_train",
                "rmse_mean",
                "rmse_std",
                "fit_time_mean_s",
                "fit_time_std_s",
            ],
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result["n_train"].iloc[0], 2)

    def test_run_records_fit_time_columns(self) -> None:
        self._setup_trained_mock(
            np.array([[0.0], [0.0]]),
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
        )
        runner = LatentSweepRunner(exp_cfg=MagicMock(), vendor_dir=Path("."))

        with patch(
            "oasis.learning_curve.execution.perf_counter",
            side_effect=[0.0, 0.25],
        ):
            result = runner.run(_make_payload(_make_latent_df()))

        np.testing.assert_allclose(
            result["fit_time_mean_s"].to_numpy(),
            [0.25],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result["fit_time_std_s"].to_numpy(),
            [0.0],
            atol=1e-12,
        )

    def test_run_passes_cobyla_params(self) -> None:
        self._setup_trained_mock(
            np.array([[0.0], [0.0]]),
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
        )
        runner = LatentSweepRunner(
            exp_cfg=MagicMock(),
            vendor_dir=Path("."),
            cobyla_initial_guess=0.5,
            cobyla_max_iter=50,
        )
        runner.run(_make_payload(_make_latent_df()))

        call_kwargs = self._mock_train_module.train_model.call_args
        self.assertEqual(call_kwargs.kwargs["cobyla_initial_guess"], 0.5)
        self.assertEqual(call_kwargs.kwargs["cobyla_max_iter"], 50)

    def test_run_slices_latent_df_by_split_indices(self) -> None:
        latent_df = _make_latent_df()
        captured: list[pd.DataFrame] = []
        mock_trained = MagicMock()
        mock_trained.train_params = np.array([0.5])
        mock_trained.architecture.linearize_data.return_value = (
            np.array([[0.0], [0.0]]),
            np.array([0.0, 0.0]),
        )
        mock_trained.estimator.predict.return_value = np.array([0.0, 0.0])

        def capture_train_df(train_df, untrained, **kw):
            captured.append(train_df)
            return mock_trained

        self._mock_train_module.train_model.side_effect = capture_train_df

        runner = LatentSweepRunner(exp_cfg=MagicMock(), vendor_dir=Path("."))
        runner.run(_make_payload(latent_df))

        self.assertEqual(len(captured), 1)
        self.assertEqual(list(captured[0]["reaction"]), ["A", "B"])

    def test_latent_family_factory_reads_cobyla_settings_from_vendor_config(self) -> None:
        family = _latent_config_family_factory(
            SimpleNamespace(
                latent=SimpleNamespace(
                    experiment_config_path="vendor/latent/config.toml",
                )
            )
        )

        runner = family.spec.runner
        self.assertIsInstance(runner, LatentSweepRunner)
        self.assertEqual(runner.cobyla_initial_guess, 0.1)
        self.assertEqual(runner.cobyla_max_iter, 100)


if __name__ == "__main__":
    unittest.main()
