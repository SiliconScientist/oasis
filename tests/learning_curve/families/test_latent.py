from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from types import ModuleType
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


class TestLatentVendorTiming(unittest.TestCase):
    def _load_vendor_timing_module(self):
        module_path = (
            Path(__file__).resolve().parents[3] / "vendor" / "latent" / "timing.py"
        )
        spec = importlib.util.spec_from_file_location("latent.timing", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def _load_vendor_data_module(self):
        module_path = (
            Path(__file__).resolve().parents[3] / "vendor" / "latent" / "data.py"
        )
        timing_module = self._load_vendor_timing_module()

        latent_pkg = ModuleType("latent")
        latent_pkg.__path__ = [str(module_path.parent)]

        fake_modules = {
            "latent": latent_pkg,
            "latent.timing": timing_module,
            "latent.config": ModuleType("latent.config"),
            "latent.feature_generation": ModuleType("latent.feature_generation"),
            "ase": ModuleType("ase"),
            "ase.calculators": ModuleType("ase.calculators"),
            "ase.calculators.singlepoint": ModuleType("ase.calculators.singlepoint"),
            "ase.db": ModuleType("ase.db"),
            "tqdm": ModuleType("tqdm"),
            "montegroup": ModuleType("montegroup"),
            "montegroup.ghit": ModuleType("montegroup.ghit"),
            "pymatgen": ModuleType("pymatgen"),
            "pymatgen.core": ModuleType("pymatgen.core"),
            "pymatgen.core.periodic_table": ModuleType(
                "pymatgen.core.periodic_table"
            ),
        }
        fake_modules["latent.config"].Config = object
        fake_modules["latent.config"].ExperimentConfig = object
        fake_modules["latent.feature_generation"].get_host_feature_list = MagicMock()
        fake_modules["latent.feature_generation"].findSurfaceAtoms = MagicMock()
        fake_modules["ase"].Atoms = object
        fake_modules["ase.calculators.singlepoint"].SinglePointCalculator = object
        fake_modules["ase.db"].connect = MagicMock()
        fake_modules["tqdm"].tqdm = lambda iterable=None, **kwargs: iterable
        fake_modules["montegroup.ghit"].GHIT = MagicMock()
        fake_modules["pymatgen.core.periodic_table"].Element = MagicMock()

        with patch.dict(sys.modules, fake_modules):
            spec = importlib.util.spec_from_file_location("latent.data", module_path)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
        return module

    def test_output_data_writes_generation_timing_sidecar(self) -> None:
        vendor_data = self._load_vendor_data_module()
        vendor_timing = self._load_vendor_timing_module()

        with TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "latent.csv"
            cfg = SimpleNamespace(
                data_generation=SimpleNamespace(data_filepath=str(csv_path))
            )
            expected = pd.DataFrame({"x": [1.0], "y": [2.0]})

            with patch.object(
                vendor_data,
                "make_data",
                return_value=expected,
            ), patch.object(
                vendor_data,
                "perf_counter",
                side_effect=[10.0, 12.5],
            ):
                result = vendor_data.output_data(cfg=cfg, exp_cfg=object())

            pd.testing.assert_frame_equal(result, expected)
            sidecar_path = vendor_timing.generation_timing_sidecar_path(csv_path)
            self.assertTrue(csv_path.is_file())
            self.assertTrue(sidecar_path.is_file())
            payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["generation_time_s"], 2.5)
            self.assertEqual(payload["output_csv_path"], str(csv_path))


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
