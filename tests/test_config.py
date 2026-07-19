from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

try:
    from oasis.config import Config, get_config, load_config_data

    HAS_CONFIG = True
except ModuleNotFoundError:
    HAS_CONFIG = False


@unittest.skipUnless(HAS_CONFIG, "requires config dependencies")
class ConfigParsingTests(unittest.TestCase):
    def test_load_config_data_merges_two_explicit_files_with_later_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            base_path = tmp / "base.toml"
            experiment_path = tmp / "experiment.toml"
            base_path.write_text(
                "\n".join(
                    [
                        "seed = 7",
                        "",
                        "[plot]",
                        'output_dir = "data/results/mlip-plots"',
                        "",
                        "[dataset_profile]",
                        'tag = "example"',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            experiment_path.write_text(
                "\n".join(
                    [
                        "seed = 11",
                        "",
                        "[plot]",
                        'output_dir = "data/results/experiment-plots"',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            data = load_config_data([base_path, experiment_path])

        self.assertEqual(data["seed"], 11)
        self.assertEqual(data["plot"]["output_dir"], "data/results/experiment-plots")
        self.assertEqual(data["dataset_profile"]["tag"], "example")

    def test_load_config_data_uses_experiment_only_default_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            experiment_path = tmp / "experiment.toml"
            experiment_path.write_text(
                "\n".join(
                    [
                        "seed = 7",
                        "",
                        "[dataset_profile]",
                        'tag = "example"',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(tmp)
                data = load_config_data()
            finally:
                os.chdir(old_cwd)

        self.assertEqual(data["seed"], 7)
        self.assertEqual(data["dataset_profile"]["tag"], "example")

    def test_load_config_data_allows_experiment_only_default_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            experiment_path = tmp / "experiment.toml"
            experiment_path.write_text(
                "\n".join(
                    [
                        "seed = 13",
                        "",
                        "[plot]",
                        'output_dir = "data/results/plots"',
                        "",
                        "[experiment.learning_curve]",
                        "min_train = 2",
                        "max_train = 4",
                        "n_repeats = 3",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(tmp)
                data = load_config_data()
            finally:
                os.chdir(old_cwd)

        self.assertEqual(data["seed"], 13)
        self.assertEqual(data["plot"]["output_dir"], "data/results/plots")
        self.assertEqual(data["experiment"]["learning_curve"]["n_repeats"], 3)

    def test_load_config_data_rejects_missing_explicit_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            experiment_path = tmp / "experiment.toml"

            with self.assertRaisesRegex(
                FileNotFoundError,
                r"No config files found\. Looked for:",
            ):
                load_config_data([tmp / "base.toml", experiment_path])

    def test_get_config_merges_explicit_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            base_path = tmp / "base.toml"
            experiment_path = tmp / "experiment.toml"
            base_path.write_text(
                "\n".join(
                    [
                        'seed = 7',
                        "",
                        "[dataset_profile]",
                        'tag = "example"',
                        "",
                        "[datasets.example]",
                        'raw_dataset_filename = "example.json"',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            experiment_path.write_text(
                "\n".join(
                    [
                        "[plot]",
                        "",
                        "[dataset_profile]",
                        'tag = "example"',
                        "",
                        "[datasets.example]",
                        'mlip_run_dirname = "Example-Run"',
                        "",
                        "[experiment]",
                        "[experiment.learning_curve]",
                        "min_train = 2",
                        "max_train = 4",
                        "n_repeats = 3",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            cfg = get_config([base_path, experiment_path])

        self.assertEqual(cfg.seed, 7)
        self.assertEqual(cfg.resolved_dataset_path, Path("data/raw_data/example.json"))
        assert cfg.plot is not None
        self.assertEqual(cfg.plot.output_dir, Path("data/results/plots/Example-Run"))
        assert cfg.experiment is not None
        assert cfg.experiment.learning_curve is not None
        self.assertEqual(cfg.experiment.learning_curve.n_repeats, 3)

    def test_get_config_loads_experiment_only_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            experiment_path = tmp / "experiment.toml"
            experiment_path.write_text(
                "\n".join(
                    [
                        "[dataset_profile]",
                        'tag = "example"',
                        "",
                        "[datasets.example]",
                        'raw_dataset_filename = "example.json"',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            cfg = get_config(experiment_path)

        self.assertEqual(cfg.resolved_dataset_path, Path("data/raw_data/example.json"))
        self.assertEqual(cfg.resolved_mlip_results_dir, Path("data/mlips/example"))
        self.assertIsNone(cfg.experiment)
        self.assertIsNone(cfg.analysis)
        self.assertIsNone(cfg.plot)

    def test_get_config_derives_plot_output_dir_from_dataset_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            experiment_path = tmp / "experiment.toml"
            experiment_path.write_text(
                "\n".join(
                    [
                        "[dataset_profile]",
                        'tag = "example"',
                        "",
                        "[datasets.example]",
                        'mlip_run_dirname = "Example-Run"',
                        "",
                        "[plot]",
                        "zero_shot_stage_show_lone_mlip_swarm = false",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            cfg = get_config(experiment_path)

        assert cfg.plot is not None
        self.assertEqual(cfg.plot.output_dir, Path("data/results/plots/Example-Run"))
        self.assertFalse(cfg.plot.zero_shot_stage_show_lone_mlip_swarm)

    def test_get_config_parses_screening_policy_diagnostic_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            experiment_path = tmp / "experiment.toml"
            experiment_path.write_text(
                "\n".join(
                    [
                        "[dataset_profile]",
                        'tag = "example"',
                        "",
                        "[datasets.example]",
                        'raw_dataset_filename = "example.json"',
                        "",
                        "[experiment.learning_curve]",
                        "min_train = 2",
                        "max_train = 4",
                        "n_repeats = 3",
                        "",
                        "[experiment.screening]",
                        "screen_fraction = 0.25",
                        'policy_names = ["min_screening_rmse", "combined_screening_rmse_miscalibration"]',
                        "combined_miscalibration_lambda = 2.5",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            cfg = get_config(experiment_path)

        assert cfg.experiment is not None
        assert cfg.experiment.screening is not None
        self.assertEqual(
            cfg.experiment.screening.policy_names,
            [
                "min_screening_rmse",
                "combined_screening_rmse_miscalibration",
            ],
        )
        self.assertEqual(cfg.experiment.screening.combined_miscalibration_lambda, 2.5)
        self.assertEqual(
            cfg.experiment.screening.plot_baselines.low_data_domain.method_name,
            "residual",
        )
        self.assertEqual(
            cfg.experiment.screening.plot_baselines.high_data_domain.method_name,
            "kernel_ridge",
        )

    def test_get_config_parses_screening_plot_baseline_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            experiment_path = tmp / "experiment.toml"
            experiment_path.write_text(
                "\n".join(
                    [
                        "[dataset_profile]",
                        'tag = "example"',
                        "",
                        "[datasets.example]",
                        'raw_dataset_filename = "example.json"',
                        "",
                        "[experiment.learning_curve]",
                        "min_train = 2",
                        "max_train = 4",
                        "n_repeats = 3",
                        "",
                        "[experiment.screening]",
                        "screen_fraction = 0.25",
                        "",
                        "[experiment.screening.plot_baselines.low_data_domain]",
                        "enabled = false",
                        'method_name = "weighted_simplex"',
                        'label = "Low-data weighted simplex"',
                        "",
                        "[experiment.screening.plot_baselines.high_data_domain]",
                        "enabled = true",
                        'method_name = "ridge"',
                        'label = "Late ridge"',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            cfg = get_config(experiment_path)

        assert cfg.experiment is not None
        assert cfg.experiment.screening is not None
        self.assertFalse(cfg.experiment.screening.plot_baselines.low_data_domain.enabled)
        self.assertEqual(
            cfg.experiment.screening.plot_baselines.low_data_domain.method_name,
            "weighted_simplex",
        )
        self.assertEqual(
            cfg.experiment.screening.plot_baselines.low_data_domain.label,
            "Low-data weighted simplex",
        )
        self.assertTrue(cfg.experiment.screening.plot_baselines.high_data_domain.enabled)
        self.assertEqual(
            cfg.experiment.screening.plot_baselines.high_data_domain.method_name,
            "ridge",
        )
        self.assertEqual(
            cfg.experiment.screening.plot_baselines.high_data_domain.label,
            "Late ridge",
        )

    def test_load_config_data_loads_experiment_only_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            experiment_path = tmp / "experiment.toml"
            experiment_path.write_text(
                "\n".join(
                    [
                        "seed = 23",
                        "",
                        "[dataset_profile]",
                        'tag = "example_oh"',
                        "",
                        "[datasets.example_oh]",
                        'raw_dataset_filename = "ExampleOH_adsorption.json"',
                        'processed_basename = "example_oh"',
                        "",
                        "[analysis]",
                        "run_adsorption_analysis = false",
                        'out_dir = "data/mlips_by_prefix"',
                        'prefixes = ["ol"]',
                        "",
                        "[plot]",
                        'output_dir = "data/results/plots"',
                        "",
                        "[experiment.learning_curve]",
                        "min_train = 2",
                        "max_train = 4",
                        "n_repeats = 3",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            data = load_config_data(experiment_path)

        self.assertEqual(data["seed"], 23)
        self.assertEqual(data["dataset_profile"]["tag"], "example_oh")
        self.assertEqual(data["analysis"]["prefixes"], ["ol"])
        self.assertEqual(data["experiment"]["learning_curve"]["max_train"], 4)

    def test_load_config_data_normalizes_top_level_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            experiment_path = tmp / "experiment.toml"
            experiment_path.write_text(
                "\n".join(
                    [
                        "seed = 7",
                        "",
                        "[ingest]",
                        'source = "data/raw_vasp/systems"',
                        'dataset_name = "test"',
                        "",
                        "[ingest.stoich]",
                        'elements = ["H"]',
                        'basis_species = ["H2"]',
                        "",
                        "[ingest.stoich.basis_composition]",
                        'H2 = { H = 2 }',
                        "",
                        "[mlip]",
                        "dev_n = 1",
                        "dev_run = false",
                        'dataset = "data/raw_data/example.json"',
                        "",
                        "[mlip.models]",
                        'enabled = ["mace"]',
                        "",
                        "[mlip.rootstock]",
                        'root = "."',
                        "",
                        "[mlip.rootstock.models.mace]",
                        'model = "mace"',
                        'mlip_name = "mace-test"',
                        "",
                        "[experiment.learning_curve]",
                        "min_train = 2",
                        "max_train = 4",
                        "n_repeats = 3",
                        "",
                        "[models]",
                        "use_ridge = true",
                        "use_kernel_ridge = false",
                        "use_lasso = false",
                        "use_elastic_net = false",
                        "use_residual = true",
                        "",
                        "[models.moe]",
                        "enabled = true",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            data = load_config_data(experiment_path)
            cfg = get_config(experiment_path)

        self.assertTrue(data["experiment"]["learning_curve"]["models"]["use_ridge"])
        assert cfg.experiment is not None
        assert cfg.experiment.learning_curve is not None
        assert cfg.experiment.learning_curve.models is not None
        self.assertTrue(cfg.experiment.learning_curve.models.use_ridge)
        self.assertTrue(cfg.experiment.learning_curve.models.moe.enabled)
        self.assertFalse(cfg.experiment.learning_curve.models.use_weighted_linear)
        self.assertFalse(cfg.experiment.learning_curve.models.use_weighted_simplex)

    def test_load_config_data_preserves_top_level_models_after_shared_defaults_merge(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            experiment_path = tmp / "experiment.toml"
            experiment_path.write_text(
                "\n".join(
                    [
                        "[ingest]",
                        'source = "data/raw_vasp/systems"',
                        'dataset_name = "test"',
                        "",
                        "[ingest.stoich]",
                        'elements = ["H"]',
                        'basis_species = ["H2"]',
                        "",
                        "[ingest.stoich.basis_composition]",
                        'H2 = { H = 2 }',
                        "",
                        "[mlip]",
                        "dev_n = 1",
                        "dev_run = false",
                        'dataset = "data/raw_data/example.json"',
                        "",
                        "[mlip.models]",
                        'enabled = ["mace"]',
                        "",
                        "[mlip.rootstock]",
                        'root = "."',
                        "",
                        "[mlip.rootstock.models.mace]",
                        'model = "mace"',
                        'mlip_name = "mace-test"',
                        "",
                        "[experiment.defaults]",
                        "validation_fraction = 0.2",
                        "",
                        "[experiment.learning_curve]",
                        "min_train = 2",
                        "max_train = 4",
                        "n_repeats = 3",
                        "",
                        "[models]",
                        "use_ridge = true",
                        "use_kernel_ridge = false",
                        "use_lasso = false",
                        "use_elastic_net = false",
                        "use_residual = true",
                        "use_weighted_linear = false",
                        "use_weighted_simplex = false",
                        "",
                        "[models.moe]",
                        "enabled = false",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            data = load_config_data(experiment_path)
            cfg = get_config(experiment_path)

        self.assertIn("models", data["experiment"]["learning_curve"])
        assert cfg.experiment is not None
        assert cfg.experiment.learning_curve is not None
        assert cfg.experiment.learning_curve.models is not None
        self.assertTrue(cfg.experiment.learning_curve.models.use_ridge)
        self.assertTrue(cfg.experiment.learning_curve.models.use_residual)
        self.assertFalse(cfg.experiment.learning_curve.models.use_weighted_linear)
        self.assertFalse(cfg.experiment.learning_curve.models.use_weighted_simplex)
        self.assertFalse(cfg.experiment.learning_curve.models.moe.enabled)

    def test_load_config_data_applies_shared_top_level_tuning_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            experiment_path = tmp / "experiment.toml"
            experiment_path.write_text(
                "\n".join(
                    [
                        "seed = 7",
                        "",
                        "[ingest]",
                        'source = "data/raw_vasp/systems"',
                        'dataset_name = "test"',
                        "",
                        "[ingest.stoich]",
                        'elements = ["H"]',
                        'basis_species = ["H2"]',
                        "",
                        "[ingest.stoich.basis_composition]",
                        'H2 = { H = 2 }',
                        "",
                        "[mlip]",
                        "dev_n = 1",
                        "dev_run = false",
                        'dataset = "data/raw_data/example.json"',
                        "",
                        "[mlip.models]",
                        'enabled = ["mace"]',
                        "",
                        "[mlip.rootstock]",
                        'root = "."',
                        "",
                        "[mlip.rootstock.models.mace]",
                        'model = "mace"',
                        'mlip_name = "mace-test"',
                        "",
                        "[experiment.learning_curve]",
                        "min_train = 2",
                        "max_train = 4",
                        "n_repeats = 3",
                        "",
                        "[models]",
                        "use_ridge = false",
                        "use_kernel_ridge = false",
                        "use_lasso = false",
                        "use_elastic_net = false",
                        "use_residual = true",
                        "use_probe_gnn = true",
                        "",
                        "[models.moe]",
                        "enabled = true",
                        "",
                        "[tuning.optuna]",
                        "n_trials = 12",
                        'sampler = "tpe"',
                        "",
                        "[models.probe_gnn.tuning.optuna]",
                        "n_trials = 5",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            cfg = get_config(experiment_path)

        assert cfg.experiment is not None
        assert cfg.experiment.learning_curve is not None
        assert cfg.experiment.learning_curve.models is not None
        shared_optuna = cfg.experiment.learning_curve.models.tuning.optuna
        moe_optuna = cfg.experiment.learning_curve.models.moe.tuning.optuna
        probe_optuna = cfg.experiment.learning_curve.models.probe_gnn.tuning.optuna
        assert shared_optuna is not None
        assert moe_optuna is not None
        assert probe_optuna is not None
        self.assertTrue(cfg.experiment.learning_curve.models.use_probe_gnn)
        self.assertTrue(cfg.experiment.learning_curve.models.probe_gnn.enabled)
        self.assertEqual(shared_optuna.n_trials, 12)
        self.assertEqual(shared_optuna.sampler, "tpe")
        self.assertEqual(moe_optuna.n_trials, 12)
        self.assertEqual(moe_optuna.sampler, "tpe")
        self.assertEqual(probe_optuna.n_trials, 5)
        self.assertEqual(probe_optuna.sampler, "tpe")

    def test_load_config_data_maps_gnn_boolean_aliases_to_enabled_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            experiment_path = tmp / "experiment.toml"
            experiment_path.write_text(
                "\n".join(
                    [
                        "seed = 7",
                        "",
                        "[ingest]",
                        'source = "data/raw_vasp/systems"',
                        'dataset_name = "test"',
                        "",
                        "[ingest.stoich]",
                        'elements = ["H"]',
                        'basis_species = ["H2"]',
                        "",
                        "[ingest.stoich.basis_composition]",
                        'H2 = { H = 2 }',
                        "",
                        "[mlip]",
                        "dev_n = 1",
                        "dev_run = false",
                        'dataset = "data/raw_data/example.json"',
                        "",
                        "[mlip.models]",
                        'enabled = ["mace"]',
                        "",
                        "[mlip.rootstock]",
                        'root = "."',
                        "",
                        "[mlip.rootstock.models.mace]",
                        'model = "mace"',
                        'mlip_name = "mace-test"',
                        "",
                        "[experiment.learning_curve]",
                        "min_train = 2",
                        "max_train = 4",
                        "n_repeats = 3",
                        "",
                        "[models]",
                        "use_ridge = false",
                        "use_kernel_ridge = false",
                        "use_lasso = false",
                        "use_elastic_net = false",
                        "use_residual = true",
                        "use_gnn_direct = true",
                        "use_probe_gnn = true",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            cfg = get_config(experiment_path)

        assert cfg.experiment is not None
        assert cfg.experiment.learning_curve is not None
        assert cfg.experiment.learning_curve.models is not None
        self.assertTrue(cfg.experiment.learning_curve.models.use_gnn_direct)
        self.assertTrue(cfg.experiment.learning_curve.models.use_probe_gnn)
        self.assertTrue(cfg.experiment.learning_curve.models.gnn_direct.enabled)
        self.assertTrue(cfg.experiment.learning_curve.models.probe_gnn.enabled)

    def test_dataset_profile_derives_common_paths(self) -> None:
        cfg = Config(
            **{
                "ingest": {
                    "source": "data/raw_vasp/systems",
                    "dataset_name": "test",
                    "stoich": {
                        "elements": ["H"],
                        "basis_species": ["H2"],
                        "basis_composition": {"H2": {"H": 2}},
                    },
                },
                "dataset_profile": {
                    "tag": "mamun_oh",
                },
                "mlip": {
                    "dev_n": 1,
                    "dev_run": False,
                    "models": {"enabled": []},
                    "rootstock": {"root": ".", "models": {}},
                },
                "experiment": {
                    "learning_curve": {
                        "min_train": 2,
                        "max_train": 4,
                        "n_repeats": 3,
                    }
                },
                "analysis": {
                    "run_adsorption_analysis": False,
                    "out_dir": "data/mlips_by_prefix",
                    "prefixes": ["ol"],
                },
            }
        )

        self.assertEqual(cfg.mlip.dataset, "data/raw_data/mamun_oh.json")
        assert cfg.experiment is not None
        assert cfg.experiment.learning_curve is not None
        self.assertEqual(
            cfg.experiment.learning_curve.results_bundle_path,
            Path("data/results/learning_curve/mamun_oh.json"),
        )
        self.assertIsNone(cfg.experiment.learning_curve.graph_dataset)
        assert cfg.probe_features is not None
        self.assertEqual(
            cfg.probe_features.dataset_path,
            Path("data/raw_data/mamun_oh_with_probe_ids.json"),
        )
        self.assertEqual(
            cfg.probe_features.mlip_results_dir,
            Path("data/mlips/mamun_oh_unique_probes"),
        )
        self.assertTrue(cfg.probe_features.dataset_path.name.endswith("_with_probe_ids.json"))
        self.assertTrue(cfg.probe_features.mlip_results_dir.name.endswith("_unique_probes"))
        assert cfg.analysis is not None
        self.assertEqual(cfg.analysis.base_dir, Path("data/mlips/mamun_oh"))
        self.assertEqual(
            cfg.analysis.comparison_plot_path,
            Path("data/results/plots/mamun_oh_mae_comparison.png"),
        )

    def test_dataset_profile_derives_zero_shot_bundle_path(self) -> None:
        cfg = Config(
            **{
                "ingest": {
                    "source": "data/raw_vasp/systems",
                    "dataset_name": "test",
                    "stoich": {
                        "elements": ["H"],
                        "basis_species": ["H2"],
                        "basis_composition": {"H2": {"H": 2}},
                    },
                },
                "dataset_profile": {
                    "tag": "mamun_oh",
                },
                "mlip": {
                    "dev_n": 1,
                    "dev_run": False,
                    "models": {"enabled": []},
                    "rootstock": {"root": ".", "models": {}},
                },
                "experiment": {
                    "zero_shot": {
                        "mlip_selection": {
                            "exclude_anomalous_mlips": True,
                            "minimum_quorum": 2,
                        }
                    }
                },
                "analysis": {
                    "run_adsorption_analysis": False,
                    "out_dir": "data/mlips_by_prefix",
                    "prefixes": ["ol"],
                },
            }
        )

        assert cfg.experiment is not None
        assert cfg.experiment.zero_shot is not None
        self.assertEqual(
            cfg.experiment.zero_shot.results_bundle_path,
            Path("data/results/zero_shot/mamun_oh.json"),
        )

    def test_named_dataset_profile_derives_real_mamun_oh_paths(self) -> None:
        cfg = Config(
            **{
                "ingest": {
                    "source": "data/raw_vasp/systems",
                    "dataset_name": "test",
                    "stoich": {
                        "elements": ["H"],
                        "basis_species": ["H2"],
                        "basis_composition": {"H2": {"H": 2}},
                    },
                },
                "dataset_profile": {
                    "tag": "mamun_oh",
                },
                "datasets": {
                    "mamun_oh": {
                        "raw_dataset_filename": "MamunHighT2019_oh_adsorption.json",
                        "processed_basename": "mamun_oh",
                        "mlip_run_dirname": "oh_mamun",
                        "analysis_run_dirname": "ch3_oh_mamun",
                        "summary_run_dirname": "ch3_oh_shifted_mamun",
                    }
                },
                "mlip": {
                    "dev_n": 1,
                    "dev_run": False,
                    "models": {"enabled": []},
                    "rootstock": {"root": ".", "models": {}},
                },
                "experiment": {
                    "learning_curve": {
                        "min_train": 2,
                        "max_train": 4,
                        "n_repeats": 3,
                    }
                },
                "analysis": {
                    "run_adsorption_analysis": False,
                    "out_dir": "data/mlips_by_prefix",
                    "prefixes": ["ol"],
                },
            }
        )

        self.assertEqual(
            cfg.mlip.dataset,
            "data/raw_data/MamunHighT2019_oh_adsorption.json",
        )
        assert cfg.experiment is not None
        assert cfg.experiment.learning_curve is not None
        self.assertEqual(
            cfg.experiment.learning_curve.results_bundle_path,
            Path("data/results/learning_curve/mamun_oh.json"),
        )
        self.assertIsNone(cfg.experiment.learning_curve.graph_dataset)
        assert cfg.probe_features is not None
        self.assertEqual(
            cfg.probe_features.dataset_path,
            Path("data/raw_data/MamunHighT2019_oh_adsorption_with_probe_ids.json"),
        )
        assert cfg.analysis is not None
        self.assertEqual(cfg.analysis.base_dir, Path("data/mlips/oh_mamun"))
        self.assertEqual(
            cfg.analysis.calculating_path,
            Path("data/mlips/ch3_oh_mamun"),
        )
        self.assertEqual(
            cfg.analysis.summary_workbook_path,
            Path("data/results/ch3_oh_shifted_mamun/oasis_Benchmarking_Analysis.xlsx"),
        )
        self.assertEqual(
            cfg.analysis.comparison_workbook_path,
            Path("data/results/ch3_oh_mamun/oasis_Benchmarking_Analysis.xlsx"),
        )

    def test_named_dataset_profile_honors_analysis_base_dir_override(self) -> None:
        cfg = Config(
            **{
                "ingest": {
                    "source": "data/raw_vasp/systems",
                    "dataset_name": "test",
                    "stoich": {
                        "elements": ["H"],
                        "basis_species": ["H2"],
                        "basis_composition": {"H2": {"H": 2}},
                    },
                },
                "dataset_profile": {
                    "tag": "mamun_oh",
                },
                "datasets": {
                    "mamun_oh": {
                        "raw_dataset_filename": "MamunHighT2019_oh_adsorption.json",
                        "mlip_run_dirname": "OH-BMA",
                        "analysis_base_dir": "data/mlips/backup_copy/OH-BMA",
                    }
                },
                "mlip": {
                    "dev_n": 1,
                    "dev_run": False,
                    "models": {"enabled": []},
                    "rootstock": {"root": ".", "models": {}},
                },
                "experiment": {
                    "learning_curve": {
                        "min_train": 2,
                        "max_train": 4,
                        "n_repeats": 3,
                    }
                },
                "analysis": {
                    "run_adsorption_analysis": False,
                    "out_dir": "data/mlips_by_prefix",
                    "prefixes": ["ol"],
                },
            }
        )

        assert cfg.analysis is not None
        self.assertEqual(
            cfg.analysis.base_dir,
            Path("data/mlips/backup_copy/OH-BMA"),
        )

    def test_screening_mode_derives_separate_screening_results_bundle_path(self) -> None:
        cfg = Config(
            **{
                "ingest": {
                    "source": "data/raw_vasp/systems",
                    "dataset_name": "test",
                    "stoich": {
                        "elements": ["H"],
                        "basis_species": ["H2"],
                        "basis_composition": {"H2": {"H": 2}},
                    },
                },
                "dataset_profile": {"tag": "mamun_oh"},
                "datasets": {
                    "mamun_oh": {
                        "raw_dataset_filename": "MamunHighT2019_oh_adsorption.json",
                        "processed_basename": "mamun_oh",
                    }
                },
                "mlip": {
                    "dev_n": 1,
                    "dev_run": False,
                    "models": {"enabled": []},
                    "rootstock": {"root": ".", "models": {}},
                },
                "experiment": {
                    "learning_curve": {
                        "min_train": 2,
                        "max_train": 4,
                        "n_repeats": 3,
                    },
                    "screening": {
                        "screen_fraction": 0.25,
                    },
                },
                "analysis": {
                    "run_adsorption_analysis": False,
                    "out_dir": "data/mlips_by_prefix",
                    "prefixes": ["ol"],
                },
            }
        )

        assert cfg.experiment is not None
        assert cfg.experiment.learning_curve is not None
        assert cfg.experiment.screening is not None
        self.assertEqual(
            cfg.experiment.learning_curve.results_bundle_path,
            Path("data/results/learning_curve/mamun_oh.json"),
        )
        self.assertEqual(
            cfg.experiment.screening.results_bundle_path,
            Path("data/results/screening/mamun_oh.json"),
        )

    def test_dataset_profile_templates_support_two_distinct_naming_patterns(self) -> None:
        base_payload = {
            "ingest": {
                "source": "data/raw_vasp/systems",
                "dataset_name": "test",
                "stoich": {
                    "elements": ["H"],
                    "basis_species": ["H2"],
                    "basis_composition": {"H2": {"H": 2}},
                },
            },
            "datasets": {
                "simple": {},
                "tol_probe": {
                    "raw_dataset_filename": "KHLOHC_origin_tolstar_adsorption.json",
                    "processed_basename": "khlohc_tol",
                    "probe_results_dirname": "khlohc_toluene_unique_probes",
                    "mlip_run_dirname": "khlohc_tol_results",
                    "analysis_run_dirname": "khlohc_tol_analysis",
                    "summary_run_dirname": "khlohc_tol_shifted",
                },
            },
            "mlip": {
                "dev_n": 1,
                "dev_run": False,
                "models": {"enabled": []},
                "rootstock": {"root": ".", "models": {}},
            },
            "experiment": {
                "learning_curve": {
                    "min_train": 2,
                    "max_train": 4,
                    "n_repeats": 3,
                }
            },
            "analysis": {
                "run_adsorption_analysis": False,
                "comparison_plot_path": "data/results/plots/shared.png",
                "out_dir": "data/mlips_by_prefix",
                "prefixes": ["ol"],
            },
        }

        simple_cfg = Config(
            **{
                **base_payload,
                "dataset_profile": {"tag": "simple"},
            }
        )
        custom_cfg = Config(
            **{
                **base_payload,
                "dataset_profile": {"tag": "tol_probe"},
            }
        )

        self.assertEqual(simple_cfg.mlip.dataset, "data/raw_data/simple.json")
        assert simple_cfg.probe_features is not None
        self.assertEqual(
            simple_cfg.probe_features.dataset_path,
            Path("data/raw_data/simple_with_probe_ids.json"),
        )
        self.assertIsNone(simple_cfg.experiment.learning_curve.graph_dataset)

        self.assertEqual(
            custom_cfg.mlip.dataset,
            "data/raw_data/KHLOHC_origin_tolstar_adsorption.json",
        )
        assert custom_cfg.probe_features is not None
        self.assertEqual(
            custom_cfg.probe_features.dataset_path,
            Path("data/raw_data/KHLOHC_origin_tolstar_adsorption_with_probe_ids.json"),
        )
        self.assertEqual(
            custom_cfg.probe_features.mlip_results_dir,
            Path("data/mlips/khlohc_toluene_unique_probes"),
        )
        self.assertTrue(
            custom_cfg.probe_features.dataset_path.name.endswith("_with_probe_ids.json")
        )
        self.assertTrue(
            custom_cfg.probe_features.mlip_results_dir.name.endswith("_unique_probes")
        )
        assert custom_cfg.experiment is not None
        assert custom_cfg.experiment.learning_curve is not None
        self.assertIsNone(custom_cfg.experiment.learning_curve.graph_dataset)
        assert custom_cfg.analysis is not None
        self.assertEqual(
            custom_cfg.analysis.base_dir,
            Path("data/mlips/khlohc_tol_results"),
        )
        self.assertEqual(
            custom_cfg.analysis.calculating_path,
            Path("data/mlips/khlohc_tol_analysis"),
        )
        self.assertEqual(
            custom_cfg.analysis.summary_workbook_path,
            Path("data/results/khlohc_tol_shifted/oasis_Benchmarking_Analysis.xlsx"),
        )

    def test_dataset_profile_explicit_paths_override_defaults(self) -> None:
        cfg = Config(
            **{
                "ingest": {
                    "source": "data/raw_vasp/systems",
                    "dataset_name": "test",
                    "stoich": {
                        "elements": ["H"],
                        "basis_species": ["H2"],
                        "basis_composition": {"H2": {"H": 2}},
                    },
                },
                "dataset_profile": {
                    "tag": "mamun_oh",
                },
                "mlip": {
                    "dev_n": 1,
                    "dev_run": False,
                    "dataset": "data/raw_data/explicit.json",
                    "models": {"enabled": []},
                    "rootstock": {"root": ".", "models": {}},
                },
                "probe_features": {
                    "dataset_path": "data/raw_data/explicit_probe.json",
                    "mlip_results_dir": "data/mlips/explicit_probes",
                },
                "experiment": {
                    "learning_curve": {
                        "min_train": 2,
                        "max_train": 4,
                        "n_repeats": 3,
                        "results_bundle_path": "data/results/learning_curve/explicit.json",
                        "graph_dataset": {
                            "path": "data/processed/explicit.parquet",
                        },
                    }
                },
                "analysis": {
                    "run_adsorption_analysis": False,
                    "base_dir": "data/mlips/explicit",
                    "comparison_plot_path": "data/results/plots/explicit.png",
                    "out_dir": "data/mlips_by_prefix",
                    "prefixes": ["ol"],
                },
            }
        )

        self.assertEqual(cfg.mlip.dataset, "data/raw_data/explicit.json")
        assert cfg.probe_features is not None
        self.assertEqual(
            cfg.probe_features.dataset_path,
            Path("data/raw_data/explicit_probe.json"),
        )
        self.assertEqual(
            cfg.probe_features.mlip_results_dir,
            Path("data/mlips/explicit_probes"),
        )
        assert cfg.experiment is not None
        assert cfg.experiment.learning_curve is not None
        self.assertEqual(
            cfg.experiment.learning_curve.results_bundle_path,
            Path("data/results/learning_curve/explicit.json"),
        )
        assert cfg.experiment.learning_curve.graph_dataset is not None
        self.assertEqual(
            cfg.experiment.learning_curve.graph_dataset.path,
            Path("data/processed/explicit.parquet"),
        )
        assert cfg.analysis is not None
        self.assertEqual(cfg.analysis.base_dir, Path("data/mlips/explicit"))
        self.assertEqual(
            cfg.analysis.comparison_plot_path,
            Path("data/results/plots/explicit.png"),
        )

    def test_fully_explicit_legacy_config_parses_without_dataset_profile(self) -> None:
        cfg = Config(
            **{
                "ingest": {
                    "source": "data/raw_vasp/systems",
                    "dataset_name": "legacy",
                    "stoich": {
                        "elements": ["H"],
                        "basis_species": ["H2"],
                        "basis_composition": {"H2": {"H": 2}},
                    },
                },
                "mlip": {
                    "dev_n": 1,
                    "dev_run": False,
                    "dataset": "data/raw_data/legacy.json",
                    "models": {"enabled": []},
                    "rootstock": {"root": ".", "models": {}},
                },
                "probe_features": {
                    "dataset_path": "data/raw_data/legacy_with_probe_ids.json",
                    "mlip_results_dir": "data/mlips/legacy_unique_probes",
                },
                "experiment": {
                    "learning_curve": {
                        "min_train": 2,
                        "max_train": 4,
                        "n_repeats": 3,
                        "results_bundle_path": "data/results/learning_curve/legacy.json",
                        "graph_dataset": {
                            "path": "data/processed/legacy.parquet",
                            "join_key": "reaction",
                        },
                    }
                },
                "analysis": {
                    "run_adsorption_analysis": False,
                    "calculating_path": "data/mlips/legacy_analysis",
                    "summary_workbook_path": "data/results/legacy_shifted/oasis_Benchmarking_Analysis.xlsx",
                    "comparison_workbook_path": "data/results/legacy/oasis_Benchmarking_Analysis.xlsx",
                    "comparison_plot_path": "data/results/plots/legacy.png",
                    "base_dir": "data/mlips/legacy",
                    "out_dir": "data/mlips_by_prefix",
                    "prefixes": ["ol"],
                },
            }
        )

        self.assertIsNone(cfg.dataset_profile)
        self.assertEqual(cfg.mlip.dataset, "data/raw_data/legacy.json")
        assert cfg.probe_features is not None
        self.assertEqual(
            cfg.probe_features.dataset_path,
            Path("data/raw_data/legacy_with_probe_ids.json"),
        )
        assert cfg.experiment is not None
        assert cfg.experiment.learning_curve is not None
        self.assertEqual(
            cfg.experiment.learning_curve.results_bundle_path,
            Path("data/results/learning_curve/legacy.json"),
        )
        assert cfg.analysis is not None
        self.assertEqual(cfg.analysis.base_dir, Path("data/mlips/legacy"))

    def test_missing_named_dataset_profile_gives_clear_error(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"dataset_profile\.tag 'missing' was not found in \[datasets\]",
        ):
            Config(
                **{
                    "ingest": {
                        "source": "data/raw_vasp/systems",
                        "dataset_name": "test",
                        "stoich": {
                            "elements": ["H"],
                            "basis_species": ["H2"],
                            "basis_composition": {"H2": {"H": 2}},
                        },
                    },
                    "dataset_profile": {
                        "tag": "missing",
                    },
                    "datasets": {
                        "mamun_oh": {
                            "raw_dataset_filename": "MamunHighT2019_oh_adsorption.json",
                        }
                    },
                    "mlip": {
                        "dev_n": 1,
                        "dev_run": False,
                        "models": {"enabled": []},
                        "rootstock": {"root": ".", "models": {}},
                    },
                }
            )

    def test_analysis_requires_resolved_workbook_paths(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "analysis.summary_workbook_path, analysis.comparison_workbook_path",
        ):
            Config(
                **{
                    "ingest": {
                        "source": "data/raw_vasp/systems",
                        "dataset_name": "test",
                        "stoich": {
                            "elements": ["H"],
                            "basis_species": ["H2"],
                            "basis_composition": {"H2": {"H": 2}},
                        },
                    },
                    "mlip": {
                        "dev_n": 1,
                        "dev_run": False,
                        "models": {"enabled": []},
                        "rootstock": {"root": ".", "models": {}},
                    },
                    "analysis": {
                        "run_adsorption_analysis": False,
                        "base_dir": "data/mlips/manual",
                        "out_dir": "data/mlips_by_prefix",
                        "prefixes": ["ol"],
                    },
                }
            )

    def test_graph_dataset_requires_explicit_or_derived_path(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "experiment.learning_curve.graph_dataset.path must be provided explicitly or derived from dataset_profile.tag",
        ):
            Config(
                **{
                    "ingest": {
                        "source": "data/raw_vasp/systems",
                        "dataset_name": "test",
                        "stoich": {
                            "elements": ["H"],
                            "basis_species": ["H2"],
                            "basis_composition": {"H2": {"H": 2}},
                        },
                    },
                    "mlip": {
                        "dev_n": 1,
                        "dev_run": False,
                        "models": {"enabled": []},
                        "rootstock": {"root": ".", "models": {}},
                    },
                    "experiment": {
                        "learning_curve": {
                            "min_train": 2,
                            "max_train": 4,
                            "n_repeats": 3,
                            "graph_dataset": {
                                "join_key": "reaction",
                            },
                        }
                    },
                }
            )

    def test_analysis_requires_derived_calculating_path_when_adsorption_enabled(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "analysis.calculating_path must be provided explicitly or derived from dataset_profile.tag",
        ):
            Config(
                **{
                    "ingest": {
                        "source": "data/raw_vasp/systems",
                        "dataset_name": "test",
                        "stoich": {
                            "elements": ["H"],
                            "basis_species": ["H2"],
                            "basis_composition": {"H2": {"H": 2}},
                        },
                    },
                    "mlip": {
                        "dev_n": 1,
                        "dev_run": False,
                        "models": {"enabled": []},
                        "rootstock": {"root": ".", "models": {}},
                    },
                    "analysis": {
                        "run_adsorption_analysis": True,
                        "base_dir": "data/mlips/manual",
                        "summary_workbook_path": "data/results/manual/oasis_Benchmarking_Analysis.xlsx",
                        "comparison_workbook_path": "data/results/manual_compare/oasis_Benchmarking_Analysis.xlsx",
                        "comparison_plot_path": "data/results/plots/manual.png",
                        "out_dir": "data/mlips_by_prefix",
                        "prefixes": ["ol"],
                    },
                }
            )

    def test_get_config_loads_named_dataset_profile_from_toml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config_path = tmp / "config.toml"
            experiment_path = tmp / "experiment.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[dataset_profile]",
                        'tag = "mamun_oh"',
                        "",
                        "[datasets.mamun_oh]",
                        'raw_dataset_filename = "MamunHighT2019_oh_adsorption.json"',
                        'processed_basename = "mamun_oh"',
                        'mlip_run_dirname = "oh_mamun"',
                        'analysis_run_dirname = "ch3_oh_mamun"',
                        'summary_run_dirname = "ch3_oh_shifted_mamun"',
                        "",
                        "[ingest]",
                        'source = "data/raw_vasp/systems"',
                        'dataset_name = "test"',
                        "",
                        "[ingest.stoich]",
                        'elements = ["H"]',
                        'basis_species = ["H2"]',
                        "",
                        "[ingest.stoich.basis_composition]",
                        'H2 = { H = 2 }',
                        "",
                        "[mlip]",
                        "dev_n = 1",
                        "dev_run = false",
                        "",
                        "[mlip.models]",
                        'enabled = ["mace"]',
                        "",
                        "[mlip.rootstock]",
                        'root = "."',
                        "",
                        "[mlip.rootstock.models.mace]",
                        'model = "mace"',
                        'mlip_name = "mace-test"',
                        "",
                        "[analysis]",
                        "run_adsorption_analysis = false",
                        'out_dir = "data/mlips_by_prefix"',
                        'prefixes = ["ol"]',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            experiment_path.write_text(
                "\n".join(
                    [
                        "[experiment]",
                        "[experiment.learning_curve]",
                        "min_train = 2",
                        "max_train = 4",
                        "n_repeats = 3",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            cfg = get_config([config_path, experiment_path])

        self.assertEqual(
            cfg.mlip.dataset,
            "data/raw_data/MamunHighT2019_oh_adsorption.json",
        )
        assert cfg.experiment is not None
        assert cfg.experiment.learning_curve is not None
        self.assertEqual(
            cfg.experiment.learning_curve.results_bundle_path,
            Path("data/results/learning_curve/mamun_oh.json"),
        )
        assert cfg.analysis is not None
        self.assertEqual(cfg.analysis.base_dir, Path("data/mlips/oh_mamun"))

    def test_get_config_derives_graph_dataset_path_from_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config_path = tmp / "config.toml"
            experiment_path = tmp / "experiment.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[dataset_profile]",
                        'tag = "mamun_oh"',
                        "",
                        "[datasets.mamun_oh]",
                        'raw_dataset_filename = "MamunHighT2019_oh_adsorption.json"',
                        'processed_basename = "mamun_oh"',
                        "",
                        "[ingest]",
                        'source = "data/raw_vasp/systems"',
                        'dataset_name = "test"',
                        "",
                        "[ingest.stoich]",
                        'elements = ["H"]',
                        'basis_species = ["H2"]',
                        "",
                        "[ingest.stoich.basis_composition]",
                        'H2 = { H = 2 }',
                        "",
                        "[mlip]",
                        "dev_n = 1",
                        "dev_run = false",
                        "",
                        "[mlip.models]",
                        'enabled = ["mace"]',
                        "",
                        "[mlip.rootstock]",
                        'root = "."',
                        "",
                        "[mlip.rootstock.models.mace]",
                        'model = "mace"',
                        'mlip_name = "mace-test"',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            experiment_path.write_text(
                "\n".join(
                    [
                        "[experiment]",
                        "[experiment.learning_curve]",
                        "min_train = 2",
                        "max_train = 4",
                        "n_repeats = 3",
                        "",
                        "[experiment.learning_curve.graph_dataset]",
                        'join_key = "reaction"',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            cfg = get_config([config_path, experiment_path])

        assert cfg.experiment is not None
        assert cfg.experiment.learning_curve is not None
        assert cfg.experiment.learning_curve.graph_dataset is not None
        self.assertEqual(
            cfg.experiment.learning_curve.graph_dataset.path,
            Path("data/processed/mamun_oh.parquet"),
        )
        self.assertEqual(
            cfg.experiment.learning_curve.graph_dataset.join_key,
            "reaction",
        )

    def test_learning_curve_graph_dataset_section_parses(self) -> None:
        cfg = Config(
            **{
                "ingest": {
                    "source": "data/raw_vasp/systems",
                    "dataset_name": "test",
                    "stoich": {
                        "elements": ["H"],
                        "basis_species": ["H2"],
                        "basis_composition": {"H2": {"H": 2}},
                    },
                },
                "mlip": {
                    "dev_n": 1,
                    "dev_run": False,
                    "models": {"enabled": []},
                    "rootstock": {"root": ".", "models": {}},
                },
                "experiment": {
                    "learning_curve": {
                        "min_train": 2,
                        "max_train": 4,
                        "n_repeats": 3,
                        "graph_dataset": {
                            "path": "data/graphs/reactions.json",
                            "join_key": "reaction_id",
                        },
                    }
                },
            }
        )

        assert cfg.experiment is not None
        assert cfg.experiment.learning_curve is not None
        assert cfg.experiment.learning_curve.graph_dataset is not None
        self.assertEqual(
            cfg.experiment.learning_curve.graph_dataset.path,
            Path("data/graphs/reactions.json"),
        )
        self.assertEqual(
            cfg.experiment.learning_curve.graph_dataset.join_key,
            "reaction_id",
        )

    def test_learning_curve_graph_dataset_join_key_defaults(self) -> None:
        cfg = Config(
            **{
                "ingest": {
                    "source": "data/raw_vasp/systems",
                    "dataset_name": "test",
                    "stoich": {
                        "elements": ["H"],
                        "basis_species": ["H2"],
                        "basis_composition": {"H2": {"H": 2}},
                    },
                },
                "mlip": {
                    "dev_n": 1,
                    "dev_run": False,
                    "models": {"enabled": []},
                    "rootstock": {"root": ".", "models": {}},
                },
                "experiment": {
                    "learning_curve": {
                        "min_train": 2,
                        "max_train": 4,
                        "n_repeats": 3,
                        "graph_dataset": {
                            "path": "data/graphs/reactions.json",
                        },
                    }
                },
            }
        )

        assert cfg.experiment is not None
        assert cfg.experiment.learning_curve is not None
        assert cfg.experiment.learning_curve.graph_dataset is not None
        self.assertEqual(cfg.experiment.learning_curve.graph_dataset.join_key, "reaction")

    def test_learning_curve_split_sizing_explicit_values_parse(self) -> None:
        cfg = Config(
            **{
                "ingest": {
                    "source": "data/raw_vasp/systems",
                    "dataset_name": "test",
                    "stoich": {
                        "elements": ["H"],
                        "basis_species": ["H2"],
                        "basis_composition": {"H2": {"H": 2}},
                    },
                },
                "mlip": {
                    "dev_n": 1,
                    "dev_run": False,
                    "models": {"enabled": []},
                    "rootstock": {"root": ".", "models": {}},
                },
                "experiment": {
                    "learning_curve": {
                        "min_train": 2,
                        "max_train": 4,
                        "n_repeats": 3,
                        "min_test_size": 3,
                    },
                    "screening": {
                        "budget_mode": "screening_fraction",
                        "screen_fraction": 0.25,
                        "min_screen_size": 2,
                        "validation_fraction": 0.35,
                        "min_val_size": 2,
                        "min_tuning_val_size": 4,
                        "min_inner_train_size": 5,
                    },
                },
            }
        )

        assert cfg.experiment is not None
        assert cfg.experiment.learning_curve is not None
        assert cfg.experiment.screening is not None
        self.assertEqual(cfg.experiment.learning_curve.budget_mode, "full_remainder_test")
        self.assertIsNone(cfg.experiment.learning_curve.screen_fraction)
        self.assertEqual(cfg.experiment.learning_curve.min_screen_size, 1)
        self.assertEqual(cfg.experiment.learning_curve.validation_fraction, 0.2)
        self.assertEqual(cfg.experiment.learning_curve.min_val_size, 1)
        self.assertEqual(cfg.experiment.learning_curve.min_tuning_val_size, 1)
        self.assertEqual(cfg.experiment.learning_curve.min_inner_train_size, 1)
        self.assertEqual(cfg.experiment.screening.budget_mode, "screening_fraction")
        self.assertEqual(cfg.experiment.screening.screen_fraction, 0.25)
        self.assertEqual(cfg.experiment.screening.min_screen_size, 2)
        self.assertEqual(cfg.experiment.screening.validation_fraction, 0.35)
        self.assertEqual(cfg.experiment.screening.min_val_size, 2)
        self.assertEqual(cfg.experiment.screening.min_tuning_val_size, 4)
        self.assertEqual(cfg.experiment.screening.min_inner_train_size, 5)
        self.assertEqual(cfg.experiment.learning_curve.min_test_size, 3)

    def test_learning_curve_results_artifact_config_explicit_values_parse(self) -> None:
        cfg = Config(
            **{
                "ingest": {
                    "source": "data/raw_vasp/systems",
                    "dataset_name": "test",
                    "stoich": {
                        "elements": ["H"],
                        "basis_species": ["H2"],
                        "basis_composition": {"H2": {"H": 2}},
                    },
                },
                "mlip": {
                    "dev_n": 1,
                    "dev_run": False,
                    "models": {"enabled": []},
                    "rootstock": {"root": ".", "models": {}},
                },
                "experiment": {
                    "learning_curve": {
                        "min_train": 2,
                        "max_train": 4,
                        "n_repeats": 3,
                    },
                    "screening": {
                        "results_bundle_path": "data/results/screening/example.json",
                        "reuse_results": True,
                        "force_refresh_methods": ["moe", "probe_gnn"],
                        "force_refresh_train_sizes": {"ridge": [5, 10]},
                    },
                },
            }
        )

        assert cfg.experiment is not None
        assert cfg.experiment.learning_curve is not None
        assert cfg.experiment.screening is not None
        self.assertEqual(
            cfg.experiment.screening.results_bundle_path,
            Path("data/results/screening/example.json"),
        )
        self.assertTrue(cfg.experiment.screening.reuse_results)
        self.assertEqual(
            cfg.experiment.screening.force_refresh_methods,
            ["moe", "probe_gnn"],
        )
        self.assertEqual(
            cfg.experiment.screening.force_refresh_train_sizes,
            {"ridge": [5, 10]},
        )

    def test_learning_curve_config_accepts_mixed_sweep_sizes_and_fractions(self) -> None:
        cfg = Config(
            **{
                "ingest": {
                    "source": "data/raw_vasp/systems",
                    "dataset_name": "test",
                    "stoich": {
                        "elements": ["H"],
                        "basis_species": ["H2"],
                        "basis_composition": {"H2": {"H": 2}},
                    },
                },
                "mlip": {
                    "dev_n": 1,
                    "dev_run": False,
                    "models": {"enabled": []},
                    "rootstock": {"root": ".", "models": {}},
                },
                "experiment": {
                    "learning_curve": {
                        "n_repeats": 3,
                        "sweep_sizes": [1, 2, 3, 20],
                        "sweep_fractions": [0.05, 0.1, 0.2],
                    }
                },
            }
        )

        assert cfg.experiment is not None
        assert cfg.experiment.learning_curve is not None
        self.assertEqual(cfg.experiment.learning_curve.sweep_sizes, [1, 2, 3, 20])
        self.assertEqual(
            cfg.experiment.learning_curve.sweep_fractions,
            [0.05, 0.1, 0.2],
        )

    def test_experiment_defaults_shared_by_learning_curve_and_screening(self) -> None:
        cfg = Config(
            **{
                "ingest": {
                    "source": "data/raw_vasp/systems",
                    "dataset_name": "test",
                    "stoich": {
                        "elements": ["H"],
                        "basis_species": ["H2"],
                        "basis_composition": {"H2": {"H": 2}},
                    },
                },
                "mlip": {
                    "dev_n": 1,
                    "dev_run": False,
                    "models": {"enabled": []},
                    "rootstock": {"root": ".", "models": {}},
                },
                "experiment": {
                    "defaults": {
                        "validation_fraction": 0.3,
                        "min_val_size": 2,
                        "min_tuning_val_size": 4,
                        "min_inner_train_size": 5,
                        "reuse_results": True,
                        "force_refresh_methods": ["ridge"],
                    },
                    "learning_curve": {
                        "min_train": 2,
                        "max_train": 4,
                        "n_repeats": 3,
                    },
                    "screening": {
                        "budget_mode": "screening_fraction",
                        "screen_fraction": 0.25,
                        "min_screen_size": 2,
                    },
                },
            }
        )

        assert cfg.experiment is not None
        assert cfg.experiment.learning_curve is not None
        assert cfg.experiment.screening is not None
        self.assertEqual(cfg.experiment.learning_curve.validation_fraction, 0.3)
        self.assertEqual(cfg.experiment.learning_curve.min_val_size, 2)
        self.assertEqual(cfg.experiment.learning_curve.min_tuning_val_size, 4)
        self.assertEqual(cfg.experiment.learning_curve.min_inner_train_size, 5)
        self.assertTrue(cfg.experiment.learning_curve.reuse_results)
        self.assertEqual(cfg.experiment.learning_curve.force_refresh_methods, ["ridge"])
        self.assertEqual(cfg.experiment.screening.validation_fraction, 0.3)
        self.assertEqual(cfg.experiment.screening.min_val_size, 2)
        self.assertEqual(cfg.experiment.screening.min_tuning_val_size, 4)
        self.assertEqual(cfg.experiment.screening.min_inner_train_size, 5)
        self.assertTrue(cfg.experiment.screening.reuse_results)
        self.assertEqual(cfg.experiment.screening.force_refresh_methods, ["ridge"])

    def test_experiment_defaults_allow_section_level_overrides(self) -> None:
        cfg = Config(
            **{
                "ingest": {
                    "source": "data/raw_vasp/systems",
                    "dataset_name": "test",
                    "stoich": {
                        "elements": ["H"],
                        "basis_species": ["H2"],
                        "basis_composition": {"H2": {"H": 2}},
                    },
                },
                "mlip": {
                    "dev_n": 1,
                    "dev_run": False,
                    "models": {"enabled": []},
                    "rootstock": {"root": ".", "models": {}},
                },
                "experiment": {
                    "defaults": {
                        "validation_fraction": 0.3,
                        "min_val_size": 2,
                        "min_tuning_val_size": 4,
                        "min_inner_train_size": 5,
                        "reuse_results": False,
                        "force_refresh_methods": ["ridge"],
                    },
                    "learning_curve": {
                        "min_train": 2,
                        "max_train": 4,
                        "n_repeats": 3,
                        "validation_fraction": 0.4,
                        "reuse_results": True,
                    },
                    "screening": {
                        "budget_mode": "screening_fraction",
                        "screen_fraction": 0.25,
                        "min_screen_size": 2,
                        "force_refresh_methods": ["weighted_simplex"],
                    },
                },
            }
        )

        assert cfg.experiment is not None
        assert cfg.experiment.learning_curve is not None
        assert cfg.experiment.screening is not None
        self.assertEqual(cfg.experiment.learning_curve.validation_fraction, 0.4)
        self.assertTrue(cfg.experiment.learning_curve.reuse_results)
        self.assertEqual(cfg.experiment.learning_curve.force_refresh_methods, ["ridge"])
        self.assertEqual(cfg.experiment.screening.validation_fraction, 0.3)
        self.assertFalse(cfg.experiment.screening.reuse_results)
        self.assertEqual(
            cfg.experiment.screening.force_refresh_methods,
            ["weighted_simplex"],
        )

    def test_learning_curve_mlip_selection_defaults_parse(self) -> None:
        cfg = Config(
            **{
                "ingest": {
                    "source": "data/raw_vasp/systems",
                    "dataset_name": "test",
                    "stoich": {
                        "elements": ["H"],
                        "basis_species": ["H2"],
                        "basis_composition": {"H2": {"H": 2}},
                    },
                },
                "mlip": {
                    "dev_n": 1,
                    "dev_run": False,
                    "models": {"enabled": []},
                    "rootstock": {"root": ".", "models": {}},
                },
                "experiment": {
                    "learning_curve": {
                        "min_train": 2,
                        "max_train": 4,
                        "n_repeats": 3,
                    }
                },
            }
        )

        assert cfg.experiment is not None
        assert cfg.experiment.learning_curve is not None
        self.assertEqual(cfg.experiment.learning_curve.mlip_selection.enabled, [])
        self.assertFalse(
            cfg.experiment.learning_curve.mlip_selection.exclude_anomalous_mlips
        )
        self.assertEqual(cfg.experiment.learning_curve.mlip_selection.minimum_quorum, 0)

    def test_learning_curve_mlip_selection_explicit_values_parse(self) -> None:
        cfg = Config(
            **{
                "ingest": {
                    "source": "data/raw_vasp/systems",
                    "dataset_name": "test",
                    "stoich": {
                        "elements": ["H"],
                        "basis_species": ["H2"],
                        "basis_composition": {"H2": {"H": 2}},
                    },
                },
                "mlip": {
                    "dev_n": 1,
                    "dev_run": False,
                    "models": {"enabled": []},
                    "rootstock": {"root": ".", "models": {}},
                },
                "experiment": {
                    "learning_curve": {
                        "min_train": 2,
                        "max_train": 4,
                        "n_repeats": 3,
                        "mlip_selection": {
                            "enabled": ["mace", "uma"],
                            "exclude_anomalous_mlips": True,
                            "minimum_quorum": 3,
                        },
                    }
                },
            }
        )

        assert cfg.experiment is not None
        assert cfg.experiment.learning_curve is not None
        self.assertEqual(cfg.experiment.learning_curve.mlip_selection.enabled, ["mace", "uma"])
        self.assertTrue(
            cfg.experiment.learning_curve.mlip_selection.exclude_anomalous_mlips
        )
        self.assertEqual(cfg.experiment.learning_curve.mlip_selection.minimum_quorum, 3)

    def test_learning_curve_mlip_selection_accepts_legacy_keys(
        self,
    ) -> None:
        cfg = Config(
            **{
                "ingest": {
                    "source": "data/raw_vasp/systems",
                    "dataset_name": "test",
                    "stoich": {
                        "elements": ["H"],
                        "basis_species": ["H2"],
                        "basis_composition": {"H2": {"H": 2}},
                    },
                },
                "mlip": {
                    "dev_n": 1,
                    "dev_run": False,
                    "models": {"enabled": []},
                    "rootstock": {"root": ".", "models": {}},
                },
                "experiment": {
                    "learning_curve": {
                        "min_train": 2,
                        "max_train": 4,
                        "n_repeats": 3,
                        "mlip_selection": {
                            "exclude_anomalous": True,
                            "label_allowlist": [],
                            "strict_inference_anomaly": False,
                            "minimum_quorum": 2,
                        },
                    }
                },
            }
        )

        self.assertTrue(cfg.experiment.learning_curve.mlip_selection.exclude_anomalous_mlips)
        self.assertEqual(cfg.experiment.learning_curve.mlip_selection.minimum_quorum, 2)

    def test_experiment_defaults_mlip_selection_applies_to_zero_shot_and_learning_curve(
        self,
    ) -> None:
        cfg = Config(
            **{
                "ingest": {
                    "source": "data/raw_vasp/systems",
                    "dataset_name": "test",
                    "stoich": {
                        "elements": ["H"],
                        "basis_species": ["H2"],
                        "basis_composition": {"H2": {"H": 2}},
                    },
                },
                "mlip": {
                    "dev_n": 1,
                    "dev_run": False,
                    "models": {"enabled": []},
                    "rootstock": {"root": ".", "models": {}},
                },
                "experiment": {
                    "defaults": {
                        "mlip_selection": {
                            "enabled": ["mace", "uma"],
                            "exclude_anomalous_mlips": True,
                            "minimum_quorum": 3,
                        }
                    },
                    "zero_shot": {},
                    "learning_curve": {
                        "min_train": 2,
                        "max_train": 4,
                        "n_repeats": 3,
                    },
                },
            }
        )

        assert cfg.experiment is not None
        assert cfg.experiment.zero_shot is not None
        assert cfg.experiment.learning_curve is not None
        self.assertEqual(cfg.experiment.zero_shot.mlip_selection.enabled, ["mace", "uma"])
        self.assertEqual(
            cfg.experiment.learning_curve.mlip_selection.enabled,
            ["mace", "uma"],
        )
        self.assertTrue(cfg.experiment.zero_shot.mlip_selection.exclude_anomalous_mlips)
        self.assertEqual(cfg.experiment.zero_shot.mlip_selection.minimum_quorum, 3)

    def test_learning_curve_split_sizing_defaults_parse(self) -> None:
        cfg = Config(
            **{
                "ingest": {
                    "source": "data/raw_vasp/systems",
                    "dataset_name": "test",
                    "stoich": {
                        "elements": ["H"],
                        "basis_species": ["H2"],
                        "basis_composition": {"H2": {"H": 2}},
                    },
                },
                "mlip": {
                    "dev_n": 1,
                    "dev_run": False,
                    "models": {"enabled": []},
                    "rootstock": {"root": ".", "models": {}},
                },
                "experiment": {
                        "learning_curve": {
                            "min_train": 2,
                            "max_train": 4,
                            "n_repeats": 3,
                        }
                    },
                }
        )

        assert cfg.experiment is not None
        assert cfg.experiment.learning_curve is not None
        self.assertEqual(cfg.experiment.learning_curve.budget_mode, "full_remainder_test")
        self.assertIsNone(cfg.experiment.learning_curve.screen_fraction)
        self.assertEqual(cfg.experiment.learning_curve.min_screen_size, 1)
        self.assertEqual(cfg.experiment.learning_curve.validation_fraction, 0.2)
        self.assertEqual(cfg.experiment.learning_curve.min_val_size, 1)
        self.assertEqual(cfg.experiment.learning_curve.min_tuning_val_size, 1)
        self.assertFalse(cfg.experiment.learning_curve.calibration_enabled)
        self.assertEqual(cfg.experiment.learning_curve.calibration_method, "scalar_scale")
        self.assertEqual(cfg.experiment.learning_curve.calibration_fraction, 0.2)
        self.assertEqual(cfg.experiment.learning_curve.min_cal_size, 1)
        self.assertEqual(cfg.experiment.learning_curve.min_inner_train_size, 1)
        self.assertEqual(cfg.experiment.learning_curve.min_test_size, 1)

    def test_experiment_calibration_settings_parse_and_default_share(self) -> None:
        cfg = Config(
            **{
                "seed": 13,
                "ingest": {
                    "source": "data/raw_vasp/systems",
                    "dataset_name": "test",
                    "stoich": {
                        "elements": ["H"],
                        "basis_species": ["H2"],
                        "basis_composition": {"H2": {"H": 2}},
                    },
                },
                "mlip": {
                    "dev_n": 1,
                    "dev_run": False,
                    "models": {"enabled": []},
                    "rootstock": {"root": ".", "models": {}},
                },
                "experiment": {
                    "defaults": {
                        "calibration_enabled": False,
                        "calibration_method": "scalar_scale",
                        "calibration_fraction": 0.3,
                        "min_cal_size": 2,
                    },
                    "learning_curve": {
                        "min_train": 2,
                        "max_train": 4,
                        "n_repeats": 2,
                    },
                    "screening": {
                        "screen_fraction": 0.2,
                    },
                },
            }
        )

        assert cfg.experiment is not None
        assert cfg.experiment.learning_curve is not None
        assert cfg.experiment.screening is not None
        self.assertFalse(cfg.experiment.learning_curve.calibration_enabled)
        self.assertEqual(cfg.experiment.learning_curve.calibration_method, "scalar_scale")
        self.assertEqual(cfg.experiment.learning_curve.calibration_fraction, 0.3)
        self.assertEqual(cfg.experiment.learning_curve.min_cal_size, 2)
        self.assertFalse(cfg.experiment.screening.calibration_enabled)
        self.assertEqual(cfg.experiment.screening.calibration_fraction, 0.3)
        self.assertEqual(cfg.experiment.screening.min_cal_size, 2)
        self.assertIsNone(cfg.experiment.learning_curve.results_bundle_path)
        self.assertFalse(cfg.experiment.learning_curve.reuse_results)
        self.assertEqual(cfg.experiment.learning_curve.force_refresh_methods, [])
        self.assertEqual(cfg.experiment.learning_curve.force_refresh_train_sizes, {})

    def test_plot_curve_window_section_parses(self) -> None:
        cfg = Config(
            **{
                "ingest": {
                    "source": "data/raw_vasp/systems",
                    "dataset_name": "test",
                    "stoich": {
                        "elements": ["H"],
                        "basis_species": ["H2"],
                        "basis_composition": {"H2": {"H": 2}},
                    },
                },
                "mlip": {
                    "dev_n": 1,
                    "dev_run": False,
                    "models": {"enabled": []},
                    "rootstock": {"root": ".", "models": {}},
                },
                "plot": {
                    "output_dir": "data/results/plots",
                    "zero_shot_stage_show_lone_mlip_swarm": False,
                    "curve_window": {
                        "full_dataset_window": True,
                        "all": True,
                        "min_x": 10,
                        "max_x": 50,
                        "include_x": [10, 30, 50],
                        "oracle_all_datasets_log_x": True,
                    },
                },
            }
        )

        assert cfg.plot is not None
        self.assertFalse(cfg.plot.zero_shot_stage_show_lone_mlip_swarm)
        self.assertTrue(cfg.plot.curve_window.full_dataset_window)
        self.assertTrue(cfg.plot.curve_window.all)
        self.assertEqual(cfg.plot.curve_window.min_x, 10)
        self.assertEqual(cfg.plot.curve_window.max_x, 50)
        self.assertEqual(cfg.plot.curve_window.include_x, [10, 30, 50])
        self.assertTrue(cfg.plot.curve_window.oracle_all_datasets_log_x)

    def test_plot_fixed_split_section_parses(self) -> None:
        cfg = Config(
            **{
                "ingest": {
                    "source": "data/raw_vasp/systems",
                    "dataset_name": "test",
                    "stoich": {
                        "elements": ["H"],
                        "basis_species": ["H2"],
                        "basis_composition": {"H2": {"H": 2}},
                    },
                },
                "mlip": {
                    "dev_n": 1,
                    "dev_run": False,
                    "models": {"enabled": []},
                    "rootstock": {"root": ".", "models": {}},
                },
                "plot": {
                    "output_dir": "data/results/plots",
                    "fixed_split": {
                        "train_fraction": 0.5,
                    },
                },
            }
        )

        assert cfg.plot is not None
        self.assertEqual(cfg.plot.fixed_split.train_fraction, 0.5)

    def test_screening_section_overrides_legacy_learning_curve_screening_fields(self) -> None:
        cfg = Config(
            **{
                "ingest": {
                    "source": "data/raw_vasp/systems",
                    "dataset_name": "test",
                    "stoich": {
                        "elements": ["H"],
                        "basis_species": ["H2"],
                        "basis_composition": {"H2": {"H": 2}},
                    },
                },
                "mlip": {
                    "dev_n": 1,
                    "dev_run": False,
                    "models": {"enabled": []},
                    "rootstock": {"root": ".", "models": {}},
                },
                "experiment": {
                    "learning_curve": {
                        "min_train": 2,
                        "max_train": 4,
                        "n_repeats": 3,
                        "budget_mode": "full_remainder_test",
                        "screen_fraction": 0.1,
                        "min_screen_size": 9,
                    },
                    "screening": {
                        "budget_mode": "screening_fraction",
                        "screen_fraction": 0.25,
                        "min_screen_size": 2,
                    },
                },
            }
        )

        assert cfg.experiment is not None
        assert cfg.experiment.learning_curve is not None
        assert cfg.experiment.screening is not None
        self.assertEqual(cfg.experiment.learning_curve.budget_mode, "full_remainder_test")
        self.assertEqual(cfg.experiment.learning_curve.screen_fraction, 0.1)
        self.assertEqual(cfg.experiment.learning_curve.min_screen_size, 9)
        self.assertEqual(cfg.experiment.screening.budget_mode, "screening_fraction")
        self.assertEqual(cfg.experiment.screening.screen_fraction, 0.25)
        self.assertEqual(cfg.experiment.screening.min_screen_size, 2)

    def test_screening_section_requires_learning_curve_section(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "experiment.screening requires experiment.learning_curve",
        ):
            Config(
                **{
                    "ingest": {
                        "source": "data/raw_vasp/systems",
                        "dataset_name": "test",
                        "stoich": {
                            "elements": ["H"],
                            "basis_species": ["H2"],
                            "basis_composition": {"H2": {"H": 2}},
                        },
                    },
                    "mlip": {
                        "dev_n": 1,
                        "dev_run": False,
                        "models": {"enabled": []},
                        "rootstock": {"root": ".", "models": {}},
                    },
                    "experiment": {
                        "screening": {
                            "screen_fraction": 0.25,
                        }
                    },
                }
            )

    def test_learning_curve_models_section_parses(self) -> None:
        cfg = Config(
            **{
                "ingest": {
                    "source": "data/raw_vasp/systems",
                    "dataset_name": "test",
                    "stoich": {
                        "elements": ["H"],
                        "basis_species": ["H2"],
                        "basis_composition": {"H2": {"H": 2}},
                    },
                },
                "mlip": {
                    "dev_n": 1,
                    "dev_run": False,
                    "models": {"enabled": []},
                    "rootstock": {"root": ".", "models": {}},
                },
                "plot": {
                    "output_dir": "data/results/plots",
                },
                "experiment": {
                        "learning_curve": {
                            "min_train": 2,
                            "max_train": 4,
                            "n_repeats": 3,
                            "models": {
                                "use_ridge": True,
                                "use_kernel_ridge": False,
                            "use_lasso": True,
                            "use_elastic_net": False,
                            "use_residual": True,
                            "use_weighted_linear": True,
                            "use_weighted_simplex": False,
                            "moe": {
                                "enabled": False,
                                "gate_type": "mlip_baseline",
                                "hidden_dims": [16, 8],
                                "training": {
                                    "batch_size": 64,
                                    "eval_batch_size": 128,
                                    "epochs": 250,
                                    "lr": 5e-4,
                                    "weight_decay": 1e-5,
                                    "num_workers": 4,
                                    "device": "cuda:0",
                                    "seed": 17,
                                },
                                "tuning": {
                                    "optuna": {
                                        "n_trials": 25,
                                        "sampler": "tpe",
                                        "pruner": "median",
                                        "timeout_s": 600,
                                        "seed": 7,
                                    }
                                },
                            },
                        },
                    },
                },
            }
        )

        assert cfg.plot is not None
        assert cfg.experiment is not None
        assert cfg.experiment.learning_curve is not None
        assert cfg.experiment.learning_curve.models is not None
        self.assertTrue(cfg.experiment.learning_curve.models.use_ridge)
        self.assertTrue(cfg.experiment.learning_curve.models.use_lasso)
        self.assertTrue(cfg.experiment.learning_curve.models.use_weighted_linear)
        self.assertEqual(cfg.experiment.learning_curve.models.moe.gate_type, "mlip_baseline")
        self.assertEqual(cfg.experiment.learning_curve.models.moe.hidden_dims, [16, 8])
        self.assertEqual(cfg.experiment.learning_curve.models.moe.training.batch_size, 64)
        self.assertEqual(
            cfg.experiment.learning_curve.models.moe.training.eval_batch_size,
            128,
        )
        self.assertEqual(cfg.experiment.learning_curve.models.moe.training.epochs, 250)
        self.assertEqual(cfg.experiment.learning_curve.models.moe.training.lr, 5e-4)
        self.assertEqual(
            cfg.experiment.learning_curve.models.moe.training.weight_decay,
            1e-5,
        )
        self.assertEqual(
            cfg.experiment.learning_curve.models.moe.training.num_workers,
            4,
        )
        self.assertEqual(cfg.experiment.learning_curve.models.moe.training.device, "cuda:0")
        self.assertEqual(cfg.experiment.learning_curve.models.moe.training.seed, 17)
        assert cfg.experiment.learning_curve.models.moe.tuning.optuna is not None
        self.assertEqual(
            cfg.experiment.learning_curve.models.moe.tuning.optuna.n_trials, 25
        )
        self.assertEqual(
            cfg.experiment.learning_curve.models.moe.tuning.optuna.sampler, "tpe"
        )
        self.assertEqual(
            cfg.experiment.learning_curve.models.moe.tuning.optuna.pruner, "median"
        )
        self.assertEqual(
            cfg.experiment.learning_curve.models.moe.tuning.optuna.timeout_s, 600
        )
        self.assertEqual(cfg.experiment.learning_curve.models.moe.tuning.optuna.seed, 7)
        self.assertEqual(cfg.experiment.learning_curve.min_train, 2)
        self.assertEqual(cfg.experiment.learning_curve.max_train, 4)
        self.assertEqual(cfg.experiment.learning_curve.n_repeats, 3)

    def test_learning_curve_models_moe_defaults_parse(self) -> None:
        cfg = Config(
            **{
                "seed": 23,
                "ingest": {
                    "source": "data/raw_vasp/systems",
                    "dataset_name": "test",
                    "stoich": {
                        "elements": ["H"],
                        "basis_species": ["H2"],
                        "basis_composition": {"H2": {"H": 2}},
                    },
                },
                "mlip": {
                    "dev_n": 1,
                    "dev_run": False,
                    "models": {"enabled": []},
                    "rootstock": {"root": ".", "models": {}},
                },
                "experiment": {
                        "learning_curve": {
                            "min_train": 2,
                            "max_train": 4,
                            "n_repeats": 3,
                            "models": {
                                "use_ridge": True,
                                "use_kernel_ridge": False,
                            "use_lasso": False,
                            "use_elastic_net": False,
                            "use_residual": True,
                        },
                    }
                },
            }
        )

        assert cfg.experiment is not None
        assert cfg.experiment.learning_curve is not None
        assert cfg.experiment.learning_curve.models is not None
        self.assertFalse(cfg.experiment.learning_curve.models.use_weighted_linear)
        self.assertFalse(cfg.experiment.learning_curve.models.use_weighted_simplex)
        self.assertFalse(cfg.experiment.learning_curve.models.moe.enabled)
        self.assertIsNone(cfg.experiment.learning_curve.models.tuning.optuna)
        self.assertEqual(cfg.experiment.learning_curve.models.moe.hidden_dims, [])
        self.assertEqual(cfg.experiment.learning_curve.models.moe.training.batch_size, 32)
        self.assertIsNone(cfg.experiment.learning_curve.models.moe.training.eval_batch_size)
        self.assertIsNone(cfg.experiment.learning_curve.models.moe.training.epochs)
        self.assertEqual(cfg.experiment.learning_curve.models.moe.training.lr, 1e-3)
        self.assertEqual(
            cfg.experiment.learning_curve.models.moe.training.weight_decay,
            0.0,
        )
        self.assertEqual(cfg.experiment.learning_curve.models.moe.training.num_workers, 0)
        self.assertIsNone(cfg.experiment.learning_curve.models.moe.training.device)
        self.assertEqual(cfg.experiment.learning_curve.models.moe.training.seed, 23)
        self.assertIsNone(cfg.experiment.learning_curve.models.moe.tuning.optuna)

    def test_learning_curve_models_moe_optuna_defaults_parse(self) -> None:
        cfg = Config(
            **{
                "seed": 29,
                "ingest": {
                    "source": "data/raw_vasp/systems",
                    "dataset_name": "test",
                    "stoich": {
                        "elements": ["H"],
                        "basis_species": ["H2"],
                        "basis_composition": {"H2": {"H": 2}},
                    },
                },
                "mlip": {
                    "dev_n": 1,
                    "dev_run": False,
                    "models": {"enabled": []},
                    "rootstock": {"root": ".", "models": {}},
                },
                "experiment": {
                        "learning_curve": {
                            "min_train": 2,
                            "max_train": 4,
                            "n_repeats": 3,
                            "models": {
                                "use_ridge": False,
                                "use_kernel_ridge": False,
                            "use_lasso": False,
                            "use_elastic_net": False,
                            "use_residual": True,
                            "moe": {
                                "enabled": True,
                                "tuning": {
                                    "optuna": {
                                        "n_trials": 50,
                                    }
                                },
                            },
                        },
                    }
                },
            }
        )

        assert cfg.experiment is not None
        assert cfg.experiment.learning_curve is not None
        assert cfg.experiment.learning_curve.models is not None
        self.assertIsNone(cfg.experiment.learning_curve.models.tuning.optuna)
        assert cfg.experiment.learning_curve.models.moe.tuning.optuna is not None
        self.assertEqual(cfg.experiment.learning_curve.models.moe.tuning.optuna.n_trials, 50)
        self.assertIsNone(cfg.experiment.learning_curve.models.moe.tuning.optuna.sampler)
        self.assertIsNone(cfg.experiment.learning_curve.models.moe.tuning.optuna.pruner)
        self.assertIsNone(cfg.experiment.learning_curve.models.moe.tuning.optuna.timeout_s)
        self.assertEqual(cfg.experiment.learning_curve.models.moe.tuning.optuna.seed, 29)

    def test_learning_curve_model_seeds_preserve_explicit_overrides(self) -> None:
        cfg = Config(
            **{
                "seed": 23,
                "ingest": {
                    "source": "data/raw_vasp/systems",
                    "dataset_name": "test",
                    "stoich": {
                        "elements": ["H"],
                        "basis_species": ["H2"],
                        "basis_composition": {"H2": {"H": 2}},
                    },
                },
                "mlip": {
                    "dev_n": 1,
                    "dev_run": False,
                    "models": {"enabled": []},
                    "rootstock": {"root": ".", "models": {}},
                },
                "experiment": {
                    "learning_curve": {
                        "min_train": 2,
                        "max_train": 4,
                        "n_repeats": 3,
                        "models": {
                            "use_ridge": False,
                            "use_kernel_ridge": False,
                            "use_lasso": False,
                            "use_elastic_net": False,
                            "use_residual": True,
                            "probe_gnn": {
                                "enabled": True,
                                "training": {"epochs": 25, "seed": 7},
                                "tuning": {"optuna": {"n_trials": 5, "seed": 11}},
                            },
                        },
                    }
                },
            }
        )

        assert cfg.experiment is not None
        assert cfg.experiment.learning_curve is not None
        assert cfg.experiment.learning_curve.models is not None
        probe = cfg.experiment.learning_curve.models.probe_gnn
        self.assertIsNone(cfg.experiment.learning_curve.models.tuning.optuna)
        self.assertEqual(probe.training.seed, 7)
        assert probe.tuning.optuna is not None
        self.assertEqual(probe.tuning.optuna.seed, 11)

    def test_top_level_optuna_config_is_available_as_shared_learning_curve_tuning(self) -> None:
        cfg = Config(
            **{
                "seed": 29,
                "tuning": {
                    "optuna": {
                        "n_trials": 50,
                    }
                },
                "ingest": {
                    "source": "data/raw_vasp/systems",
                    "dataset_name": "test",
                    "stoich": {
                        "elements": ["H"],
                        "basis_species": ["H2"],
                        "basis_composition": {"H2": {"H": 2}},
                    },
                },
                "mlip": {
                    "dev_n": 1,
                    "dev_run": False,
                    "models": {"enabled": []},
                    "rootstock": {"root": ".", "models": {}},
                },
                "experiment": {
                    "learning_curve": {
                        "min_train": 2,
                        "max_train": 4,
                        "n_repeats": 3,
                        "models": {
                            "use_ridge": False,
                            "use_kernel_ridge": False,
                            "use_lasso": False,
                            "use_elastic_net": False,
                            "use_residual": True,
                            "use_gnn_direct": True,
                        },
                    }
                },
            }
        )

        assert cfg.experiment is not None
        assert cfg.experiment.learning_curve is not None
        assert cfg.experiment.learning_curve.models is not None
        shared_optuna = cfg.experiment.learning_curve.models.tuning.optuna
        assert shared_optuna is not None
        self.assertEqual(shared_optuna.n_trials, 50)
        self.assertEqual(shared_optuna.seed, 29)

    def test_global_device_is_inherited_by_torch_learned_families(self) -> None:
        cfg = Config(
            **{
                "seed": 7,
                "device": "cuda",
                "ingest": {
                    "source": "data/raw_vasp/systems",
                    "dataset_name": "test",
                    "stoich": {
                        "elements": ["H"],
                        "basis_species": ["H2"],
                        "basis_composition": {"H2": {"H": 2}},
                    },
                },
                "mlip": {
                    "dev_n": 1,
                    "dev_run": False,
                    "models": {"enabled": []},
                    "rootstock": {"root": ".", "models": {}},
                },
                "experiment": {
                    "learning_curve": {
                        "min_train": 2,
                        "max_train": 4,
                        "n_repeats": 3,
                        "models": {
                            "use_ridge": False,
                            "use_kernel_ridge": False,
                            "use_lasso": False,
                            "use_elastic_net": False,
                            "use_residual": True,
                            "moe": {"enabled": True, "gate_type": "gnn"},
                            "probe_gnn": {"enabled": True},
                            "gnn_direct": {"enabled": True},
                        },
                    }
                },
            }
        )

        assert cfg.experiment is not None
        assert cfg.experiment.learning_curve is not None
        assert cfg.experiment.learning_curve.models is not None
        models = cfg.experiment.learning_curve.models
        self.assertEqual(models.moe.training.device, "cuda")
        self.assertEqual(models.probe_gnn.training.device, "cuda")
        self.assertEqual(models.gnn_direct.training.device, "cuda")

    def test_plot_can_parse_without_model_toggles(self) -> None:
        cfg = Config(
            **{
                "ingest": {
                    "source": "data/raw_vasp/systems",
                    "dataset_name": "test",
                    "stoich": {
                        "elements": ["H"],
                        "basis_species": ["H2"],
                        "basis_composition": {"H2": {"H": 2}},
                    },
                },
                "mlip": {
                    "dev_n": 1,
                    "dev_run": False,
                    "models": {"enabled": []},
                    "rootstock": {"root": ".", "models": {}},
                },
                "plot": {
                    "output_dir": "data/results/plots",
                },
                "experiment": {
                        "learning_curve": {
                            "min_train": 2,
                            "max_train": 4,
                            "n_repeats": 3,
                        }
                    },
                }
        )

        assert cfg.plot is not None
        assert cfg.experiment is not None
        assert cfg.experiment.learning_curve is not None
        self.assertEqual(cfg.plot.output_dir.as_posix(), "data/results/plots")
        self.assertIsNone(cfg.experiment.learning_curve.models)
        self.assertEqual(cfg.experiment.learning_curve.min_train, 2)
        self.assertEqual(cfg.experiment.learning_curve.max_train, 4)
        self.assertEqual(cfg.experiment.learning_curve.n_repeats, 3)

    def test_experiment_can_be_omitted(self) -> None:
        cfg = Config(
            **{
                "ingest": {
                    "source": "data/raw_vasp/systems",
                    "dataset_name": "test",
                    "stoich": {
                        "elements": ["H"],
                        "basis_species": ["H2"],
                        "basis_composition": {"H2": {"H": 2}},
                    },
                },
                "mlip": {
                    "dev_n": 1,
                    "dev_run": False,
                    "models": {"enabled": []},
                    "rootstock": {"root": ".", "models": {}},
                },
                "plot": {
                    "output_dir": "data/results/plots",
                },
            }
        )

        self.assertIsNone(cfg.experiment)

    def test_plot_can_be_omitted(self) -> None:
        cfg = Config(
            **{
                "ingest": {
                    "source": "data/raw_vasp/systems",
                    "dataset_name": "test",
                    "stoich": {
                        "elements": ["H"],
                        "basis_species": ["H2"],
                        "basis_composition": {"H2": {"H": 2}},
                    },
                },
                "mlip": {
                    "dev_n": 1,
                    "dev_run": False,
                    "models": {"enabled": []},
                    "rootstock": {"root": ".", "models": {}},
                },
            }
        )

        self.assertIsNone(cfg.plot)

    def test_probe_gnn_config_parses(self) -> None:
        cfg = Config(
            **{
                "ingest": {
                    "source": "data/raw_vasp/systems",
                    "dataset_name": "test",
                    "stoich": {
                        "elements": ["H"],
                        "basis_species": ["H2"],
                        "basis_composition": {"H2": {"H": 2}},
                    },
                },
                "mlip": {
                    "dev_n": 1,
                    "dev_run": False,
                    "models": {"enabled": []},
                    "rootstock": {"root": ".", "models": {}},
                },
                "experiment": {
                    "learning_curve": {
                        "min_train": 2,
                        "max_train": 4,
                        "n_repeats": 3,
                        "models": {
                            "use_ridge": False,
                            "use_kernel_ridge": False,
                            "use_lasso": False,
                            "use_elastic_net": False,
                            "use_residual": False,
                            "probe_gnn": {
                                "enabled": True,
                                "hidden_dims": [32, 32],
                                "training": {"epochs": 25, "seed": 7},
                            },
                        },
                    }
                },
            }
        )

        assert cfg.experiment is not None
        assert cfg.experiment.learning_curve is not None
        assert cfg.experiment.learning_curve.models is not None
        probe = cfg.experiment.learning_curve.models.probe_gnn
        self.assertTrue(probe.enabled)
        self.assertEqual(probe.hidden_dims, [32, 32])
        self.assertEqual(probe.training.epochs, 25)
        self.assertEqual(probe.training.seed, 7)

    def test_probe_gnn_config_defaults(self) -> None:
        cfg = Config(
            **{
                "ingest": {
                    "source": "data/raw_vasp/systems",
                    "dataset_name": "test",
                    "stoich": {
                        "elements": ["H"],
                        "basis_species": ["H2"],
                        "basis_composition": {"H2": {"H": 2}},
                    },
                },
                "mlip": {
                    "dev_n": 1,
                    "dev_run": False,
                    "models": {"enabled": []},
                    "rootstock": {"root": ".", "models": {}},
                },
                "experiment": {
                    "learning_curve": {
                        "min_train": 2,
                        "max_train": 4,
                        "n_repeats": 3,
                        "models": {
                            "use_ridge": False,
                            "use_kernel_ridge": False,
                            "use_lasso": False,
                            "use_elastic_net": False,
                            "use_residual": False,
                        },
                    }
                },
            }
        )

        assert cfg.experiment is not None
        assert cfg.experiment.learning_curve is not None
        assert cfg.experiment.learning_curve.models is not None
        probe = cfg.experiment.learning_curve.models.probe_gnn
        self.assertFalse(probe.enabled)
        self.assertEqual(probe.hidden_dims, [])

    def test_latent_config_parses_timing_path(self) -> None:
        cfg = Config(
            **{
                "ingest": {
                    "source": "data/raw_vasp/systems",
                    "dataset_name": "test",
                    "stoich": {
                        "elements": ["H"],
                        "basis_species": ["H2"],
                        "basis_composition": {"H2": {"H": 2}},
                    },
                },
                "mlip": {
                    "dev_n": 1,
                    "dev_run": False,
                    "models": {"enabled": []},
                    "rootstock": {"root": ".", "models": {}},
                },
                "experiment": {
                    "learning_curve": {
                        "min_train": 2,
                        "max_train": 4,
                        "n_repeats": 3,
                        "models": {
                            "use_ridge": False,
                            "use_kernel_ridge": False,
                            "use_lasso": False,
                            "use_elastic_net": False,
                            "use_residual": False,
                            "use_latent": True,
                            "latent": {
                                "experiment_config_path": "vendor/latent/config.toml",
                                "csv_path": "vendor/latent/data/processed/example_latent.csv",
                                "timing_path": "vendor/latent/data/processed/example_latent.csv.timing.json",
                            },
                        },
                    }
                },
            }
        )

        assert cfg.experiment is not None
        assert cfg.experiment.learning_curve is not None
        assert cfg.experiment.learning_curve.models is not None
        latent = cfg.experiment.learning_curve.models.latent
        assert latent is not None
        self.assertEqual(
            latent.timing_path,
            Path("vendor/latent/data/processed/example_latent.csv.timing.json"),
        )
