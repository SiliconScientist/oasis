from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

try:
    from oasis.config import Config, get_config

    HAS_CONFIG = True
except ModuleNotFoundError:
    HAS_CONFIG = False


@unittest.skipUnless(HAS_CONFIG, "requires config dependencies")
class ConfigParsingTests(unittest.TestCase):
    def test_get_config_merges_mlip_and_experiment_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            mlip_path = tmp / "mlip.toml"
            experiment_path = tmp / "experiment.toml"
            mlip_path.write_text(
                "\n".join(
                    [
                        'seed = 7',
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
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            experiment_path.write_text(
                "\n".join(
                    [
                        "[plot]",
                        'output_dir = "data/results/plots"',
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

            cfg = get_config([mlip_path, experiment_path])

        self.assertEqual(cfg.seed, 7)
        self.assertEqual(cfg.mlip.dataset, "data/raw_data/example.json")
        assert cfg.plot is not None
        self.assertEqual(cfg.plot.output_dir, Path("data/results/plots"))
        assert cfg.experiment is not None
        assert cfg.experiment.learning_curve is not None
        self.assertEqual(cfg.experiment.learning_curve.n_repeats, 3)

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
                            "validation_fraction": 0.35,
                            "min_val_size": 2,
                            "min_tuning_val_size": 4,
                            "min_inner_train_size": 5,
                            "min_test_size": 3,
                    }
                },
            }
        )

        assert cfg.experiment is not None
        assert cfg.experiment.learning_curve is not None
        self.assertEqual(cfg.experiment.learning_curve.validation_fraction, 0.35)
        self.assertEqual(cfg.experiment.learning_curve.min_val_size, 2)
        self.assertEqual(cfg.experiment.learning_curve.min_tuning_val_size, 4)
        self.assertEqual(cfg.experiment.learning_curve.min_inner_train_size, 5)
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
                        "results_bundle_path": "data/results/learning_curve/example.json",
                        "reuse_results": True,
                        "force_refresh_methods": ["moe", "probe_gnn"],
                        "force_refresh_train_sizes": {"ridge": [5, 10]},
                    }
                },
            }
        )

        assert cfg.experiment is not None
        assert cfg.experiment.learning_curve is not None
        self.assertEqual(
            cfg.experiment.learning_curve.results_bundle_path,
            Path("data/results/learning_curve/example.json"),
        )
        self.assertTrue(cfg.experiment.learning_curve.reuse_results)
        self.assertEqual(
            cfg.experiment.learning_curve.force_refresh_methods,
            ["moe", "probe_gnn"],
        )
        self.assertEqual(
            cfg.experiment.learning_curve.force_refresh_train_sizes,
            {"ridge": [5, 10]},
        )

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
        self.assertEqual(cfg.experiment.learning_curve.validation_fraction, 0.2)
        self.assertEqual(cfg.experiment.learning_curve.min_val_size, 1)
        self.assertEqual(cfg.experiment.learning_curve.min_tuning_val_size, 1)
        self.assertEqual(cfg.experiment.learning_curve.min_inner_train_size, 1)
        self.assertEqual(cfg.experiment.learning_curve.min_test_size, 1)
        self.assertIsNone(cfg.experiment.learning_curve.results_bundle_path)
        self.assertFalse(cfg.experiment.learning_curve.reuse_results)
        self.assertEqual(cfg.experiment.learning_curve.force_refresh_methods, [])
        self.assertEqual(cfg.experiment.learning_curve.force_refresh_train_sizes, {})

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
                    "filters": {
                        "adsorbate": "OH",
                        "anomaly_label": "!inference_anomaly",
                        "reaction_contains": ["Pt"],
                    },
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
        self.assertEqual(cfg.plot.filters.adsorbate, "OH")
        self.assertEqual(cfg.plot.filters.anomaly_label, "!inference_anomaly")
        self.assertEqual(cfg.plot.filters.reaction_contains, ["Pt"])

    def test_learning_curve_models_moe_defaults_parse(self) -> None:
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
        self.assertEqual(cfg.experiment.learning_curve.models.moe.hidden_dims, [])
        self.assertEqual(cfg.experiment.learning_curve.models.moe.training.batch_size, 32)
        self.assertIsNone(cfg.experiment.learning_curve.models.moe.training.eval_batch_size)
        self.assertEqual(cfg.experiment.learning_curve.models.moe.training.epochs, 100)
        self.assertEqual(cfg.experiment.learning_curve.models.moe.training.lr, 1e-3)
        self.assertEqual(
            cfg.experiment.learning_curve.models.moe.training.weight_decay,
            0.0,
        )
        self.assertEqual(cfg.experiment.learning_curve.models.moe.training.num_workers, 0)
        self.assertIsNone(cfg.experiment.learning_curve.models.moe.training.device)
        self.assertIsNone(cfg.experiment.learning_curve.models.moe.training.seed)
        self.assertIsNone(cfg.experiment.learning_curve.models.moe.tuning.optuna)

    def test_learning_curve_models_moe_optuna_defaults_parse(self) -> None:
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
        assert cfg.experiment.learning_curve.models.moe.tuning.optuna is not None
        self.assertEqual(cfg.experiment.learning_curve.models.moe.tuning.optuna.n_trials, 50)
        self.assertIsNone(cfg.experiment.learning_curve.models.moe.tuning.optuna.sampler)
        self.assertIsNone(cfg.experiment.learning_curve.models.moe.tuning.optuna.pruner)
        self.assertIsNone(cfg.experiment.learning_curve.models.moe.tuning.optuna.timeout_s)
        self.assertIsNone(cfg.experiment.learning_curve.models.moe.tuning.optuna.seed)

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
        self.assertIsNone(cfg.plot.filters.adsorbate)
        self.assertIsNone(cfg.plot.filters.anomaly_label)
        self.assertIsNone(cfg.plot.filters.reaction_contains)

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
