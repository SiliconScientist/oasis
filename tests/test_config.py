from __future__ import annotations

import unittest

from oasis.config import Config


class ConfigParsingTests(unittest.TestCase):
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
                        "trim": False,
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
                                "gate_type": "mlp",
                                "gating_mode": "soft",
                                "hidden_dims": [16, 8],
                                "training": {},
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
        self.assertEqual(cfg.experiment.learning_curve.models.moe.gate_type, "mlp")
        self.assertEqual(cfg.experiment.learning_curve.models.moe.gating_mode, "soft")
        self.assertEqual(cfg.experiment.learning_curve.models.moe.hidden_dims, [16, 8])
        self.assertEqual(cfg.experiment.learning_curve.min_train, 2)
        self.assertEqual(cfg.experiment.learning_curve.max_train, 4)
        self.assertEqual(cfg.experiment.learning_curve.n_repeats, 3)
        self.assertFalse(cfg.experiment.learning_curve.trim)
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
                        "trim": False,
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
                        "trim": True,
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
        self.assertTrue(cfg.experiment.learning_curve.trim)
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
