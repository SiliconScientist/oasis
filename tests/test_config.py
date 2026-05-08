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
                    "min_train": 2,
                    "max_train": 4,
                    "n_repeats": 3,
                    "trim": False,
                },
                "learning_curve_models": {
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
            }
        )

        assert cfg.learning_curve_models is not None
        self.assertTrue(cfg.learning_curve_models.use_ridge)
        self.assertTrue(cfg.learning_curve_models.use_lasso)
        self.assertTrue(cfg.learning_curve_models.use_weighted_linear)
        self.assertEqual(cfg.learning_curve_models.moe.gate_type, "mlp")
        self.assertEqual(cfg.learning_curve_models.moe.gating_mode, "soft")
        self.assertEqual(cfg.learning_curve_models.moe.hidden_dims, [16, 8])

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
                "learning_curve_models": {
                    "use_ridge": True,
                    "use_kernel_ridge": False,
                    "use_lasso": False,
                    "use_elastic_net": False,
                    "use_residual": True,
                },
            }
        )

        assert cfg.learning_curve_models is not None
        self.assertFalse(cfg.learning_curve_models.use_weighted_linear)
        self.assertFalse(cfg.learning_curve_models.use_weighted_simplex)
        self.assertFalse(cfg.learning_curve_models.moe.enabled)
        self.assertEqual(cfg.learning_curve_models.moe.hidden_dims, [])
