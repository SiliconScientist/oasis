from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from oasis.candidate_ranking_runner import run_candidate_ranking_from_config
from oasis.experiment_config import CandidateRankingConfig
from oasis.config import get_config


class CandidateRankingConfigTests(unittest.TestCase):
    def test_get_config_loads_candidate_ranking_section(self) -> None:
        with patch(
            "oasis.config.load_config_data",
            return_value={
                "ingest": {
                    "source": "data/raw_data/source.json",
                    "dataset_name": "example",
                    "stoich": {
                        "elements": ["H"],
                        "basis_species": ["H2"],
                        "basis_composition": {"H2": {"H": 2}},
                    },
                },
                "mlip": {
                    "dev_n": 1,
                    "dev_run": False,
                    "dataset": "data/raw_data/example.json",
                    "models": {"enabled": ["mace"]},
                    "rootstock": {"root": "vendor/rootstock", "models": {}},
                },
                "dataset_profile": {"tag": "mamun_oh"},
                "datasets": {
                    "mamun_oh": {
                        "raw_dataset_filename": "MamunHighT2019_oh_adsorption.json",
                        "mlip_run_dirname": "OH-BMA",
                    }
                },
                "candidate_ranking": {
                    "method": "zero_shot",
                    "target_binding_energy": -0.5,
                    "top_k": 5,
                    "uncertainty_weight": 0.4,
                    "method_configs": {
                        "two_shot": {"acquisition_weight": 0.3},
                    },
                },
            },
        ):
            cfg = get_config("experiment.toml")

        assert cfg.candidate_ranking is not None
        self.assertEqual(cfg.candidate_ranking.method, "zero_shot")
        self.assertEqual(cfg.candidate_ranking.top_k, 5)
        self.assertEqual(cfg.candidate_ranking.uncertainty_weight, 0.4)
        self.assertEqual(
            cfg.candidate_ranking.method_configs["two_shot"]["acquisition_weight"],
            0.3,
        )
        self.assertEqual(cfg.candidate_ranking.results_dir, Path("data/mlips/OH-BMA"))

    def test_candidate_ranking_resolved_method_config_merges_future_namespace(self) -> None:
        cfg = CandidateRankingConfig(
            method="two_shot",
            results_dir="data/mlips/example",
            target_binding_energy=-0.1,
            uncertainty_weight=0.6,
            method_configs={
                "two_shot": {"acquisition_weight": 0.3},
            },
        )

        resolved = cfg.resolved_method_config()

        self.assertEqual(resolved["uncertainty_weight"], 0.6)
        self.assertEqual(resolved["acquisition_weight"], 0.3)

    def test_run_candidate_ranking_from_config_uses_registry_selected_method(self) -> None:
        ranking_cfg = CandidateRankingConfig(
            method="zero_shot",
            results_dir="data/mlips/example",
            target_binding_energy=-0.2,
            top_k=3,
        )
        fake_cfg = SimpleNamespace(
            dataset_profile=SimpleNamespace(tag="example"),
            candidate_ranking=ranking_cfg,
        )
        with patch("oasis.candidate_ranking_runner.get_config", return_value=fake_cfg), patch(
            "oasis.candidate_ranking_runner.rank_candidates_from_results_dir"
        ) as mock_rank, patch("builtins.print") as mock_print:
            mock_rank.return_value = type(
                "_Result",
                (),
                {
                    "ranked_candidates": [],
                },
            )()
            run_candidate_ranking_from_config(["experiment.toml"])

        mock_rank.assert_called_once()
        self.assertEqual(mock_rank.call_args.kwargs["method_name"], "zero_shot")
        self.assertEqual(
            mock_rank.call_args.kwargs["target_binding_energy"],
            -0.2,
        )
        mock_print.assert_called_once()
