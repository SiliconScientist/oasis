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
                    "predictors": ["residual", "weighted_simplex", "ridge"],
                    "target_binding_energy": -0.5,
                    "top_k": 5,
                    "uncertainty_weight": 0.4,
                    "predictor_configs": {
                        "ridge": {"alpha_grid": [0.1, 1.0]},
                    },
                },
            },
        ):
            cfg = get_config("experiment.toml")

        assert cfg.candidate_ranking is not None
        self.assertEqual(
            cfg.candidate_ranking.predictors,
            ["residual", "weighted_simplex", "ridge"],
        )
        self.assertEqual(cfg.candidate_ranking.top_k, 5)
        self.assertEqual(cfg.candidate_ranking.uncertainty_weight, 0.4)
        self.assertEqual(
            cfg.candidate_ranking.predictor_configs["ridge"]["alpha_grid"],
            [0.1, 1.0],
        )
        self.assertEqual(cfg.candidate_ranking.results_dir, Path("data/mlips/OH-BMA"))

    def test_candidate_ranking_resolved_predictor_config_merges_future_namespace(self) -> None:
        cfg = CandidateRankingConfig(
            predictors=["residual", "ridge"],
            results_dir="data/mlips/example",
            target_binding_energy=-0.1,
            uncertainty_weight=0.6,
            predictor_configs={
                "ridge": {"alpha_grid": [0.1, 1.0]},
            },
        )

        resolved = cfg.resolved_predictor_config("ridge")

        self.assertEqual(resolved["uncertainty_weight"], 0.6)
        self.assertEqual(resolved["alpha_grid"], [0.1, 1.0])

    def test_candidate_ranking_requires_nonempty_predictor_list(self) -> None:
        with self.assertRaisesRegex(ValueError, "predictors must not be empty"):
            CandidateRankingConfig(
                predictors=[],
                results_dir="data/mlips/example",
                target_binding_energy=-0.1,
            )

    def test_run_candidate_ranking_from_config_uses_predictor_list_surface(self) -> None:
        ranking_cfg = CandidateRankingConfig(
            predictors=["residual", "weighted_simplex", "ridge"],
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
        self.assertEqual(
            list(mock_rank.call_args.kwargs["predictor_names"]),
            ["residual", "weighted_simplex", "ridge"],
        )
        self.assertEqual(
            mock_rank.call_args.kwargs["target_binding_energy"],
            -0.2,
        )
        mock_print.assert_called_once()

    def test_run_candidate_ranking_from_config_loads_validated_references_when_configured(
        self,
    ) -> None:
        ranking_cfg = CandidateRankingConfig(
            predictors=["residual", "ridge"],
            results_dir="data/mlips/example",
            validated_references_path="data/validated_references/example.json",
            target_binding_energy=-0.2,
            top_k=3,
        )
        fake_cfg = SimpleNamespace(
            dataset_profile=SimpleNamespace(tag="example"),
            candidate_ranking=ranking_cfg,
        )
        with patch("oasis.candidate_ranking_runner.get_config", return_value=fake_cfg), patch(
            "oasis.candidate_ranking_runner.rank_candidates_from_results_dir_and_references"
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
        self.assertEqual(
            mock_rank.call_args.kwargs["validated_references_path"],
            Path("data/validated_references/example.json"),
        )
        mock_print.assert_called_once()
