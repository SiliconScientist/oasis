from __future__ import annotations

import unittest
from unittest.mock import patch

from oasis.__main__ import main


class MainTests(unittest.TestCase):
    def test_main_runs_experiment_with_default_config_discovery_on_empty_cli(self) -> None:
        with patch("oasis.experiment_runner.run_experiment_from_config") as mock_run:
            main([])

        mock_run.assert_called_once_with(None)

    def test_main_treats_non_mlip_args_as_config_paths(self) -> None:
        with patch("oasis.experiment_runner.run_experiment_from_config") as mock_run:
            main(["experiment.toml"])

        mock_run.assert_called_once_with(["experiment.toml"])

    def test_main_dispatches_candidate_ranking_cli(self) -> None:
        with patch(
            "oasis.candidate_ranking_runner.run_candidate_ranking_from_config"
        ) as mock_run:
            main(["rank-candidates", "experiment.toml"])

        mock_run.assert_called_once_with(["experiment.toml"])
