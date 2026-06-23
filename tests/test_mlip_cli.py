from __future__ import annotations

import importlib
import runpy
import sys
import unittest
from unittest.mock import patch

from oasis.__main__ import main
from oasis.mlip.cli import main as mlip_main


class MainDispatchTests(unittest.TestCase):
    def test_main_forwards_mlip_args_to_dedicated_cli(self) -> None:
        with patch("oasis.mlip.cli.main") as mock_mlip_main:
            main(["mlip", "run-one", "--line", "task", "--config", "config.toml"])

        mock_mlip_main.assert_called_once_with(
            ["run-one", "--line", "task", "--config", "config.toml"]
        )

    def test_main_forwards_non_mlip_args_to_experiment_runner(self) -> None:
        with patch("oasis.experiment_runner.run_experiment_from_config") as mock_run:
            main(["experiment.toml"])

        mock_run.assert_called_once_with(["experiment.toml"])

    def test_importing_main_does_not_load_experiment_modules(self) -> None:
        sys.modules.pop("oasis.__main__", None)
        before_import = set(sys.modules)

        importlib.import_module("oasis.__main__")

        for name in (
            "oasis.analysis",
            "oasis.exp",
            "oasis.experiment_runner",
            "oasis.graphs",
            "oasis.plot",
            "oasis.probe",
            "oasis.probe_features",
        ):
            self.assertNotIn(name, set(sys.modules) - before_import)


class MlipCliTests(unittest.TestCase):
    def test_importing_mlip_cli_does_not_load_experiment_modules(self) -> None:
        sys.modules.pop("oasis.mlip.cli", None)
        before_import = set(sys.modules)

        importlib.import_module("oasis.mlip.cli")

        for name in (
            "oasis.analysis",
            "oasis.exp",
            "oasis.experiment_runner",
            "oasis.graphs",
            "oasis.learning_curve.registry",
            "oasis.learning_curve.results_io",
            "oasis.learning_curve.runners",
            "oasis.plot",
            "oasis.probe",
            "oasis.probe_features",
        ):
            self.assertNotIn(name, set(sys.modules) - before_import)

    def test_submit_delegates_to_submit_jobs(self) -> None:
        with patch("oasis.mlip.submit.submit_jobs") as mock_submit_jobs:
            mlip_main(
                ["submit", "--config", "config.toml", "--run-tag", "tag", "dataset.json"]
            )

        mock_submit_jobs.assert_called_once_with(
            config_path="config.toml",
            run_tag="tag",
            datasets=["dataset.json"],
        )

    def test_run_one_delegates_to_runner(self) -> None:
        with patch("oasis.mlip.runner.run_one_task") as mock_run_one_task:
            mlip_main(["run-one", "--line", "task-line", "--config", "config.toml"])

        mock_run_one_task.assert_called_once_with(
            line="task-line",
            config_path="config.toml",
        )

    def test_make_tasks_delegates_to_task_writer(self) -> None:
        with patch("oasis.mlip.tasks.make_tasks") as mock_make_tasks:
            mlip_main(
                [
                    "make-tasks",
                    "--config",
                    "config.toml",
                    "--run-tag",
                    "tag",
                    "--out",
                    "tasks.txt",
                    "dataset.json",
                ]
            )

        mock_make_tasks.assert_called_once_with(
            config_path="config.toml",
            run_tag="tag",
            out_path="tasks.txt",
            datasets=["dataset.json"],
        )

    def test_python_dash_m_oasis_mlip_delegates_to_cli_main(self) -> None:
        with patch("oasis.mlip.cli.main") as mock_mlip_main:
            with patch.object(sys, "argv", ["python", "submit", "--config", "config.toml"]):
                runpy.run_module("oasis.mlip", run_name="__main__")

        mock_mlip_main.assert_called_once_with()
