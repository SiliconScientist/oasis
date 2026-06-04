from __future__ import annotations

import unittest
from unittest.mock import patch

from oasis.__main__ import main
from oasis.mlip.cli import main as mlip_main


class MainDispatchTests(unittest.TestCase):
    def test_main_forwards_mlip_args_to_dedicated_cli(self) -> None:
        with patch("oasis.mlip.cli.main") as mock_mlip_main:
            main(["mlip", "run-one", "--line", "task", "--config", "mlip.toml"])

        mock_mlip_main.assert_called_once_with(
            ["run-one", "--line", "task", "--config", "mlip.toml"]
        )


class MlipCliTests(unittest.TestCase):
    def test_submit_delegates_to_submit_jobs(self) -> None:
        with patch("oasis.mlip.cli.submit_jobs") as mock_submit_jobs:
            mlip_main(
                ["submit", "--config", "mlip.toml", "--run-tag", "tag", "dataset.json"]
            )

        mock_submit_jobs.assert_called_once_with(
            config_path="mlip.toml",
            run_tag="tag",
            datasets=["dataset.json"],
        )

    def test_run_one_delegates_to_runner(self) -> None:
        with patch("oasis.mlip.cli.run_one_task") as mock_run_one_task:
            mlip_main(["run-one", "--line", "task-line", "--config", "mlip.toml"])

        mock_run_one_task.assert_called_once_with(
            line="task-line",
            config_path="mlip.toml",
        )

    def test_make_tasks_delegates_to_task_writer(self) -> None:
        with patch("oasis.mlip.cli.make_tasks") as mock_make_tasks:
            mlip_main(
                [
                    "make-tasks",
                    "--config",
                    "mlip.toml",
                    "--run-tag",
                    "tag",
                    "--out",
                    "tasks.txt",
                    "dataset.json",
                ]
            )

        mock_make_tasks.assert_called_once_with(
            config_path="mlip.toml",
            run_tag="tag",
            out_path="tasks.txt",
            datasets=["dataset.json"],
        )
