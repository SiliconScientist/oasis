from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from oasis.submission import (
    create_submission_run_dir,
    prepare_submission,
    snapshot_cli_argv,
)


class SubmissionTests(unittest.TestCase):
    def test_create_submission_run_dir_uses_ignored_slurm_output_subdir(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)

            run_dir = create_submission_run_dir(run_tag="oasis", root_dir=root)

        self.assertEqual(run_dir.parent, (root / "slurm_output" / "runs").resolve())
        self.assertTrue(run_dir.name.startswith("oasis."))

    def test_prepare_submission_snapshots_default_experiment_config(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = root / "experiment.toml"
            config_path.write_text("[experiment]\n", encoding="utf-8")
            original_cwd = Path.cwd()
            try:
                import os

                os.chdir(root)
                plan = prepare_submission([])
            finally:
                os.chdir(original_cwd)

            self.assertEqual(len(plan.argv), 1)
            snapshot_path = Path(plan.argv[0])
            self.assertNotEqual(snapshot_path, config_path.resolve())
            self.assertEqual(snapshot_path.parent, plan.run_dir)
            self.assertEqual(snapshot_path.read_text(encoding="utf-8"), "[experiment]\n")
            self.assertIn(str(snapshot_path), plan.argv_path.read_text(encoding="utf-8"))

    def test_prepare_submission_snapshots_rank_candidates_config(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = root / "experiment.toml"
            config_path.write_text("[candidate_ranking]\n", encoding="utf-8")
            original_cwd = Path.cwd()
            try:
                import os

                os.chdir(root)
                plan = prepare_submission(["rank-candidates", "experiment.toml"])
            finally:
                os.chdir(original_cwd)

            self.assertEqual(plan.argv[0], "rank-candidates")
            snapshot_path = Path(plan.argv[1])
            self.assertEqual(snapshot_path.parent, plan.run_dir)
            self.assertEqual(
                snapshot_path.read_text(encoding="utf-8"),
                "[candidate_ranking]\n",
            )

    def test_snapshot_cli_argv_defaults_rank_candidates_to_experiment_toml(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_dir = root / "slurm_output" / "runs" / "oasis.1"
            run_dir.mkdir(parents=True)
            config_path = root / "experiment.toml"
            config_path.write_text("[candidate_ranking]\n", encoding="utf-8")
            original_cwd = Path.cwd()
            try:
                import os

                os.chdir(root)
                argv = snapshot_cli_argv(["rank-candidates"], run_dir=run_dir)
            finally:
                os.chdir(original_cwd)

            self.assertEqual(argv[0], "rank-candidates")
            self.assertEqual(
                Path(argv[1]).read_text(encoding="utf-8"),
                "[candidate_ranking]\n",
            )
