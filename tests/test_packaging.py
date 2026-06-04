from __future__ import annotations

from pathlib import Path
import tomllib
import unittest


class PackagingMetadataTests(unittest.TestCase):
    def test_project_metadata_remains_oasis_experiment_repo(self) -> None:
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        with pyproject.open("rb") as handle:
            data = tomllib.load(handle)

        self.assertEqual(data["project"]["name"], "oasis")
        self.assertNotIn("scripts", data["project"])
        self.assertIn("Experiment workflows", data["project"]["description"])
