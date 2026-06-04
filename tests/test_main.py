from __future__ import annotations

import unittest

from oasis.__main__ import main


class MainTests(unittest.TestCase):
    def test_main_rejects_empty_cli(self) -> None:
        with self.assertRaises(SystemExit) as exc_info:
            main([])

        self.assertEqual(str(exc_info.exception), main.__globals__["_EXPERIMENT_CLI_ERROR"])

    def test_main_rejects_non_mlip_cli(self) -> None:
        with self.assertRaises(SystemExit) as exc_info:
            main(["experiment"])

        self.assertEqual(str(exc_info.exception), main.__globals__["_EXPERIMENT_CLI_ERROR"])
