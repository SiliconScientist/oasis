from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from oasis.dataset_overrides import (
    configured_dataset_profile_tag,
    configured_dataset_tags,
    render_dataset_override,
    write_dataset_overrides,
)
import scripts.submit_configured_datasets as submit_configured_datasets


class DatasetOverrideTests(unittest.TestCase):
    def test_configured_dataset_tags_reads_declared_dataset_keys(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "experiment.toml"
            config_path.write_text(
                "\n".join(
                    [
                        '[dataset_profile]',
                        'tag = "mamun_oh"',
                        '',
                        '[datasets.mamun_oh]',
                        'raw_dataset_filename = "MamunHighT2019_oh_adsorption.json"',
                        '',
                        '[datasets.khlohc]',
                        'raw_dataset_filename = "KHLOHC_origin_tolstar_adsorption.json"',
                        '',
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                configured_dataset_tags(config_path),
                ["mamun_oh", "khlohc"],
            )

    def test_render_dataset_override_emits_minimal_dataset_profile(self) -> None:
        self.assertEqual(
            render_dataset_override("bio_mass"),
            '[dataset_profile]\ntag = "bio_mass"\n',
        )

    def test_configured_dataset_profile_tag_reads_selected_tag(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "experiment.toml"
            config_path.write_text(
                "\n".join(
                    [
                        '[dataset_profile]',
                        'tag = "mamun_oh"',
                        '',
                        '[datasets.mamun_oh]',
                        'raw_dataset_filename = "MamunHighT2019_oh_adsorption.json"',
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                configured_dataset_profile_tag(config_path),
                "mamun_oh",
            )

    def test_write_dataset_overrides_creates_one_file_per_tag(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            written = write_dataset_overrides(
                ["mamun_oh", "rodrigo"],
                output_dir=tmp_dir,
            )

            self.assertEqual(
                [path.name for path in written],
                ["mamun_oh.override.toml", "rodrigo.override.toml"],
            )
            self.assertEqual(
                written[1].read_text(encoding="utf-8"),
                '[dataset_profile]\ntag = "rodrigo"\n',
            )

    def test_submit_helper_passes_only_base_and_override_configs(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = root / "demo.toml"
            config_path.write_text(
                "\n".join(
                    [
                        '[datasets.mamun_oh]',
                        'raw_dataset_filename = "MamunHighT2019_oh_adsorption.json"',
                        '',
                    ]
                ),
                encoding="utf-8",
            )
            out_dir = root / "overrides"

            with patch.object(submit_configured_datasets.subprocess, "run") as run_mock:
                with patch(
                    "sys.argv",
                    [
                        "submit_configured_datasets.py",
                        str(config_path),
                        "--out-dir",
                        str(out_dir),
                        "--submit",
                    ],
                ):
                    submit_configured_datasets.main()

            run_mock.assert_called_once_with(
                [
                    "./submit.sh",
                    str(config_path),
                    str(out_dir / "mamun_oh.override.toml"),
                ],
                check=True,
            )


if __name__ == "__main__":
    unittest.main()
