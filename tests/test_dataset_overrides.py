from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from oasis.dataset_overrides import (
    configured_dataset_tags,
    render_dataset_override,
    write_dataset_overrides,
)


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


if __name__ == "__main__":
    unittest.main()
