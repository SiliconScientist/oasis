from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from oasis.mlip.timing import (
    MlipGenerationTimingSummary,
    generation_timing_summary_from_result_dict,
    load_generation_timing_summaries,
    load_generation_timing_summary,
    load_probe_generation_timing_summaries,
    probe_generation_timing_summary_from_result_dict,
)


class MlipTimingTests(unittest.TestCase):
    def test_generation_timing_summary_from_result_dict_aggregates_final_timings(
        self,
    ) -> None:
        summary = generation_timing_summary_from_result_dict(
            {
                "calculation_settings": {
                    "chemical_bond_cutoff": 1.25,
                    "n_crit_relax": 200,
                },
                "rxn-1->OH*": {
                    "final": {
                        "time_total_slab": 1.5,
                        "time_total_adslab": 5.0,
                        "steps_total_slab": 3,
                        "steps_total_adslab": 10,
                        "time_per_step": 0.5,
                    }
                },
                "rxn-2->O*": {
                    "final": {
                        "time_total_slab": 2.5,
                        "time_total_adslab": 7.0,
                        "steps_total_slab": 5,
                        "steps_total_adslab": 14,
                        "time_per_step": 0.5,
                    }
                },
            },
            model_name="mace",
        )

        self.assertEqual(
            summary,
            MlipGenerationTimingSummary(
                model_name="mace",
                reaction_count=2,
                generation_time_total_s=16.0,
                generation_time_slab_s=4.0,
                generation_time_adslab_s=12.0,
                generation_steps_total=32,
                generation_steps_slab=8,
                generation_steps_adslab=24,
                time_per_step_s=0.5,
            ),
        )

    def test_load_generation_timing_summary_uses_result_file_model_name(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / "mace_result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "rxn-1": {
                            "final": {
                                "time_total_slab": 1.0,
                                "time_total_adslab": 4.0,
                                "steps_total_slab": 2,
                                "steps_total_adslab": 8,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            summary = load_generation_timing_summary(result_path)

        self.assertEqual(summary.model_name, "mace")
        self.assertEqual(summary.generation_time_total_s, 5.0)
        self.assertEqual(summary.generation_steps_total, 10)
        self.assertEqual(summary.time_per_step_s, 0.5)

    def test_load_generation_timing_summaries_returns_mapping_by_model_name(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            mace_path = base_dir / "mace_result.json"
            orb_path = base_dir / "orb_result.json"
            mace_path.write_text(
                json.dumps(
                    {
                        "rxn-1": {
                            "final": {
                                "time_total_slab": 1.0,
                                "time_total_adslab": 2.0,
                                "steps_total_slab": 2,
                                "steps_total_adslab": 4,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            orb_path.write_text(
                json.dumps(
                    {
                        "rxn-1": {
                            "final": {
                                "time_total_slab": 3.0,
                                "time_total_adslab": 9.0,
                                "steps_total_slab": 3,
                                "steps_total_adslab": 9,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            summaries = load_generation_timing_summaries([mace_path, orb_path])

        self.assertEqual(sorted(summaries), ["mace", "orb"])
        self.assertEqual(summaries["mace"].generation_time_total_s, 3.0)
        self.assertEqual(summaries["orb"].generation_time_total_s, 12.0)

    def test_probe_generation_timing_summary_from_result_dict_aggregates_unique_probe_entries(
        self,
    ) -> None:
        summary = probe_generation_timing_summary_from_result_dict(
            {
                "rxn-1->OH*": {
                    "final": {
                        "time_total_slab": 99.0,
                        "time_total_adslab": 99.0,
                        "steps_total_slab": 99,
                        "steps_total_adslab": 99,
                    }
                },
                "unique_probe_1": {
                    "final": {
                        "time_total_slab": 1.5,
                        "time_total_adslab": 5.0,
                        "steps_total_slab": 3,
                        "steps_total_adslab": 10,
                    }
                },
                "unique_probe_2": {
                    "final": {
                        "time_total_slab": 2.5,
                        "time_total_adslab": 7.0,
                        "steps_total_slab": 5,
                        "steps_total_adslab": 14,
                    }
                },
            },
            model_name="mace",
        )

        self.assertEqual(summary.reaction_count, 2)
        self.assertEqual(summary.generation_time_total_s, 16.0)
        self.assertEqual(summary.generation_steps_total, 32)
        self.assertEqual(summary.time_per_step_s, 0.5)

    def test_load_probe_generation_timing_summaries_returns_mapping_by_model_name(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            mace_path = base_dir / "mace_result.json"
            orb_path = base_dir / "orb_result.json"
            mace_path.write_text(
                json.dumps(
                    {
                        "unique_probe_1": {
                            "final": {
                                "time_total_slab": 1.0,
                                "time_total_adslab": 2.0,
                                "steps_total_slab": 2,
                                "steps_total_adslab": 4,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            orb_path.write_text(
                json.dumps(
                    {
                        "unique_probe_1": {
                            "final": {
                                "time_total_slab": 3.0,
                                "time_total_adslab": 9.0,
                                "steps_total_slab": 3,
                                "steps_total_adslab": 9,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            summaries = load_probe_generation_timing_summaries([mace_path, orb_path])

        self.assertEqual(sorted(summaries), ["mace", "orb"])
        self.assertEqual(summaries["mace"].generation_time_total_s, 3.0)
        self.assertEqual(summaries["orb"].generation_time_total_s, 12.0)
