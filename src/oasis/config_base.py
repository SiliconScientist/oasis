from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional

from oasis.mlip_config import mlip_results_dir, raw_dataset_path
from pydantic import BaseModel, Field

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib


class DatasetProfilePathsConfig(BaseModel):
    dataset: Optional[Path] = None
    probe_dataset_path: Optional[Path] = None
    probe_mlip_results_dir: Optional[Path] = None
    results_bundle_path: Optional[Path] = None
    graph_dataset_path: Optional[Path] = None
    calculating_path: Optional[Path] = None
    summary_workbook_path: Optional[Path] = None
    comparison_workbook_path: Optional[Path] = None
    comparison_plot_path: Optional[Path] = None
    analysis_base_dir: Optional[Path] = None


class NamedDatasetConfig(BaseModel):
    raw_dataset_filename: Optional[str] = None
    processed_basename: Optional[str] = None
    probe_results_dirname: Optional[str] = None
    mlip_run_dirname: Optional[str] = None
    analysis_run_dirname: Optional[str] = None
    summary_run_dirname: Optional[str] = None

    def raw_dataset_filename_or_default(self, tag: str) -> str:
        return self.raw_dataset_filename or f"{tag}.json"

    def processed_basename_or_default(self, tag: str) -> str:
        return self.processed_basename or tag

    def probe_results_dirname_or_default(self, tag: str) -> str:
        return self.probe_results_dirname or f"{tag}_unique_probes"

    def mlip_run_dirname_or_default(self, tag: str) -> str:
        return self.mlip_run_dirname or tag

    def analysis_run_dirname_or_default(self, tag: str) -> str:
        return self.analysis_run_dirname or self.mlip_run_dirname_or_default(tag)

    def summary_run_dirname_or_default(self, tag: str) -> str:
        return self.summary_run_dirname or self.analysis_run_dirname_or_default(tag)


class DatasetProfileConfig(BaseModel):
    tag: str
    paths: DatasetProfilePathsConfig = Field(default_factory=DatasetProfilePathsConfig)

def probe_dataset_path(raw_dataset_filename: str) -> Path:
    source_path = Path(raw_dataset_filename)
    suffix = source_path.suffix or ".json"
    return Path("data/raw_data") / f"{source_path.stem}_with_probe_ids{suffix}"


def processed_graph_dataset_path(processed_basename: str) -> Path:
    return Path("data/processed") / f"{processed_basename}.parquet"


def learning_curve_bundle_path(processed_basename: str) -> Path:
    return Path("data/results/learning_curve") / f"{processed_basename}.json"


def probe_results_dir(probe_results_dirname: str) -> Path:
    return Path("data/mlips") / probe_results_dirname

def analysis_workbook_path(run_dirname: str) -> Path:
    return Path("data/results") / run_dirname / "oasis_Benchmarking_Analysis.xlsx"


def comparison_plot_path(tag: str) -> Path:
    return Path("data/results/plots") / f"{tag}_mae_comparison.png"


def derive_dataset_profile_paths(
    tag: str,
    named_profile: NamedDatasetConfig | None = None,
) -> DatasetProfilePathsConfig:
    named_profile = named_profile or NamedDatasetConfig()
    raw_dataset_filename = named_profile.raw_dataset_filename_or_default(tag)
    processed_basename = named_profile.processed_basename_or_default(tag)
    probe_results_dirname = named_profile.probe_results_dirname_or_default(tag)
    mlip_run_dirname = named_profile.mlip_run_dirname_or_default(tag)
    analysis_run_dirname = named_profile.analysis_run_dirname_or_default(tag)
    summary_run_dirname = named_profile.summary_run_dirname_or_default(tag)

    return DatasetProfilePathsConfig(
        dataset=raw_dataset_path(raw_dataset_filename),
        probe_dataset_path=probe_dataset_path(raw_dataset_filename),
        probe_mlip_results_dir=probe_results_dir(probe_results_dirname),
        results_bundle_path=learning_curve_bundle_path(processed_basename),
        graph_dataset_path=processed_graph_dataset_path(processed_basename),
        calculating_path=mlip_results_dir(analysis_run_dirname),
        summary_workbook_path=analysis_workbook_path(summary_run_dirname),
        comparison_workbook_path=analysis_workbook_path(analysis_run_dirname),
        comparison_plot_path=comparison_plot_path(tag),
        analysis_base_dir=mlip_results_dir(mlip_run_dirname),
    )


def load_toml_file(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def deep_merge_dicts(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            merged[key] = deep_merge_dicts(existing, value)
        else:
            merged[key] = value
    return merged
