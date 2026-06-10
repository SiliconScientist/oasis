from pathlib import Path

from oasis.app_config import Config
from oasis.config_base import deep_merge_dicts, load_toml_file
from oasis.experiment_config import (
    AnalysisConfig,
    DatasetProfileConfig,
    DatasetProfilePathsConfig,
    ExperimentConfig,
    ScreeningExperimentConfig,
    GnnDirectConfig,
    GraphDatasetInputConfig,
    LatentModelConfig,
    LearningCurveBudgetMode,
    LearningCurveExperimentConfig,
    LearningCurveModelsConfig,
    MoEConfig,
    MoETuningConfig,
    MoETrainingConfig,
    NamedDatasetConfig,
    PlotConfig,
    PlotFiltersConfig,
    ProbeFeatureConfig,
    ProbeGnnConfig,
    derive_dataset_profile_paths,
)
from oasis.mlip_config import (
    IngestConfig,
    MLIPConfig,
    MLIPModelsConfig,
    RootstockConfig,
    RootstockModelConfig,
    StoichConfig,
    default_catbench_folder,
    fill_mlip_dataset_path,
)

DEFAULT_CONFIG_PATHS = (Path("mlip.toml"), Path("experiment.toml"))


def load_config_data(
    config_paths: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
) -> dict:
    use_default_discovery = config_paths is None
    raw_paths = DEFAULT_CONFIG_PATHS if use_default_discovery else config_paths
    if isinstance(raw_paths, (str, Path)):
        path_list = [Path(raw_paths)]
    else:
        path_list = [Path(path) for path in raw_paths]

    merged: dict = {}
    missing: list[Path] = []
    for path in path_list:
        if not path.exists():
            missing.append(path)
            continue
        merged = deep_merge_dicts(merged, load_toml_file(path))

    if not merged:
        searched = ", ".join(str(path) for path in path_list)
        raise FileNotFoundError(f"No config files found. Looked for: {searched}")
    if missing and not use_default_discovery:
        missing_str = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Explicit config file(s) not found: {missing_str}")
    return merged


def get_config(
    config_paths: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
) -> Config:
    return Config(**load_config_data(config_paths))
