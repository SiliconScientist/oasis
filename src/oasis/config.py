from pathlib import Path

from oasis.app_config import Config
from oasis.config_base import deep_merge_dicts, load_toml_file
from oasis.experiment_config import (
    AnalysisConfig,
    CandidateRankingConfig,
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
    PlotCurveWindowConfig,
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

DEFAULT_CONFIG_PATHS = (Path("experiment.toml"),)


def _normalize_experiment_layout(raw_cfg: dict) -> dict:
    raw_cfg = dict(raw_cfg)
    top_level_models_cfg = raw_cfg.get("models")
    top_level_tuning_cfg = raw_cfg.get("tuning")
    experiment_cfg = raw_cfg.get("experiment")
    if (
        not isinstance(experiment_cfg, dict)
        and not isinstance(top_level_models_cfg, dict)
        and not isinstance(top_level_tuning_cfg, dict)
    ):
        return raw_cfg
    if not isinstance(experiment_cfg, dict):
        experiment_cfg = {}
        raw_cfg["experiment"] = experiment_cfg

    learning_curve_cfg = experiment_cfg.get("learning_curve")
    if not isinstance(learning_curve_cfg, dict):
        learning_curve_cfg = {}
        experiment_cfg["learning_curve"] = learning_curve_cfg

    shared_experiment_cfg = experiment_cfg.pop("defaults", None)
    if isinstance(shared_experiment_cfg, dict):
        for section_name in ("learning_curve", "screening"):
            section_cfg = experiment_cfg.get(section_name)
            if not isinstance(section_cfg, dict):
                continue
            experiment_cfg[section_name] = deep_merge_dicts(
                shared_experiment_cfg,
                section_cfg,
            )
        learning_curve_cfg = experiment_cfg.get("learning_curve")
        if not isinstance(learning_curve_cfg, dict):
            learning_curve_cfg = {}
            experiment_cfg["learning_curve"] = learning_curve_cfg

    top_level_models_cfg = raw_cfg.pop("models", None)
    if isinstance(top_level_models_cfg, dict):
        learning_curve_models = learning_curve_cfg.get("models")
        if not isinstance(learning_curve_models, dict):
            learning_curve_models = {}
        learning_curve_cfg["models"] = deep_merge_dicts(
            top_level_models_cfg,
            learning_curve_models,
        )

    if isinstance(experiment_cfg.get("models"), dict):
        learning_curve_models = learning_curve_cfg.get("models")
        if not isinstance(learning_curve_models, dict):
            learning_curve_models = {}
        learning_curve_cfg["models"] = deep_merge_dicts(
            experiment_cfg["models"],
            learning_curve_models,
        )
        experiment_cfg.pop("models", None)

    models_cfg = learning_curve_cfg.get("models")
    if isinstance(models_cfg, dict):
        for alias_key, family_key in (
            ("use_probe_gnn", "probe_gnn"),
            ("use_gnn_direct", "gnn_direct"),
        ):
            if alias_key not in models_cfg:
                continue
            family_cfg = models_cfg.get(family_key)
            if not isinstance(family_cfg, dict):
                family_cfg = {}
                models_cfg[family_key] = family_cfg
            family_cfg.setdefault("enabled", bool(models_cfg[alias_key]))

    shared_optuna_cfg: dict | None = None
    top_level_tuning_cfg = raw_cfg.pop("tuning", None)
    if isinstance(top_level_tuning_cfg, dict) and isinstance(
        top_level_tuning_cfg.get("optuna"), dict
    ):
        shared_optuna_cfg = dict(top_level_tuning_cfg["optuna"])
    tuning_cfg = experiment_cfg.get("tuning")
    if isinstance(tuning_cfg, dict) and isinstance(tuning_cfg.get("optuna"), dict):
        shared_optuna_cfg = (
            deep_merge_dicts(shared_optuna_cfg, tuning_cfg["optuna"])
            if shared_optuna_cfg is not None
            else dict(tuning_cfg["optuna"])
        )
        experiment_cfg.pop("tuning", None)
    if shared_optuna_cfg is None:
        return raw_cfg

    if not isinstance(models_cfg, dict):
        return raw_cfg

    for family_name in ("moe", "probe_gnn", "gnn_direct"):
        family_cfg = models_cfg.get(family_name)
        if not isinstance(family_cfg, dict):
            continue
        tuning_section = family_cfg.get("tuning")
        if not isinstance(tuning_section, dict):
            tuning_section = {}
            family_cfg["tuning"] = tuning_section
        family_optuna_cfg = tuning_section.get("optuna")
        if not isinstance(family_optuna_cfg, dict):
            family_optuna_cfg = {}
        tuning_section["optuna"] = deep_merge_dicts(shared_optuna_cfg, family_optuna_cfg)
    return raw_cfg


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
    return _normalize_experiment_layout(merged)


def get_config(
    config_paths: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
) -> Config:
    return Config(**load_config_data(config_paths))
