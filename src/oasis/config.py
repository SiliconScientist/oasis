from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

GateType = Literal["mlip_baseline", "gnn", "schnet"]
GatingMode = Literal["dense", "top_k"]

from oasis.tune import OptunaTuningConfig
from pydantic import BaseModel, Field

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib


class StoichConfig(BaseModel):
    elements: list[str]
    basis_species: list[str]
    basis_composition: Dict[str, Dict[str, int]]


class IngestConfig(BaseModel):
    source: Path
    dataset_name: str
    catbench_folder: Optional[Path] = None
    stoich: StoichConfig


class MLIPModelsConfig(BaseModel):
    enabled: List[str]


class RootstockModelConfig(BaseModel):
    model: str
    mlip_name: str
    checkpoint: Optional[str] = None
    output_model: Optional[str] = None
    model_version: Optional[str] = None
    metadata: Dict[str, str] = Field(default_factory=dict)


class RootstockConfig(BaseModel):
    root: Path
    python: Optional[Path] = None
    models: Dict[str, RootstockModelConfig]


class MLIPConfig(BaseModel):
    dev_n: int
    dev_run: bool
    dataset: Optional[str] = None
    optimizer: str = "LBFGS"
    models: MLIPModelsConfig
    rootstock: RootstockConfig


class AnalysisConfig(BaseModel):
    calculating_path: Optional[Path] = None
    summary_workbook_path: Optional[Path] = None
    comparison_workbook_path: Optional[Path] = None
    comparison_plot_path: Optional[Path] = None
    run_adsorption_analysis: bool = False
    base_dir: Path
    out_dir: Path
    prefixes: List[str]


class MoETrainingConfig(BaseModel):
    batch_size: int = 32
    eval_batch_size: Optional[int] = None
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.0
    num_workers: int = 0
    device: Optional[str] = None
    seed: Optional[int] = None


class MoETuningConfig(BaseModel):
    optuna: Optional[OptunaTuningConfig] = None


class MoEConfig(BaseModel):
    enabled: bool = False
    gate_type: GateType = "mlip_baseline"
    gating_mode: GatingMode = "dense"
    top_k: int = 2
    hidden_dims: List[int] = Field(default_factory=list)
    n_rbf: int = 20
    r_max: float = 6.0
    training: MoETrainingConfig = Field(default_factory=MoETrainingConfig)
    tuning: MoETuningConfig = Field(default_factory=MoETuningConfig)


class LatentModelConfig(BaseModel):
    experiment_config_path: Path
    csv_path: Path
    cobyla_initial_guess: float = 0.1
    cobyla_max_iter: int = 100


class ProbeGnnConfig(BaseModel):
    enabled: bool = False
    hidden_dims: List[int] = Field(default_factory=list)
    training: MoETrainingConfig = Field(default_factory=MoETrainingConfig)
    tuning: MoETuningConfig = Field(default_factory=MoETuningConfig)


class GnnDirectConfig(BaseModel):
    enabled: bool = False
    hidden_dims: List[int] = Field(default_factory=list)
    training: MoETrainingConfig = Field(default_factory=MoETrainingConfig)
    tuning: MoETuningConfig = Field(default_factory=MoETuningConfig)


class LearningCurveModelsConfig(BaseModel):
    use_ridge: bool
    use_kernel_ridge: bool
    use_lasso: bool
    use_elastic_net: bool
    use_residual: bool
    use_weighted_linear: bool = False
    use_weighted_simplex: bool = False
    use_graph_mean: bool = False
    moe: MoEConfig = Field(default_factory=MoEConfig)
    probe_gnn: ProbeGnnConfig = Field(default_factory=ProbeGnnConfig)
    gnn_direct: GnnDirectConfig = Field(default_factory=GnnDirectConfig)
    use_latent: bool = False
    latent: Optional[LatentModelConfig] = None


class PlotFiltersConfig(BaseModel):
    adsorbate: Optional[str] = None
    anomaly_label: Optional[str] = None
    reaction_contains: Optional[List[str]] = None


class GraphDatasetInputConfig(BaseModel):
    path: Path
    join_key: str = "reaction"


class LearningCurveExperimentConfig(BaseModel):
    """Learning-curve sweep configuration.

    `min_train` and `max_train` define the outer training-budget sweep. Methods
    that require validation may spend part of that budget on inner validation,
    while the test split remains reserved for outer evaluation only.
    """

    min_train: int
    max_train: int
    step: int = 1
    n_repeats: int
    validation_fraction: float = 0.2
    min_val_size: int = 1
    min_tuning_val_size: int = 1
    min_inner_train_size: int = 1
    min_test_size: int = 1
    results_bundle_path: Optional[Path] = None
    reuse_results: bool = False
    force_refresh_methods: list[str] = Field(default_factory=list)
    force_refresh_train_sizes: dict[str, list[int]] = Field(default_factory=dict)
    graph_dataset: Optional[GraphDatasetInputConfig] = None
    models: Optional[LearningCurveModelsConfig] = None


class ExperimentConfig(BaseModel):
    learning_curve: Optional[LearningCurveExperimentConfig] = None


class PlotConfig(BaseModel):
    """Plot configuration."""

    output_dir: Path
    filters: PlotFiltersConfig = Field(default_factory=PlotFiltersConfig)


class ProbeFeatureConfig(BaseModel):
    dataset_path: Path
    mlip_results_dir: Path


class Config(BaseModel):
    seed: Optional[int] = None
    dev_run: Optional[bool] = None
    train: Optional[bool] = None
    evaluate: Optional[bool] = None
    ingest: IngestConfig
    mlip: MLIPConfig
    analysis: Optional[AnalysisConfig] = None
    probe_features: Optional[ProbeFeatureConfig] = None
    experiment: Optional[ExperimentConfig] = None
    plot: Optional[PlotConfig] = None

    def model_post_init(self, __context: Any) -> None:
        self.init_paths()

    def init_paths(self):
        catbench_folder = (
            self.ingest.source.parent / f"{self.ingest.source.name}_catbench"
        )
        self.ingest.catbench_folder = catbench_folder
        self._inherit_global_seed()

    def _inherit_global_seed(self) -> None:
        if self.seed is None or self.experiment is None:
            return
        learning_curve = self.experiment.learning_curve
        if learning_curve is None or learning_curve.models is None:
            return

        models_cfg = learning_curve.models
        for family_cfg_name in ("moe", "probe_gnn", "gnn_direct"):
            family_cfg = getattr(models_cfg, family_cfg_name, None)
            if family_cfg is None:
                continue

            training_cfg = getattr(family_cfg, "training", None)
            if training_cfg is not None and training_cfg.seed is None:
                training_cfg.seed = self.seed

            tuning_cfg = getattr(family_cfg, "tuning", None)
            optuna_cfg = getattr(tuning_cfg, "optuna", None)
            if optuna_cfg is not None and optuna_cfg.seed is None:
                optuna_cfg.seed = self.seed


DEFAULT_CONFIG_PATHS = (Path("mlip.toml"), Path("experiment.toml"))


def _load_toml_file(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _deep_merge_dicts(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge_dicts(existing, value)
        else:
            merged[key] = value
    return merged


def load_config_data(
    config_paths: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
) -> dict[str, Any]:
    raw_paths = config_paths if config_paths is not None else DEFAULT_CONFIG_PATHS
    if isinstance(raw_paths, (str, Path)):
        path_list = [Path(raw_paths)]
    else:
        path_list = [Path(path) for path in raw_paths]

    merged: dict[str, Any] = {}
    missing: list[Path] = []
    for path in path_list:
        if not path.exists():
            missing.append(path)
            continue
        merged = _deep_merge_dicts(merged, _load_toml_file(path))

    if not merged:
        searched = ", ".join(str(path) for path in path_list)
        raise FileNotFoundError(f"No config files found. Looked for: {searched}")
    if missing and config_paths is not None:
        missing_str = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Config file(s) not found: {missing_str}")
    return merged


def get_config(
    config_paths: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
) -> Config:
    return Config(**load_config_data(config_paths))
