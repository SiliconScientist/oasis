from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

GateType = Literal["mlip_baseline", "gnn", "schnet"]
GatingMode = Literal["dense", "top_k"]

from oasis.config_base import (
    DatasetProfileConfig,
    DatasetProfilePathsConfig,
    NamedDatasetConfig,
    deep_merge_dicts,
    derive_dataset_profile_paths,
    load_toml_file,
)
from oasis.tune import OptunaTuningConfig
from pydantic import BaseModel, Field


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
    base_dir: Optional[Path] = None
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
    path: Optional[Path] = None
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
    dataset_profile: Optional[DatasetProfileConfig] = None
    datasets: dict[str, NamedDatasetConfig] = Field(default_factory=dict)
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
        self._apply_dataset_profile()
        self._validate_derived_paths()
        self._inherit_global_seed()

    def _apply_dataset_profile(self) -> None:
        profile = self.dataset_profile
        if profile is None:
            return

        named_profile = self.datasets.get(profile.tag)
        if named_profile is None and self.datasets:
            known_tags = ", ".join(sorted(self.datasets))
            raise ValueError(
                f"dataset_profile.tag {profile.tag!r} was not found in [datasets]. "
                f"Known tags: {known_tags}"
            )
        derived_paths = self._derived_dataset_profile_paths(profile.tag, named_profile)
        profile_paths = derived_paths.model_copy(
            update=profile.paths.model_dump(exclude_none=True)
        )

        if self.mlip.dataset is None and profile_paths.dataset is not None:
            self.mlip.dataset = str(profile_paths.dataset)

        learning_curve = (
            self.experiment.learning_curve
            if self.experiment is not None
            else None
        )
        if learning_curve is not None:
            if (
                learning_curve.results_bundle_path is None
                and profile_paths.results_bundle_path is not None
            ):
                learning_curve.results_bundle_path = profile_paths.results_bundle_path

            if (
                learning_curve.graph_dataset is not None
                and learning_curve.graph_dataset.path is None
                and profile_paths.graph_dataset_path is not None
            ):
                learning_curve.graph_dataset.path = profile_paths.graph_dataset_path

        if (
            self.probe_features is None
            and profile_paths.probe_dataset_path is not None
            and profile_paths.probe_mlip_results_dir is not None
        ):
            self.probe_features = ProbeFeatureConfig(
                dataset_path=profile_paths.probe_dataset_path,
                mlip_results_dir=profile_paths.probe_mlip_results_dir,
            )
        elif self.probe_features is not None:
            if profile_paths.probe_dataset_path is not None:
                self._fill_none(
                    self.probe_features,
                    "dataset_path",
                    profile_paths.probe_dataset_path,
                )
            if profile_paths.probe_mlip_results_dir is not None:
                self._fill_none(
                    self.probe_features,
                    "mlip_results_dir",
                    profile_paths.probe_mlip_results_dir,
                )

        if self.analysis is not None:
            self._fill_none(
                self.analysis,
                "calculating_path",
                profile_paths.calculating_path,
            )
            self._fill_none(
                self.analysis,
                "summary_workbook_path",
                profile_paths.summary_workbook_path,
            )
            self._fill_none(
                self.analysis,
                "comparison_workbook_path",
                profile_paths.comparison_workbook_path,
            )
            self._fill_none(
                self.analysis,
                "comparison_plot_path",
                profile_paths.comparison_plot_path,
            )
            self._fill_none(
                self.analysis,
                "base_dir",
                profile_paths.analysis_base_dir,
            )

    @staticmethod
    def _fill_none(model: BaseModel, field_name: str, value: Any) -> None:
        if value is None or getattr(model, field_name) is not None:
            return
        setattr(model, field_name, value)

    @classmethod
    def _derived_dataset_profile_paths(
        cls,
        tag: str,
        named_profile: NamedDatasetConfig | None = None,
    ) -> DatasetProfilePathsConfig:
        return derive_dataset_profile_paths(tag, named_profile)

    def _validate_derived_paths(self) -> None:
        learning_curve = (
            self.experiment.learning_curve
            if self.experiment is not None
            else None
        )
        if learning_curve is not None and learning_curve.graph_dataset is not None:
            if learning_curve.graph_dataset.path is None:
                raise ValueError(
                    "experiment.learning_curve.graph_dataset.path must be provided "
                    "explicitly or derived from dataset_profile.tag"
                )

        if self.analysis is not None:
            if self.analysis.base_dir is None:
                raise ValueError(
                    "analysis.base_dir must be provided explicitly or derived from "
                    "dataset_profile.tag"
                )
            missing_analysis_fields: list[str] = []
            if self.analysis.summary_workbook_path is None:
                missing_analysis_fields.append("analysis.summary_workbook_path")
            if self.analysis.comparison_workbook_path is None:
                missing_analysis_fields.append("analysis.comparison_workbook_path")
            if self.analysis.comparison_plot_path is None:
                missing_analysis_fields.append("analysis.comparison_plot_path")
            if self.analysis.run_adsorption_analysis and self.analysis.calculating_path is None:
                missing_analysis_fields.append("analysis.calculating_path")
            if missing_analysis_fields:
                missing = ", ".join(missing_analysis_fields)
                raise ValueError(
                    f"{missing} must be provided explicitly or derived from dataset_profile.tag"
                )

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

def load_config_data(
    config_paths: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
) -> dict[str, Any]:
    use_default_discovery = config_paths is None
    raw_paths = DEFAULT_CONFIG_PATHS if use_default_discovery else config_paths
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
