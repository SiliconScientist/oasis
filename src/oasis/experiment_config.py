from pathlib import Path
from typing import List, Literal, Optional

from oasis.mlip_config import mlip_results_dir, raw_dataset_path
from oasis.tune import OptunaTuningConfig
from pydantic import BaseModel, Field

GateType = Literal["mlip_baseline", "gnn", "schnet"]
GatingMode = Literal["dense", "top_k"]
LearningCurveBudgetMode = Literal["full_remainder_test", "screening_fraction"]


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
    epochs: int | None = None
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
    use_gnn_direct: bool = False
    use_probe_gnn: bool = False
    moe: MoEConfig = Field(default_factory=MoEConfig)
    probe_gnn: ProbeGnnConfig = Field(default_factory=ProbeGnnConfig)
    gnn_direct: GnnDirectConfig = Field(default_factory=GnnDirectConfig)
    use_latent: bool = False
    latent: Optional[LatentModelConfig] = None


class PlotFiltersConfig(BaseModel):
    adsorbate: Optional[str] = None
    anomaly_label: Optional[str] = None
    reaction_contains: Optional[List[str]] = None


class PlotCurveWindowConfig(BaseModel):
    full_dataset_window: bool = False
    all: bool = False
    min_x: Optional[int] = None
    max_x: Optional[int] = None
    include_x: Optional[List[int]] = None
    include_fractions: Optional[List[float]] = None


class GraphDatasetInputConfig(BaseModel):
    path: Optional[Path] = None
    join_key: str = "reaction"


class MlipSelectionConfig(BaseModel):
    exclude_anomalous: bool = False
    label_allowlist: List[str] = Field(default_factory=lambda: ["normal"])
    strict_inference_anomaly: bool = False


class LearningCurveExperimentConfig(BaseModel):
    min_train: int | None = None
    max_train: int | None = None
    step: int = 1
    sweep_sizes: List[int] = Field(default_factory=list)
    sweep_fractions: List[float] = Field(default_factory=list)
    n_repeats: int
    budget_mode: LearningCurveBudgetMode = "full_remainder_test"
    screen_fraction: float | None = None
    min_screen_size: int = 1
    validation_fraction: float = 0.2
    min_val_size: int = 1
    min_tuning_val_size: int = 1
    min_inner_train_size: int = 1
    min_test_size: int = 1
    results_bundle_path: Optional[Path] = None
    reuse_results: bool = False
    force_refresh_methods: list[str] = Field(default_factory=list)
    force_refresh_train_sizes: dict[str, list[int]] = Field(default_factory=dict)
    mlip_selection: MlipSelectionConfig = Field(default_factory=MlipSelectionConfig)
    graph_dataset: Optional[GraphDatasetInputConfig] = None
    models: Optional[LearningCurveModelsConfig] = None


class ScreeningExperimentConfig(BaseModel):
    budget_mode: LearningCurveBudgetMode = "screening_fraction"
    screen_fraction: float | None = None
    min_screen_size: int = 1
    validation_fraction: float = 0.2
    min_val_size: int = 1
    min_tuning_val_size: int = 1
    min_inner_train_size: int = 1
    results_bundle_path: Optional[Path] = None
    reuse_results: bool = False
    force_refresh_methods: list[str] = Field(default_factory=list)
    force_refresh_train_sizes: dict[str, list[int]] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    learning_curve: Optional[LearningCurveExperimentConfig] = None
    screening: Optional[ScreeningExperimentConfig] = None

    def validate_screening_dependency(self) -> None:
        if self.screening is not None and self.learning_curve is None:
            raise ValueError(
                "experiment.screening requires experiment.learning_curve to define "
                "the training grid and model families."
            )

    def derived_screening_learning_curve(
        self,
    ) -> Optional[LearningCurveExperimentConfig]:
        if self.screening is None:
            return None
        self.validate_screening_dependency()
        assert self.learning_curve is not None

        screening = self.screening
        return self.learning_curve.model_copy(
            deep=True,
            update={
                "budget_mode": screening.budget_mode,
                "screen_fraction": screening.screen_fraction,
                "min_screen_size": screening.min_screen_size,
                "validation_fraction": screening.validation_fraction,
                "min_val_size": screening.min_val_size,
                "min_tuning_val_size": screening.min_tuning_val_size,
                "min_inner_train_size": screening.min_inner_train_size,
                "results_bundle_path": screening.results_bundle_path,
                "reuse_results": screening.reuse_results,
                "force_refresh_methods": list(screening.force_refresh_methods),
                "force_refresh_train_sizes": {
                    method_name: list(sweep_sizes)
                    for method_name, sweep_sizes in screening.force_refresh_train_sizes.items()
                },
            },
        )


class PlotConfig(BaseModel):
    output_dir: Path
    filters: PlotFiltersConfig = Field(default_factory=PlotFiltersConfig)
    curve_window: PlotCurveWindowConfig = Field(default_factory=PlotCurveWindowConfig)


class ProbeFeatureConfig(BaseModel):
    dataset_path: Path
    mlip_results_dir: Path


def probe_dataset_path(raw_dataset_filename: str) -> Path:
    source_path = Path(raw_dataset_filename)
    suffix = source_path.suffix or ".json"
    return Path("data/raw_data") / f"{source_path.stem}_with_probe_ids{suffix}"


def processed_graph_dataset_path(processed_basename: str) -> Path:
    return Path("data/processed") / f"{processed_basename}.parquet"


def learning_curve_bundle_path(processed_basename: str) -> Path:
    return Path("data/results/learning_curve") / f"{processed_basename}.json"


def screening_bundle_path(processed_basename: str) -> Path:
    return Path("data/results/screening") / f"{processed_basename}.json"


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
