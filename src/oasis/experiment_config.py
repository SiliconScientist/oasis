from pathlib import Path
from typing import Any, List, Literal, Optional

from oasis.mlip_config import mlip_results_dir, raw_dataset_path
from oasis.tune import OptunaTuningConfig
from pydantic import BaseModel, Field, model_validator

GateType = Literal["mlip_baseline", "gnn", "schnet"]
GatingMode = Literal["dense", "top_k"]
LearningCurveBudgetMode = Literal["full_remainder_test", "screening_fraction"]
CalibrationMethod = Literal["scalar_scale"]


class DatasetProfilePathsConfig(BaseModel):
    dataset: Optional[Path] = None
    # External probe artifacts consumed by Oasis when probe-aware methods run.
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
    probe_dataset_filename: Optional[str] = None
    probe_mlip_results_dir: Optional[str] = None
    # Directory name for external probe MLIP results, typically produced by Moira.
    probe_results_dirname: Optional[str] = None
    mlip_run_dirname: Optional[str] = None
    analysis_run_dirname: Optional[str] = None
    summary_run_dirname: Optional[str] = None

    def raw_dataset_filename_or_default(self, tag: str) -> str:
        return self.raw_dataset_filename or f"{tag}.json"

    def processed_basename_or_default(self, tag: str) -> str:
        return self.processed_basename or tag

    def probe_dataset_path_or_default(self, tag: str) -> Path:
        if self.probe_dataset_filename is not None:
            return Path(self.probe_dataset_filename)
        raw_dataset_filename = self.raw_dataset_filename_or_default(tag)
        return probe_dataset_path(raw_dataset_filename)

    def probe_mlip_results_dir_or_default(self, tag: str) -> Path:
        if self.probe_mlip_results_dir is not None:
            return Path(self.probe_mlip_results_dir)
        probe_results_dirname = self.probe_results_dirname_or_default(tag)
        return probe_results_dir(probe_results_dirname)

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
    timing_path: Path | None = None


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


class PlotCurveWindowConfig(BaseModel):
    full_dataset_window: bool = False
    all: bool = False
    min_x: Optional[int] = None
    max_x: Optional[int] = None
    include_x: Optional[List[int]] = None
    include_fractions: Optional[List[float]] = None


class PlotFixedSplitConfig(BaseModel):
    train_fraction: float = Field(default=0.8, gt=0.0, lt=1.0)


class GraphDatasetInputConfig(BaseModel):
    path: Optional[Path] = None
    join_key: str = "reaction"


class MlipSelectionConfig(BaseModel):
    enabled: List[str] = Field(default_factory=list)
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
    calibration_enabled: bool = True
    calibration_method: CalibrationMethod = "scalar_scale"
    calibration_fraction: float = 0.2
    min_cal_size: int = 1
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
    calibration_enabled: bool = True
    calibration_method: CalibrationMethod = "scalar_scale"
    calibration_fraction: float = 0.2
    min_cal_size: int = 1
    min_inner_train_size: int = 1
    results_bundle_path: Optional[Path] = None
    reuse_results: bool = False
    force_refresh_methods: list[str] = Field(default_factory=list)
    force_refresh_train_sizes: dict[str, list[int]] = Field(default_factory=dict)


class ExperimentDefaultsConfig(BaseModel):
    validation_fraction: float | None = None
    min_val_size: int | None = None
    min_tuning_val_size: int | None = None
    calibration_enabled: bool | None = None
    calibration_method: CalibrationMethod | None = None
    calibration_fraction: float | None = None
    min_cal_size: int | None = None
    min_inner_train_size: int | None = None
    reuse_results: bool | None = None
    force_refresh_methods: list[str] | None = None


class ExperimentConfig(BaseModel):
    defaults: Optional[ExperimentDefaultsConfig] = None
    learning_curve: Optional[LearningCurveExperimentConfig] = None
    screening: Optional[ScreeningExperimentConfig] = None

    @model_validator(mode="before")
    @classmethod
    def apply_shared_defaults(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        shared_defaults = data.get("defaults")
        if not isinstance(shared_defaults, dict):
            return data

        merged = dict(data)
        for section_name in ("learning_curve", "screening"):
            section_cfg = merged.get(section_name)
            if not isinstance(section_cfg, dict):
                continue
            merged[section_name] = {
                **shared_defaults,
                **section_cfg,
            }
        return merged

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
    curve_window: PlotCurveWindowConfig = Field(default_factory=PlotCurveWindowConfig)
    fixed_split: PlotFixedSplitConfig = Field(default_factory=PlotFixedSplitConfig)


class ProbeFeatureConfig(BaseModel):
    """External probe artifacts required by probe-aware Oasis methods."""

    dataset_path: Path
    mlip_results_dir: Path


class CandidateRankingConfig(BaseModel):
    predictors: List[str] = Field(
        default_factory=lambda: ["residual", "weighted_simplex", "ridge"]
    )
    selected_predictor: Optional[str] = None
    target_binding_energy: float
    top_k: int = 10
    results_dir: Optional[Path] = None
    validated_references_path: Optional[Path] = None
    diagnostics_output_dir: Optional[Path] = None
    score_function: str = "target_uncertainty_cost"
    target_distance_weight: float = 1.0
    uncertainty_weight: float = 1.0
    target_uncertainty_alpha: float = 0.75
    supporting_signal_weights: dict[str, float] = Field(default_factory=dict)
    exclude_anomalous: bool = False
    label_allowlist: List[str] = Field(default_factory=lambda: ["normal"])
    strict_inference_anomaly: bool = False
    min_valid_mlips: int = 2
    predictor_configs: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_predictor_surface(self) -> "CandidateRankingConfig":
        if not self.predictors:
            raise ValueError("candidate_ranking.predictors must not be empty.")
        if (
            self.selected_predictor is not None
            and self.selected_predictor not in self.predictors
        ):
            raise ValueError(
                "candidate_ranking.selected_predictor must also appear in "
                "candidate_ranking.predictors."
            )
        if (
            self.validated_references_path is not None
            and self.selected_predictor is None
        ):
            raise ValueError(
                "candidate_ranking.selected_predictor is required when "
                "validated_references_path is provided."
            )
        return self

    def resolved_predictor_config(self, predictor_name: str | None = None) -> dict[str, Any]:
        predictor_specific = (
            {}
            if predictor_name is None
            else dict(self.predictor_configs.get(predictor_name, {}))
        )
        return {
            "score_function": self.score_function,
            "target_distance_weight": self.target_distance_weight,
            "uncertainty_weight": self.uncertainty_weight,
            "target_uncertainty_alpha": self.target_uncertainty_alpha,
            "supporting_signal_weights": dict(self.supporting_signal_weights),
            "exclude_anomalous": self.exclude_anomalous,
            "label_allowlist": list(self.label_allowlist),
            "strict_inference_anomaly": self.strict_inference_anomaly,
            "min_valid_mlips": self.min_valid_mlips,
            **predictor_specific,
        }


def probe_dataset_path(raw_dataset_filename: str) -> Path:
    """Default location for an external probe-annotated dataset artifact."""
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
    """Default location for an external probe MLIP results directory."""
    return Path("data/mlips") / probe_results_dirname


def analysis_workbook_path(run_dirname: str) -> Path:
    return Path("data/results") / run_dirname / "oasis_Benchmarking_Analysis.xlsx"


def comparison_plot_path(tag: str) -> Path:
    return Path("data/results/plots") / f"{tag}_mae_comparison.png"


def derive_dataset_profile_paths(
    tag: str,
    named_profile: NamedDatasetConfig | None = None,
) -> DatasetProfilePathsConfig:
    """Derive Oasis defaults, including dataset-specific probe artifact paths."""
    named_profile = named_profile or NamedDatasetConfig()
    raw_dataset_filename = named_profile.raw_dataset_filename_or_default(tag)
    processed_basename = named_profile.processed_basename_or_default(tag)
    mlip_run_dirname = named_profile.mlip_run_dirname_or_default(tag)
    analysis_run_dirname = named_profile.analysis_run_dirname_or_default(tag)
    summary_run_dirname = named_profile.summary_run_dirname_or_default(tag)

    return DatasetProfilePathsConfig(
        dataset=raw_dataset_path(raw_dataset_filename),
        probe_dataset_path=named_profile.probe_dataset_path_or_default(tag),
        probe_mlip_results_dir=named_profile.probe_mlip_results_dir_or_default(tag),
        results_bundle_path=learning_curve_bundle_path(processed_basename),
        graph_dataset_path=processed_graph_dataset_path(processed_basename),
        calculating_path=mlip_results_dir(analysis_run_dirname),
        summary_workbook_path=analysis_workbook_path(summary_run_dirname),
        comparison_workbook_path=analysis_workbook_path(analysis_run_dirname),
        comparison_plot_path=comparison_plot_path(tag),
        analysis_base_dir=mlip_results_dir(mlip_run_dirname),
    )
