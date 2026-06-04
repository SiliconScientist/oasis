from pathlib import Path
from typing import Any, Optional

from oasis.experiment_config import (
    AnalysisConfig,
    DatasetProfileConfig,
    DatasetProfilePathsConfig,
    ExperimentConfig,
    NamedDatasetConfig,
    PlotConfig,
    ProbeFeatureConfig,
    derive_dataset_profile_paths,
)
from oasis.mlip_config import (
    IngestConfig,
    MLIPConfig,
    default_catbench_folder,
    fill_mlip_dataset_path,
)
from pydantic import BaseModel, Field


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

    def init_paths(self) -> None:
        self.ingest.catbench_folder = default_catbench_folder(self.ingest.source)
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

        fill_mlip_dataset_path(self.mlip, dataset_path=profile_paths.dataset)

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
            if (
                self.analysis.run_adsorption_analysis
                and self.analysis.calculating_path is None
            ):
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
