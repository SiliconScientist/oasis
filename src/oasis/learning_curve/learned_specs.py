from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from oasis.learning_curve.families.graph_mean import (
    GraphMeanLearnedTrialTuningSpec,
    _graph_mean_study_factory,
)
from oasis.learning_curve.families.moe import MlipBaselineGateTuningSpec
from oasis.learning_curve.runners import (
    ConfiguredSweepModelFamily,
    FunctionalSweepRunner,
    SweepFamilySpec,
    SweepModelFamily,
    WeightedLinearSweepRunner,
    WeightedSimplexSweepRunner,
)
from oasis.sweep import SweepModelCapabilities, TrainValTestSweepRunnerInput
from oasis.tune import LearnedTrialTuningSpec, TrialTuningSpec


def _probe_gnn_config_tuning_spec_factory(model_cfg: Any) -> LearnedTrialTuningSpec:
    from oasis.learning_curve.families.probe_gnn import ProbeGnnTuningSpec

    probe_cfg = getattr(model_cfg, "probe_gnn", None)
    training_cfg = getattr(probe_cfg, "training", None)
    hidden_dims_list = getattr(probe_cfg, "hidden_dims", [])
    return ProbeGnnTuningSpec(
        training_cfg=training_cfg,
        hidden_dims=tuple(hidden_dims_list),
    )


def _probe_gnn_config_runner_kwargs(model_cfg: Any) -> dict[str, Any]:
    from oasis.tune import study_factory_from_optuna_cfg

    optuna_cfg = getattr(
        getattr(getattr(model_cfg, "probe_gnn", None), "tuning", None), "optuna", None
    )
    if optuna_cfg is None:
        return {"n_trials": 10}
    return {
        "n_trials": optuna_cfg.n_trials,
        "timeout_s": optuna_cfg.timeout_s,
        "study_factory": study_factory_from_optuna_cfg(optuna_cfg),
    }


def _gnn_direct_config_tuning_spec_factory(model_cfg: Any) -> LearnedTrialTuningSpec:
    from oasis.learning_curve.families.probe_gnn import GnnDirectTuningSpec

    cfg = getattr(model_cfg, "gnn_direct", None)
    training_cfg = getattr(cfg, "training", None)
    hidden_dims_list = getattr(cfg, "hidden_dims", [])
    return GnnDirectTuningSpec(
        training_cfg=training_cfg,
        hidden_dims=tuple(hidden_dims_list),
    )


def _gnn_direct_config_runner_kwargs(model_cfg: Any) -> dict[str, Any]:
    from oasis.tune import study_factory_from_optuna_cfg

    optuna_cfg = getattr(
        getattr(getattr(model_cfg, "gnn_direct", None), "tuning", None), "optuna", None
    )
    if optuna_cfg is None:
        return {"n_trials": 10}
    return {
        "n_trials": optuna_cfg.n_trials,
        "timeout_s": optuna_cfg.timeout_s,
        "study_factory": study_factory_from_optuna_cfg(optuna_cfg),
    }


@dataclass(frozen=True, slots=True)
class LearnedFamilyRegistrationSpec:
    """Registry spec for non-sklearn learning-curve families."""

    name: str
    capabilities: SweepModelCapabilities
    family_factory: Callable[[], SweepModelFamily] | None = None
    result_field: str | None = None
    trial_tuning_spec: TrialTuningSpec | None = None
    learned_trial_tuning_spec: LearnedTrialTuningSpec | None = None
    optuna_n_trials: int | None = None
    optuna_timeout_s: int | None = None
    optuna_study_factory: Callable[[TrainValTestSweepRunnerInput], Any] | None = None
    selection_metadata_field: str | None = None
    uq_summary_field: str | None = None
    config_key: str | None = None
    is_enabled: Callable[[Any], bool] | None = None
    config_tuning_spec_factory: Callable[[Any], LearnedTrialTuningSpec] | None = None
    config_runner_kwargs_factory: Callable[[Any], dict[str, Any]] | None = None
    config_family_factory: Callable[[Any], SweepModelFamily] | None = None
    default_enabled: bool = True


def _latent_config_family_factory(model_cfg: Any) -> SweepModelFamily:
    import importlib.util
    from pathlib import Path

    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    latent_cfg = model_cfg.latent
    config_path = Path(latent_cfg.experiment_config_path).resolve()
    vendor_dir = config_path.parent

    with open(config_path, "rb") as f:
        raw_cfg = tomllib.load(f)
    experiment_path = vendor_dir / raw_cfg["experiment_path"]
    model_fitting_cfg = raw_cfg.get("model_fitting", {})
    cobyla_initial_guess = model_fitting_cfg.get("cobyla_initial_guess", 0.1)
    cobyla_max_iter = model_fitting_cfg.get("cobyla_max_iter", 100)

    spec = importlib.util.spec_from_file_location(
        "latent_vendor_config", vendor_dir / "config.py"
    )
    vendor_config_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vendor_config_mod)
    ns = vars(vendor_config_mod)
    for model in (
        vendor_config_mod.XGBoostSearchOptions,
        vendor_config_mod.XGBoostConfig,
        vendor_config_mod.ExperimentConfig,
    ):
        model.model_rebuild(_types_namespace=ns)

    with open(experiment_path, "rb") as f:
        exp_data = tomllib.load(f)
    exp_cfg = vendor_config_mod.ExperimentConfig(**exp_data)
    from oasis.learning_curve.families.latent import LatentSweepRunner

    runner = LatentSweepRunner(
        exp_cfg=exp_cfg,
        vendor_dir=vendor_dir,
        cobyla_initial_guess=cobyla_initial_guess,
        cobyla_max_iter=cobyla_max_iter,
    )
    return ConfiguredSweepModelFamily(
        SweepFamilySpec(
            result_field="latent_df",
            runner=runner,
            capabilities=SweepModelCapabilities(),
        )
    )


def _moe_config_runner_kwargs(model_cfg: Any) -> dict[str, Any]:
    from oasis.tune import study_factory_from_optuna_cfg

    optuna_cfg = getattr(
        getattr(getattr(model_cfg, "moe", None), "tuning", None), "optuna", None
    )
    if optuna_cfg is None:
        return {"n_trials": 10}
    return {
        "n_trials": optuna_cfg.n_trials,
        "timeout_s": optuna_cfg.timeout_s,
        "study_factory": study_factory_from_optuna_cfg(optuna_cfg),
    }


def _moe_config_tuning_spec_factory(model_cfg: Any) -> LearnedTrialTuningSpec:
    from oasis.learning_curve.families.gating_policy import DenseGatingPolicy, TopKGatingPolicy
    from oasis.learning_curve.families.gnn_gate import GnnGateTuningSpec

    moe_cfg = getattr(model_cfg, "moe", None)
    gate_type = getattr(moe_cfg, "gate_type", "mlip_baseline")
    gating_mode = getattr(moe_cfg, "gating_mode", "dense")
    top_k = getattr(moe_cfg, "top_k", 2)

    policy = TopKGatingPolicy(k=top_k) if gating_mode == "top_k" else DenseGatingPolicy()

    if gate_type == "mlip_baseline":
        return MlipBaselineGateTuningSpec(policy=policy)
    if gate_type == "gnn":
        training_cfg = getattr(moe_cfg, "training", None)
        hidden_dims_list = getattr(moe_cfg, "hidden_dims", [])
        return GnnGateTuningSpec(
            training_cfg=training_cfg,
            hidden_dims=tuple(hidden_dims_list),
            policy=policy,
        )
    if gate_type == "schnet":
        from oasis.learning_curve.families.schnet_gate import SchNetGateTuningSpec

        training_cfg = getattr(moe_cfg, "training", None)
        hidden_dims_list = getattr(moe_cfg, "hidden_dims", [])
        n_rbf = getattr(moe_cfg, "n_rbf", 20)
        r_max = getattr(moe_cfg, "r_max", 6.0)
        return SchNetGateTuningSpec(
            training_cfg=training_cfg,
            hidden_dims=tuple(hidden_dims_list),
            n_rbf=n_rbf,
            r_max=r_max,
            policy=policy,
        )
    raise ValueError(f"Unknown MoE gate type: {gate_type!r}")


def learned_family_registration_specs() -> tuple[LearnedFamilyRegistrationSpec, ...]:
    from oasis.config import MoETrainingConfig
    from oasis.learning_curve.execution import residual_sweep
    from oasis.learning_curve.families.probe_gnn import GnnDirectTuningSpec, ProbeGnnTuningSpec
    from oasis.learning_curve.registry import (
        _gnn_direct_enabled,
        _latent_enabled,
        _moe_enabled,
        _probe_gnn_enabled,
    )

    return (
        LearnedFamilyRegistrationSpec(
            name="residual",
            config_key="use_residual",
            capabilities=SweepModelCapabilities(),
            family_factory=lambda: ConfiguredSweepModelFamily(
                SweepFamilySpec(
                    result_field="resid_df",
                    runner=FunctionalSweepRunner(base_runner=residual_sweep),
                    uq_summary_field="resid_uq_df",
                )
            ),
        ),
        LearnedFamilyRegistrationSpec(
            name="weighted_linear",
            config_key="use_weighted_linear",
            capabilities=SweepModelCapabilities(),
            family_factory=lambda: ConfiguredSweepModelFamily(
                SweepFamilySpec(
                    result_field="weighted_linear_df",
                    runner=WeightedLinearSweepRunner(fit_intercept=True),
                    uq_summary_field="weighted_linear_uq_df",
                )
            ),
        ),
        LearnedFamilyRegistrationSpec(
            name="weighted_simplex",
            config_key="use_weighted_simplex",
            capabilities=SweepModelCapabilities(),
            family_factory=lambda: ConfiguredSweepModelFamily(
                SweepFamilySpec(
                    result_field="weighted_simplex_df",
                    runner=WeightedSimplexSweepRunner(),
                    uq_summary_field="weighted_simplex_uq_df",
                )
            ),
        ),
        LearnedFamilyRegistrationSpec(
            name="graph_mean",
            config_key="use_graph_mean",
            capabilities=SweepModelCapabilities(requires_validation=True),
            result_field="graph_mean_df",
            learned_trial_tuning_spec=GraphMeanLearnedTrialTuningSpec(),
            optuna_n_trials=3,
            optuna_study_factory=_graph_mean_study_factory,
            selection_metadata_field="graph_mean_selection_df",
            default_enabled=False,
        ),
        LearnedFamilyRegistrationSpec(
            name="moe",
            is_enabled=_moe_enabled,
            capabilities=SweepModelCapabilities(requires_validation=True),
            learned_trial_tuning_spec=MlipBaselineGateTuningSpec(),
            config_tuning_spec_factory=_moe_config_tuning_spec_factory,
            config_runner_kwargs_factory=_moe_config_runner_kwargs,
            result_field="moe_df",
            selection_metadata_field="moe_selection_df",
            uq_summary_field="moe_uq_df",
            optuna_n_trials=10,
            default_enabled=False,
        ),
        LearnedFamilyRegistrationSpec(
            name="probe_gnn",
            is_enabled=_probe_gnn_enabled,
            capabilities=SweepModelCapabilities(requires_validation=True),
            learned_trial_tuning_spec=ProbeGnnTuningSpec(training_cfg=MoETrainingConfig()),
            config_tuning_spec_factory=_probe_gnn_config_tuning_spec_factory,
            config_runner_kwargs_factory=_probe_gnn_config_runner_kwargs,
            result_field="probe_gnn_df",
            selection_metadata_field="probe_gnn_selection_df",
            optuna_n_trials=10,
            default_enabled=False,
        ),
        LearnedFamilyRegistrationSpec(
            name="gnn_direct",
            is_enabled=_gnn_direct_enabled,
            capabilities=SweepModelCapabilities(requires_validation=True),
            learned_trial_tuning_spec=GnnDirectTuningSpec(training_cfg=MoETrainingConfig()),
            config_tuning_spec_factory=_gnn_direct_config_tuning_spec_factory,
            config_runner_kwargs_factory=_gnn_direct_config_runner_kwargs,
            result_field="gnn_direct_df",
            selection_metadata_field="gnn_direct_selection_df",
            optuna_n_trials=10,
            default_enabled=False,
        ),
        LearnedFamilyRegistrationSpec(
            name="latent",
            is_enabled=_latent_enabled,
            capabilities=SweepModelCapabilities(),
            config_family_factory=_latent_config_family_factory,
            result_field="latent_df",
            default_enabled=False,
        ),
    )
