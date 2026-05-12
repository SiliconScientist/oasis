from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from oasis.learning_curve.families.graph_mean import (
    GraphMeanLearnedTrialTuningSpec,
    _graph_mean_study_factory,
)
from oasis.learning_curve.runners import (
    ConfiguredSweepModelFamily,
    FunctionalSweepRunner,
    PlaceholderLearnedSweepModelFamily,
    SweepFamilySpec,
    SweepModelFamily,
    WeightedLinearSweepRunner,
    WeightedSimplexSweepRunner,
)
from oasis.sweep import SweepModelCapabilities, TrainValTestSweepRunnerInput
from oasis.tune import LearnedTrialTuningSpec, TrialTuningSpec


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
    config_key: str | None = None
    is_enabled: Callable[[Any], bool] | None = None
    default_enabled: bool = True


def learned_family_registration_specs() -> tuple[LearnedFamilyRegistrationSpec, ...]:
    from oasis.learning_curve.registry import _moe_enabled
    from oasis.method import residual_sweep

    return (
        LearnedFamilyRegistrationSpec(
            name="residual",
            config_key="use_residual",
            capabilities=SweepModelCapabilities(),
            family_factory=lambda: ConfiguredSweepModelFamily(
                SweepFamilySpec(
                    result_field="resid_df",
                    runner=FunctionalSweepRunner(base_runner=residual_sweep),
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
            family_factory=lambda: PlaceholderLearnedSweepModelFamily(
                name="moe",
                declared_capabilities=SweepModelCapabilities(requires_validation=True),
            ),
            default_enabled=False,
        ),
    )
