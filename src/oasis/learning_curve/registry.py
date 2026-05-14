from __future__ import annotations

from collections.abc import Collection
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from oasis.learning_curve.learned_specs import learned_family_registration_specs
from oasis.learning_curve.runners import (
    ConfiguredSweepModelFamily,
    FunctionalSweepRunner,
    PlaceholderLearnedSweepModelFamily,
    SupervisedModelSweepRunner,
    SweepExperimentRunner,
    SweepFamilySpec,
    SweepModelFamily,
    ValidationAwareSweepExperimentRunner,
    WeightedLinearSweepRunner,
    WeightedSimplexSweepRunner,
)
from oasis.learning_curve.sklearn_specs import sklearn_sweep_model_specs
from oasis.sweep import SweepModelCapabilities
from oasis.tune import (
    LearnedOptunaModelSelectionSweepRunner,
    OptunaModelSelectionSweepRunner,
    SupervisedModelSelectionSweepRunner,
)


@dataclass(frozen=True, slots=True)
class LearningCurveModelRegistration:
    name: str
    is_enabled: Callable[[Any], bool]
    family_factory: Callable[[], SweepModelFamily]
    default_enabled: bool = True
    config_factory: Callable[[Any], SweepModelFamily] | None = None


def _config_flag_enabled(config_key: str) -> Callable[[Any], bool]:
    return lambda config_section: bool(getattr(config_section, config_key, False))


def _moe_enabled(config_section: Any) -> bool:
    return bool(getattr(getattr(config_section, "moe", None), "enabled", False))


def _latent_enabled(config_section: Any) -> bool:
    return bool(getattr(config_section, "use_latent", False))


def _is_enabled_for_learned_family_spec(
    spec: Any,
) -> Callable[[Any], bool]:
    if spec.is_enabled is not None:
        return spec.is_enabled
    if spec.config_key is not None:
        return _config_flag_enabled(spec.config_key)
    raise ValueError(
        f"learned family registration '{spec.name}' must define config_key or is_enabled"
    )


def _family_factory_for_learned_family_spec(
    spec: Any,
) -> Callable[[], SweepModelFamily]:
    def build_family() -> SweepModelFamily:
        if spec.family_factory is not None:
            family = spec.family_factory()
        elif spec.config_family_factory is not None:
            return PlaceholderLearnedSweepModelFamily(
                name=spec.name,
                declared_capabilities=spec.capabilities,
            )
        else:
            family = _configured_trial_tuned_family_for_learned_family_spec(spec)
        if family.capabilities() != spec.capabilities:
            raise ValueError(
                f"learned family registration '{spec.name}' declared capabilities "
                f"{spec.capabilities!r} but factory produced {family.capabilities()!r}"
            )
        return family

    return build_family


def _configured_trial_tuned_family_for_learned_family_spec(
    spec: Any,
) -> SweepModelFamily:
    if (
        spec.trial_tuning_spec is not None
        and spec.learned_trial_tuning_spec is not None
    ):
        raise ValueError(
            f"learned family registration '{spec.name}' cannot define both "
            "trial_tuning_spec and learned_trial_tuning_spec"
        )
    if spec.trial_tuning_spec is None and spec.learned_trial_tuning_spec is None:
        raise ValueError(
            f"learned family registration '{spec.name}' must define family_factory "
            "or a trial tuning spec"
        )
    if spec.result_field is None:
        raise ValueError(
            f"learned family registration '{spec.name}' must define result_field "
            "for trial-tuned registration"
        )
    if spec.optuna_n_trials is None:
        raise ValueError(
            f"learned family registration '{spec.name}' must define optuna_n_trials "
            "for trial-tuned registration"
        )
    runner_kwargs = {
        "n_trials": spec.optuna_n_trials,
        "timeout_s": spec.optuna_timeout_s,
    }
    if spec.optuna_study_factory is not None:
        runner_kwargs["study_factory"] = spec.optuna_study_factory
    runner = (
        OptunaModelSelectionSweepRunner(
            spec.trial_tuning_spec,
            **runner_kwargs,
        )
        if spec.trial_tuning_spec is not None
        else LearnedOptunaModelSelectionSweepRunner(
            spec.learned_trial_tuning_spec,
            **runner_kwargs,
        )
    )
    return ConfiguredSweepModelFamily(
        SweepFamilySpec(
            result_field=spec.result_field,
            runner=runner,
            selection_metadata_field=spec.selection_metadata_field,
            capabilities=spec.capabilities,
        )
    )


def _config_factory_for_learned_family_spec(
    spec: Any,
) -> Callable[[Any], SweepModelFamily] | None:
    if spec.config_family_factory is not None:
        return spec.config_family_factory
    if spec.config_tuning_spec_factory is None:
        return None

    def build_config_family(model_cfg: Any) -> SweepModelFamily:
        tuning_spec = spec.config_tuning_spec_factory(model_cfg)
        if spec.config_runner_kwargs_factory is not None:
            runner_kwargs: dict[str, Any] = spec.config_runner_kwargs_factory(model_cfg)
        else:
            runner_kwargs = {
                "n_trials": spec.optuna_n_trials,
                "timeout_s": spec.optuna_timeout_s,
            }
            if spec.optuna_study_factory is not None:
                runner_kwargs["study_factory"] = spec.optuna_study_factory
        runner = LearnedOptunaModelSelectionSweepRunner(tuning_spec, **runner_kwargs)
        family = ConfiguredSweepModelFamily(
            SweepFamilySpec(
                result_field=spec.result_field,
                runner=runner,
                selection_metadata_field=spec.selection_metadata_field,
                capabilities=spec.capabilities,
            )
        )
        if family.capabilities() != spec.capabilities:
            raise ValueError(
                f"learned family registration '{spec.name}' declared capabilities "
                f"{spec.capabilities!r} but config factory produced {family.capabilities()!r}"
            )
        return family

    return build_config_family


def learned_family_registration(
    spec: Any,
) -> LearningCurveModelRegistration:
    return LearningCurveModelRegistration(
        name=spec.name,
        is_enabled=_is_enabled_for_learned_family_spec(spec),
        family_factory=_family_factory_for_learned_family_spec(spec),
        config_factory=_config_factory_for_learned_family_spec(spec),
        default_enabled=spec.default_enabled,
    )


def default_sweep_model_families(
    enabled_model_names: Collection[str] | None = None,
    *,
    config: Any | None = None,
) -> tuple[SweepModelFamily, ...]:
    registrations = learning_curve_model_registry()
    enabled_names = (
        set(
            registration.name
            for registration in registrations
            if registration.default_enabled
        )
        if enabled_model_names is None
        else set(enabled_model_names)
    )
    return tuple(
        registration.config_factory(config)
        if registration.config_factory is not None and config is not None
        else registration.family_factory()
        for registration in registrations
        if registration.name in enabled_names
    )


def enabled_learning_curve_model_names_from_config(
    model_cfg: Any | None,
) -> tuple[str, ...]:
    registrations = learning_curve_model_registry()
    if model_cfg is None:
        return tuple(
            registration.name
            for registration in registrations
            if registration.default_enabled
        )
    return tuple(
        registration.name
        for registration in registrations
        if registration.is_enabled(model_cfg)
    )


def _sklearn_runner_for_spec(
    spec: Any,
) -> SweepExperimentRunner | ValidationAwareSweepExperimentRunner:
    if spec.trial_tuning_spec is not None:
        if spec.optuna_n_trials is None:
            raise ValueError("optuna-backed sklearn spec must declare optuna_n_trials")
        runner_kwargs = {
            "n_trials": spec.optuna_n_trials,
            "timeout_s": spec.optuna_timeout_s,
            "refit_policy": spec.selection_refit_policy,
        }
        if spec.optuna_study_factory is not None:
            runner_kwargs["study_factory"] = spec.optuna_study_factory
        return OptunaModelSelectionSweepRunner(
            spec.trial_tuning_spec,
            **runner_kwargs,
        )
    if spec.hyperparameter_spec is not None:
        return SupervisedModelSelectionSweepRunner(
            spec.hyperparameter_spec,
            refit_policy=spec.selection_refit_policy,
        )
    return SupervisedModelSweepRunner(spec.model_factory)


def _sklearn_capabilities_for_spec(
    spec: Any,
) -> SweepModelCapabilities:
    if spec.hyperparameter_spec is None and spec.trial_tuning_spec is None:
        return SweepModelCapabilities()
    return SweepModelCapabilities(requires_validation=True)


def _sklearn_selection_metadata_field_for_spec(
    spec: Any,
) -> str | None:
    return spec.selection_metadata_field


def learning_curve_model_registry() -> tuple[LearningCurveModelRegistration, ...]:
    sklearn_registrations = tuple(
        LearningCurveModelRegistration(
            name=name,
            is_enabled=_config_flag_enabled(config_key),
            family_factory=lambda spec=spec: ConfiguredSweepModelFamily(
                SweepFamilySpec(
                    result_field=spec.result_field,
                    selection_metadata_field=_sklearn_selection_metadata_field_for_spec(
                        spec
                    ),
                    runner=_sklearn_runner_for_spec(spec),
                    capabilities=_sklearn_capabilities_for_spec(spec),
                )
            ),
        )
        for name, config_key, spec in sklearn_sweep_model_specs()
    )
    return sklearn_registrations + tuple(
        learned_family_registration(spec)
        for spec in learned_family_registration_specs()
    )


def sklearn_model_families(
    specs: tuple[tuple[str, str, Any], ...],
) -> tuple[SweepModelFamily, ...]:
    return tuple(
        ConfiguredSweepModelFamily(
            SweepFamilySpec(
                result_field=spec.result_field,
                selection_metadata_field=_sklearn_selection_metadata_field_for_spec(
                    spec
                ),
                runner=_sklearn_runner_for_spec(spec),
                capabilities=_sklearn_capabilities_for_spec(spec),
            )
        )
        for _, _, spec in specs
    )
