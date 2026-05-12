from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from oasis.learning_curve.execution import (
    _normalize_runner_output,
    _select_runner_call,
    residual_sweep,
    sweep_learned_model,
    sweep_learned_model_with_validation,
    sweep_model,
    sweep_model_with_validation,
    sweep_results_frame,
    weighted_linear_sweep,
    weighted_simplex_sweep,
)
from oasis.learning_curve.families.graph_mean import (
    _graph_feature_means,
    _graph_feature_means_from_batches,
    _graph_mean_study_factory,
    _GRAPH_MEAN_BATCH_ADAPTER,
    _FixedScaleStudy,
    _FixedScaleTrial,
    _targets_from_batches,
    GraphMeanConstantModel,
    GraphMeanLearnedTrialTuningSpec,
)
from oasis.learning_curve.learned_specs import (
    LearnedFamilyRegistrationSpec,
    learned_family_registration_specs,
)
from oasis.learning_curve.registry import (
    _configured_trial_tuned_family_for_learned_family_spec,
    _config_flag_enabled,
    _family_factory_for_learned_family_spec,
    _is_enabled_for_learned_family_spec,
    _moe_enabled,
    _sklearn_capabilities_for_spec,
    _sklearn_runner_for_spec,
    _sklearn_selection_metadata_field_for_spec,
    default_sweep_model_families,
    enabled_learning_curve_model_names_from_config,
    learned_family_registration,
    learning_curve_model_registry,
    sklearn_model_families,
    LearningCurveModelRegistration,
)
from oasis.learning_curve.runners import (
    ConfiguredSweepModelFamily,
    FunctionalSweepRunner,
    LearnedModelSweepRunner,
    PlaceholderLearnedSweepModelFamily,
    SupervisedModelSweepRunner,
    SweepFamilySpec,
    SweepModelFamily,
    TrainTestLearnedEstimator,
    TrainValTestLearnedEstimator,
    ValidationAwareLearnedModelSweepRunner,
    ValidationAwareSupervisedModelSweepRunner,
    WeightedLinearSweepRunner,
    WeightedSimplexSweepRunner,
)
from oasis.learning_curve.sklearn_specs import (
    RidgeOptunaTrialTuningSpec,
    SklearnSweepModelSpec,
    sklearn_sweep_model_specs,
    _ridge_optuna_study_factory,
)
from oasis.sweep import SweepModelCapabilities, TrainValTestSweepRunnerInput
from oasis.sweep import SweepDatasetBatchLoaderAdapter, TrainEvalLoaderPolicy
from oasis.tune import (
    GridHyperparameterSpec,
    HyperparameterSpec,
    LearnedOptunaModelSelectionSweepRunner,
    LearnedTrialTuningSpec,
    SelectionRefitPolicy,
    TrialTuningSpec,
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso, Ridge
