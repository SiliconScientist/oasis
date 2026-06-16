from __future__ import annotations

import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pandas as pd

from oasis.exp import (
    build_sweep_split_collection,
    generate_inner_validation_sweep_splits,
    generate_sweep_splits,
    run_learning_curve_experiments,
)
from oasis.sweep import (
    GraphDatasetView,
    GraphRecord,
    LearningCurveResults,
    LoaderAdapterInput,
    SweepDataset,
    SweepDatasetBatchLoaderAdapter,
    SweepFamilyRequirements,
    SweepModelCapabilities,
    SweepRunnerPayload,
    SweepSplit,
    SweepSplitCollection,
    SweepRunPayload,
    TrainEvalLoaderPolicy,
    TrainTestSweepRunnerInput,
    TrainValTestSweepRunnerInput,
)
from oasis.tune import (
    OptunaModelSelectionSweepRunner,
    SupervisedModelSelectionSweepRunner,
    TrialTuningSpec,
    _select_candidate_factory_by_validation,
    sweep_model_with_hyperparameter_selection,
    sweep_model_with_optuna_selection,
)
from tests.support import regression_dataset, regression_train_test_payload, weighted_fixed_payload, weighted_toy_dataset

try:
    import oasis.learning_curve.sklearn_specs as sklearn_specs_module
    from oasis.learning_curve.execution import (
        residual_sweep,
        sweep_learned_model,
        sweep_learned_model_with_validation,
        sweep_model,
        sweep_model_with_validation,
        weighted_linear_sweep,
        weighted_simplex_sweep,
    )
    from oasis.learning_curve.families.graph_mean import GraphMeanLearnedTrialTuningSpec
    from oasis.learning_curve.learned_specs import LearnedFamilyRegistrationSpec
    from oasis.learning_curve.registry import (
        default_sweep_model_families,
        enabled_learning_curve_model_names_from_config,
        learned_family_registration,
        learning_curve_model_registry,
        sklearn_model_families,
    )
    from oasis.learning_curve.runners import (
        ConfiguredSweepModelFamily,
        LearnedModelSweepRunner,
        SupervisedModelSweepRunner,
        SweepFamilySpec,
        ValidationAwareLearnedModelSweepRunner,
        ValidationAwareSupervisedModelSweepRunner,
    )
    from oasis.learning_curve.sklearn_specs import sklearn_sweep_model_specs
    from oasis.tune import LearnedOptunaModelSelectionSweepRunner

    HAS_METHOD = True
except ModuleNotFoundError:
    HAS_METHOD = False

try:
    import sklearn  # noqa: F401

    HAS_SKLEARN = True
except ModuleNotFoundError:
    HAS_SKLEARN = False


@unittest.skipUnless(HAS_METHOD, "requires method dependencies")
class SweepOutputRegressionTests(unittest.TestCase):
    _regression_dataset = staticmethod(regression_dataset)
    _train_test_payload = staticmethod(regression_train_test_payload)

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_sklearn_methods_are_registered_from_specs(self) -> None:
        specs = tuple(
            spec
            for spec in sklearn_sweep_model_specs()
            if spec[0] in {"ridge", "lasso", "elastic"}
        )

        families = sklearn_model_families(specs)

        self.assertTrue(
            all(isinstance(family, ConfiguredSweepModelFamily) for family in families)
        )
        self.assertEqual(
            [family.spec.result_field for family in families],
            ["ridge_df", "lasso_df", "elastic_df"],
        )

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_non_sklearn_methods_are_configured_as_first_class_families(self) -> None:
        families = default_sweep_model_families(
            ["residual", "weighted_linear", "weighted_simplex"]
        )

        self.assertTrue(
            all(isinstance(family, ConfiguredSweepModelFamily) for family in families)
        )
        self.assertEqual(
            [family.spec.result_field for family in families],
            [
                "resid_df",
                "weighted_linear_df",
                "weighted_simplex_df",
            ],
        )

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_registry_drives_enabled_model_assembly_from_config(self) -> None:
        registry = learning_curve_model_registry()
        self.assertEqual(
            [registration.name for registration in registry],
            [
                "ridge",
                "kernel_ridge",
                "lasso",
                "elastic",
                "residual",
                "weighted_linear",
                "weighted_simplex",
                "graph_mean",
                "moe",
                "probe_gnn",
                "gnn_direct",
                "latent",
            ],
        )

        model_cfg = SimpleNamespace(
            experiment=SimpleNamespace(
                learning_curve=SimpleNamespace(
                    models=SimpleNamespace(
                        use_ridge=True,
                        use_kernel_ridge=False,
                        use_lasso=True,
                        use_elastic_net=False,
                        use_residual=True,
                        use_weighted_linear=True,
                        use_weighted_simplex=True,
                        use_graph_mean=True,
                    )
                )
            )
        )

        self.assertEqual(
            enabled_learning_curve_model_names_from_config(
                model_cfg.experiment.learning_curve.models
            ),
            (
                "ridge",
                "lasso",
                "residual",
                "weighted_linear",
                "weighted_simplex",
                "graph_mean",
            ),
        )

        self.assertEqual(
            enabled_learning_curve_model_names_from_config(None),
            (
                "ridge",
                "kernel_ridge",
                "lasso",
                "elastic",
                "residual",
                "weighted_linear",
                "weighted_simplex",
            ),
        )

        built_in_requirements = [registration.family_factory().requirements() for registration in registry]
        self.assertEqual(
            built_in_requirements,
            [
                SweepFamilyRequirements(requires_inner_validation=True),
                SweepFamilyRequirements(requires_inner_validation=True),
                SweepFamilyRequirements(requires_inner_validation=True),
                SweepFamilyRequirements(requires_inner_validation=True),
                SweepFamilyRequirements(),
                SweepFamilyRequirements(),
                SweepFamilyRequirements(),
                SweepFamilyRequirements(requires_inner_validation=True),
                SweepFamilyRequirements(requires_inner_validation=True),
                SweepFamilyRequirements(requires_inner_validation=True),
                SweepFamilyRequirements(requires_inner_validation=True),
                SweepFamilyRequirements(),
            ],
        )
        built_in_capabilities = [registration.family_factory().capabilities() for registration in registry]
        self.assertEqual(
            built_in_capabilities,
            [
                SweepModelCapabilities(requires_validation=True),
                SweepModelCapabilities(requires_validation=True),
                SweepModelCapabilities(requires_validation=True),
                SweepModelCapabilities(requires_validation=True),
                SweepModelCapabilities(),
                SweepModelCapabilities(),
                SweepModelCapabilities(),
                SweepModelCapabilities(requires_validation=True),
                SweepModelCapabilities(requires_validation=True),
                SweepModelCapabilities(requires_validation=True),
                SweepModelCapabilities(requires_validation=True),
                SweepModelCapabilities(),
            ],
        )

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_learned_family_registration_supports_stub_non_sklearn_family(self) -> None:
        class StubLearnedFamily:
            def capabilities(self) -> SweepModelCapabilities:
                return SweepModelCapabilities(requires_validation=True)

            def requirements(self) -> SweepFamilyRequirements:
                return SweepFamilyRequirements(requires_inner_validation=True)

            def run(self, payload) -> LearningCurveResults:
                del payload
                return LearningCurveResults.empty()

        registration = learned_family_registration(
            LearnedFamilyRegistrationSpec(
                name="stub_learned",
                config_key="use_stub_learned",
                capabilities=SweepModelCapabilities(requires_validation=True),
                family_factory=StubLearnedFamily,
            )
        )

        self.assertEqual(registration.name, "stub_learned")
        self.assertTrue(registration.is_enabled(SimpleNamespace(use_stub_learned=True)))
        self.assertFalse(registration.is_enabled(SimpleNamespace(use_stub_learned=False)))

        family = registration.family_factory()
        self.assertEqual(
            family.capabilities(),
            SweepModelCapabilities(requires_validation=True),
        )
        self.assertEqual(
            family.requirements(),
            SweepFamilyRequirements(requires_inner_validation=True),
        )

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_moe_registration_instantiates_cleanly(self) -> None:
        registry = {
            registration.name: registration
            for registration in learning_curve_model_registry()
        }

        moe_family = registry["moe"].family_factory()

        self.assertIsInstance(moe_family, ConfiguredSweepModelFamily)
        self.assertEqual(
            moe_family.capabilities(),
            SweepModelCapabilities(requires_validation=True),
        )
        self.assertEqual(
            moe_family.requirements(),
            SweepFamilyRequirements(requires_inner_validation=True),
        )

    @staticmethod
    def _graph_dataset() -> SweepDataset:
        graph_view = GraphDatasetView.from_records(
            (
                GraphRecord(
                    sample_id="s0",
                    node_features=np.array([[1.0], [1.0]]),
                    edge_index=np.array([[0], [1]], dtype=np.int64),
                ),
                GraphRecord(
                    sample_id="s1",
                    node_features=np.array([[2.0], [2.0]]),
                    edge_index=np.array([[0], [1]], dtype=np.int64),
                ),
                GraphRecord(
                    sample_id="s2",
                    node_features=np.array([[3.0], [3.0]]),
                    edge_index=np.array([[0], [1]], dtype=np.int64),
                ),
                GraphRecord(
                    sample_id="s3",
                    node_features=np.array([[4.0], [4.0]]),
                    edge_index=np.array([[0], [1]], dtype=np.int64),
                ),
                GraphRecord(
                    sample_id="s4",
                    node_features=np.array([[5.0], [5.0]]),
                    edge_index=np.array([[0], [1]], dtype=np.int64),
                ),
                GraphRecord(
                    sample_id="s5",
                    node_features=np.array([[6.0], [6.0]]),
                    edge_index=np.array([[0], [1]], dtype=np.int64),
                ),
            )
        )
        return SweepDataset(
            mlip_features=np.zeros((6, 1), dtype=float),
            targets=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            sample_ids=np.array(["s0", "s1", "s2", "s3", "s4", "s5"]),
            graph_view=graph_view,
        )

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_graph_mean_family_is_registered_as_learned_validation_family(self) -> None:
        graph_mean_registration = next(
            registration
            for registration in learning_curve_model_registry()
            if registration.name == "graph_mean"
        )
        graph_mean_family = graph_mean_registration.family_factory()

        self.assertEqual(
            graph_mean_family.capabilities(),
            SweepModelCapabilities(requires_validation=True),
        )
        self.assertEqual(
            graph_mean_family.requirements(),
            SweepFamilyRequirements(requires_inner_validation=True),
        )
        self.assertIsInstance(graph_mean_family, ConfiguredSweepModelFamily)
        self.assertIsInstance(
            graph_mean_family.spec.runner,
            LearnedOptunaModelSelectionSweepRunner,
        )

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_graph_mean_family_runs_end_to_end_with_standard_result_shape(self) -> None:
        dataset = self._graph_dataset()

        results = run_learning_curve_experiments(
            dataset,
            min_train=2,
            max_train=4,
            n_repeats=1,
            seed=41,
            enabled_model_names=["graph_mean"],
        )

        self.assertIsNotNone(results.graph_mean_df)
        self.assertIsNotNone(results.graph_mean_selection_df)
        self.assertEqual(
            results.graph_mean_df.columns.tolist(),
            ["n_train", "rmse_mean", "rmse_std"],
        )
        self.assertEqual(results.graph_mean_df["n_train"].tolist(), [2, 3, 4])
        self.assertEqual(
            results.graph_mean_selection_df.columns.tolist(),
            ["n_train", "best_validation_score", "offset", "scale", "trial_count"],
        )
        self.assertEqual(
            results.graph_mean_selection_df["n_train"].tolist(),
            results.graph_mean_df["n_train"].tolist(),
        )

    def test_graph_mean_learned_trial_tuning_spec_uses_loader_adapter_seam(self) -> None:
        spec = GraphMeanLearnedTrialTuningSpec()
        seen_split_names: list[str] = []

        class FakeSplit:
            def loaders(self, loader_adapter):
                seen_split_names.extend(
                    [
                        "train",
                        "val",
                        "test",
                    ]
                )
                train_dataset = SweepDataset(
                    mlip_features=np.empty((2, 0), dtype=float),
                    targets=np.array([2.0, 4.0]),
                    sample_ids=np.array(["s0", "s1"]),
                    graph_view=GraphDatasetView.from_records(
                        (
                            GraphRecord(
                                sample_id="s0",
                                node_features=np.array([[1.0], [1.0]]),
                                edge_index=np.array([[0], [1]], dtype=np.int64),
                            ),
                            GraphRecord(
                                sample_id="s1",
                                node_features=np.array([[2.0], [2.0]]),
                                edge_index=np.array([[0], [1]], dtype=np.int64),
                            ),
                        )
                    ),
                )
                val_dataset = SweepDataset(
                    mlip_features=np.empty((1, 0), dtype=float),
                    targets=np.array([3.0]),
                    sample_ids=np.array(["s2"]),
                    graph_view=GraphDatasetView.from_records(
                        (
                            GraphRecord(
                                sample_id="s2",
                                node_features=np.array([[3.0], [3.0]]),
                                edge_index=np.array([[0], [1]], dtype=np.int64),
                            ),
                        )
                    ),
                )
                test_dataset = SweepDataset(
                    mlip_features=np.empty((1, 0), dtype=float),
                    targets=np.array([0.0]),
                    sample_ids=np.array(["s3"]),
                    graph_view=GraphDatasetView.from_records(
                        (
                            GraphRecord(
                                sample_id="s3",
                                node_features=np.array([[4.0], [4.0]]),
                                edge_index=np.array([[0], [1]], dtype=np.int64),
                            ),
                        )
                    ),
                )
                return SimpleNamespace(
                    train=loader_adapter.build_loader(
                        LoaderAdapterInput(
                            dataset=train_dataset,
                            split_name="train",
                            batching=loader_adapter.batching_for_split(
                                split_name="train"
                            ),
                        )
                    ),
                    val=loader_adapter.build_loader(
                        LoaderAdapterInput(
                            dataset=val_dataset,
                            split_name="val",
                            batching=loader_adapter.batching_for_split(split_name="val"),
                        )
                    ),
                    test=loader_adapter.build_loader(
                        LoaderAdapterInput(
                            dataset=test_dataset,
                            split_name="test",
                            batching=loader_adapter.batching_for_split(
                                split_name="test"
                            ),
                        )
                    ),
                )

            def dataset_subsets(self):
                raise AssertionError("dataset_subsets should not be used here")

        objective = spec.build_trial_objective(FakeSplit())
        trial = SimpleNamespace(params={"scale": 1.0})
        model = spec.fit_selected_model(
            FakeSplit(),
            trial,
            refit_policy="train_plus_val",
        )
        preds = spec.predict(
            model,
            SweepDataset(
                mlip_features=np.empty((1, 0), dtype=float),
                targets=np.array([0.0]),
                sample_ids=np.array(["s9"]),
                graph_view=GraphDatasetView.from_records(
                    (
                        GraphRecord(
                            sample_id="s9",
                            node_features=np.array([[5.0], [5.0]]),
                            edge_index=np.array([[0], [1]], dtype=np.int64),
                        ),
                    )
                ),
            ),
        )

        self.assertAlmostEqual(objective(trial), 0.0)
        self.assertEqual(seen_split_names, ["train", "val", "test", "train", "val", "test"])
        self.assertAlmostEqual(model.scale, 1.0)
        self.assertAlmostEqual(model.offset, 1.0)
        np.testing.assert_allclose(preds, np.array([6.0]))

    def test_configured_learned_family_uses_batched_loaders_and_touches_outer_test_once(
        self,
    ) -> None:
        seen_objective_batches: list[tuple[tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...]]] = []
        seen_fit_batches: list[tuple[tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...]]] = []
        seen_test_predictions: list[tuple[str, ...]] = []

        class ConstantPredictor:
            def __init__(self, constant: float) -> None:
                self.constant = constant

        class FakeLearnedTrialSpec:
            _loader_adapter = SweepDatasetBatchLoaderAdapter(
                policy=TrainEvalLoaderPolicy(
                    batch_size=2,
                    eval_batch_size=1,
                    train_shuffle=True,
                    eval_shuffle=False,
                )
            )

            def build_trial_objective(self, split):
                loaders = split.loaders(self._loader_adapter)
                seen_objective_batches.append(
                    (
                        tuple(batch.sample_ids for batch in loaders.train),
                        tuple(batch.sample_ids for batch in loaders.val),
                    )
                )
                y_val = np.concatenate([batch.targets for batch in loaders.val])

                def objective(trial):
                    preds = np.full(len(y_val), trial.params["constant"], dtype=float)
                    return float(np.sqrt(np.mean((y_val - preds) ** 2)))

                return objective

            def fit_selected_model(self, split, best_trial, *, refit_policy):
                loaders = split.loaders(self._loader_adapter)
                seen_fit_batches.append(
                    (
                        tuple(batch.sample_ids for batch in loaders.train),
                        tuple(batch.sample_ids for batch in loaders.val),
                    )
                )
                del split, refit_policy
                return ConstantPredictor(float(best_trial.params["constant"]))

            def predict(self, model, dataset):
                seen_test_predictions.append(tuple(dataset.sample_ids.tolist()))
                return np.full(len(dataset), model.constant, dtype=float)

            def trial_metadata(self, best_trial, model):
                return {
                    "constant": float(best_trial.params["constant"]),
                    "selected": float(model.constant),
                }

        class FakeTrial:
            def __init__(self, constant: float) -> None:
                self.params = {"constant": constant}
                self.value: float | None = None

        class FakeStudy:
            def __init__(self, trials):
                self.trials = tuple(trials)
                self.best_trial = None
                self.sampler = SimpleNamespace(__class__=SimpleNamespace(__name__="FakeSampler"))
                self.pruner = SimpleNamespace(__class__=SimpleNamespace(__name__="FakePruner"))

            def optimize(self, objective, *, n_trials, timeout):
                del timeout
                best_value = np.inf
                for trial in self.trials[:n_trials]:
                    value = objective(trial)
                    trial.value = value
                    if value < best_value:
                        best_value = value
                        self.best_trial = trial

        family = ConfiguredSweepModelFamily(
            spec=SweepFamilySpec(
                result_field="graph_mean_df",
                selection_metadata_field="graph_mean_selection_df",
                runner=LearnedOptunaModelSelectionSweepRunner(
                    FakeLearnedTrialSpec(),
                    n_trials=2,
                    study_factory=lambda split: FakeStudy(
                        (FakeTrial(0.0), FakeTrial(4.0))
                    ),
                ),
                capabilities=SweepModelCapabilities(requires_validation=True),
            )
        )
        payload = SweepRunPayload(
            dataset=SweepDataset(
                mlip_features=np.arange(24, dtype=float).reshape(8, 3),
                targets=np.array([0.0, 0.0, 1.0, 1.0, 4.0, 4.0, 9.0, 9.0]),
                sample_ids=np.array(["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"]),
            ),
            split_collection=SweepSplitCollection(
                splits=(
                    SweepSplit(
                        sweep_size=5,
                        train_idx=np.array([0, 1, 2, 3]),
                        val_idx=np.array([4, 5]),
                        test_idx=np.array([6, 7]),
                    ),
                ),
                planning_requirements=SweepFamilyRequirements(
                    requires_inner_validation=True
                ),
            ),
        )

        results = family.run(payload)

        self.assertEqual(
            seen_objective_batches,
            [((("s2", "s0"), ("s1", "s3")), (("s4",), ("s5",)))],
        )
        self.assertEqual(
            seen_fit_batches,
            [((("s2", "s0"), ("s1", "s3")), (("s4",), ("s5",)))],
        )
        self.assertEqual(seen_test_predictions, [("s6", "s7")])
        self.assertIsNotNone(results.graph_mean_df)
        self.assertIsNotNone(results.graph_mean_selection_df)
        self.assertEqual(results.graph_mean_df["n_train"].tolist(), [5])

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_learned_family_registration_supports_trial_tuned_stub_family(self) -> None:
        class FakeTrialSpec:
            def build_trial_objective(self, split):
                y_val = split.dataset.targets[split.val_idx]

                def objective(trial):
                    preds = np.full(len(y_val), trial.params["constant"], dtype=float)
                    return float(np.sqrt(np.mean((y_val - preds) ** 2)))

                return objective

            def fit_selected_model(self, split, best_trial, *, refit_policy):
                del split, refit_policy

                class ConstantModel:
                    def __init__(self, constant: float) -> None:
                        self.constant = constant

                    def predict(self, X):
                        return np.full(len(X), self.constant, dtype=float)

                return ConstantModel(float(best_trial.params["constant"]))

            def trial_metadata(self, best_trial, model):
                return {
                    "constant": float(best_trial.params["constant"]),
                    "selected": float(model.constant),
                }

        class FakeTrial:
            def __init__(self, constant: float) -> None:
                self.params = {"constant": constant}
                self.value: float | None = None

        class FakeStudy:
            def __init__(self) -> None:
                self.trials = (FakeTrial(0.0), FakeTrial(3.0))
                self.best_trial = None

            def optimize(self, objective, *, n_trials, timeout):
                del timeout
                best_value = np.inf
                for trial in self.trials[:n_trials]:
                    objective_value = objective(trial)
                    trial.value = objective_value
                    if objective_value < best_value:
                        best_value = objective_value
                        self.best_trial = trial

        registration = learned_family_registration(
            LearnedFamilyRegistrationSpec(
                name="stub_trial_learned",
                config_key="use_stub_trial_learned",
                capabilities=SweepModelCapabilities(requires_validation=True),
                result_field="ridge_df",
                trial_tuning_spec=FakeTrialSpec(),
                optuna_n_trials=2,
                optuna_study_factory=lambda split: FakeStudy(),
                selection_metadata_field="ridge_selection_df",
            )
        )
        self.assertIsInstance(FakeTrialSpec(), TrialTuningSpec)

        self.assertEqual(registration.name, "stub_trial_learned")
        self.assertTrue(
            registration.is_enabled(SimpleNamespace(use_stub_trial_learned=True))
        )
        self.assertFalse(
            registration.is_enabled(SimpleNamespace(use_stub_trial_learned=False))
        )

        family = registration.family_factory()
        self.assertIsInstance(family, ConfiguredSweepModelFamily)
        self.assertEqual(
            family.capabilities(),
            SweepModelCapabilities(requires_validation=True),
        )
        self.assertEqual(
            family.requirements(),
            SweepFamilyRequirements(requires_inner_validation=True),
        )
        self.assertIsInstance(family.spec.runner, OptunaModelSelectionSweepRunner)

        dataset = SweepDataset(
            mlip_features=np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]),
            targets=np.array([0.0, 0.0, 1.0, 3.0, 4.0, 5.0]),
        )
        payload = SweepRunPayload(
            dataset=dataset,
            split_collection=SweepSplitCollection(
                splits=(
                    SweepSplit(
                        sweep_size=4,
                        train_idx=np.array([0, 1]),
                        val_idx=np.array([2, 3]),
                        test_idx=np.array([4, 5]),
                    ),
                ),
                planning_requirements=SweepFamilyRequirements(
                    requires_inner_validation=True
                ),
            ),
        )
        result = family.run(payload)

        self.assertIsNotNone(result.ridge_df)
        self.assertIsNotNone(result.ridge_selection_df)
        self.assertEqual(
            result.ridge_df.columns.tolist(),
            ["n_train", "rmse_mean", "rmse_std"],
        )
        self.assertEqual(
            result.ridge_selection_df.columns.tolist(),
            [
                "n_train",
                "best_validation_score",
                "constant",
                "selected",
                "trial_count",
            ],
        )

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_config_family_factory_bypasses_trial_tuning_pathway(self) -> None:
        from oasis.learning_curve.registry import _config_factory_for_learned_family_spec

        sentinel_family = object()

        def stub_factory(model_cfg: Any) -> Any:
            return sentinel_family

        spec = LearnedFamilyRegistrationSpec(
            name="stub_config_factory",
            config_key="use_stub",
            capabilities=SweepModelCapabilities(),
            config_family_factory=stub_factory,
        )
        factory = _config_factory_for_learned_family_spec(spec)
        self.assertIs(factory, stub_factory)
        self.assertIs(factory(SimpleNamespace()), sentinel_family)

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_config_factory_returns_none_when_no_factory_fields_set(self) -> None:
        from oasis.learning_curve.registry import _config_factory_for_learned_family_spec
        from oasis.learning_curve.execution import residual_sweep
        from oasis.learning_curve.runners import FunctionalSweepRunner

        spec = LearnedFamilyRegistrationSpec(
            name="no_config_factory",
            config_key="use_no_config",
            capabilities=SweepModelCapabilities(),
            family_factory=lambda: ConfiguredSweepModelFamily(
                SweepFamilySpec(
                    result_field="resid_df",
                    runner=FunctionalSweepRunner(base_runner=residual_sweep),
                )
            ),
        )
        self.assertIsNone(_config_factory_for_learned_family_spec(spec))

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_default_built_in_families_still_instantiate_and_run(self) -> None:
        X, y = self._regression_dataset()

        results = run_learning_curve_experiments(
            SweepDataset(mlip_features=X, targets=y),
            min_train=2,
            max_train=4,
            n_repeats=1,
            seed=11,
            model_families=default_sweep_model_families(),
        )

        self.assertIsNotNone(results.ridge_df)
        self.assertIsNotNone(results.kernel_ridge_df)
        self.assertIsNotNone(results.lasso_df)
        self.assertIsNotNone(results.elastic_df)
        self.assertIsNotNone(results.resid_df)
        self.assertIsNotNone(results.weighted_linear_df)
        self.assertIsNotNone(results.weighted_simplex_df)

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_all_methods_consume_same_split_counts_and_keep_result_shape(self) -> None:
        X, y = self._regression_dataset()
        payload = SweepRunPayload(
            dataset=SweepDataset(mlip_features=X, targets=y),
            split_collection=SweepSplitCollection(
                splits=tuple(
                    generate_sweep_splits(
                        n_samples=len(X),
                        min_train=2,
                        max_train=4,
                        n_repeats=2,
                        rng=np.random.default_rng(7),
                    )
                )
            ),
        )
        expected_counts = [2, 3, 4]
        expected_columns = ["n_train", "rmse_mean", "rmse_std"]

        class DummyModel:
            def fit(self, X, y):
                X = np.asarray(X)
                self.coef_ = np.ones(X.shape[1], dtype=float)
                return self

            def predict(self, X):
                return np.asarray(X).mean(axis=1)

        results = [
            sweep_model(payload, lambda: DummyModel()),
            residual_sweep(payload),
            weighted_linear_sweep(payload),
            weighted_simplex_sweep(payload),
        ]

        for df in results:
            self.assertEqual(df["n_train"].tolist(), expected_counts)
            self.assertEqual(df.columns.tolist(), expected_columns)

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_registry_pipeline_matches_direct_method_outputs(self) -> None:
        X, y = self._regression_dataset()
        dataset = SweepDataset(mlip_features=X, targets=y)
        payload = self._train_test_payload(seed=13)
        sklearn_specs = {
            name: spec for name, _, spec in sklearn_sweep_model_specs()
        }
        validation_payload = SweepRunPayload(
            dataset=dataset,
            split_collection=build_sweep_split_collection(
                dataset.n_samples,
                min_train=2,
                max_train=4,
                n_repeats=2,
                seed=13,
                requirements=SweepFamilyRequirements(
                    requires_inner_validation=True,
                ),
            ),
        )

        expected = {
            "ridge_df": sweep_model_with_hyperparameter_selection(
                validation_payload,
                sklearn_specs["ridge"].hyperparameter_spec,
            ),
            "kernel_ridge_df": sweep_model_with_hyperparameter_selection(
                validation_payload,
                sklearn_specs["kernel_ridge"].hyperparameter_spec,
            ),
            "lasso_df": sweep_model_with_hyperparameter_selection(
                validation_payload,
                sklearn_specs["lasso"].hyperparameter_spec,
            ),
            "elastic_df": sweep_model_with_hyperparameter_selection(
                validation_payload,
                sklearn_specs["elastic"].hyperparameter_spec,
            ),
            "resid_df": residual_sweep(payload),
            "weighted_linear_df": weighted_linear_sweep(payload),
            "weighted_simplex_df": weighted_simplex_sweep(payload),
        }

        actual = run_learning_curve_experiments(
            dataset,
            min_train=2,
            max_train=4,
            n_repeats=2,
            seed=13,
            enabled_model_names=[
                "ridge",
                "kernel_ridge",
                "lasso",
                "elastic",
                "residual",
                "weighted_linear",
                "weighted_simplex",
            ],
        )

        for field_name, expected_df in expected.items():
            actual_df = getattr(actual, field_name)
            self.assertIsNotNone(actual_df, field_name)
            pd.testing.assert_frame_equal(actual_df, expected_df)

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_ridge_is_registered_as_validation_aware_selection_method(self) -> None:
        ridge_registration = next(
            registration
            for registration in learning_curve_model_registry()
            if registration.name == "ridge"
        )
        ridge_family = ridge_registration.family_factory()

        self.assertEqual(
            ridge_family.capabilities(),
            SweepModelCapabilities(requires_validation=True),
        )
        self.assertEqual(
            ridge_family.requirements(),
            SweepFamilyRequirements(requires_inner_validation=True),
        )
        self.assertIsInstance(
            ridge_family.spec.runner,
            SupervisedModelSelectionSweepRunner,
        )

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_ridge_mixed_sweep_integration_keeps_expected_output_shape(self) -> None:
        X, y = self._regression_dataset()

        results = run_learning_curve_experiments(
            SweepDataset(mlip_features=X, targets=y),
            min_train=2,
            max_train=4,
            n_repeats=1,
            seed=17,
            enabled_model_names=["ridge", "weighted_linear"],
        )

        self.assertIsNotNone(results.ridge_df)
        self.assertIsNotNone(results.weighted_linear_df)
        self.assertEqual(
            results.ridge_df.columns.tolist(),
            ["n_train", "rmse_mean", "rmse_std"],
        )
        self.assertEqual(results.ridge_df["n_train"].tolist(), [2, 3, 4])
        self.assertEqual(
            results.weighted_linear_df.columns.tolist(),
            ["n_train", "rmse_mean", "rmse_std"],
        )

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_ridge_skips_undersized_budgets_predictably(self) -> None:
        X, y = self._regression_dataset()

        results = run_learning_curve_experiments(
            SweepDataset(mlip_features=X, targets=y),
            min_train=1,
            max_train=3,
            n_repeats=1,
            seed=17,
            enabled_model_names=["ridge"],
        )

        self.assertIsNotNone(results.ridge_df)
        self.assertEqual(results.ridge_df["n_train"].tolist(), [2, 3])

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_ridge_respects_combined_family_and_validation_guards(self) -> None:
        X = np.array(
            [
                [1.0, 1.2, 0.8],
                [1.8, 2.1, 1.9],
                [2.7, 3.0, 2.9],
                [3.9, 4.2, 3.8],
                [5.1, 5.0, 4.9],
                [6.2, 6.0, 5.8],
                [7.1, 7.0, 6.9],
                [8.0, 8.2, 7.8],
            ]
        )
        y = np.array([1.1, 2.0, 2.9, 4.0, 5.0, 6.1, 7.0, 8.1])

        results = run_learning_curve_experiments(
            SweepDataset(mlip_features=X, targets=y),
            min_train=2,
            max_train=6,
            n_repeats=1,
            seed=17,
            enabled_model_names=["ridge"],
            min_val_size=2,
            min_tuning_val_size=3,
            min_inner_train_size=2,
            min_test_size=2,
        )

        self.assertIsNotNone(results.ridge_df)
        self.assertEqual(results.ridge_df["n_train"].tolist(), [5, 6])

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_ridge_selection_is_deterministic_for_fixed_seed(self) -> None:
        X, y = self._regression_dataset()
        dataset = SweepDataset(mlip_features=X, targets=y)

        first = run_learning_curve_experiments(
            dataset,
            min_train=2,
            max_train=4,
            n_repeats=2,
            seed=19,
            enabled_model_names=["ridge"],
        )
        second = run_learning_curve_experiments(
            dataset,
            min_train=2,
            max_train=4,
            n_repeats=2,
            seed=19,
            enabled_model_names=["ridge"],
        )

        pd.testing.assert_frame_equal(first.ridge_df, second.ridge_df)

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_selection_metadata_surfaces_without_changing_rmse_frames(self) -> None:
        X, y = self._regression_dataset()

        results = run_learning_curve_experiments(
            SweepDataset(mlip_features=X, targets=y),
            min_train=2,
            max_train=4,
            n_repeats=1,
            seed=37,
            enabled_model_names=["ridge", "kernel_ridge"],
        )

        self.assertEqual(
            results.ridge_df.columns.tolist(),
            ["n_train", "rmse_mean", "rmse_std"],
        )
        self.assertIsNotNone(results.ridge_selection_df)
        self.assertEqual(
            results.ridge_selection_df.columns.tolist(),
            [
                "n_train",
                "alpha",
            ],
        )
        self.assertEqual(
            results.ridge_selection_df["n_train"].tolist(),
            results.ridge_df["n_train"].tolist(),
        )
        self.assertTrue(
            set(results.ridge_selection_df["alpha"]).issubset({0.01, 0.1, 1.0, 10.0})
        )

        self.assertIsNotNone(results.kernel_ridge_selection_df)
        self.assertEqual(
            results.kernel_ridge_selection_df.columns.tolist(),
            ["n_train", "alpha", "gamma", "kernel"],
        )
        self.assertEqual(
            results.kernel_ridge_selection_df["n_train"].tolist(),
            results.kernel_ridge_df["n_train"].tolist(),
        )

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_ridge_train_test_runner_payload_preserves_legacy_behavior(self) -> None:
        payload = self._train_test_payload(seed=19)
        runner_payload = payload.to_runner_payload()
        ridge_spec = {
            name: spec for name, _, spec in sklearn_sweep_model_specs()
        }["ridge"]

        self.assertTrue(
            all(
                isinstance(split, TrainTestSweepRunnerInput)
                for split in runner_payload.splits
            )
        )

        legacy = sweep_model(payload, ridge_spec.model_factory)
        split_aware = sweep_model(runner_payload, ridge_spec.model_factory)

        pd.testing.assert_frame_equal(split_aware, legacy)

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_lasso_train_test_runner_payload_preserves_legacy_behavior(self) -> None:
        payload = self._train_test_payload(seed=19)
        runner_payload = payload.to_runner_payload()
        lasso_spec = {
            name: spec for name, _, spec in sklearn_sweep_model_specs()
        }["lasso"]

        self.assertTrue(
            all(
                isinstance(split, TrainTestSweepRunnerInput)
                for split in runner_payload.splits
            )
        )

        legacy = sweep_model(payload, lasso_spec.model_factory)
        split_aware = sweep_model(runner_payload, lasso_spec.model_factory)

        pd.testing.assert_frame_equal(split_aware, legacy)

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_residual_train_test_runner_payload_preserves_legacy_behavior(self) -> None:
        payload = self._train_test_payload(seed=19)
        runner_payload = payload.to_runner_payload()

        self.assertTrue(
            all(
                isinstance(split, TrainTestSweepRunnerInput)
                for split in runner_payload.splits
            )
        )

        legacy = residual_sweep(payload)
        split_aware = residual_sweep(runner_payload)

        pd.testing.assert_frame_equal(split_aware, legacy)


@unittest.skipUnless(HAS_METHOD, "requires method dependencies")
class WeightedBaselineRegressionTests(unittest.TestCase):
    _toy_dataset = staticmethod(weighted_toy_dataset)
    _fixed_payload = staticmethod(weighted_fixed_payload)

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_weighted_linear_keeps_standard_result_shape(self) -> None:
        result = weighted_linear_sweep(self._fixed_payload())

        self.assertEqual(
            result.columns.tolist(),
            ["n_train", "rmse_mean", "rmse_std"],
        )
        self.assertEqual(result["n_train"].tolist(), [4, 5])
        self.assertEqual(len(result), 2)

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_weighted_linear_is_deterministic_for_fixed_splits(self) -> None:
        payload = self._fixed_payload()

        first = weighted_linear_sweep(payload)
        second = weighted_linear_sweep(payload)

        pd.testing.assert_frame_equal(first, second)

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_weighted_linear_has_near_zero_error_on_noiseless_toy_data(self) -> None:
        result = weighted_linear_sweep(self._fixed_payload())

        np.testing.assert_allclose(result["rmse_mean"].to_numpy(), 0.0, atol=1e-12)
        np.testing.assert_allclose(result["rmse_std"].to_numpy(), 0.0, atol=1e-12)

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_weighted_simplex_keeps_standard_result_shape(self) -> None:
        result = weighted_simplex_sweep(self._fixed_payload())

        self.assertEqual(
            result.columns.tolist(),
            ["n_train", "rmse_mean", "rmse_std"],
        )
        self.assertEqual(result["n_train"].tolist(), [4, 5])
        self.assertEqual(len(result), 2)

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_weighted_simplex_is_deterministic_for_fixed_splits(self) -> None:
        payload = self._fixed_payload()

        first = weighted_simplex_sweep(payload)
        second = weighted_simplex_sweep(payload)

        pd.testing.assert_frame_equal(first, second)

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_weighted_simplex_exactly_recovers_convex_toy_target(
        self,
    ) -> None:
        X, _ = self._toy_dataset()
        y = 0.75 * X[:, 0] + 0.25 * X[:, 1]
        payload = SweepRunPayload(
            dataset=SweepDataset(mlip_features=X, targets=y),
            split_collection=self._fixed_payload().split_collection,
        )

        result = weighted_simplex_sweep(payload)

        np.testing.assert_allclose(result["rmse_mean"].to_numpy(), 0.0, atol=1e-12)
        np.testing.assert_allclose(result["rmse_std"].to_numpy(), 0.0, atol=1e-12)

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_weighted_simplex_rejects_single_mlip_feature(self) -> None:
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        payload = SweepRunPayload(
            dataset=SweepDataset(mlip_features=X, targets=y),
            split_collection=self._fixed_payload().split_collection,
        )

        with self.assertRaisesRegex(
            ValueError,
            "weighted_simplex requires at least 2 MLIP feature columns; got 1.",
        ):
            weighted_simplex_sweep(payload)

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_weighted_simplex_rejects_zero_mlip_features(self) -> None:
        X = np.empty((6, 0), dtype=float)
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        graph_view = GraphDatasetView.from_records(
            tuple(
                GraphRecord(
                    sample_id=i,
                    node_features=np.array([[1.0]]),
                    edge_index=np.empty((2, 0), dtype=np.int64),
                )
                for i in range(6)
            )
        )
        payload = SweepRunPayload(
            dataset=SweepDataset(
                mlip_features=X,
                targets=y,
                sample_ids=np.arange(6),
                graph_view=graph_view,
            ),
            split_collection=self._fixed_payload().split_collection,
        )

        with self.assertRaisesRegex(
            ValueError,
            "weighted_simplex requires at least 2 MLIP feature columns; got 0.",
        ):
            weighted_simplex_sweep(payload)

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_weighted_linear_ignores_unrelated_auxiliary_views(self) -> None:
        X, y = self._toy_dataset()
        split_collection = self._fixed_payload().split_collection
        base_payload = SweepRunPayload(
            dataset=SweepDataset(mlip_features=X, targets=y),
            split_collection=split_collection,
        )
        payload_with_auxiliary_views = SweepRunPayload(
            dataset=SweepDataset(
                mlip_features=X,
                targets=y,
                auxiliary_views={
                    "graphs": np.array(
                        [{"edges": 1}, {"edges": 2}, {"edges": 3}, {"edges": 4}, {"edges": 5}, {"edges": 6}],
                        dtype=object,
                    )
                },
            ),
            split_collection=split_collection,
        )

        baseline_result = weighted_linear_sweep(base_payload)
        result_with_auxiliary_views = weighted_linear_sweep(
            payload_with_auxiliary_views
        )

        pd.testing.assert_frame_equal(
            baseline_result,
            result_with_auxiliary_views,
        )


@unittest.skipUnless(HAS_METHOD, "requires method dependencies")
class BoundaryTests(unittest.TestCase):
    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_sklearn_specs_declare_validation_search_spaces(self) -> None:
        specs = {name: spec for name, _, spec in sklearn_sweep_model_specs()}

        self.assertIsNotNone(specs["ridge"].hyperparameter_spec)
        self.assertIsNotNone(specs["kernel_ridge"].hyperparameter_spec)
        self.assertIsNotNone(specs["lasso"].hyperparameter_spec)
        self.assertIsNotNone(specs["elastic"].hyperparameter_spec)

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_lasso_selection_can_choose_different_hyperparameters_by_split(self) -> None:
        class SpyLasso:
            def __init__(self, alpha: float, max_iter: int = 10000) -> None:
                del max_iter
                self.alpha = alpha
                self.coef_ = np.array([alpha], dtype=float)

            def fit(self, X, y):
                del X, y
                return self

            def predict(self, X):
                signature = tuple(float(v) for v in np.ravel(X))
                if signature == (2.0,):
                    return np.full(len(X), self.alpha, dtype=float)
                return np.full(len(X), 1.0 - self.alpha, dtype=float)

        with patch("oasis.learning_curve.sklearn_specs.Lasso", SpyLasso):
            lasso_spec = {
                name: spec for name, _, spec in sklearn_sweep_model_specs()
            }["lasso"]
            dataset = SweepDataset(
                mlip_features=np.array([[0.0], [1.0], [2.0], [3.0], [4.0]]),
                targets=np.array([0.0, 0.0, 0.001, 0.99, 0.0]),
            )
            first_split = TrainValTestSweepRunnerInput(
                dataset=dataset,
                sweep_size=3,
                train_idx=np.array([0, 1]),
                val_idx=np.array([2]),
                test_idx=np.array([4]),
            )
            second_split = TrainValTestSweepRunnerInput(
                dataset=dataset,
                sweep_size=3,
                train_idx=np.array([0, 1]),
                val_idx=np.array([3]),
                test_idx=np.array([4]),
            )

            first_model = _select_candidate_factory_by_validation(
                first_split,
                lasso_spec.hyperparameter_spec,
            )()
            second_model = _select_candidate_factory_by_validation(
                second_split,
                lasso_spec.hyperparameter_spec,
            )()

        self.assertNotEqual(first_model.alpha, second_model.alpha)

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_elastic_selection_can_choose_different_hyperparameters_by_split(self) -> None:
        class SpyElasticNet:
            def __init__(
                self,
                alpha: float,
                l1_ratio: float,
                max_iter: int = 20000,
            ) -> None:
                del max_iter
                self.alpha = alpha
                self.l1_ratio = l1_ratio
                self.coef_ = np.array([alpha + l1_ratio], dtype=float)

            def fit(self, X, y):
                del X, y
                return self

            def predict(self, X):
                signature = tuple(float(v) for v in np.ravel(X))
                if signature == (2.0,):
                    return np.full(len(X), self.alpha + self.l1_ratio, dtype=float)
                return np.full(
                    len(X),
                    abs((self.alpha + self.l1_ratio) - 1.1),
                    dtype=float,
                )

        with patch("oasis.learning_curve.sklearn_specs.ElasticNet", SpyElasticNet):
            elastic_spec = {
                name: spec for name, _, spec in sklearn_sweep_model_specs()
            }["elastic"]
            dataset = SweepDataset(
                mlip_features=np.array([[0.0], [1.0], [2.0], [3.0], [4.0]]),
                targets=np.array([0.0, 0.0, 0.201, 0.299, 0.0]),
            )
            first_split = TrainValTestSweepRunnerInput(
                dataset=dataset,
                sweep_size=3,
                train_idx=np.array([0, 1]),
                val_idx=np.array([2]),
                test_idx=np.array([4]),
            )
            second_split = TrainValTestSweepRunnerInput(
                dataset=dataset,
                sweep_size=3,
                train_idx=np.array([0, 1]),
                val_idx=np.array([3]),
                test_idx=np.array([4]),
            )

            first_model = _select_candidate_factory_by_validation(
                first_split,
                elastic_spec.hyperparameter_spec,
            )()
            second_model = _select_candidate_factory_by_validation(
                second_split,
                elastic_spec.hyperparameter_spec,
            )()

        self.assertNotEqual(
            (first_model.alpha, first_model.l1_ratio),
            (second_model.alpha, second_model.l1_ratio),
        )

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_lasso_and_elastic_skip_undersized_budgets_predictably(self) -> None:
        X, y = SweepOutputRegressionTests._regression_dataset()

        results = run_learning_curve_experiments(
            SweepDataset(mlip_features=X, targets=y),
            min_train=1,
            max_train=3,
            n_repeats=1,
            seed=23,
            enabled_model_names=["lasso", "elastic"],
        )

        self.assertEqual(results.lasso_df["n_train"].tolist(), [2, 3])
        self.assertEqual(results.elastic_df["n_train"].tolist(), [2, 3])

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_train_test_baselines_remain_unaffected_by_lasso_and_elastic(self) -> None:
        X, y = SweepOutputRegressionTests._regression_dataset()
        dataset = SweepDataset(mlip_features=X, targets=y)

        baseline_only = run_learning_curve_experiments(
            dataset,
            min_train=2,
            max_train=4,
            n_repeats=1,
            seed=29,
            enabled_model_names=["weighted_linear"],
        )
        with_selection_families = run_learning_curve_experiments(
            dataset,
            min_train=2,
            max_train=4,
            n_repeats=1,
            seed=29,
            enabled_model_names=["weighted_linear", "lasso", "elastic"],
        )

        pd.testing.assert_frame_equal(
            baseline_only.weighted_linear_df,
            with_selection_families.weighted_linear_df,
        )

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_kernel_ridge_candidate_order_is_deterministic_and_bounded(self) -> None:
        kernel_ridge_spec = {
            name: spec for name, _, spec in sklearn_sweep_model_specs()
        }["kernel_ridge"]

        candidates = [
            factory() for factory in kernel_ridge_spec.hyperparameter_spec.candidate_factories()
        ]

        self.assertEqual(len(candidates), 6)
        self.assertEqual(
            [(candidate.alpha, candidate.gamma) for candidate in candidates],
            [
                (0.1, 0.1),
                (0.1, 1.0),
                (1.0, 0.1),
                (1.0, 1.0),
                (10.0, 0.1),
                (10.0, 1.0),
            ],
        )

    @unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn")
    def test_kernel_ridge_skips_expected_small_budget_ranges(self) -> None:
        X, y = SweepOutputRegressionTests._regression_dataset()

        results = run_learning_curve_experiments(
            SweepDataset(mlip_features=X, targets=y),
            min_train=1,
            max_train=3,
            n_repeats=1,
            seed=31,
            enabled_model_names=["kernel_ridge"],
        )

        self.assertIsNotNone(results.kernel_ridge_df)
        self.assertEqual(results.kernel_ridge_df["n_train"].tolist(), [2, 3])

    def test_validation_aware_supervised_runner_uses_train_val_test_path(self) -> None:
        class ValidationAwareLinearModel:
            def fit(self, X_train, y_train, X_val, y_val) -> None:
                del X_val, y_val
                self.coef_, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)

            def predict(self, X):
                return X @ self.coef_

        dataset = SweepDataset(
            mlip_features=np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [2.0, 1.0],
                    [1.0, 2.0],
                    [2.0, 2.0],
                ]
            ),
            targets=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        )
        payload = SweepRunnerPayload(
            splits=(
                TrainValTestSweepRunnerInput(
                    dataset=dataset,
                    sweep_size=4,
                    train_idx=np.array([0, 1, 2]),
                    val_idx=np.array([3]),
                    test_idx=np.array([4, 5]),
                ),
            )
        )

        result = sweep_model_with_validation(
            payload,
            ValidationAwareLinearModel,
        )
        runner_result = ValidationAwareSupervisedModelSweepRunner(
            ValidationAwareLinearModel
        ).run_with_validation(payload)

        self.assertEqual(
            result.columns.tolist(),
            ["n_train", "rmse_mean", "rmse_std"],
        )
        self.assertEqual(result["n_train"].tolist(), [4])
        pd.testing.assert_frame_equal(result, runner_result)

    def test_learned_runner_uses_full_train_test_split_input(self) -> None:
        seen_splits: list[TrainTestSweepRunnerInput] = []

        class SplitAwareLinearModel:
            def fit(self, split: TrainTestSweepRunnerInput) -> None:
                seen_splits.append(split)
                X_train = split.dataset.mlip_features[split.train_idx]
                y_train = split.dataset.targets[split.train_idx]
                self.coef_, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)

            def predict(self, X):
                return X @ self.coef_

        dataset = SweepDataset(
            mlip_features=np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [2.0, 1.0],
                    [1.0, 2.0],
                    [2.0, 2.0],
                ]
            ),
            targets=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        )
        payload = SweepRunnerPayload(
            splits=(
                TrainTestSweepRunnerInput(
                    dataset=dataset,
                    sweep_size=3,
                    train_idx=np.array([0, 1, 2]),
                    test_idx=np.array([3, 4, 5]),
                ),
            )
        )

        result = sweep_learned_model(payload, SplitAwareLinearModel)
        runner_result = LearnedModelSweepRunner(SplitAwareLinearModel).run(payload)

        self.assertEqual(len(seen_splits), 2)
        self.assertIs(seen_splits[0], payload.splits[0])
        self.assertEqual(result["n_train"].tolist(), [3])
        pd.testing.assert_frame_equal(result, runner_result)

    def test_validation_aware_learned_runner_uses_full_train_val_test_split_input(
        self,
    ) -> None:
        seen_splits: list[TrainValTestSweepRunnerInput] = []

        class SplitAwareValidationModel:
            def fit(self, split: TrainValTestSweepRunnerInput) -> None:
                seen_splits.append(split)
                fit_idx = np.concatenate([split.train_idx, split.val_idx])
                X_fit = split.dataset.mlip_features[fit_idx]
                y_fit = split.dataset.targets[fit_idx]
                self.coef_, *_ = np.linalg.lstsq(X_fit, y_fit, rcond=None)

            def predict(self, X):
                return X @ self.coef_

        dataset = SweepDataset(
            mlip_features=np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [2.0, 1.0],
                    [1.0, 2.0],
                    [2.0, 2.0],
                ]
            ),
            targets=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        )
        payload = SweepRunnerPayload(
            splits=(
                TrainValTestSweepRunnerInput(
                    dataset=dataset,
                    sweep_size=4,
                    train_idx=np.array([0, 1, 2]),
                    val_idx=np.array([3]),
                    test_idx=np.array([4, 5]),
                ),
            )
        )

        result = sweep_learned_model_with_validation(
            payload,
            SplitAwareValidationModel,
        )
        runner_result = ValidationAwareLearnedModelSweepRunner(
            SplitAwareValidationModel
        ).run_with_validation(payload)

        self.assertEqual(len(seen_splits), 2)
        self.assertIs(seen_splits[0], payload.splits[0])
        self.assertEqual(result["n_train"].tolist(), [4])
        pd.testing.assert_frame_equal(result, runner_result)

    def test_configured_family_capabilities_round_trip_to_requirements(self) -> None:
        if not HAS_SKLEARN:
            self.skipTest("requires scikit-learn")

        family = ConfiguredSweepModelFamily(
            spec=SweepFamilySpec(
                result_field="ridge_df",
                runner=SupervisedModelSweepRunner(lambda: None),
                capabilities=SweepModelCapabilities(
                    min_train_size=6,
                    requires_validation=True,
                ),
            )
        )

        self.assertEqual(
            family.capabilities(),
            SweepModelCapabilities(
                min_train_size=6,
                requires_validation=True,
            ),
        )
        self.assertEqual(
            family.requirements(),
            SweepFamilyRequirements(
                min_train_size=6,
                requires_inner_validation=True,
            ),
        )
