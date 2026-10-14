# Oasis documentation

[Project introduction and quickstart](../README.md)

Start with [Running experiments](getting_started.md) to prepare inputs and run a
study. Configuration snippets in the reference guides are fragments, not complete
standalone experiments; use [experiment.example.toml](../experiment.example.toml)
as the annotated configuration template.

| Guide | Contents |
| --- | --- |
| [Running experiments](getting_started.md) | Input preparation, environment, entrypoints, tests, legacy MLIP commands |
| [GPU setup](gpu_setup.md) | Tested CUDA environment and smoke-run instructions |
| [Learning curves and artifacts](learning_curves.md) | Sweeps, model/tuning settings, incremental results, plots and screening diagnostics |
| [Candidate ranking](candidate_ranking.md) | Predictor selection, target energies, validated-reference JSON |
| [Uncertainty](uncertainty.md) | Spread semantics, metrics, held-out calibration |
| [Anomaly filtering](anomaly_filtering.md) | Baseline/filtered OH-BMA comparisons and cache identity |
| [Graph artifacts](graph_artifacts.md) | Graph alignment, Parquet schema, failure modes |
| [Evaluation protocols](evaluation_protocol.md) | Learned-model interfaces, split safety, feasibility rules |
| [Comparative caching](comparative_learning_cache_model.md) | Primitive and derived caches, invalidation, completeness |
| [Timing protocol](time_accuracy_protocol.md) | Generation/training cost definitions and interpretation |

## Documentation gaps

- The example configuration requires study data and upstream MLIP artifacts.
  A small, self-contained end-to-end tutorial with expected outputs is still needed.
- Input preparation currently relies on the configuration template and the
  [Moira project](https://github.com/SiliconScientist/Moirai). A consolidated
  input-artifact schema and acquisition guide would make onboarding easier.
- The annotated template and source configuration models cover more options than
  these guides; there is no exhaustive generated configuration reference.
