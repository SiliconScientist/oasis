# oasis
Experiment workflows for refining ML interatomic potentials with sparse DFT data.

## Environment

Install the project and test dependencies with:

```bash
pip install -e .[dev]
```

If you use `uv`, the equivalent is:

```bash
uv pip install -e ".[dev]"
```

Run the test suite with:

```bash
PYTHONPATH=src python -m unittest
```

## Entrypoints

Oasis owns experiment workflows. MLIP ingestion/prediction now lives in
[Moirai](https://github.com/SiliconScientist/Moirai).

Experiment execution is config-driven:

```python
from oasis.experiment_runner import run_experiment_from_config

run_experiment_from_config(["mlip.toml", "experiment.toml"])
```

If you already have a parsed config object, call `run_experiment(cfg)` instead.

For MLIP ingestion/prediction, use Moirai. The old in-repo MLIP module path
remains documented here only as a temporary compatibility surface:

```bash
python -m oasis.mlip submit --config mlip.toml
python -m oasis.mlip make-tasks --config mlip.toml --run-tag dev --out slurm_output/tasks.txt
python -m oasis.mlip run-one --config mlip.toml --line "mace example data/raw_data/example.json data/results/mlips/dev/example/mace.json"
```

`python -m oasis` is not an experiment runner. It only forwards `mlip ...` and
otherwise exits with guidance to use the MLIP module entrypoint.

## Related Repo

MLIP ingestion/prediction repo:

- `Moirai`: https://github.com/SiliconScientist/Moirai

Oasis remains experiment-first.

Targeted graph/config test commands:

```bash
PYTHONPATH=src python -m unittest tests.test_exp tests.test_graphs tests.test_config
PYTHONPATH=src python -m unittest tests.test_graphs tests.test_exp
PYTHONPATH=src python -m unittest tests.test_config
```

## Learning-Curve Result Artifacts

Learning-curve runs can persist one bundle file per dataset so you do not need
to retrain every model just to regenerate plots.

In the tag-first config style, the bundle path is derived from
`dataset_profile.tag` plus `[datasets.<tag>]`, so `experiment.toml` only needs
the naming inputs that differ from the default tag-based convention:

```toml
[dataset_profile]
tag = "example_oh"

[datasets.example_oh]
# Omit this when the processed basename is exactly the tag.
# processed_basename = "example_oh"

[experiment.learning_curve]
reuse_results = true

[plot.curve_window]
full_dataset_window = true
```

`full_dataset_window = true` only disables the `min_x` / `max_x` crop. It does
not disable `include_x` or `include_fractions`.

If you want to keep the full sweep on disk but only plot selected fractional
budgets, use:

```toml
[plot.curve_window]
include_fractions = [0.1, 0.2, 0.5, 0.9]
```

These fractions are resolved against the filtered dataset size, then passed to
the plotting layer as explicit `n_train` points.

For sparse paper sweeps, you can replace the contiguous `min_train` /
`max_train` / `step` grid with explicit points:

```toml
[experiment.learning_curve]
sweep_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
n_repeats = 10
reuse_results = true
```

Oasis resolves `sweep_fractions` against the active dataset size, deduplicates
any collisions after rounding down to counts, and reuses cached rows by the
resolved `n_train` / `n_budget` values.

Model toggles now live in a flatter top-level config layout:

```toml
[models]
use_ridge = true
use_residual = true
use_gnn_direct = false
use_probe_gnn = false

[models.moe]
enabled = true
gate_type = "gnn"

[tuning.optuna]
n_trials = 10
sampler = "tpe"
```

`[models]` is normalized into the learning-curve model config, and
`[tuning.optuna]` provides shared Optuna defaults for `moe`,
`gnn_direct`, and `probe_gnn`. Per-model tuning blocks still override the
shared defaults when needed.

`gnn_direct` and `probe_gnn` can now be toggled directly in `[models]` via
`use_gnn_direct` and `use_probe_gnn`. Their dedicated subsections are only
needed for advanced overrides such as `hidden_dims` or training settings.

For Optuna-backed learned models (`moe`, `gnn_direct`, `probe_gnn`, and the
SchNet gate), you can omit `training.epochs` to let Optuna tune epochs from a
built-in candidate set. Keep `training.epochs` only when you want to pin it.

Workflow:

- Build results incrementally. Run one method or one `n_train` range now, then
  add more methods or more sweep sizes to the same bundle later.
- Fill gaps later. With `reuse_results = true`, Oasis inspects cached rows per
  method and runs only the missing `n_train` points needed for the current
  request.
- Reuse cached points. Plot regeneration and later sweep runs reuse compatible
  cached rows instead of retraining them.
- Mixed coverage is allowed. Different methods may have different feasible
  `n_train` ranges in the same bundle, and plotting uses whatever rows exist
  for each method.
- Refresh conflicts explicitly. Use `force_refresh_methods` to rerun a whole
  method, or `force_refresh_train_sizes` to rerun only selected `n_train`
  points for a method.

Example iterative workflow:

```toml
[dataset_profile]
tag = "example_oh"

[datasets.example_oh]
# Omit this when the processed basename is exactly the tag.
# processed_basename = "example_oh"

[experiment.learning_curve]
reuse_results = true

# First pass: run one method or one range.
# enabled methods example:
# use_ridge = true
# use_weighted_simplex = false

# Later, enable another method or widen the sweep bounds.
# Oasis keeps the old rows and fills only the gaps.

force_refresh_methods = ["moe", "probe_gnn"]
force_refresh_train_sizes = { ridge = [20, 30] }
```

## Uncertainty Metrics

Learning-curve runs can also persist uncertainty-summary companions for methods
that emit a predictive spread. Oasis currently uses three aggregate metrics:

- `miscalibration_area`: area between the empirical calibration curve and the
  ideal curve. Lower is better.
- `sharpness`: average predicted spread magnitude. Lower means tighter
  predictions, though sharpness alone is not sufficient without calibration.
- `dispersion`: coefficient of variation of the predicted spread. This measures
  how much the spread varies across samples.

Method semantics matter:

- `residual`, `weighted_simplex`, and other explicitly uncertainty-aware
  methods should be interpreted as uncertainty summaries.
- `ridge` contributes a useful spread proxy for ranking or relative comparison,
  but it is not a calibrated probabilistic uncertainty and should not be
  interpreted as one.

## OH-BMA Comparison Workflow

For the anomaly-aware MLIP-selection branch, keep two experiment presets and
run them against the same `OH-BMA` result set:

- baseline: anomaly-aware MLIP filtering off
- filtered: anomaly-aware MLIP filtering on

Minimal config pattern:

```toml
[dataset_profile]
tag = "mamun_oh"

[datasets.mamun_oh]
raw_dataset_filename = "MamunHighT2019_oh_adsorption.json"
mlip_run_dirname = "OH-BMA"

[experiment.learning_curve]
reuse_results = true

[experiment.learning_curve.mlip_selection]
exclude_anomalous = false
label_allowlist = ["normal"]
strict_inference_anomaly = false
```

Filtered variant:

```toml
[dataset_profile]
tag = "mamun_oh"

[datasets.mamun_oh]
raw_dataset_filename = "MamunHighT2019_oh_adsorption.json"
mlip_run_dirname = "OH-BMA"

[experiment.learning_curve]
reuse_results = true

[experiment.learning_curve.mlip_selection]
exclude_anomalous = true
label_allowlist = ["normal"]
strict_inference_anomaly = false
```

Notes:

- Oasis now auto-suffixes outputs and caches with `anomalyaware_on` or
  `anomalyaware_off`, so the two runs do not overwrite each other.
- The learning-curve bundle and aligned graph artifact are also separated by
  that suffix.
- If you want a stricter comparison, set
  `strict_inference_anomaly = true` in the filtered preset to drop MLIPs based
  on per-MLIP inference-detail flags rather than labels alone.

Bundle identity is dataset/filter oriented, not exact-sweep oriented. Reuse
requires the same dataset/filter identity and seed, but the bundle may contain:

- different enabled-method subsets across runs
- different `min_train` / `max_train` / `step` ranges across runs
- different `n_repeats` across runs

Those sweep settings are still preserved as per-point provenance in the saved
artifact so you can audit how each row was produced.

## Graph Artifact Contract

Configured graph-backed learning-curve runs use
`experiment.learning_curve.graph_dataset` in `experiment.toml`.

Example:

```toml
[dataset_profile]
tag = "example_oh"

[datasets.example_oh]
# Omit this when the processed basename is exactly the tag.
# processed_basename = "example_oh"

[experiment.learning_curve.graph_dataset]
join_key = "reaction"
```

Behavior:

- This section is optional. If you omit it, Oasis rebuilds graphs in memory for
  the current run and does not write an aligned Parquet cache.
- `graph_dataset.path` is derived from the active dataset profile.
- If `graph_dataset.path` exists, Oasis reuses that saved aligned artifact.
- If it does not exist, Oasis loads atoms from `mlip.dataset`, converts them to
  graphs, aligns them to the filtered MLIP frame, and saves the artifact to
  `graph_dataset.path`.

The saved artifact is a Parquet file containing:

- the aligned MLIP wide-frame columns, including `reference_ads_eng`
- the join-key column, typically `reaction`
- one row per sample after filtering/alignment
- serialized graph payload columns:
  `graph_sample_id`, `graph_node_features`, `graph_edge_index`,
  `graph_node_positions`, `graph_edge_features`, and `graph_graph_features`

Join-key expectations:

- `graph_dataset.join_key` must name a column present in the MLIP wide frame.
- The join key must be unique per row in the filtered frame.
- Graph sample IDs must match the filtered frame exactly after alignment.
- Extra graph sample IDs and missing graph sample IDs are treated as errors.

Failure modes:

- Missing join-key column: `wide_df is missing required join column`
- Missing target column: `wide_df is missing required target column: reference_ads_eng`
- Duplicate frame IDs: `wide_df contains duplicate <join_key> values`
- Missing graphs for frame rows: `missing graphs for <join_key> values`
- Extra graphs not present in the frame: `graph_view contains extra sample_ids`
- No MLIP feature columns: `No MLIP prediction columns found`
- Too few rows for an experiment run: `Not enough data to evaluate (need >5 samples)`
- Parquet graph artifact support requires `polars`

## Learned Sweep Contract

Learned-model families run against precomputed sweep splits rather than owning
their own splitting logic.

- Train/test families receive `TrainTestSweepRunnerInput`.
- Validation-aware learned families receive `TrainValTestSweepRunnerInput`.
- `split.dataset_subsets()` returns aligned `SweepDataset` views for each split,
  including features, targets, sample IDs, graphs, and auxiliary views.
- `split.loader_inputs(...)` separates split membership from batching and
  collation policy.
- `split.loaders(...)` is the adapter seam for framework-specific loader
  creation, including graph loaders and mixed-modal learned families.
- Model selection may use only train/val data.
- Outer test data must remain held out until one final evaluation after
  selection and any optional refit.

Batching and split-safety guarantees:

- Split subsets are materialized first. Batching happens only after the
  train/val/test membership is fixed.
- A batch is always derived from exactly one split-local `SweepDataset` subset.
  No emitted batch may mix train, val, or test examples.
- Train batching and eval batching are configured separately through
  `TrainEvalLoaderPolicy`.
- Train loaders may shuffle. Validation and test loaders default to no shuffle.
- `eval_batch_size` may differ from `batch_size`, so validation/test throughput
  can be tuned independently of training behavior.
- The default helper path, `SweepDatasetBatchLoaderAdapter`, emits deterministic
  split-safe batches from `SweepDataset` subsets.
- If train shuffling is enabled, batch order may change inside the train split,
  but train batches still contain only train examples.
- Validation/test loaders remain stable unless a caller explicitly opts into
  different behavior.
- Held-out outer-test data must not be consumed during candidate ranking. It is
  reserved for one final post-selection evaluation pass.

## Split Feasibility Policy

Learning-curve sweep sizes are outer training budgets.

- For train/test-only families, `sweep_size` is the full training set size.
- For validation-aware families, `sweep_size` must cover both inner training
  and inner validation.
- `test_idx` is always an outer holdout and is never part of `sweep_size`.

Validation-aware runs size validation as:

```text
max(
  floor(validation_fraction * sweep_size),
  min_val_size,
  min_tuning_val_size,
)
```

A validation-aware sweep point is emitted only if all of these can be satisfied
together:

- caller-requested `min_train` / `max_train`
- family-level `min_train_size`
- `min_tuning_val_size` and `min_val_size`
- `min_inner_train_size`
- `min_test_size`

That means some requested sweep sizes may be skipped, and some whole sweep
regions may collapse to an empty split collection, when the budget is too small
to leave:

- enough validation samples for meaningful scoring
- enough remaining inner-train samples after validation
- enough outer-test samples for final evaluation

This behavior is intentional. Oasis now prefers dropping infeasible
validation-aware points over producing train/val/test splits that are too small
to support sensible model selection.
