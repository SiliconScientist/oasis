# oasis
Refining ML interatomic potentials with sparse DFT data in low-data regimes.

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

Targeted graph/config test commands:

```bash
PYTHONPATH=src python -m unittest tests.test_exp tests.test_graphs tests.test_config
PYTHONPATH=src python -m unittest tests.test_graphs tests.test_exp
PYTHONPATH=src python -m unittest tests.test_config
```

## Learning-Curve Result Artifacts

Learning-curve runs can persist one artifact per method so you do not need to
retrain every model just to regenerate plots.

Enable this in `config.toml`:

```toml
[experiment.learning_curve]
results_artifact_dir = "data/results/learning_curve"
reuse_results = true
```

If you prefer one file per dataset instead of one file per method:

```toml
[experiment.learning_curve]
results_bundle_path = "data/results/learning_curve/khlohc_tol.json"
reuse_results = true
```

Workflow:

- Run once with `results_artifact_dir` set. Oasis writes one JSON artifact per
  enabled method after the sweep completes.
- Regenerate plots later with `reuse_results = true`. If the saved artifact
  metadata matches the current sweep definition, Oasis reloads cached
  `LearningCurveResults` instead of retraining.
- Partial cache hits are supported. If only some enabled methods are cached,
  Oasis reuses those artifacts and trains only the missing methods.
- Reuse also tolerates cached supersets. If a saved artifact contains results
  for more methods than are currently enabled, Oasis reuses the matching subset
  instead of failing.
- Selective refresh is supported through `force_refresh_methods`. This reruns
  only the named enabled methods and overwrites just their artifacts.

Example selective refresh:

```toml
[experiment.learning_curve]
results_artifact_dir = "data/results/learning_curve"
reuse_results = true
force_refresh_methods = ["moe", "probe_gnn"]
```

Artifact compatibility is strict. Reuse happens only when the saved artifact
matches the current sweep metadata, including seed, min/max train, step,
repeats, enabled methods, and active plot filters.

## Graph Artifact Contract

Configured graph-backed learning-curve runs use
`experiment.learning_curve.graph_dataset` in `config.toml`.

Example:

```toml
[experiment.learning_curve.graph_dataset]
path = "data/processed/oh_mamun.parquet"
join_key = "reaction"
```

Behavior:

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
