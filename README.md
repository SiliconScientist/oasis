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
- `split.loaders(...)` is the adapter seam for framework-specific loader
  creation.
- Model selection may use only train/val data.
- Outer test data must remain held out until one final evaluation after
  selection and any optional refit.
