# Graph artifact contract

[Documentation index](README.md) · [Project introduction](../README.md)

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
