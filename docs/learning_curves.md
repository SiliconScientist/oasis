# Learning curves, configuration, and result artifacts

[Documentation index](README.md) · [Project introduction](../README.md)

Timing definitions are in [the timing protocol](time_accuracy_protocol.md).
The examples below are fragments to merge into a complete experiment config.

Learning-curve runs can persist one bundle file per dataset so you do not need
to retrain every model just to regenerate plots.

## Dataset paths and sweep sizes

In the tag-first config style, the bundle path is derived from
`dataset_profile.tag` plus `[datasets.<tag>]`, so `experiment.toml` only needs
the naming inputs that differ from the default tag-based convention:

```toml
[dataset_profile]
tag = "example_oh"

[datasets.example_oh]
# Omit this when the processed basename is exactly the tag.
# processed_basename = "example_oh"

[experiment.defaults]
reuse_results = false

[experiment.learning_curve]
[plot.curve_window]
full_dataset_window = true
```

`full_dataset_window = true` only disables the `min_x` / `max_x` crop. It does
not disable `include_sizes`, which selects the training sizes shown in
absolute-size plots.

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
[experiment.defaults]
reuse_results = false

[experiment.learning_curve]
sweep_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
n_repeats = 10
```

Oasis resolves `sweep_fractions` against the active dataset size, deduplicates
any collisions after rounding down to counts, and reuses cached rows by the
resolved `n_train` / `n_budget` values.

## Models and shared defaults

Model toggles live in a flatter top-level config layout:

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
`gnn_direct`, and `probe_gnn`.

`gnn_direct` and `probe_gnn` can now be toggled directly in `[models]` via
`use_gnn_direct` and `use_probe_gnn`.

Shared split/cache defaults can live in `[experiment.defaults]`. Values set
there apply to configured `[experiment.zero_shot]`, `[experiment.learning_curve]`,
and `[experiment.screening]` sections,
while section-local values still win when they differ.

See [uncertainty calibration](uncertainty.md#calibration) for the opt-in protocol.

For Optuna-backed learned models (`moe`, `gnn_direct`, `probe_gnn`, and the
SchNet gate), you can omit `training.epochs` to let Optuna tune epochs from a
built-in candidate set. Keep `training.epochs` only when you want to pin it.

## Incremental results

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

## Screening diagnostics and plots

When both `[experiment.learning_curve]` and `[experiment.screening]` are
configured, Oasis also writes a derived screening-policy diagnostic based on
shared outer train/test splits across methods. This diagnostic is separate from
the normal learning-curve bundle and separate from screening CV bundles.

Interpretation:

- Screening CV RMSE is used only to select a method within each shared outer
  training split.
- Final regret is computed only from held-out outer-test RMSE on that same
  split.
- Screening CV RMSE is not compared directly to held-out oracle RMSE.

The diagnostic JSON is cached under the screening bundle root (see
[comparative cache stages](comparative_learning_cache_model.md)).
Tables and figures are written to `plot.output_dir`:

- `policy_selection_diagnostic_detail_<suffix>.csv`
- `policy_selection_diagnostic_summary_<suffix>.csv`
- `policy_selected_vs_oracle_<suffix>.png`
- `policy_regret_<suffix>.png`

Learning-curve figures use these base names; budget-axis plots add
`_absolute` and `_fraction` span suffixes:

- `learning_curve_<suffix>.png`

When multiple datasets are configured under `[datasets]`, Oasis also writes:

- `learning_curve_oracle_all_datasets_<suffix>.png`

Interpretation of `learning_curve_oracle_all_datasets_<suffix>.png`:

- The legend is datasets, not methods.
- Each curve is the oracle envelope for one dataset across `n_train`.
- At each `n_train`, the plotted RMSE is the best value among the currently
  enabled learning-curve methods only.
- Disabled methods are excluded from the oracle selection even if cached result
  frames for those methods are present in the bundle.

The detail table is keyed by `budget` and `repeat` and includes
`oracle_method`, `screening_selected_method`, both held-out RMSE values,
`regret`, `screening_cv_rmse`, and `agreement`.

## Iterative configuration example

```toml
[dataset_profile]
tag = "example_oh"

[datasets.example_oh]
# Omit this when the processed basename is exactly the tag.
# processed_basename = "example_oh"

[experiment.defaults]
reuse_results = true

[experiment.learning_curve]
# First pass: run one method or one range.
# Later, enable another method or widen the sweep bounds.
# Oasis keeps the old rows and fills only the gaps.

force_refresh_methods = ["moe", "probe_gnn"]
force_refresh_train_sizes = { ridge = [20, 30] }

[models]
use_ridge = true
use_weighted_simplex = false
```
