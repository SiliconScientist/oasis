# Evaluation and split contracts

[Documentation index](README.md) · [Project introduction](../README.md)

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

Calibration-aware families also reserve calibration samples within the outer
training budget; see [the calibration protocol](uncertainty.md#calibration).
