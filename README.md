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
python -m pytest
```

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
