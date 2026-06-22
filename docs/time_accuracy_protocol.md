# Time-vs-Accuracy Protocol for Learning-Curve Methods

This note defines the timing protocol for the
`feat/time-accuracy-plot` branch and follow-on method-specific timing work.

## Scope

This protocol applies to Oasis learning-curve methods.

Methods using standard MLIP feature tables:

- `ridge`
- `kernel_ridge`
- `lasso`
- `elastic`
- `residual`
- `weighted_linear`
- `weighted_simplex`
- `graph_mean`
- `moe`
- `gnn_direct`

Methods with method-specific generation inputs:

- `probe_gnn`
- `latent`

## Timing Definitions

For each plotted point, record:

- `generation_time_s`: time spent producing the MLIP-derived inputs used by the
  method.
- `training_time_s`: time spent on Oasis-side model selection and fitting for
  that point.
- `total_time_s`: `generation_time_s + training_time_s`.

This branch treats upstream feature extraction as part of generation, not
training.

For tuned methods, `training_time_s` includes the full Oasis-side selection
workflow used for that plotted point:

- validation-time candidate or trial evaluation
- chosen-model refit work
- any method-specific training work inside that selection/refit path

It does not include outer-test prediction or plotting.

## MLIP Time Semantics

The existing MLIP artifacts already include timing fields from CatBench result
JSON files. Those timings reflect MLIP-side structure relaxation / inference
work and belong in `generation_time_s`.

For this branch, use the summed MLIP compute time saved in the artifacts, not
end-to-end wall-clock job duration.

That means:

- we do not attempt to measure scheduler delay or cluster queue time
- we do not subtract or model parallelization speedups
- we treat the saved MLIP compute totals as the comparable cost basis

This is deliberate. Summed compute time is more stable across reruns than
parallel wall-clock time and is easier to compare across MLIPs with different
execution patterns.

## Parallelism Policy

MLIP extraction may run in parallel, and some MLIPs are slower than others.
For now, the protocol does not try to reconstruct effective parallel wall time.

Instead, charge each MLIP by its total recorded compute time. In other words,
parallel execution changes throughput, but not the cost definition used by this
plot.

## How Generation Time Is Charged

Charge generation time over the full evaluation dataset associated with the
plotted point, not per outer split replicate.

Operationally, this means:

- generation time is attached to the dataset artifacts used for the point
- generation time is constant across methods that consume the same MLIP inputs
- training time varies by method and train size

This keeps the comparison simple:

- generation captures the cost of obtaining the MLIP-based features
- training captures the extra cost of fitting the downstream Oasis method

## Method-Specific Generation Sources

Different methods may charge generation time to different upstream artifacts.

For standard MLIP-table methods (`ridge`, `kernel_ridge`, `lasso`, `elastic`,
`residual`, `weighted_linear`, `weighted_simplex`, `graph_mean`, `moe`, and
`gnn_direct`):

- `generation_time_s` is charged to the selected MLIP result JSON artifacts.

For `probe_gnn`:

- `generation_time_s` is charged to probe-feature generation.
- This includes the upstream work required to build the probe-augmented inputs
  consumed by the model.
- It does not include Probe GNN model fitting; that belongs in
  `training_time_s`.

For `latent`:

- `generation_time_s` is charged to latent feature-table creation.
- This includes the upstream work required to build the latent CSV or equivalent
  latent input artifact consumed by Oasis.
- It does not include downstream latent model fitting inside the Oasis sweep;
  that belongs in `training_time_s`.

## Interpretation

Under this protocol, a point answers:

"If these MLIP features already define the candidate input set for this
evaluation point, how much generation cost plus downstream training cost is
required to reach this error level?"

The intent is not to estimate true cluster wall time. The intent is to compare
methods on a consistent cost basis.
