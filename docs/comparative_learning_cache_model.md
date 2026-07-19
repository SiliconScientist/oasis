# Comparative Learning Cache Model

Comparative learning now uses a staged cache model instead of treating
`policy_selection_diagnostic.json` as the only reusable artifact.

## Stages

The runner routes comparative learning through these stages:

1. Load or run learning-curve results.
2. Load or run screening results.
3. Load or build primitive comparison rows.
4. Derive policy summaries from primitive rows.
5. Render plots.

The important split is between primitive row generation and policy reduction.

## Artifacts

Comparative learning writes three artifact types under the screening bundle root:

- `policy_selection_outer_repeat_metrics_*.json`
- `policy_selection_screening_rows_*.json`
- `policy_selection_diagnostic_*.json`

The first two are primitive artifacts.
The last one is derived.

### Primitive outer repeat metrics

`policy_selection_outer_repeat_metrics_*.json` stores outer-test RMSE rows keyed by:

- `method`
- `budget`
- `repeat`

This artifact is reusable across policy changes because it does not depend on:

- `policy_names`
- `combined_miscalibration_lambda`

### Primitive screening rows

`policy_selection_screening_rows_*.json` stores screening diagnostic rows keyed by:

- `method`
- `budget`
- `repeat`
- `split_fingerprint`

It also stores explicit completeness metadata, including infeasible points.

### Derived policy diagnostic

`policy_selection_diagnostic_*.json` stores:

- detail rows
- summary rows
- outer metrics snapshot used for the derived result

This artifact depends on policy configuration and is invalidated by:

- `policy_names`
- `combined_miscalibration_lambda`

## Invalidation rules

Primitive and derived invalidation are now separated.

Policy-only changes invalidate only the derived artifact.
Primitive artifacts remain reusable if the learning/screening configuration still matches.

Method-set changes are handled asymmetrically:

- If enabled methods shrink, cached primitive rows are filtered.
- If enabled methods expand, compatible cached primitive rows are reused and only missing methods are recomputed.

## Completeness

Primitive artifacts no longer treat “file exists” as “cache complete.”

Each primitive artifact carries completeness metadata over `(method, budget, repeat)` points.
For screening rows, completeness can include both:

- present point keys
- infeasible point keys

A primitive cache is complete only when every expected point is covered by one of those sets.

## Infeasible screening points

Some method/budget points cannot produce a valid screening split under the configured screening constraints.
Those points are recorded explicitly as infeasible instead of being left as silent misses.

This matters for reuse:

- infeasible points are treated as complete for primitive cache coverage
- infeasible points are not retried on later rebuilds
- infeasible methods are excluded from oracle/selected-policy comparison at that point

Ordinary failures are still different from infeasibility.
A failed point is incomplete and should continue to trigger recompute.

## Practical debugging

The runner now logs cache decisions by stage, for example:

- primitive outer metrics cache hit/miss
- primitive screening rows cache hit/miss
- policy diagnostic artifact cache hit/miss

When a derived artifact misses but both primitive caches are complete, the runner performs a pure reduction step instead of rerunning model families.
