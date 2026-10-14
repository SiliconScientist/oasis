# OH-BMA comparison workflow

[Documentation index](README.md) · [Project introduction](../README.md)

To compare anomaly-aware MLIP selection, keep two experiment presets and
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
exclude_anomalous_mlips = false
minimum_quorum = 2
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
exclude_anomalous_mlips = true
minimum_quorum = 2
```

Notes:

- Oasis now auto-suffixes outputs and caches with `anomalyaware_on` or
  `anomalyaware_off`, so the two runs do not overwrite each other.
- The learning-curve bundle and aligned graph artifact are also separated by
  that suffix.
- `minimum_quorum` controls how many non-anomalous MLIPs a sample must retain.
  These presets explicitly use `2`; the configuration default is `0`.
- Older README examples included `label_allowlist` and
  `strict_inference_anomaly` in learning-curve MLIP selection. These are not
  fields of the current `MlipSelectionConfig`; they remain available in
  `[candidate_ranking]`, where `strict_inference_anomaly = true` enables
  filtering based on inference-detail flags. Do not use them to control
  learning-curve filtering.

Bundle identity is dataset/filter oriented, not exact-sweep oriented. Reuse
requires the same dataset/filter identity and seed, but the bundle may contain:

- different enabled-method subsets across runs
- different `min_train` / `max_train` / `step` ranges across runs
- different `n_repeats` across runs

Those sweep settings are still preserved as per-point provenance in the saved
artifact so you can audit how each row was produced.
