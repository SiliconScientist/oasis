# Oasis

Oasis studies how a small set of quantum-mechanical reference calculations can
improve predictions of adsorption energy: how strongly an atom or molecule
binds to a surface, a useful quantity when comparing candidate catalysts.

Density functional theory (DFT) provides reference energies, but its computational
cost limits how many structures can be evaluated. Machine-learned interatomic
potentials (MLIPs) predict atomic energies and forces from models trained on
reference calculations. They can make screening more affordable, but their
accuracy can vary across chemical systems.

Oasis compares ways to combine or correct MLIP predictions using sparse DFT
data. Its experiments help assess how prediction error changes as more reference
data become available, and how to select methods and rank candidates under
limited data budgets. It provides evaluation workflows, not a guarantee that a
particular model will improve on every dataset.

## Capabilities

- Compare unfitted baselines, regression corrections, ensembles, and graph-based
  learned models through learning curves.
- Evaluate screening policies and rank candidates near a target binding energy,
  with or without validated reference energies.
- Examine predictive spread, optional uncertainty calibration, and the tradeoff
  between computation time and prediction error.
- Reuse compatible result bundles, extend sweeps incrementally, and regenerate
  plots without repeating every fit.

Experiments use explicit configurations, seeds, and recorded per-point
provenance. Model selection uses training/validation data; outer test data stay
held out for final evaluation. Uncertainty estimates and timing comparisons have
specific limits, documented in the [technical guides](docs/README.md).

Oasis focuses on experiments. MLIP ingestion and prediction are maintained in
[Moira](https://github.com/SiliconScientist/Moirai).

<a id="environment"></a>

## Quickstart

Requires Python 3.10 or later. From a checkout of this repository, install the
project and development dependencies:

```bash
pip install -e ".[dev]"
```

For an environment resolved from the committed lockfile, use `uv sync --extra dev`
instead, then activate `.venv`. See [GPU setup](docs/gpu_setup.md) for CUDA.

Copy [experiment.example.toml](experiment.example.toml) to `my-experiment.toml`.
Set its dataset paths, model choices, and device for your study. You must supply
reference data and precomputed MLIP results; the template includes placeholder
paths and optional feature artifacts. Follow [input preparation](docs/getting_started.md#prepare-a-study)
before running:

```bash
python -m oasis my-experiment.toml
```

The run produces result artifacts and plots under `data/results/` by default.
For the Python API, configuration details, and test commands, see
[Running experiments](docs/getting_started.md).

## Documentation

The [documentation index](docs/README.md) is the starting point for technical
reference material. Former README sections are linked below with their original
anchors retained.

<a id="entrypoints"></a>
<a id="related-repo"></a>

- [Running experiments](docs/getting_started.md): setup, Python/CLI entrypoints,
  tests, and legacy MLIP commands.
- [Candidate ranking](docs/candidate_ranking.md): predictor selection and
  validated-reference inputs.

<a id="learning-curve-result-artifacts"></a>

- [Learning curves and result artifacts](docs/learning_curves.md): sweeps,
  model configuration, caching, and plot interpretation.

<a id="uncertainty-metrics"></a>

- [Uncertainty and calibration](docs/uncertainty.md): metrics and scientific caveats.

<a id="oh-bma-comparison-workflow"></a>

- [Anomaly-filtering comparisons](docs/anomaly_filtering.md): OH-BMA presets.

<a id="graph-artifact-contract"></a>

- [Graph artifacts](docs/graph_artifacts.md): alignment, storage, and errors.

<a id="learned-sweep-contract"></a>
<a id="split-feasibility-policy"></a>

- [Evaluation protocols](docs/evaluation_protocol.md): held-out splits, batching,
  and feasible training budgets.
- [Comparative caching](docs/comparative_learning_cache_model.md) and
  [timing protocol](docs/time_accuracy_protocol.md): reuse rules and cost definitions.
