# Running experiments

[Documentation index](README.md) · [Project introduction](../README.md)

## Environment

See the [README installation instructions](../README.md#quickstart) and
[GPU setup](gpu_setup.md). For an existing environment, the editable-install
alternative to `uv sync --extra dev` is:

```bash
uv pip install -e ".[dev]"
```

## Prepare a study

Copy [experiment.example.toml](../experiment.example.toml) to a study-specific
file, such as `my-experiment.toml`. It is a configuration template, not a
self-contained demonstration dataset. Before running:

- Set `dataset_profile.tag` and the matching `[datasets.<tag>]` entry.
- Supply the raw structures/reference data and precomputed MLIP result JSONs.
  `raw_dataset_filename` resolves under `data/raw_data/`; `mlip_run_dirname`
  resolves under `data/mlips/`.
- Review `[models]`. The template enables `use_latent`, which needs the separate
  CSV configured in `[models.latent]`; disable it if that artifact is unavailable.
  Graph and probe methods also need their corresponding structure/feature inputs.
- Choose `device = "cpu"` or configure a supported GPU environment. The template
  currently requests `"cuda"`. Set `dev_run = true` for a small smoke run.
- Choose training budgets and repeats appropriate to the dataset. See
  [split feasibility](evaluation_protocol.md#split-feasibility-policy).

Plots default to `data/results/plots/<mlip_run_dirname>`; learning-curve bundles
use `data/results/learning_curve/<processed_basename>.json` before run-specific
suffixes are applied. See [results and caching](learning_curves.md).

## Entrypoints


Oasis owns experiment workflows. MLIP ingestion/prediction now lives in
[Moira](https://github.com/SiliconScientist/Moirai).

Experiment execution is config-driven:

```python
from oasis.experiment_runner import run_experiment_from_config

run_experiment_from_config(["experiment.toml"])
```

If you already have a parsed config object, call `run_experiment(cfg)` instead.

For MLIP ingestion/prediction, use Moira. The old in-repo MLIP module path
remains documented here only as a temporary compatibility surface:

```bash
python -m oasis.mlip submit --config path/to/mlip-config.toml
python -m oasis.mlip make-tasks --config path/to/mlip-config.toml --run-tag dev --out slurm_output/tasks.txt
python -m oasis.mlip run-one --config path/to/mlip-config.toml --line "mace example data/raw_data/example.json data/results/mlips/dev/example/mace.json"
```

The CLI also runs experiments:

```bash
python -m oasis experiment.toml
```

With no arguments, it discovers `experiment.toml` in the working directory.
Multiple config paths are merged in order; later values override earlier ones.
Run commands from the repository root so relative data paths resolve correctly.


## Tests

Run the suite from the repository root in the installed environment:

```bash
PYTHONPATH=src python -m unittest
```

Targeted graph/config checks:

```bash
PYTHONPATH=src python -m unittest tests.test_exp tests.test_graphs tests.test_config
PYTHONPATH=src python -m unittest tests.test_graphs tests.test_exp
PYTHONPATH=src python -m unittest tests.test_config
```
