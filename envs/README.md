All virtual environments created with Python 3.13.0.

Per-model environments should contain only model-specific dependencies.
CatBench is provided from the shared vendored source at `vendor/catbench`
via `mlip.catbench_source` and injected into adapter subprocesses with
`PYTHONPATH`.
