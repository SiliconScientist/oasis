Place a patched CatBench source checkout in this directory.

Expected layout:

- `vendor/catbench/catbench/__init__.py`
- `vendor/catbench/catbench/...`

`oasis` will prepend `vendor/catbench` to `PYTHONPATH` for adapter subprocesses.

If you store the checkout somewhere else, set `mlip.catbench_source` in `config.toml`.
