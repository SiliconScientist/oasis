# JCP round-one saved-data analysis

Original data and caches are read-only. This revision adds deterministic audits,
not MLIP/DFT calculations or model refits. See `revision_outputs/round1-eight-datasets/manifest.json`
for input fingerprints and the exact analysis commit; `table_export_manifest.json`
identifies the separately committed formatting code. Numerical outputs include
all sample IDs/predictions/masks, filtering results, paired ten/seven-model
comparisons, anomaly counts, off-scale values and descriptive bias/scatter.

Run with the existing scientific Python environment, from this repository:

```sh
python -m unittest discover -s tests/revision -v
MPLCONFIGDIR=/private/tmp/oasis-revision-mpl python scripts/revision/round1.py \
  --data-root /path/to/original/oasis/data --output /path/to/new/audit-directory
python scripts/revision/export_tables.py /path/to/new/audit-directory /path/to/new/tables.tex
```

The analysis rejects output within the historical data tree, an existing output
directory, modified tracked source code, unexpected model identities, reference
mismatches, ambiguous duplicate inputs, and failure to reproduce saved stage/model
values. It does not infer the original relaxation software version from the
current code. A successful reconstruction establishes numerical agreement, not
proof of every historical execution detail. No original file is overwritten.

The Tol signature contains both FIRE and L-BFGS MACE files. Only FIRE matches the
saved individual-model value; the full reconstructed ensemble stages also agree.
Both candidate hashes and selection evidence are retained. OH-BMA n1094 and Tol
n302 are five-model alternatives; the NHx-n2367 file identifies OCxHx internally.

The main result contract uses the intersection of the ten- and seven-model
quorum-two populations. Separate retained counts report coverage on the same
complete ten-model universe. Descriptive centered errors are not deployed or
fitted corrections. `common_model_valid` statistics use model-specific subsets;
only `common_unfiltered` has the identical per-model population.

Validation: eight unit tests passed; all eight datasets reproduce three stages
and ten individual-model values within 1e-10; all 101 recorded input fingerprints
were unchanged after execution. Historical and scientific interpretation limits
are discussed in the companion manuscript repository's provenance checkpoint.
