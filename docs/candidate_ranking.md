# Candidate ranking

[Documentation index](README.md) · [Project introduction](../README.md)

The `rank-candidates` command stays stable:

```bash
python -m oasis rank-candidates experiment.toml
```

Its config is predictor-driven. Use `[candidate_ranking]` with an ordered
`predictors = ["residual", "weighted_simplex", "ridge"]` list plus an optional
`validated_references_path` JSON file. If no validated references are supplied,
candidate ranking falls back automatically to the unfitted ensemble baseline.
If validated references are supplied, Oasis infers the shot count from that
file, writes CV diagnostic plots for the feasible predictors, and uses the
explicitly configured `selected_predictor` for the final screening decision.

Minimal config shape:

```toml
[candidate_ranking]
predictors = ["residual", "weighted_simplex", "ridge"]
selected_predictor = "residual"
target_binding_energy = -0.5
top_k = 10
# validated_references_path = "data/validated_references/example.json"
```

Validated-reference files are JSON lists like:

```json
[
  {"adslab_id": "adslab-000001", "adsorption_energy": -0.42},
  {"adslab_id": "adslab-000017", "adsorption_energy": -0.58}
]
```


These are configuration fragments; combine them with a dataset profile as in
[the example configuration](../experiment.example.toml).
