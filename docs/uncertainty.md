# Uncertainty metrics and calibration

[Documentation index](README.md) · [Project introduction](../README.md)

Learning-curve runs can also persist uncertainty-summary companions for methods
that emit a predictive spread. Oasis currently uses three aggregate metrics:

- `miscalibration_area`: area between the empirical calibration curve and the
  ideal curve. Lower is better.
- `sharpness`: average predicted spread magnitude. Lower means tighter
  predictions, though sharpness alone is not sufficient without calibration.
- `dispersion`: coefficient of variation of the predicted spread. This measures
  how much the spread varies across samples.

Method semantics matter:

- `residual`, `weighted_simplex`, and other explicitly uncertainty-aware
  methods should be interpreted as uncertainty summaries.
- `ridge` contributes a useful spread proxy for ranking or relative comparison,
  but it is not a calibrated probabilistic uncertainty and should not be
  interpreted as one.


## Calibration

Spread-based uncertainty can now be post-hoc recalibrated for supported
families. Calibration is opt-in; enable it explicitly when you want scalar
scaling:

```toml
[experiment.defaults]
calibration_enabled = true
calibration_method = "scalar_scale"
calibration_fraction = 0.2
min_cal_size = 1
```

Protocol:

- Hyperparameter-free spread methods use `train/cal/test`.
- Validation-tuned spread methods use `train/val/cal/test`.
- The calibration fit uses only the held-out calibration subset.
- Final RMSE and coverage are still reported only on the outer test subset.

When calibration is enabled, UQ summary frames report
`uncertainty_kind = "calibrated"`. Historical `spread_only` runs are not
directly comparable to calibrated runs unless they are rerun under the same
protocol.
