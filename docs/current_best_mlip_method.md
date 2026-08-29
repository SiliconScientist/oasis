# Current-Best-MLIP Baseline

## Method

The current-best-MLIP baseline selects one ML interatomic potential (MLIP) at
each learning-curve split. Given predictions from `M` MLIPs for the currently
labeled training set, it computes the training root-mean-square error (RMSE)
of every MLIP and selects

```text
m* = argmin_m RMSE(y_train, y_hat_train,m).
```

The selected MLIP alone produces the held-out test predictions for that split.
Selection uses only labeled training samples; held-out test targets are never
used to choose the MLIP. Because the labeled set grows during an active
learning workflow, the selected MLIP can change as additional reference
calculations become available.

## Constant Spread Proxy

This method is fundamentally a point-prediction baseline: it does not combine
multiple MLIPs and it does not produce structure-specific uncertainty. Oasis
nevertheless attaches a diagnostic spread proxy so it can be compared in the
same UQ summaries as ensemble methods.

For a split whose selected MLIP has training RMSE `s`, every test prediction is
assigned the same spread:

```text
sigma_i = s for every test structure i.
```

For nominal central coverage `c`, Oasis evaluates the implied interval

```text
y_hat_i +/- z_c sigma_i,
```

where `z_c` is the standard-normal central-coverage quantile. It evaluates
nominal coverages from 0.1 to 0.9, computes the absolute gap between observed
and nominal coverage at each value, and integrates that gap with the trapezoid
rule to obtain miscalibration area.

This is deliberately recorded as `spread_only`, not as a calibrated predictive
interval. The training RMSE determines its common width; it does not claim to
estimate heteroscedastic uncertainty for individual structures.

## Dispersion Interpretation

The spread proxy is constant within a split, so its normalized dispersion is

```text
std(sigma_i) / mean(sigma_i) = 0.
```

Zero dispersion is expected and informative rather than a numerical failure.
It makes the method's limitation visible: unlike an ensemble or another
structure-aware uncertainty model, current-best-MLIP cannot widen or narrow its
interval in response to the input structure. Its sharpness reports the overall
training-residual scale, while its dispersion reports no per-structure
adaptability.

## Inference-Cost Motivation

The method can substantially reduce prediction cost after MLIP selection.
Consider 10 MLIPs, 10 labeled samples, and 100 unlabeled samples. First obtain
all 10 MLIP predictions for the 10 labeled samples and use their known labels
to select the lowest-training-RMSE MLIP. The current-best-MLIP strategy then
runs only that one MLIP on the 100 unlabeled samples: 100 model evaluations.

In contrast, a 10-member ensemble needs predictions from all 10 MLIPs on the
100 unlabeled samples to form its ensemble mean and disagreement-based
uncertainty: 1,000 model evaluations. Conditional on the initial labeled-set
predictions already being available, this is a 10-fold reduction in unlabeled
inference evaluations for the example.

The trade-off is explicit. Ensembling can improve predictive robustness and
provides structure-dependent disagreement, whereas current-best-MLIP uses a
single selected model and only exposes a global, constant residual-scale
diagnostic.

## Representative OC20-OCxHx Observation

On the representative OC20-OCxHx learning curve, current-best-MLIP does not
outperform the mean-residual baseline at any of the 36 evaluated training
sizes. Its RMSE is higher than the residual baseline by at least `0.0339 eV`
at every point in this run. This comparison is specific to the saved
OC20-OCxHx experiment and should not be interpreted as a universal ordering of
the methods.

In the all-datasets dominant-oracle analysis, current-best-MLIP is dominant
only at train size 12. Its lone marker is not visually distinguishable in the
current plot because it uses Matplotlib's default blue, which closely matches
the configured Ridge color. Thus, the figure appears to show that it is never
dominant even though the underlying data contains that one dominant point.

The high-data behavior illustrates the baseline's limited adaptability. For
training sizes of 946 or more, it selected the same MLIP in all 270 repeated
splits and its mean RMSE remained in a narrow `0.631-0.644 eV` range. Thus,
once one existing MLIP is consistently preferred, additional labeled data does
not provide the method with a mechanism to learn a more expressive correction
or a structure-dependent prediction interval. In contrast, the mean-residual
method uses the labeled data to correct the MLIP predictions and remains better
throughout this representative curve.
