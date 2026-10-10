# Temporal models and coherent intervals

## Run

After the base setup and data preparation in the README, install pinned PyTorch
in the same separate environment. The CPU version needs no GPU:

```sh
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/temporal.txt -e .
grid-prob backtest --data runs/dataset --output runs/probabilistic --device cpu
grid-prob predict --data runs/dataset --model runs/probabilistic/models/2014-10-01/transformer-42 --family transformer --origin 2014-10-01T00:00:00
```

For the measured NVIDIA run, install PyTorch from
`https://download.pytorch.org/whl/cu128` instead and use `--device cuda`.
The tested GPU is an RTX 5070. Models are trained from scratch; no pretrained
weights or externally hosted inference services are needed.

`--resume` continues the same run from epoch checkpoints. It requires identical
configuration, input-data checksum, source files and device. Checkpoints preserve
optimizer state, selected weights and random-number state. To change any of those
inputs, choose a new output directory. Runtime after resume measures that
invocation only. The prediction command loads checksummed model weights and
rejects origins earlier than the recorded test boundary.

## Fixed experiment

[configs/probabilistic.json](../configs/probabilistic.json) uses the baseline's
four 28-day test periods and four horizons. The new comparison gives both model
families the same earlier training cutoff: 28 days before each test boundary.
The following 14 days select hyperparameters or epochs; the final 14 days are
reserved for calibration. Every selection target precedes calibration and every
calibration target precedes the test. No post-selection refit consumes the
calibration observations. This is a different training protocol from the original
baseline; compare the matched-cutoff models in this experiment directly.

The transformer takes 336 observed hours across all six nodes, divided into
14 daily patches. Two attention layers use dimension 48, four heads and dropout
0.1. Pooled and last-patch representations combine with known target calendar
cycles and observed weekly lags to predict all four horizons. A median residual
and two positive widths give ordered, nonnegative 10/50/90 quantiles. Training
uses scaled pinball loss, AdamW, 30 epochs, batches of 64, and gradient clipping.
Epoch choice uses selection loss only. All three seeds, 42–44, are reported;
none is selected using test performance.

The tree comparator keeps the baseline's two depth candidates, boosting rounds,
learning rate and history features. Both families use training-only node scales
and labels, and selection origins every six hours with all requested horizon
labels available before calibration. Calibration and testing use hourly origins.
Models remain fixed within each test period, while observed history updates.
Observed future weather and future demand are never inputs.

## Reconciliation and calibration

Let `S` map the five leaves to the six-node hierarchy. We estimate a full residual
covariance `W` from selection errors, with Ledoit–Wolf shrinkage and a small
diagonal ridge. The projection is
`P = S (S' W^-1 S)^-1 S' W^-1`. It combines regional and aggregate information
while preserving coherent inputs. See the
[forecast reconciliation reference](https://otexts.com/fpp3/reconciliation.html).
After projection, negative leaves are clipped and reaggregated; this changes the
unconstrained linear solution when clipping occurs. Forecasts reconcile only on
the dataset's algebraic source-clock hierarchy, not a verified UTC grid total.

Separate calibration observations determine a simultaneous regional band.
For each horizon, the score is the maximum absolute reconciled-point error over
the five leaves, divided by each leaf's raw half-interval width. A small
training-scale floor prevents division by zero. The sorted score at rank
`ceil((n+1)*0.8)` sets the radius multiplier. This follows split-conformal
[calibration principles](https://arxiv.org/abs/2107.07511); temporal dependence
and distribution shifts mean the usual exchangeability guarantee cannot be
assumed here. Coverage is measured, not promised.

Each regional band is centered on its reconciled point, clipped below at zero.
The aggregate lower bound sums regional lower bounds; the upper bound sums upper
bounds. If all five regional outcomes lie inside their bands, the aggregate lies
inside this envelope. These are **simultaneous bands**, not marginal 10th/90th
quantiles. Summing marginal quantiles would not provide that interpretation.
The aggregate envelope can be conservative and wide.

Three outputs isolate the effects: `raw` quantiles, `reconciled` points with
shifted uncalibrated bands, and `calibrated` simultaneous bands. Point, lower and
upper forecasts in the calibrated output each add up across the hierarchy.
Only raw forecasts receive a quantile pinball score; the other outputs are
evaluated as points and intervals. Interval score uses nominal alpha 0.2 to
compare width and misses, alongside separate simultaneous-leaf coverage.

## Evidence and limits

The fixed run produced 774,144 forecasts in 121.57 seconds. The transformer has
57,762 parameters and used at most 74.92 MiB of allocated GPU memory. Raw tree
MASE is 0.751 versus transformer 1.041; after reconciliation, 0.771 versus 1.090.
The three transformer seeds' raw MASE values are 1.075, 1.036 and 1.011.

After calibration, simultaneous regional coverage is 73.6% for trees and 71.1%
for the transformer, below the intended 80%. Individual node coverage averages
93.6% and 92.4%, including the conservative aggregate envelope; these marginal
averages do not establish that all regions are covered together. Average widths
increase from 680 to 1,825 MW for trees and 964 to 2,160 MW for the transformer.
Point coherence errors fall to floating-point rounding, but accuracy decreases
slightly and the wider bands increase the nominal marginal interval score.
No architecture or calibration changes were made after inspecting these scores.

The results directory preserves every forecast as `lower`, `point` and `upper`,
plus its fold, family, seed, origin, horizon and node. It includes per-node MASE,
coverage, width, interval score and peak error; simultaneous leaf coverage and
point coherence are reported separately. README/report averages give equal
weight to folds, seeds within each family, and all six nodes.

The saved-model verification covers real forecasts on CPU and GPU, allowing
small floating-point differences between devices. Tests cover independent
future-data perturbations, exact CPU checkpoint continuation, covariance
projection identities, calibration rank, aggregate-band consistency, invalid
data and command-line inference. The original baseline result files remain
unchanged. Current models and calibration records are local reproducible
artifacts; checksums for them are included with the measured results.

The dataset is small in geographic scope and old; the naive source clock lacks
verified physical alignment. Calibration examples overlap in time, rare peaks
can lie outside the calibration distribution, and the wide simultaneous bands
trade sharpness for joint coverage. This is a benchmark, not a grid operations
system or a guarantee of future calibration.
