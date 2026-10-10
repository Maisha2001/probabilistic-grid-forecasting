# Evaluation protocol

The fixed [configuration](../configs/baseline.json) evaluates origins in the
first 28 days of January, April, July and October 2014. Each origin predicts
1, 6, 24 and 168 hours ahead. Target windows move with the horizon and do not
overlap across folds. This is four seasonal blocks, not a full-year score.

At each fold, candidate models train strictly before the preceding 28-day
validation block. Hyperparameters are selected by mean pinball loss on scaled
validation demand; all validation targets precede the test boundary. The winner
is refitted using only the last year's available history before that boundary.
Each model is then fixed for the fold. Observed demand history updates at every
hourly test origin. The latest ending interval is assumed immediately available;
actual source reporting delays are not modeled.

## Models

- **Weekly residual:** the observed value one week before the target, plus the
  empirical 10th, 50th or 90th percentile of weekly differences from the training
  prefix. The median includes a fitted residual offset; it is not a pure
  unadjusted seasonal-naive point forecast.
- **Quantile XGBoost:** one pooled model per horizon with regional one-hot
  features, observed lags, past 24/168-hour summaries, target-aligned weekly
  lags and known calendar cycles. Each node is normalized by its training mean.
  Depth 3 or 5 is chosen on validation data, with 120 boosting rounds, learning
  rate 0.07 and seed 42. Training/validation origins are sampled every six hours;
  every test hour is evaluated. No future demand or observed future weather
  enters features. Preprocessing statistics are fit only on training history.

Both models output quantiles 0.1, 0.5 and 0.9. Negative predictions are clipped
at zero and quantiles sorted before scoring. The report records crossings before
sorting: 4,016 tree node-origin forecasts crossed in this measured run. Sorting
enforces order; it does not calibrate coverage. Node forecasts are independent,
including the aggregate. Do not sum marginal quantiles to obtain uncertainty
for a sum of correlated series.

## Metrics and evidence

`results/baseline/predictions.parquet` stores every origin, target, horizon,
node, method, actual and quantile. `metrics.json` includes:

- MAE and weekly MASE using a training-only denominator for each fold/node.
- Mean pinball loss over the three quantiles; empirical 80% interval coverage
  and average width.
- Error at the actual peak and absolute difference between predicted and actual
  maxima, separately for each node, fold and horizon.
- Mean and maximum absolute difference between the root median and summed leaf
  medians, measuring forecast coherence without conflating marginal quantiles.
- Validation choices, model/data/source checksums, package versions and runtime.

The HTML report and README use equal-weight averages across folds and all six
nodes. This gives the aggregate its own weight alongside the five leaves; inspect
the per-node scores when comparing region-level performance. MW scores depend
on node scale. Training-mean scaling gives each node comparable influence during
hyperparameter selection, while the report also preserves scores in MW.

Saved model files and their metadata live under the run's `models/` directory.
The prediction command rejects origins before the model's training boundary.
Tests perturb future demand to verify unchanged features, selected parameters
and fitted model bytes; they also check known metric values, data failures and
saved-model predictions. CI uses generated fixtures; the real-data result files
are from the full local run. A fixed seed and dependency pins aid reproduction;
small floating-point differences across platforms can remain.

## Interpretation

Trees improve mean point error over the seasonal residual baseline at all four
horizons, but their 80% intervals cover only 61–73% on average by horizon.
Mean absolute median incoherence is 195.5 MW across folds and horizons, with
a maximum of 3,618.5 MW. The weekly model is also slightly incoherent because
its separately fitted median residuals are not additive.

There is no claim of calibrated uncertainty, coherent forecasts, production
readiness, or state-of-the-art performance. Results use one seed and one pinned
historical dataset. This protocol updates inputs with observations between
origins; it does not test a 28-day open-loop forecast. Timezone ambiguity and
the limited demand definition further constrain interpretation.
