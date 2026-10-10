# Forward evaluation and robustness

## Reproduce

Use the separate Python 3.11 environment and pinned PyTorch installation from
the [model guide](PROBABILISTIC.md). Then:

```sh
python -m pip install -r requirements/tracking.txt -e .
grid-forecast fetch --output data/raw
grid-forecast prepare --raw data/raw --output runs/data-forward --end 2015-03-01
grid-experiment run --data runs/data-forward --output runs/forward --device cpu
grid-experiment verify --data runs/data-forward --output runs/forward
grid-experiment track --data runs/data-forward --output runs/forward --tracking runs/tracking
python scripts/export_experiment.py --data runs/data-forward --run runs/forward --output runs/evidence
```

Open `runs/evidence/report.html`. Use `--device cuda` with the pinned CUDA build
for the measured GPU path. Add `--resume` to continue an interrupted run; it
requires identical configuration, data, source, package versions and device.
A completed resume verifies the sealed run without retraining. Use a new output
directory for a separate repeat; do not edit a sealed experiment.

The local MLflow command writes SQLite metadata and artifacts under the specified
tracking directory. It explicitly sets the local tracking and artifact locations,
disables MLflow telemetry, and needs no account or server. Each `track` invocation
registers a new run and returns its ID and stored metrics. The lightweight pinned
client does not include the full MLflow web UI. See the official
[local database guide](https://www.mlflow.org/docs/latest/ml/tracking/tutorials/local-database/).

## Forward protocol

[forward.json](../configs/forward.json) retains the published transformer,
tree candidates, seeds, selection and calibration settings. Only the evaluation
dates and duration change: 21 days of hourly origins starting January 1 and
February 1, 2015. All four horizons are retained. These targets had not been scored
when this protocol was fixed. The earlier 2014 benchmark remains unchanged and
is not described as a fresh test.

Training ends 28 days before each boundary; selection and calibration occupy
separate later blocks. Targets must precede the next block. February training
and calibration may use January observations already available by then. Models
stay fixed within each test period; observations update at every forecast origin.
Neither diagnostics nor stress tests select models or alter calibration.

The extended data contain 27,720 complete hourly observations per node, from
source days January 2012 through February 2015. The same upstream commit, file
checksums, GPL-3 attribution and source-clock convention apply. Missing March
2015 readings are outside the selected window. No target values are imputed.

## Observation failures and diagnostics

Five scenarios use the same 42 daily origins per fitted family/seed across both
folds: unchanged readings, all five regions missing for six hours, missing for
24 hours, independent 10% missing entries during the last week, and a 10% upward
measurement bias during the last 24 hours. Masks are seeded by origin and shared
across models. These are synthetic observation failures; future labels stay real
and unchanged. They are not simulations of a physical change in demand.

Missing leaf readings use the corresponding observed or previously repaired
weekly lag, falling back to the preceding reading when a full week is unavailable.
The algorithm moves forward only. Leading gaps, negative/infinite readings and
more than 24 consecutive missing hours per region are rejected. The aggregate is
rebuilt from repaired leaves so it cannot reveal the hidden values. Repair is an
explicit prediction-time policy; data ingestion continues to reject missing data.
Intervals are not recalibrated for outages and can fail badly during them.

Peak demand means a target exceeds that node's training-window 95th percentile.
Thresholds never use test demand. Reports include slice sizes, errors and coverage.
Population stability index uses training decile bins with 0.5-count smoothing;
it and mean changes compare test-origin demand with the fitting window. These are
descriptive, seasonally confounded comparisons, not tuned alerts or significance
tests. Daily joint coverage and calibration-score lag-one correlation expose
temporal dependence without claiming an independent-sample confidence interval.

## Measured findings

| Calibrated model | MASE | Joint regional coverage | Mean width, MW |
| --- | ---: | ---: | ---: |
| Tree | 0.879 | 72.6% | 2,239 |
| Transformer, three seeds | 1.202 | 88.7% | 4,328 |

These are equal-weight fold/seed/horizon/node averages, with joint coverage
computed over the five leaves. Trees remain more accurate; the transformer's
higher coverage comes with much wider intervals. The transformer exceeding 80%
here does not establish reliable calibration across dates, horizons or failures.
Raw tree and transformer MASE are 0.826 and 1.142 respectively.

On matched midnight stress origins, tree MASE rises from 0.464 to 0.767 under
a 24-hour outage, while joint coverage falls from 73.8% to 37.5%. Transformer MASE
rises from 0.701 to 0.843 and joint coverage falls from 89.9% to 74.4%. A 10%
upward reading bias increases tree MASE to 0.962. These daily samples are easier
than the full hourly test and must not be compared directly to its headline MASE.
The report retains all five scenarios, horizons, seeds and folds.

Tree individual-node coverage averages 87.9% in nonempty peak slices versus
95.4% in nonpeak slices; these marginal values differ from joint coverage.
Calibration-score lag-one correlation averages 0.478 for trees and 0.892 for the
transformer, supporting caution about exchangeability. In January, QLD mean
demand is 16.5% above its fitting-window mean and TAS is 19.4% below; seasonality
and archive conventions can contribute, so this does not establish a cause.

The 290,304-row forward comparison took 55.30 seconds on an RTX 5070, with
74.92 MiB peak allocated GPU memory. The 20,160-row stress experiment took
14.82 seconds. Runtime excludes environment setup and downloading data.

## Versioning and evidence

`protocol.json` records settings, source hashes, package versions, data identity
and device before fitting. `seal.json` inventories measured outputs and model
files, including preprocessing metadata and calibration. Dataset bytes and the
data manifest are both checked. Verification rejects changed files and unsafe
inventory paths. Checkpoints are recovery files and are excluded from the final
seal; model weights and all inference metadata are included.

An experiment version identifies one execution, including its timing records.
A separate repeat can have a different version while yielding identical
predictions. Public [evidence](../results/forward) includes forecasts, stress
outputs, slice diagnostics, the original seal and independent verification.
Raw data and trained weights remain local and are regenerated by the commands
above. Checksums detect accidental corruption; they are not digital signatures.

Tests cover future-data isolation, causal repair, immutable inventories, rejected
configuration changes, exact CPU retraining, prediction replay, and local tracking.
The export command independently recomputes error/coverage groups, verifies the
hierarchy, compares clean stress predictions with the full test, and reloads real
models on CPU. The published validation file records tolerances and check counts.
The measured GPU repeat reproduced every forecast and stress row exactly, along
with 22 model-weight and calibration files. Independent checks recomputed 576
metric groups, matched 4,032 clean stress forecasts, and reloaded 384 node
forecasts on CPU. The largest CPU difference was 0.0026 MW; hierarchy errors
were zero in the checked outputs. All 37 local tests passed.
