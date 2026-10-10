# Probabilistic grid forecasting

Forecast hourly electricity demand for five Australian regional series and their
derived aggregate, with 1-, 6-, 24- and 168-hour horizons. The runnable baseline
compares seasonal residual forecasts with quantile XGBoost using rolling origins.

## Run

Use Python 3.11 in a separate environment. No GPU, credentials or paid services
are required. On Windows, replace the activation command with
`.venv\Scripts\Activate.ps1`.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/dev.txt -e .
python -m pytest -q
grid-forecast fetch --output data/raw
grid-forecast prepare --raw data/raw --output runs/dataset
grid-forecast backtest --data runs/dataset --output runs/baseline
grid-forecast predict --data runs/dataset --model runs/baseline/models/2014-10-01/h24 --origin 2014-10-01T00:00:00
```

Open `runs/baseline/report.html` for the measured comparison. Downloads total
about 38 MB. Existing run directories are protected; use a new output path to
repeat an experiment. Models are generated locally and verified by checksum.

## Measured baseline

Four 28-day evaluation periods in 2014, six nodes, hourly forecast origins.
MASE is mean absolute error scaled by training-only weekly differences; lower
is better. Values below give equal weight to each fold and node, including the
derived aggregate.

| Horizon | Seasonal MASE | Tree MASE | Tree 80% coverage |
| --- | ---: | ---: | ---: |
| 1 hour | 1.160 | 0.379 | 64.7% |
| 6 hours | 1.161 | 0.915 | 61.1% |
| 24 hours | 1.167 | 0.747 | 72.9% |
| 168 hours | 1.299 | 1.072 | 63.8% |

The tree forecasts improve point error but their intervals under-cover. Regional
and aggregate forecasts are independent and are not yet reconciled. These are
historical benchmark results, not deployment guarantees. Full scores, peak errors,
coherence errors, validation choices and all 129,024 predictions are in
[results/baseline](results/baseline). The run took 31.58 seconds on four CPU
threads, with 346 MiB RSS sampled after horizon fits.

## Data and method

Pinned `tsibbledata` source files provide `OperationalLessIndustrial` demand for
NSW, QLD, SA, TAS and VIC. This is not total national demand. The 2012–2014 window
contains 26,304 complete hourly observations per node. `ALL_FIVE` sums the five
recorded series on their original day/period clock; a shared UTC or civil-time
mapping is not verified. No weather inputs are used.

- [Data, attribution and clock conventions](docs/DATA.md)
- [Evaluation protocol and limitations](docs/EVALUATION.md)

Source code is in `src/`, tests in `tests/`, configurations in `configs/`, and
dependency pins in `requirements/`. Code and distributed derived results use
[GPL-3.0-only](LICENSE); upstream data attribution is documented separately.
