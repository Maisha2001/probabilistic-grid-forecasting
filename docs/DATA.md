# Data

The source is the five regional `demand.csv` files under
[`tidyverts/tsibbledata/data-raw/vic_elec`](https://github.com/tidyverts/tsibbledata/tree/ccfcb321bb4f0e12b7287f5a8f752097e35a20a5/data-raw/vic_elec),
pinned to commit `ccfcb321bb4f0e12b7287f5a8f752097e35a20a5`. Rob J Hyndman
[describes their format and provenance](https://robjhyndman.com/hyndsight/electrictsibbles/).
Credit the Australian Energy Market Operator for the original electricity data
and the tsibbledata authors and contributors for this archived distribution.

The pinned repository's [DESCRIPTION](https://github.com/tidyverts/tsibbledata/blob/ccfcb321bb4f0e12b7287f5a8f752097e35a20a5/DESCRIPTION)
declares `GPL-3`. This project retains that provenance and distributes its code
and derived benchmark files under GPL-3.0-only. It does not assert a separate
license for current AEMO website products. Raw files and fitted models are not
bundled; the documented commands reproduce them from pinned public inputs.

## Contract

`OperationalLessIndustrial` is demand in MW. Source `Date` is an Excel day
offset from 1899-12-30; `Period` is an integer from 1 to 48. The raw files have
no explicit timezone column. Preserve their naive clock rather than assuming
UTC or applying daylight-saving corrections. An interval-end label is
`Date + Period × 30 minutes`. Average adjacent half-hours to create one hourly
mean, labeled at the second interval's end. This is a mean power, not an energy
sum. These end labels make the observed interval explicit at prediction time.

Select source days from 2012-01-01 inclusive to 2015-01-01 exclusive. Each
region has 52,608 half-hours, producing 26,304 hours from 2012-01-01 01:00 to
2015-01-01 00:00. This fixed window was selected before test-model evaluation.
No selected values are missing; the broader raw archives include missing rows
outside this window. No interpolation or synthetic replacement is performed.

The hierarchy has five leaves, NSW, QLD, SA, TAS and VIC, and one derived root,
`ALL_FIVE`. DuckDB creates the root as their sum at each source-clock label.
This supports a reproducible algebraic hierarchy. It does **not** establish
that all regional records represent the same UTC instant, or that their sum
is official national or NEM total demand. Physical regional reconciliation
and weather matching require independently verified timestamp metadata.

## Validation and files

Downloads must match five SHA-256 checksums in `src/grid_forecast/data.py`.
Preparation checks columns, integer days and periods, the complete requested
half-hour grid, nonnegative finite demand, hourly completeness, node identity
and exact aggregate consistency within floating-point tolerance. Identical
duplicate records are removed and counted; conflicting duplicates fail.

Prepared data is `demand.parquet`, with a manifest containing source hashes,
audit counts, hierarchy and the output checksum. Training and prediction verify
that checksum. Data is stored as long rows: `timestamp`, `node`, `demand_mw`.

The data is old, geographically limited and excludes parts of underlying demand.
It is suitable for a bounded forecasting benchmark; it does not validate modern
grid operations, new regions, unusual future events or source publication delays.
