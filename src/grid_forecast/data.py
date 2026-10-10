from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import duckdb
import numpy as np
import pandas as pd

from .io import file_hash, read_json, write_json

SOURCE_REF = "ccfcb321bb4f0e12b7287f5a8f752097e35a20a5"
STATES = ("NSW", "QLD", "SA", "TAS", "VIC")
NODES = (*STATES, "ALL_FIVE")
HASHES = {
    "NSW": "de5da1dc68210c690ebe2e2382172727c0800b8b7efb463146eaa3625ea200f7",
    "QLD": "444bccd10643cdd76e03c1d32e037d534f02549fd776173bb23a1a8b1bbdc777",
    "SA": "145cff8fc5ebf2603a16fd14c6119e793155cefbdf66ed1595a0fdae279ee60e",
    "TAS": "1f911ed71310bbbca6b31ab3ede7c6d1779e6b7ebf76cb0c17a014950a9ffaf6",
    "VIC": "bbc2c0f75636cb95e974438962fd858d5ce23e533f80e292c1b6b95f22c5aa37",
}


def source_url(state):
    if state not in STATES:
        raise ValueError("Unknown state")
    return f"https://raw.githubusercontent.com/tidyverts/tsibbledata/{SOURCE_REF}/data-raw/vic_elec/{state}2015/demand.csv"


def fetch(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for state in STATES:
        target = output / f"{state}.csv"
        if target.exists():
            if file_hash(target) != HASHES[state]:
                raise ValueError(f"Existing {state} file fails checksum; choose a clean directory")
            continue
        temporary = target.with_suffix(".part")
        with urlopen(source_url(state), timeout=60) as source, temporary.open("wb") as dest:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                dest.write(block)
        if file_hash(temporary) != HASHES[state]:
            raise ValueError(f"Downloaded {state} file fails checksum")
        temporary.replace(target)


def clean_state(frame, state, start, end):
    required = ["Date", "Period", "OperationalLessIndustrial"]
    if not set(required).issubset(frame):
        raise ValueError("Missing source columns")
    data = frame[required].copy()
    for key in ("Date", "Period"):
        if not np.isfinite(data[key]).all() or not (data[key] == data[key].round()).all():
            raise ValueError("Invalid day or period")
    if not data.Period.between(1, 48).all():
        raise ValueError("Periods must be 1..48")
    duplicate_count = int(data.duplicated().sum())
    data = data.drop_duplicates()
    if data.duplicated(["Date", "Period"]).any():
        raise ValueError("Conflicting duplicate periods")
    days = pd.to_datetime(data.Date, unit="D", origin="1899-12-30")
    selected = data.loc[(days >= pd.Timestamp(start)) & (days < pd.Timestamp(end))].copy()
    selected["timestamp"] = days.loc[selected.index] + pd.to_timedelta(selected.Period * 30, unit="m")
    selected = selected.sort_values("timestamp")
    expected = pd.date_range(pd.Timestamp(start) + pd.Timedelta(minutes=30), end, freq="30min")
    if not pd.DatetimeIndex(selected.timestamp).equals(expected):
        raise ValueError(f"{state}: missing or unexpected half-hour intervals")
    values = selected.OperationalLessIndustrial.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{state}: invalid demand in selected period")
    hourly = pd.DataFrame({"timestamp": expected[1::2], "node": state,
                           "demand_mw": values.reshape(-1, 2).mean(axis=1)})
    audit = {"source_rows": len(frame), "identical_duplicates_removed": duplicate_count,
             "selected_half_hours": len(selected), "hourly_rows": len(hourly),
             "source_missing_demand_rows": int(frame.OperationalLessIndustrial.isna().sum())}
    return hourly, audit


def assemble(frames, start, end):
    if set(frames) != set(STATES):
        raise ValueError("Exactly five named state series are required")
    rows, audit = [], {}
    for state in STATES:
        hourly, audit[state] = clean_state(frames[state], state, start, end)
        rows.append(hourly)
    regional = pd.concat(rows, ignore_index=True)
    with duckdb.connect() as db:
        db.register("regional", regional)
        all_nodes = db.sql("""
            SELECT timestamp, node, demand_mw FROM regional
            UNION ALL
            SELECT timestamp, 'ALL_FIVE' AS node, sum(demand_mw) AS demand_mw
            FROM regional GROUP BY timestamp
            ORDER BY timestamp, node
        """).df()
    matrix_from_frame(all_nodes)
    return all_nodes, audit


def matrix_from_frame(frame):
    if set(frame.node) != set(NODES) or frame.duplicated(["timestamp", "node"]).any():
        raise ValueError("Invalid hierarchy nodes or duplicate timestamps")
    wide = frame.pivot(index="timestamp", columns="node", values="demand_mw").sort_index()
    index = pd.DatetimeIndex(wide.index)
    if len(index) < 2 or index.tz is not None or not index.equals(pd.date_range(index[0], index[-1], freq="h")):
        raise ValueError("Expected regular, timezone-naive hourly source labels")
    values = wide.loc[:, NODES].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Demand must be complete, finite and nonnegative")
    if not np.allclose(values[:, -1], values[:, :-1].sum(axis=1), rtol=1e-12, atol=1e-8):
        raise ValueError("Aggregate does not equal the five regional series")
    return index, values


def prepare(raw, output, start="2012-01-01", end="2015-01-01"):
    raw, output = Path(raw), Path(output)
    if output.exists():
        raise ValueError("Output directory exists; choose a new path")
    for state in STATES:
        if file_hash(raw / f"{state}.csv") != HASHES[state]:
            raise ValueError(f"Source checksum mismatch: {state}")
    frames = {state: pd.read_csv(raw / f"{state}.csv") for state in STATES}
    frame, audit = assemble(frames, start, end)
    output.mkdir(parents=True)
    frame.to_parquet(output / "demand.parquet", index=False)
    write_json(output / "manifest.json", {
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(), "source_ref": SOURCE_REF,
        "source_sha256": HASHES, "source_license": "GPL-3, upstream tsibbledata DESCRIPTION",
        "parquet_sha256": file_hash(output / "demand.parquet"), "nodes": list(NODES),
        "hierarchy": {"ALL_FIVE": list(STATES)}, "start_day_inclusive": start, "end_day_exclusive": end,
        "timestamp_convention": "Interval-end labels on the original date/period clock; no UTC or civil-time conversion.",
        "target": "Hourly mean OperationalLessIndustrial (MW); ALL_FIVE is its derived five-series sum.",
        "audit": audit, "rows": len(frame), "missing_value_policy": "Reject; no interpolation or imputation."})


def load_dataset(path):
    path = Path(path)
    manifest = read_json(path / "manifest.json")
    if file_hash(path / "demand.parquet") != manifest["parquet_sha256"]:
        raise ValueError("Prepared data checksum mismatch")
    index, values = matrix_from_frame(pd.read_parquet(path / "demand.parquet"))
    return index, values, manifest
