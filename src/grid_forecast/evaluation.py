from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import platform
import time

import duckdb
import numpy as np
import pandas as pd
import psutil

from .data import NODES, load_dataset
from .features import ordered_quantiles, pinball
from .io import file_hash, write_json
from .model import predict_model, select_and_fit
from .report import render_report


def validate_config(config):
    if config["quantiles"] != [0.1, 0.5, 0.9]:
        raise ValueError("This baseline uses the declared 10/50/90 quantiles")
    for key in ("test_days", "validation_days", "training_hours", "training_stride", "evaluation_stride", "threads"):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"Invalid {key}")
    horizons = config["horizons"]
    if not horizons or len(set(horizons)) != len(horizons) or any(not isinstance(h, int) or not 1 <= h <= 168 for h in horizons):
        raise ValueError("Distinct horizons must lie in 1..168")
    folds = pd.DatetimeIndex(config["folds"])
    if folds.tz is not None or not folds.is_monotonic_increasing or folds.has_duplicates:
        raise ValueError("Fold origins must be increasing naive source-clock labels")
    if len(folds) > 1 and np.any(np.diff(folds.asi8)/3.6e12 < config["test_days"]*24+max(horizons)):
        raise ValueError("Test target windows overlap between folds")
    if not config["candidates"]:
        raise ValueError("No candidates")


def prediction_rows(fold, index, values, origins, horizon, method, predicted):
    count, nodes = len(origins), len(NODES)
    return pd.DataFrame({"fold": fold, "origin": np.repeat(index[origins].to_numpy(), nodes),
                         "target": np.repeat(index[origins+horizon].to_numpy(), nodes),
                         "horizon": horizon, "node": np.tile(NODES, count), "method": method,
                         "actual": values[origins+horizon].reshape(-1),
                         "q10": predicted[:, :, 0].reshape(-1), "q50": predicted[:, :, 1].reshape(-1),
                         "q90": predicted[:, :, 2].reshape(-1)})


def summarize(predictions, scales):
    records = []
    for (fold, method, horizon, node), frame in predictions.groupby(["fold", "method", "horizon", "node"], sort=True):
        actual = frame.actual.to_numpy()
        quantiles = frame[["q10", "q50", "q90"]].to_numpy()
        error = np.abs(actual-quantiles[:, 1])
        peak = int(np.argmax(actual))
        records.append({"fold": fold, "method": method, "horizon": int(horizon), "node": node, "count": len(frame),
                        "mae_mw": float(error.mean()), "mase": float(error.mean()/scales[fold][node]),
                        "pinball_mw": float(pinball(actual, quantiles).mean()),
                        "coverage_80": float(((actual>=quantiles[:, 0]) & (actual<=quantiles[:, 2])).mean()),
                        "interval_width_mw": float((quantiles[:, 2]-quantiles[:, 0]).mean()),
                        "actual_peak_mw": float(actual[peak]), "error_at_actual_peak_mw": float(error[peak]),
                        "max_forecast_peak_error_mw": float(abs(quantiles[:, 1].max()-actual.max()))})
    with duckdb.connect() as db:
        db.register("predictions", predictions)
        coherence = db.sql("""
            WITH errors AS (
                SELECT fold, method, horizon, origin,
                    abs(sum(CASE WHEN node='ALL_FIVE' THEN q50 ELSE -q50 END)) AS error_mw
                FROM predictions GROUP BY fold, method, horizon, origin
            )
            SELECT fold, method, horizon, avg(error_mw) AS mean_absolute_mw,
                   max(error_mw) AS max_absolute_mw
            FROM errors GROUP BY fold, method, horizon ORDER BY fold, method, horizon
        """).df().to_dict("records")
    return records, coherence


def backtest(dataset, config, output):
    validate_config(config)
    output = Path(output)
    if output.exists():
        raise ValueError("Run directory exists; choose a new path")
    index, values, manifest = load_dataset(dataset)
    fold_positions = index.get_indexer(pd.DatetimeIndex(config["folds"]))
    if (fold_positions < 0).any() or (fold_positions+config["test_days"]*24+max(config["horizons"]) > len(index)).any():
        raise ValueError("Data does not cover requested fold targets")
    started = time.perf_counter()
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    rows, fits, scales = [], [], {}
    peak_rss = psutil.Process().memory_info().rss
    for fold_label, boundary in zip(config["folds"], fold_positions):
        fold = fold_label[:10]
        origins = np.arange(boundary, boundary+config["test_days"]*24, config["evaluation_stride"])
        for horizon in config["horizons"]:
            fit_start = time.perf_counter()
            model, metadata, choices = select_and_fit(index, values, int(boundary), horizon, config,
                                                     output / "models" / fold / f"h{horizon}")
            predicted, crossings = predict_model(model, metadata, index, values, origins)
            seasonal = ordered_quantiles(values[origins+horizon-168, :, None] + np.asarray(metadata["seasonal_residual_quantiles"])[None])
            for method, result in (("xgboost_quantile", predicted), ("weekly_residual", seasonal)):
                rows.append(prediction_rows(fold, index, values, origins, horizon, method, result))
            scales[fold] = dict(zip(NODES, metadata["mase_scale"]))
            fits.append({"fold": fold, "horizon": horizon, "validation": choices, "selected_parameters": metadata["selected_parameters"],
                         "model_sha256": metadata["model_sha256"], "training_rows": metadata["training_rows"],
                         "last_training_timestamp": metadata["last_training_timestamp"], "crossing_rows_before_sort": crossings,
                         "seconds": time.perf_counter()-fit_start})
            peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
            print(f"{fold} h{horizon}: fitted and evaluated", flush=True)
    predictions = pd.concat(rows, ignore_index=True)
    predictions.to_parquet(output / "predictions.parquet", index=False)
    metrics, coherence = summarize(predictions, scales)
    report = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "config": config,
              "data": manifest, "metrics": metrics, "median_coherence": coherence, "fits": fits,
              "mase_scales": scales, "prediction_rows": len(predictions),
              "predictions_sha256": file_hash(output/"predictions.parquet"),
              "source_sha256": {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "packages": {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "xgboost", "duckdb", "pyarrow")}},
              "compute": {"seconds": time.perf_counter()-started, "sampled_peak_rss_mib": peak_rss/2**20,
                          "device": "CPU", "threads": config["threads"], "memory_measurement": "RSS sampled after each horizon; not a continuous peak."},
              "protocol": "Models refit at each fold. Validation and training labels strictly precede fold origins. Observed history updates at each hourly test origin. Quantiles are sorted/nonnegative; independent node forecasts are not reconciled. No future weather or demand features."}
    write_json(output / "metrics.json", report)
    (output / "report.html").write_text(render_report(report), encoding="utf-8")
    return report
