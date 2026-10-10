from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import pandas as pd
import psutil
import torch

from .data import NODES, load_dataset
from .evaluation import prediction_rows, validate_config
from .features import pinball, training_statistics
from .io import file_hash, read_json, write_json
from .model import fit_model, predict_model, load_model
from .reconciliation import fit_projection, fit_bands, predict_bands, reconcile
from .temporal import fit_temporal, predict_temporal, load_temporal


def split_origins(boundary, config):
    cutoff = boundary-config["validation_days"]*24
    middle = boundary-config["calibration_days"]*24
    selection = np.arange(cutoff, middle-max(config["horizons"]), config["training_stride"])
    calibration = {h: np.arange(middle, boundary-h) for h in config["horizons"]}
    if len(selection) < 10 or any(len(x) < 10 for x in calibration.values()):
        raise ValueError("Selection/calibration blocks are too short")
    return cutoff, selection, calibration


def validate_prob_config(config):
    validate_config(config)
    n = config["neural"]
    if not 0 < config["calibration_days"] < config["validation_days"] or not 0 < config["coverage"] < 1:
        raise ValueError("Invalid calibration configuration")
    if not config["seeds"] or len(config["seeds"]) != len(set(config["seeds"])):
        raise ValueError("Seeds must be nonempty and distinct")
    for k in ["history_hours", "patch_hours", "d_model", "heads", "layers", "epochs", "batch_size"]:
        if not isinstance(n[k], int) or n[k] <= 0: raise ValueError(f"Invalid neural {k}")
    if n["history_hours"] % n["patch_hours"] or n["d_model"] % n["heads"]:
        raise ValueError("Patch and attention dimensions must divide evenly")
    split_origins(20000, config)


def fit_tree(index, values, cutoff, selection, horizon, boundary, config, output):
    best, choices = None, []
    for candidate in config["candidates"]:
        model, meta = fit_model(index, values, cutoff, horizon, candidate, config)
        forecast, _ = predict_model(model, meta, index, values, selection)
        scale = np.asarray(meta["scale"])
        loss = float(pinball(values[selection+horizon]/scale, forecast/scale[None, :, None]).mean())
        choices.append({"parameters": candidate, "selection_pinball": loss})
        if best is None or loss < best[0]: best = (loss, model, meta, candidate)
    _, model, meta, candidate = best
    output.mkdir(parents=True, exist_ok=True); model.save_model(output/"model.ubj")
    meta.update(model_sha256=file_hash(output/"model.ubj"), feature_schema="lags-calendar-node-v1",
                forecast_origin_not_before=str(index[boundary]), selected_parameters=candidate,
                last_training_timestamp=str(index[meta["last_training_target"]]), selection=choices)
    write_json(output/"model.json", meta)
    return model, meta


def evaluate_bands(rows, scales):
    metrics, joint = [], []
    keys = ["fold", "family", "seed", "method", "horizon"]
    for group, block in rows.groupby(keys, sort=True):
        metadata = dict(zip(keys, group))
        metadata = {k: int(v) if k in ["seed", "horizon"] else v for k, v in metadata.items()}
        leaves = block[block.node != "ALL_FIVE"].copy()
        leaves["covered"] = (leaves.actual >= leaves.lower) & (leaves.actual <= leaves.upper)
        pivot = block.pivot(index="origin", columns="node", values="point")
        error = abs(pivot.ALL_FIVE-pivot[list(NODES[:5])].sum(axis=1))
        joint.append({**metadata, "simultaneous_leaf_coverage": float(leaves.groupby("origin").covered.all().mean()),
                      "point_coherence_mean_mw": float(error.mean()), "point_coherence_max_mw": float(error.max())})
        for node, frame in block.groupby("node", sort=True):
            y, point, lower, upper = (frame[c].to_numpy() for c in ["actual", "point", "lower", "upper"])
            width = upper-lower; score = width+10*np.maximum(lower-y, 0)+10*np.maximum(y-upper, 0)
            loss = float(pinball(y, np.column_stack([lower, point, upper])).mean()) if metadata["method"] == "raw" else None
            metrics.append({**metadata, "node": node, "count": len(y), "mae_mw": float(abs(y-point).mean()),
                            "mase": float(abs(y-point).mean()/scales[metadata["fold"]][node]), "raw_quantile_pinball_mw": loss,
                            "coverage_80": float(((y >= lower)&(y <= upper)).mean()), "width_mw": float(width.mean()),
                            "interval_score_80_mw": float(score.mean()), "peak_error_mw": float(abs(y[np.argmax(y)]-point[np.argmax(y)]))})
    return metrics, joint


def postprocess(index, values, selection, calibration, test_origins, horizon, predict, scale, coverage):
    selected = predict(selection)
    projection = fit_projection(values[selection+horizon]-selected[..., 1])
    cal = predict(calibration)
    state = fit_bands(values[calibration+horizon], cal, projection, scale, coverage)
    state.update(last_selection_target=str(index[selection[-1]+horizon]),
                 calibration_origin_start=str(index[calibration[0]]),
                 last_calibration_target=str(index[calibration[-1]+horizon]))
    raw = predict(test_origins)
    point = reconcile(raw[..., 1], state)
    shifted = np.maximum(raw + (point-raw[..., 1])[..., None], 0)
    return state, {"raw": raw, "reconciled": shifted, "calibrated": predict_bands(raw, state)}


def result_rows(fold, family, seed, horizon, index, values, origins, forecasts):
    rows = []
    for method, output in forecasts.items():
        frame = prediction_rows(fold, index, values, origins, horizon, method, output)
        frame = frame.rename(columns={"q10": "lower", "q50": "point", "q90": "upper"})
        frame["family"], frame["seed"] = family, seed
        rows.append(frame)
    return rows


def report_html(report):
    metrics = pd.DataFrame(report["metrics"])
    table = metrics.groupby(["family", "method", "horizon"])[["mase", "coverage_80", "width_mw"]].mean().round(3).reset_index()
    return '<!doctype html><html lang="en"><meta charset="utf-8"><title>Calibrated regional forecasts</title><style>body{font:16px system-ui;max-width:1050px;margin:40px auto;padding:0 20px}td,th{padding:8px;text-align:right}p{line-height:1.6}table{border-collapse:collapse}tr{border-bottom:1px solid #ddd}</style><h1>Calibrated regional forecasts</h1><p>Equal-weight averages over folds, seeds and six nodes. Trees use one deterministic seed; the transformer uses three. Training cutoffs are matched. Calibrated intervals target simultaneous leaf coverage; aggregate limits are envelopes, not marginal quantiles. Coverage under temporal dependence is empirical.</p>' + table.to_html(index=False, border=0, escape=True) + '<p>Point forecasts reconcile exactly before floating-point rounding. Quantile medians and interval endpoints have different meanings across methods; see <a href="../../docs/PROBABILISTIC.md">protocol</a>. Detailed seed/fold scores and calibration evidence are in <a href="metrics.json">metrics.json</a>.</p></html>'


def run_probabilistic(dataset, config, output, device="cpu", resume=False):
    validate_prob_config(config)
    if device not in ["cpu", "cuda"] or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested device is unavailable")
    index, values, manifest = load_dataset(dataset)
    boundaries = index.get_indexer(pd.DatetimeIndex(config["folds"]))
    if (boundaries < 0).any() or (boundaries+config["test_days"]*24+max(config["horizons"]) > len(index)).any():
        raise ValueError("Missing fold data")
    sources = {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    identity = {"config": config, "data_sha256": manifest["parquet_sha256"], "source_sha256": sources, "device": device}
    output = Path(output)
    if output.exists():
        if not resume or read_json(output/"run.json") != identity:
            raise ValueError("Run exists; resume requires identical inputs, source and device")
    else:
        output.mkdir(parents=True); write_json(output/"run.json", identity)
    started = time.perf_counter(); rows, fits, scales = [], [], {}
    if device == "cuda": torch.cuda.reset_peak_memory_stats()
    for label, boundary in zip(config["folds"], boundaries):
        fold = label[:10]; cutoff, selection, calibration = split_origins(int(boundary), config)
        test_origins = np.arange(boundary, boundary+config["test_days"]*24, config["evaluation_stride"])
        scale, mase, _ = training_statistics(values, cutoff, config["training_hours"])
        scales[fold] = dict(zip(NODES, mase.tolist()))
        tree_states = {}
        for horizon in config["horizons"]:
            path = output/"models"/fold/"tree"/f"h{horizon}"
            model, meta = fit_tree(index, values, cutoff, selection, horizon, int(boundary), config, path)
            predict = lambda origins: predict_model(model, meta, index, values, origins)[0]
            state, forecasts = postprocess(index, values, selection, calibration[horizon], test_origins, horizon, predict, scale, config["coverage"])
            tree_states[str(horizon)] = state
            fits.append({"fold": fold, "family": "tree", "horizon": horizon, "metadata": meta})
            rows.extend(result_rows(fold, "tree", config["seed"], horizon, index, values, test_origins, forecasts))
        write_json(output/"models"/fold/"tree"/"postprocess.json", {"forecast_origin_not_before": str(index[boundary]), "horizons": tree_states})
        print(f"{fold}: matched-cutoff trees and calibration complete", flush=True)
        for seed in config["seeds"]:
            path = output/"models"/fold/f"transformer-{seed}"
            model, meta = fit_temporal(index, values, cutoff, selection, config, seed, path, device)
            states = {}
            for i, horizon in enumerate(config["horizons"]):
                predict = lambda origins: predict_temporal(model, meta, index, values, origins, device)[:, i]
                state, forecasts = postprocess(index, values, selection, calibration[horizon], test_origins, horizon, predict, scale, config["coverage"])
                states[str(horizon)] = state
                rows.extend(result_rows(fold, "transformer", seed, horizon, index, values, test_origins, forecasts))
            write_json(path/"postprocess.json", {"forecast_origin_not_before": str(index[boundary]), "horizons": states})
            fits.append({"fold": fold, "family": "transformer", "seed": seed, "metadata": meta})
            print(f"{fold}: transformer seed {seed}, epoch {meta['selected_epoch']}, calibration complete", flush=True)
    predictions = pd.concat(rows, ignore_index=True); predictions.to_parquet(output/"predictions.parquet", index=False)
    metrics, joint = evaluate_bands(predictions, scales)
    artifact_hashes = {p.relative_to(output).as_posix(): file_hash(p) for p in (output/"models").rglob("*") if p.is_file() and p.name != "checkpoint.pt"}
    report = {**identity, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "metrics": metrics, "joint_metrics": joint,
              "fits": fits, "mase_scales": scales, "prediction_rows": len(predictions), "artifact_sha256": artifact_hashes,
              "predictions_sha256": file_hash(output/"predictions.parquet"),
              "compute": {"seconds_this_invocation": time.perf_counter()-started, "device": device, "torch": torch.__version__,
                          "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                          "peak_allocated_gpu_mib": torch.cuda.max_memory_allocated()/2**20 if device == "cuda" else 0,
                          "rss_mib_after_run": psutil.Process().memory_info().rss/2**20},
              "interval_semantics": "Raw: marginal quantiles. Reconciled: shifted uncalibrated bands. Calibrated: simultaneous leaf bands and aggregate envelope; these are not marginal 10/90 quantiles."}
    write_json(output/"metrics.json", report)
    (output/"report.html").write_text(report_html(report), encoding="utf-8")
    return report


def saved_prediction(dataset, model_path, origin, family, device="cpu"):
    index, values, _ = load_dataset(dataset); origin = pd.Timestamp(origin); model_path = Path(model_path)
    state = read_json(model_path/"postprocess.json")
    if origin.tz is not None or origin not in index or origin < pd.Timestamp(state["forecast_origin_not_before"]):
        raise ValueError("Origin must be an available source label after the training/calibration boundary")
    positions = np.array([index.get_loc(origin)]); forecasts = {}
    if family == "transformer":
        model, meta = load_temporal(model_path, device)
        raw = predict_temporal(model, meta, index, values, positions, device)
    for i, horizon in enumerate(state["horizons"]):
        if family == "tree":
            model, meta = load_model(model_path/f"h{horizon}")
            current = predict_model(model, meta, index, values, positions)[0]
        else:
            current = raw[:, meta["horizons"].index(int(horizon))]
        bands = predict_bands(current, state["horizons"][horizon])[0]
        forecasts[horizon] = {"target": str(origin+pd.Timedelta(hours=int(horizon))), "nodes": dict(zip(NODES, bands.tolist()))}
    return {"origin": str(origin), "columns": ["lower", "point", "upper"],
            "interval_semantics": "simultaneous_leaf_bands_with_aggregate_envelope", "forecasts": forecasts}
