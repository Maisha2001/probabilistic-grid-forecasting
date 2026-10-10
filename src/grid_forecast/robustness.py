from pathlib import Path
import time

import numpy as np
import pandas as pd

from .data import NODES, load_dataset
from .features import training_statistics
from .io import file_hash, read_json, write_json
from .model import load_model, predict_model
from .reconciliation import predict_bands

SCENARIOS = ("clean", "missing_6h", "missing_24h", "missing_10pct", "bias_up_10pct_24h")


def causal_history(values, origin, max_gap=24):
    if not isinstance(origin, (int, np.integer)) or not 336 <= origin < len(values):
        raise ValueError("History needs an available origin and 336 prior hours")
    if not isinstance(max_gap, int) or not 1 <= max_gap <= 24:
        raise ValueError("Maximum missing run must be 1..24 hours")
    leaves = np.asarray(values[:origin+1, :5], dtype=float).copy()
    if leaves.shape[1] != 5 or np.isinf(leaves).any() or (leaves < 0).any():
        raise ValueError("Expected five nonnegative leaf histories, with NaN for missing values")
    missing = np.isnan(leaves)
    counts = missing.sum(axis=0)
    for node in range(5):
        gap = 0
        for row in range(len(leaves)):
            if not missing[row, node]:
                gap = 0
                continue
            gap += 1
            if gap > max_gap or row == 0:
                raise ValueError("Missing history exceeds causal repair policy")
            source = row-168 if row >= 168 else row-1
            leaves[row, node] = leaves[source, node]
    # An independently retained aggregate would reveal hidden regional readings.
    return np.column_stack([leaves, leaves.sum(axis=1)]), dict(zip(NODES[:5], counts.tolist()))


def observed_scenario(values, origin, scenario):
    if scenario not in SCENARIOS:
        raise ValueError("Unknown observation scenario")
    observed = values[:origin+1].copy()
    if scenario.startswith("missing_") and scenario.endswith("h"):
        hours = int(scenario.split("_")[1][:-1])
        observed[-hours:, :5] = np.nan
    elif scenario == "missing_10pct":
        mask = np.random.default_rng(42+int(origin)).random((168, 5)) < .1
        observed[-168:, :5] = np.where(mask, np.nan, observed[-168:, :5])
    elif scenario == "bias_up_10pct_24h":
        observed[-24:, :5] *= 1.1
    return causal_history(observed, origin)


def shift_summary(training, test):
    edges = np.unique(np.quantile(training, np.linspace(0, 1, 11)))
    if len(edges) < 3:
        raise ValueError("Shift diagnostics need nonconstant training demand")
    edges[0], edges[-1] = -np.inf, np.inf
    reference = np.histogram(training, edges)[0].astype(float) + .5
    observed = np.histogram(test, edges)[0].astype(float) + .5
    reference /= reference.sum(); observed /= observed.sum()
    return {"mean_change_fraction": float(test.mean()/training.mean()-1),
            "population_stability_index": float(np.sum((observed-reference)*np.log(observed/reference))),
            "training_peak_threshold_mw": float(np.quantile(training, .95))}


def diagnose(dataset, run, output):
    index, values, manifest = load_dataset(dataset)
    run, output = Path(run), Path(output)
    report = read_json(run/"metrics.json")
    if manifest["parquet_sha256"] != report["data_sha256"] or file_hash(run/"predictions.parquet") != report["predictions_sha256"]:
        raise ValueError("Diagnostic inputs do not match the measured run")
    rows = pd.read_parquet(run/"predictions.parquet")
    config = report["config"]; shifts, slices, calibration = [], [], []
    for label in config["folds"]:
        fold = label[:10]; boundary = index.get_loc(pd.Timestamp(label))
        cutoff = boundary-config["validation_days"]*24
        train = values[max(0, cutoff-config["training_hours"]):cutoff]
        test = values[boundary:boundary+config["test_days"]*24]
        thresholds = np.quantile(train, .95, axis=0)
        for i, node in enumerate(NODES):
            shifts.append({"fold": fold, "node": node, **shift_summary(train[:, i], test[:, i])})
        selected = rows[(rows.fold == fold) & (rows.method == "calibrated")]
        for (family, seed, horizon, node), frame in selected.groupby(["family", "seed", "horizon", "node"]):
            peak = frame.actual.to_numpy() > thresholds[list(NODES).index(node)]
            for name, mask in [("all", np.ones(len(frame), bool)), ("training_95pct_peak", peak), ("nonpeak", ~peak)]:
                sample = frame.loc[mask]
                slices.append({"fold": fold, "family": family, "seed": int(seed), "horizon": int(horizon), "node": node,
                               "slice": name, "count": len(sample),
                               "mae_mw": float(abs(sample.actual-sample.point).mean()) if len(sample) else None,
                               "coverage": float(((sample.actual >= sample.lower)&(sample.actual <= sample.upper)).mean()) if len(sample) else None})
        for path in sorted((run/"models"/fold).glob("*/postprocess.json")):
            family = "tree" if path.parent.name == "tree" else "transformer"
            seed = config["seed"] if family == "tree" else int(path.parent.name.split("-")[-1])
            for horizon, state in read_json(path)["horizons"].items():
                scores = np.asarray(state["calibration_scores"])
                lag = float(np.corrcoef(scores[:-1], scores[1:])[0, 1]) if np.std(scores[:-1]) > 0 and np.std(scores[1:]) > 0 else None
                match = selected[(selected.family == family)&(selected.seed == seed)&(selected.horizon == int(horizon))&(selected.node != "ALL_FIVE")].copy()
                match["covered"] = (match.actual >= match.lower)&(match.actual <= match.upper)
                daily = match.groupby("origin").covered.all().groupby(lambda x: pd.Timestamp(x).date()).mean()
                calibration.append({"fold": fold, "family": family, "seed": int(seed), "horizon": int(horizon),
                                    "calibration_count": len(scores), "calibration_lag1_correlation": lag,
                                    "calibration_empirical_coverage": float((scores <= state["multiplier"]).mean()),
                                    "test_simultaneous_leaf_coverage": float(match.groupby("origin").covered.all().mean()),
                                    "daily_test_joint_coverage": {str(k): float(v) for k, v in daily.items()}})
    result = {"data_sha256": manifest["parquet_sha256"], "prediction_sha256": report["predictions_sha256"],
              "shifts": shifts, "slices": slices, "calibration": calibration,
              "interpretation": "Descriptive diagnostics, not model selection. Peak thresholds and PSI bins use the fitting window only. Calibration observations and rolling forecast errors are temporally dependent; daily coverage is descriptive, not a confidence interval."}
    write_json(output/"diagnostics.json", result)
    return result


def stress_test(dataset, run, output, stride=24, device="cpu"):
    if not isinstance(stride, int) or stride < 1:
        raise ValueError("Stress-test stride must be positive")
    from .temporal import load_temporal, predict_temporal
    index, values, manifest = load_dataset(dataset)
    run, output = Path(run), Path(output); report = read_json(run/"metrics.json")
    if manifest["parquet_sha256"] != report["data_sha256"]:
        raise ValueError("Stress data do not match fitted run")
    config = report["config"]; records = []; start = time.perf_counter(); scales = {}
    for label in config["folds"]:
        fold = label[:10]; boundary = index.get_loc(pd.Timestamp(label))
        cutoff = boundary-config["validation_days"]*24
        _, mase, _ = training_statistics(values, cutoff, config["training_hours"])
        scales[fold] = dict(zip(NODES, mase.tolist()))
        families = [("tree", config["seed"], run/"models"/fold/"tree")]
        families += [("transformer", s, run/"models"/fold/f"transformer-{s}") for s in config["seeds"]]
        for family, seed, path in families:
            states = read_json(path/"postprocess.json")["horizons"]
            if family == "tree":
                models = {h: load_model(path/f"h{h}") for h in config["horizons"]}
            else:
                model, metadata = load_temporal(path, device)
            for origin in range(boundary, boundary+config["test_days"]*24, stride):
                for scenario in SCENARIOS:
                    observed, repaired = observed_scenario(values, origin, scenario)
                    if family == "transformer":
                        raw = predict_temporal(model, metadata, index[:origin+1], observed, [origin], device)
                    for i, horizon in enumerate(config["horizons"]):
                        prediction = predict_model(*models[horizon], index[:origin+1], observed, [origin])[0] if family == "tree" else raw[:, i]
                        forecast = predict_bands(prediction, states[str(horizon)])[0]
                        for n, node in enumerate(NODES):
                            records.append({"fold": fold, "origin": index[origin], "family": family, "seed": seed,
                                            "horizon": horizon, "node": node, "scenario": scenario,
                                            "actual": values[origin+horizon, n], "lower": forecast[n, 0],
                                            "point": forecast[n, 1], "upper": forecast[n, 2],
                                            "repaired_leaf_entries": sum(repaired.values())})
            print(f"{fold}: {family} {seed} observation stress complete", flush=True)
    rows = pd.DataFrame(records); output.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(output/"stress.parquet", index=False)
    metrics = []
    for keys, group in rows.groupby(["fold", "family", "seed", "horizon", "scenario"]):
        fold, family, seed, horizon, scenario = keys
        group = group.copy(); group["covered"] = (group.actual >= group.lower)&(group.actual <= group.upper)
        joint = group[group.node != "ALL_FIVE"].groupby("origin").covered.all().mean()
        mase = group.apply(lambda r: abs(r.actual-r.point)/scales[fold][r.node], axis=1).mean()
        metrics.append({"fold": fold, "family": family, "seed": int(seed), "horizon": int(horizon), "scenario": scenario,
                        "mase": float(mase), "simultaneous_leaf_coverage": float(joint),
                        "width_mw": float((group.upper-group.lower).mean()), "origins": group.origin.nunique()})
    result = {"data_sha256": manifest["parquet_sha256"], "stride": stride, "scenarios": list(SCENARIOS), "metrics": metrics,
              "rows": len(rows), "predictions_sha256": file_hash(output/"stress.parquet"), "seconds": time.perf_counter()-start,
              "policy": "Input perturbations only; true future labels unchanged. Causal weekly repair of missing readings, at most24 consecutive hours per leaf. Aggregate rebuilt from observed/repaired leaves. Clean and perturbed forecasts use identical daily origins and fixed fitted/calibrated models."}
    write_json(output/"stress.json", result)
    return result
