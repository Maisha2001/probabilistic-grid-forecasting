import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from grid_forecast.data import NODES
from grid_forecast.experiments import verify
from grid_forecast.io import file_hash, read_json, write_json
from grid_forecast.probabilistic import saved_prediction


def export(dataset, run, output):
    run, output = Path(run), Path(output)
    inventory = verify(dataset, run)
    if output.exists():
        raise ValueError("Choose a new evidence directory")
    report = read_json(run/"fit/metrics.json")
    predictions = pd.read_parquet(run/"fit/predictions.parquet")
    diagnostics = read_json(run/"audit/diagnostics.json")
    stress = read_json(run/"audit/stress.json")
    stress_rows = pd.read_parquet(run/"audit/stress.parquet")
    metrics = pd.DataFrame(report["metrics"])
    joint = pd.DataFrame(report["joint_metrics"])
    stress_metrics = pd.DataFrame(stress["metrics"])
    checked = 0
    for record in report["metrics"]:
        mask = np.ones(len(predictions), dtype=bool)
        for key in ["fold", "family", "seed", "method", "horizon", "node"]:
            mask &= predictions[key].to_numpy() == record[key]
        frame = predictions.loc[mask]
        mae = abs(frame.actual-frame.point).mean()
        covered = ((frame.actual >= frame.lower)&(frame.actual <= frame.upper)).mean()
        np.testing.assert_allclose([mae, covered, mae/report["mase_scales"][record["fold"]][record["node"]]],
                                   [record["mae_mw"], record["coverage_80"], record["mase"]], atol=1e-10)
        checked += 1
    calibrated = predictions[predictions.method == "calibrated"]
    coherence = 0.
    for column in ["lower", "point", "upper"]:
        wide = calibrated.pivot(index=["fold", "family", "seed", "horizon", "origin"], columns="node", values=column)
        coherence = max(coherence, float(abs(wide.ALL_FIVE-wide[list(NODES[:5])].sum(axis=1)).max()))
    if coherence > 1e-8:
        raise ValueError("Forecasts fail hierarchy verification")
    clean = stress_rows[stress_rows.scenario == "clean"]
    joined = clean.merge(calibrated, on=["fold", "family", "seed", "horizon", "origin", "node"], suffixes=("_stress", "_fit"), validate="one_to_one")
    assert len(joined) == len(clean)
    clean_difference = 0.
    for column in ["actual", "lower", "point", "upper"]:
        a, b = joined[column+"_stress"], joined[column+"_fit"]
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=.02)
        clean_difference = max(clean_difference, float(abs(a-b).max()))
    cpu_count, cpu_error = 0, 0.
    for (fold, family, seed), group in calibrated.groupby(["fold", "family", "seed"]):
        path = run/"fit/models"/fold/("tree" if family == "tree" else f"transformer-{seed}")
        for origin in [group.origin.min(), group.origin.max()]:
            restored = saved_prediction(dataset, path, str(origin), family, "cpu")
            for h, result in restored["forecasts"].items():
                expected = group[(group.origin == origin)&(group.horizon == int(h))].set_index("node")
                for node in NODES:
                    actual = np.array(result["nodes"][node])
                    reference = expected.loc[node, ["lower", "point", "upper"]].to_numpy(dtype=float)
                    np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=.02)
                    cpu_error = max(cpu_error, float(abs(actual-reference).max())); cpu_count += 1
    validation = {"experiment_version": inventory["version"], "verified_inventory_files": len(inventory["files"]),
                  "independently_recomputed_metric_groups": checked, "max_coherence_error_mw": coherence,
                  "clean_stress_forecasts_matched": len(clean), "clean_stress_max_difference_mw": clean_difference,
                  "cpu_reloaded_node_forecasts": cpu_count, "cpu_max_difference_mw": cpu_error,
                  "forecast_tolerance": {"rtol": 2e-5, "atol_mw": .02}, "verifier_sha256": file_hash(__file__)}
    output.mkdir(parents=True)
    for path, name in [("fit/metrics.json", "metrics.json"), ("fit/predictions.parquet", "predictions.parquet"),
                       ("audit/diagnostics.json", "diagnostics.json"), ("audit/stress.json", "stress.json"),
                       ("audit/stress.parquet", "stress.parquet"), ("protocol.json", "protocol.json"), ("seal.json", "seal.json")]:
        shutil.copyfile(run/path, output/name)
    write_json(output/"validation.json", validation)
    tables = [
        ("Forward forecasts", metrics.groupby(["family", "method", "horizon"])[["mase", "coverage_80", "width_mw"]].mean().reset_index()),
        ("All five regions covered together", joint.groupby(["family", "method", "horizon"])[["simultaneous_leaf_coverage"]].mean().reset_index()),
        ("Observation failures: matched daily origins", stress_metrics.groupby(["family", "scenario"])[["mase", "simultaneous_leaf_coverage", "width_mw"]].mean().reset_index()),
        ("Peak slices: thresholds fixed from training", pd.DataFrame(diagnostics["slices"]).groupby(["family", "slice"])[["mae_mw", "coverage"]].mean().reset_index()),
        ("Demand shift: training versus evaluation origins", pd.DataFrame(diagnostics["shifts"]))]
    html = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Forward evaluation and robustness</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px}td,th{padding:8px;text-align:right}p{line-height:1.6}table{border-collapse:collapse}tr{border-bottom:1px solid #ddd}</style><h1>Forward evaluation and robustness</h1><p>Two 21-day origin windows in January and February 2015, with unchanged modeling choices. Tables average folds, horizons, seeds within a family and six nodes equally where present. Calibrated bands target simultaneous regional coverage, not marginal quantiles. Stress tests change readings only; labels are unchanged.</p>'
    for title, table in tables:
        html += f"<h2>{title}</h2>"+table.round(4).to_html(index=False, border=0, escape=True)
    html += '<p>Peak-slice tables omit empty groups; detailed counts and daily joint coverage are in <a href="diagnostics.json">diagnostics</a>. Daily midnight stress origins are a limited sample. PSI is descriptive and confounded by seasonal changes. No formal coverage or shift-detection guarantee is claimed. See the <a href="../../docs/EXPERIMENTS.md">protocol and reproduction guide</a>.</p></html>'
    (output/"report.html").write_text(html, encoding="utf-8")
    print(validation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.data, args.run, args.output)
