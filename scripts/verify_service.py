import argparse
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import psutil

from grid_forecast.data import NODES, load_dataset
from grid_forecast.io import write_json
from grid_forecast.service import create_app


def verify_service(data, experiment, output):
    saved = pd.read_parquet(experiment/"fit/predictions.parquet")
    saved = saved[saved.method == "calibrated"]
    index, values, _ = load_dataset(data)
    app = create_app(data, experiment)
    started = perf_counter()
    checked, maximum_error, timings = 0, 0., {"tree": [], "transformer": []}
    with TestClient(app) as client:
        startup = perf_counter()-started
        meta = client.get("/api/meta").json()
        for (fold, family, seed), group in saved.groupby(["fold", "family", "seed"]):
            for origin in [group.origin.min(), group.origin.max()]:
                response = client.post("/api/forecast", json={"fold":fold,"family":family,"seed":int(seed),"origin":str(origin)})
                assert response.status_code == 200, response.text
                received = pd.DataFrame(response.json()["forecasts"])
                expected = group[group.origin == origin]
                matched = received.merge(expected, on=["horizon","node"], suffixes=("_api","_saved"))
                assert len(matched) == 24
                for column in ["lower","point","upper","actual"]:
                    observed = matched[column+"_api"].to_numpy()
                    reference = matched[column+"_saved"].to_numpy()
                    np.testing.assert_allclose(observed, reference, rtol=2e-5, atol=.02)
                    maximum_error = max(maximum_error, float(abs(observed-reference).max()))
                for column in ["lower","point","upper"]:
                    wide = received.pivot(index="horizon", columns="node", values=column)
                    np.testing.assert_allclose(wide.ALL_FIVE, wide[list(NODES[:5])].sum(axis=1), atol=1e-8)
                checked += len(matched)
        fold = meta["folds"][0]
        scenarios = []
        for family in meta["models"]:
            seed = meta["models"][family][0]
            for scenario in meta["scenarios"]:
                body = {"fold":fold,"family":family,"seed":seed,"origin":fold,"scenario":scenario}
                result = client.post("/api/forecast", json=body)
                assert result.status_code == 200, result.text
                payload = result.json()
                assert all(row["lower"] <= row["point"] <= row["upper"] for row in payload["forecasts"])
                scenarios.append({"family":family,"scenario":scenario,"repaired_readings":sum(payload["repaired_counts"].values())})
            for _ in range(20):
                tick = perf_counter()
                result = client.post("/api/forecast", json={"fold":fold,"family":family,"seed":seed,"origin":fold})
                assert result.status_code == 200
                timings[family].append((perf_counter()-tick)*1000)
        p = index.get_loc(pd.Timestamp(fold))
        response = client.post("/api/forecast", json={"fold":fold,"origin":fold,"history":values[p-336:p+1,:5].tolist()})
        assert response.status_code == 200 and all(row["actual"] is None for row in response.json()["forecasts"])
        report = {"checked_at_utc":datetime.now(timezone.utc).isoformat(), "experiment_version":meta["version"],
                  "forecast_rows_compared":checked, "maximum_absolute_difference_mw":maximum_error,
                  "comparison_tolerance":{"relative":2e-5,"absolute_mw":.02}, "startup_seconds":startup,
                  "process_rss_mib":psutil.Process().memory_info().rss/2**20,
                  "device":"cpu", "threads":app.state.store.config["threads"],
                  "latency_method":"20 sequential warm ASGI requests per family; four horizons and six nodes per request, including response serialization. No network latency.",
                  "latency_ms":{k:{"p50":float(np.median(v)),"p95":float(np.quantile(v,.95))} for k,v in timings.items()},
                  "scenarios":scenarios, "custom_history_actuals_suppressed":True}
    write_json(output, report)
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check a real sealed experiment through the forecasting API")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_service(args.data, args.experiment, args.output)
