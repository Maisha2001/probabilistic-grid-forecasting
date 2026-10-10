import argparse
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
import torch

from .data import NODES, load_dataset
from .experiments import verify
from .io import file_hash, read_json
from .model import load_model, predict_model
from .reconciliation import predict_bands
from .robustness import SCENARIOS, causal_history, observed_scenario
from .temporal import load_temporal, predict_temporal

Reading = Annotated[float, Field(ge=0, le=1_000_000, allow_inf_nan=False)]
Observation = Annotated[list[Reading | None], Field(min_length=5, max_length=5)]


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fold: str
    family: Literal["tree", "transformer"] = "tree"
    seed: int = 42
    origin: str
    scenario: Literal["clean", "missing_6h", "missing_24h", "missing_10pct", "bias_up_10pct_24h"] = "clean"
    history: Annotated[list[Observation] | None, Field(min_length=337, max_length=337)] = None


def source_label(value):
    try:
        result = pd.Timestamp(value)
    except (ValueError, TypeError) as error:
        raise ValueError("Use an hourly source-clock timestamp") from error
    if pd.isna(result) or result.tz is not None or result != result.floor("h"):
        raise ValueError("Use a timezone-naive hourly source-clock timestamp")
    return result


class ForecastStore:
    def __init__(self, dataset, experiment):
        self.dataset, self.experiment = Path(dataset), Path(experiment)
        self.inventory = verify(dataset, experiment)
        self.index, self.values, self.manifest = load_dataset(dataset)
        self.report = read_json(self.experiment/"fit/metrics.json")
        self.config = self.report["config"]
        for name in ["data.py", "features.py", "model.py", "temporal.py", "reconciliation.py", "robustness.py", "io.py"]:
            if file_hash(Path(__file__).parent/name) != self.report["source_sha256"][name]:
                raise ValueError(f"Inference source differs from fitted experiment: {name}")
        self.rows = pd.read_parquet(self.experiment/"fit/predictions.parquet")
        self.diagnostics = read_json(self.experiment/"audit/diagnostics.json")
        self.stress = read_json(self.experiment/"audit/stress.json")
        self.lock = Lock(); self.models = {}; self.states = {}
        torch.set_num_threads(self.config["threads"])
        for fold in [x[:10] for x in self.config["folds"]]:
            for family, seed in [("tree", self.config["seed"])]+[("transformer", s) for s in self.config["seeds"]]:
                key = (fold, family, seed)
                path = self.experiment/"fit/models"/fold/("tree" if family == "tree" else f"transformer-{seed}")
                self.states[key] = read_json(path/"postprocess.json")
                if family == "tree":
                    self.models[key] = {h: load_model(path/f"h{h}") for h in self.config["horizons"]}
                    for model, _ in self.models[key].values(): model.set_params(n_jobs=self.config["threads"])
                else:
                    self.models[key] = load_temporal(path, "cpu")

    def key(self, fold, family, seed):
        key = (fold, family, seed)
        if key not in self.models:
            raise ValueError("Choose an available fold, model and seed")
        return key

    def metadata(self):
        metrics = pd.DataFrame(self.report["metrics"])
        joint = pd.DataFrame(self.report["joint_metrics"])
        summary = metrics.groupby(["family", "method"])[["mase", "coverage_80", "width_mw"]].mean().reset_index()
        summary = summary.merge(joint.groupby(["family", "method"]).simultaneous_leaf_coverage.mean().reset_index())
        return {"version": self.inventory["version"], "nodes": list(NODES), "horizons": self.config["horizons"],
                "folds": [f[:10] for f in self.config["folds"]], "test_days": self.config["test_days"],
                "models": {"tree": [self.config["seed"]], "transformer": self.config["seeds"]},
                "scenarios": list(SCENARIOS), "first_observation": str(self.index[0]), "last_observation": str(self.index[-1]),
                "summary": summary.to_dict(orient="records"),
                "clock": "Original source-clock labels; timezone and physical UTC alignment unverified.",
                "intervals": "Nominal 80% simultaneous regional bands; aggregate bounds are envelopes, not marginal quantiles.",
                "limitations": ["Historical 2015 benchmark, not a live grid feed or operational advice.",
                                "Trees under-cover; wider transformer bands trade precision for coverage.",
                                "Missing or biased readings can sharply reduce coverage. Repair is not a reliability guarantee.",
                                "ALL_FIVE is the derived sum of five archived series, not official national demand."]}

    def backtest(self, fold, family, seed, node, horizon):
        self.key(fold, family, seed)
        if node not in NODES or horizon not in self.config["horizons"]:
            raise ValueError("Choose an available region and horizon")
        rows = self.rows[(self.rows.fold == fold)&(self.rows.family == family)&(self.rows.seed == seed)&
                         (self.rows.node == node)&(self.rows.horizon == horizon)&(self.rows.method == "calibrated")].sort_values("target")
        peak = next(x["training_peak_threshold_mw"] for x in self.diagnostics["shifts"] if x["fold"] == fold and x["node"] == node)
        metrics = next(x for x in self.report["metrics"] if all(x[k] == v for k, v in
                       {"fold": fold, "family": family, "seed": seed, "node": node, "horizon": horizon, "method": "calibrated"}.items()))
        joint = next(x for x in self.report["joint_metrics"] if all(x[k] == v for k, v in
                     {"fold": fold, "family": family, "seed": seed, "horizon": horizon, "method": "calibrated"}.items()))
        result = rows[["origin", "target", "actual", "lower", "point", "upper"]].copy()
        result["origin"] = result.origin.astype(str); result["target"] = result.target.astype(str)
        return {"version": self.inventory["version"], "rows": result.to_dict(orient="records"), "metrics": metrics,
                "joint_coverage": joint["simultaneous_leaf_coverage"], "peak_threshold": peak,
                "mase_scale": self.report["mase_scales"][fold][node]}

    def forecast(self, request):
        key = self.key(request.fold, request.family, request.seed)
        origin = source_label(request.origin)
        boundary = source_label(self.states[key]["forecast_origin_not_before"])
        if origin < boundary:
            raise ValueError("Origin precedes this model's training and calibration boundary")
        custom = request.history is not None
        if custom:
            if request.scenario != "clean":
                raise ValueError("Custom history cannot also use a replay perturbation")
            index = pd.date_range(origin-pd.Timedelta(hours=336), origin, freq="h")
            values, repaired = causal_history(np.asarray(request.history, dtype=float), 336)
            position = 336
        else:
            if origin not in self.index:
                raise ValueError("Replay origin is outside the observed archive; supply custom history for a new origin")
            position = self.index.get_loc(origin); index = self.index[:position+1]
            values, repaired = observed_scenario(self.values, position, request.scenario)
        forecasts = []
        with self.lock, torch.inference_mode():
            if request.family == "transformer":
                raw = predict_temporal(*self.models[key], index, values, [position], "cpu")
            for i, horizon in enumerate(self.config["horizons"]):
                current = predict_model(*self.models[key][horizon], index, values, [position])[0] if request.family == "tree" else raw[:, i]
                bands = predict_bands(current, self.states[key]["horizons"][str(horizon)])[0]
                target = origin+pd.Timedelta(hours=horizon)
                for n, node in enumerate(NODES):
                    actual = float(self.values[self.index.get_loc(target), n]) if not custom and target in self.index else None
                    forecasts.append({"horizon": horizon, "target": str(target), "node": node,
                                      "lower": float(bands[n, 0]), "point": float(bands[n, 1]), "upper": float(bands[n, 2]), "actual": actual})
        history = [{"timestamp": str(index[p]), "values": dict(zip(NODES, values[p].tolist()))} for p in range(max(0, position-71), position+1)]
        warnings = ["Nominal 80% bands have no guaranteed coverage on these inputs."]
        if sum(repaired.values()): warnings.append("Missing readings were repaired from past observations; interval coverage can deteriorate.")
        if request.scenario.startswith("bias"): warnings.append("Synthetic measurement bias affects inputs only; true outcomes are unchanged.")
        if origin >= boundary+pd.Timedelta(days=self.config["test_days"]): warnings.append("Origin is outside this model's evaluated window; no measured coverage is available for it.")
        return {"version": self.inventory["version"], "fold": request.fold, "family": request.family, "seed": request.seed,
                "origin": str(origin), "scenario": request.scenario, "custom_history": custom, "repaired_counts": repaired,
                "forecasts": forecasts, "history": history, "warnings": warnings}


def create_app(dataset, experiment, ui=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.store = ForecastStore(dataset, experiment)
        yield
        del app.state.store

    app = FastAPI(title="Regional demand explorer", version="1.0.0", lifespan=lifespan)

    @app.get("/api/health")
    def health():
        return {"status": "ready", "version": app.state.store.inventory["version"], "device": "cpu"}

    @app.get("/api/meta")
    def metadata():
        return app.state.store.metadata()

    @app.get("/api/backtest")
    def backtest(fold: str, family: Literal["tree", "transformer"], seed: int, node: str, horizon: int):
        try:
            return app.state.store.backtest(fold, family, seed, node, horizon)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.get("/api/stress")
    def stress():
        return app.state.store.stress

    @app.post("/api/forecast")
    def forecast(request: ForecastRequest):
        try:
            return app.state.store.forecast(request)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    if ui is not None:
        app.mount("/", StaticFiles(directory=ui, html=True), name="explorer")
    return app


def main():
    parser = argparse.ArgumentParser(description="Serve a verified forecasting experiment and explorer")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--ui", type=Path, default=Path("frontend/dist"))
    parser.add_argument("--port", type=int, default=8012)
    args = parser.parse_args()
    if not (args.ui/"index.html").is_file(): parser.exit(2, "Build the explorer first: npm --prefix frontend run build\n")
    import uvicorn
    uvicorn.run(create_app(args.data, args.experiment, args.ui), host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()
