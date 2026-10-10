import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import NODES, fetch, load_dataset, prepare
from .evaluation import backtest
from .io import read_json
from .model import load_model, predict_model


def main():
    parser = argparse.ArgumentParser(description="Reproducible regional demand forecasting")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("fetch")
    download.add_argument("--output", type=Path, default=Path("data/raw"))
    clean = commands.add_parser("prepare")
    clean.add_argument("--raw", type=Path, default=Path("data/raw"))
    clean.add_argument("--output", type=Path, default=Path("runs/dataset"))
    evaluate = commands.add_parser("backtest")
    evaluate.add_argument("--data", type=Path, default=Path("runs/dataset"))
    evaluate.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    evaluate.add_argument("--output", type=Path, default=Path("runs/baseline"))
    predict = commands.add_parser("predict")
    predict.add_argument("--data", type=Path, required=True)
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--origin", required=True)
    args = parser.parse_args()
    try:
        if args.command == "fetch":
            fetch(args.output)
        elif args.command == "prepare":
            prepare(args.raw, args.output)
        elif args.command == "backtest":
            report = backtest(args.data, read_json(args.config), args.output)
            print(json.dumps({"prediction_rows": report["prediction_rows"], "compute": report["compute"]}))
        else:
            index, values, _ = load_dataset(args.data)
            model, metadata = load_model(args.model)
            origin = pd.Timestamp(args.origin)
            if origin.tz is not None or origin not in index or origin < pd.Timestamp(metadata["forecast_origin_not_before"]):
                raise ValueError("Origin must be an available source-clock label on/after the model fit boundary")
            result, _ = predict_model(model, metadata, index, values, np.array([index.get_loc(origin)]))
            print(json.dumps({"origin": str(origin), "target": str(origin+pd.Timedelta(hours=metadata["horizon"])),
                              "model_sha256": metadata["model_sha256"], "quantiles": [0.1, 0.5, 0.9],
                              "forecasts_mw": dict(zip(NODES, result[0].tolist()))}))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
