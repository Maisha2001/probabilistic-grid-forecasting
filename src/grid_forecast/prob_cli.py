import argparse
import json
from pathlib import Path

from .io import read_json


def main():
    parser = argparse.ArgumentParser(description="Temporal models and calibrated coherent regional forecasts")
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("backtest")
    evaluate.add_argument("--data", type=Path, default=Path("runs/dataset"))
    evaluate.add_argument("--config", type=Path, default=Path("configs/probabilistic.json"))
    evaluate.add_argument("--output", type=Path, default=Path("runs/probabilistic"))
    evaluate.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    evaluate.add_argument("--resume", action="store_true")
    predict = commands.add_parser("predict")
    predict.add_argument("--data", type=Path, required=True)
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--family", choices=["tree", "transformer"], required=True)
    predict.add_argument("--origin", required=True)
    predict.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    try:
        from .probabilistic import run_probabilistic, saved_prediction
        if args.command == "backtest":
            report = run_probabilistic(args.data, read_json(args.config), args.output, args.device, args.resume)
            print(json.dumps({"prediction_rows": report["prediction_rows"], "compute": report["compute"]}))
        else:
            print(json.dumps(saved_prediction(args.data, args.model, args.origin, args.family, args.device)))
    except (ValueError, OSError, KeyError, ImportError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
