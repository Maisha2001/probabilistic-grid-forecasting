import argparse
import json
from pathlib import Path

from .experiments import run_experiment, track_local, verify
from .io import read_json


def main():
    parser = argparse.ArgumentParser(description="Versioned forecasting experiments and local tracking")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["run", "verify", "track"]:
        command = commands.add_parser(name)
        command.add_argument("--data", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "run":
            command.add_argument("--config", type=Path, default=Path("configs/forward.json"))
            command.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
            command.add_argument("--resume", action="store_true")
        elif name == "track":
            command.add_argument("--tracking", type=Path, default=Path("runs/tracking"))
    args = parser.parse_args()
    try:
        if args.command == "run":
            result = run_experiment(args.data, read_json(args.config), args.output, args.device, args.resume)
            print(json.dumps({"version": result["version"], "files_verified": len(result["files"])}))
        elif args.command == "verify":
            result = verify(args.data, args.output)
            print(json.dumps({"version": result["version"], "files_verified": len(result["files"])}))
        else:
            print(json.dumps(track_local(args.data, args.output, args.tracking)))
    except (ValueError, OSError, KeyError, ImportError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
