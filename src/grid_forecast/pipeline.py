import argparse
from pathlib import Path

from .data import fetch, load_dataset, prepare
from .experiments import run_experiment, track_local
from .io import read_json, write_json


def reproduce(root, config, device="cpu", resume=False, raw=None):
    root = Path(root)
    protocol = {"config": config, "device": device, "source_days": ["2012-01-01", "2015-03-01"]}
    if root.exists():
        if not resume or not (root/"pipeline.json").is_file() or read_json(root/"pipeline.json")["protocol"] != protocol:
            raise ValueError("Use a new root or resume the identical pipeline")
    else:
        root.mkdir(parents=True); write_json(root/"pipeline.json", {"protocol": protocol, "status": "started"})
    lock = root/"pipeline.lock"
    try:
        with lock.open("x", encoding="utf-8") as handle: handle.write("Single local pipeline invocation; remove only after confirming its process stopped.\n")
    except FileExistsError as error:
        raise ValueError("Pipeline is locked by an active or interrupted process") from error
    try:
        raw = Path(raw) if raw is not None else root/"raw"
        fetch(raw)
        data = root/"data"
        if not data.exists(): prepare(raw, data, *protocol["source_days"])
        _, _, manifest = load_dataset(data)
        if [manifest["start_day_inclusive"], manifest["end_day_exclusive"]] != protocol["source_days"]:
            raise ValueError("Prepared dataset uses a different source window")
        result = run_experiment(data, config, root/"experiment", device, resume=(root/"experiment").exists())
        receipt_path = root/"experiment/tracking.json"
        tracking = read_json(receipt_path) if receipt_path.exists() else track_local(data, root/"experiment", root/"tracking")
        if tracking["version"] != result["version"] or tracking["status"] != "FINISHED":
            raise ValueError("Tracking receipt does not match the verified experiment")
        record = {"protocol": protocol, "status": "complete", "experiment_version": result["version"], "tracking_run": tracking["run_id"]}
        write_json(root/"pipeline.json", record)
        return record
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description="Fetch, validate, train, audit and track a local forecast experiment")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/forward.json"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        print(reproduce(args.root, read_json(args.config), args.device, args.resume, args.raw))
    except (ValueError, OSError, KeyError, ImportError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
