import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path

from .data import load_dataset
from .io import file_hash, read_json, write_json


def identity_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def safe_file(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Invalid inventory path")
    result = (Path(root)/path).resolve()
    if not result.is_relative_to(Path(root).resolve()) or not result.is_file():
        raise ValueError("Inventory file missing or outside run")
    return result


def seal(dataset, output):
    output = Path(output); _, _, manifest = load_dataset(dataset)
    report = read_json(output/"fit/metrics.json")
    if manifest["parquet_sha256"] != report["data_sha256"]:
        raise ValueError("Data version differs from experiment")
    if file_hash(output/"fit/predictions.parquet") != report["predictions_sha256"]:
        raise ValueError("Prediction checksum mismatch")
    for path, expected in report["artifact_sha256"].items():
        if file_hash(safe_file(output/"fit", path)) != expected:
            raise ValueError("Model artifact checksum mismatch")
    files = [p for p in output.rglob("*") if p.is_file() and p.name not in {"checkpoint.pt", "seal.json", "tracking.json"}]
    payload = {"schema": "grid-experiment-v1", "data_sha256": manifest["parquet_sha256"],
               "data_manifest_sha256": file_hash(Path(dataset)/"manifest.json"),
               "files": {p.relative_to(output).as_posix(): file_hash(p) for p in sorted(files)}}
    document = {"version": identity_hash(payload), **payload}
    if (output/"seal.json").exists() and read_json(output/"seal.json") != document:
        raise ValueError("Sealed run is immutable; use a new output")
    write_json(output/"seal.json", document)
    return document


def verify(dataset, output):
    output = Path(output); document = read_json(output/"seal.json")
    payload = {k: v for k, v in document.items() if k != "version"}
    if document.get("schema") != "grid-experiment-v1" or identity_hash(payload) != document["version"]:
        raise ValueError("Experiment inventory identity mismatch")
    _, _, manifest = load_dataset(dataset)
    if manifest["parquet_sha256"] != document["data_sha256"] or file_hash(Path(dataset)/"manifest.json") != document["data_manifest_sha256"]:
        raise ValueError("Dataset version mismatch")
    for name, expected in document["files"].items():
        if file_hash(safe_file(output, name)) != expected:
            raise ValueError(f"Experiment checksum mismatch: {name}")
    return document


def track_local(dataset, output, directory):
    document = verify(dataset, output); output = Path(output)
    if "://" in str(directory) or str(directory).startswith("\\\\"):
        raise ValueError("Use a separate local tracking directory")
    directory = Path(directory).resolve()
    if directory.is_relative_to(output.resolve()):
        raise ValueError("Use a separate local tracking directory")
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
    os.environ["_MLFLOW_TELEMETRY_PRE_WARM_DATABRICKS_SDK"] = "false"
    os.environ["OTEL_SDK_DISABLED"] = "true"
    from mlflow import MlflowClient
    client = MlflowClient(tracking_uri="sqlite:///"+(directory/"experiments.db").as_posix())
    artifact_root = (directory/"artifacts").as_uri()
    experiment = client.get_experiment_by_name("regional-demand")
    if experiment is None:
        experiment_id = client.create_experiment("regional-demand", artifact_location=artifact_root)
    else:
        if experiment.artifact_location != artifact_root:
            raise ValueError("Tracking experiment has a nonlocal or unexpected artifact store")
        experiment_id = experiment.experiment_id
    run = client.create_run(experiment_id, tags={"mlflow.runName": document["version"][:12], "experiment_version": document["version"]})
    run_id = run.info.run_id
    try:
        config = read_json(output/"protocol.json")
        client.log_param(run_id, "data_sha256", document["data_sha256"])
        client.log_param(run_id, "protocol_sha256", identity_hash(config))
        import pandas as pd
        report = read_json(output/"fit/metrics.json")
        metrics = pd.DataFrame(report["metrics"])
        for (family, method), group in metrics.groupby(["family", "method"]):
            client.log_metric(run_id, f"{family}.{method}.mase", float(group.mase.mean()))
        for name in ["seal.json", "protocol.json", "fit/metrics.json", "audit/diagnostics.json", "audit/stress.json"]:
            parent = Path(name).parent.as_posix()
            client.log_artifact(run_id, str(output/name), artifact_path=None if parent == "." else parent)
        client.set_terminated(run_id, "FINISHED")
    except Exception:
        client.set_terminated(run_id, "FAILED")
        raise
    receipt = {"run_id": run_id, "experiment_id": experiment_id, "version": document["version"], "status": client.get_run(run_id).info.status,
               "metrics": client.get_run(run_id).data.metrics, "mlflow_version": version("mlflow-skinny"), "storage": "Local SQLite and local artifacts; no server required"}
    write_json(output/"tracking.json", receipt)
    return receipt


def run_experiment(dataset, config, output, device="cpu", resume=False):
    from .probabilistic import run_probabilistic
    from .robustness import diagnose, stress_test
    output = Path(output)
    _, _, manifest = load_dataset(dataset)
    sources = {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    packages = {p: version(p) for p in ["numpy", "pandas", "pyarrow", "duckdb", "xgboost", "scikit-learn", "torch"]}
    protocol = {"config": config, "data_sha256": manifest["parquet_sha256"], "source_sha256": sources,
                "packages": packages, "device": device, "stress_stride": 24,
                "policy": "Fixed models and calibration; diagnostic slices and observation perturbations never select hyperparameters. Future targets remain unaltered."}
    if output.exists():
        if not resume or read_json(output/"protocol.json") != protocol:
            raise ValueError("Resume needs identical protocol, data, source, environment and device")
        if (output/"seal.json").exists():
            return verify(dataset, output)
    else:
        output.mkdir(parents=True); write_json(output/"protocol.json", protocol)
    run_probabilistic(dataset, config, output/"fit", device, resume=(output/"fit").exists())
    diagnose(dataset, output/"fit", output/"audit")
    stress_test(dataset, output/"fit", output/"audit", device=device)
    return seal(dataset, output)
