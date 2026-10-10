import copy
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from grid_forecast.data import NODES
from grid_forecast.experiments import run_experiment, verify, seal, safe_file, track_local
from grid_forecast.io import file_hash, read_json, write_json


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("experiment")
    index = pd.date_range("2012-01-01T01:00", periods=2000, freq="h")
    t = np.arange(len(index))
    leaves = np.column_stack([(i+1)*100 + 20*np.sin(t/24) + np.random.default_rng(i).normal(0, 2, len(t)) for i in range(5)])
    values = np.column_stack([leaves, leaves.sum(axis=1)])
    data = root/"data"; data.mkdir()
    pd.DataFrame({"timestamp": np.repeat(index, 6), "node": np.tile(NODES, len(index)), "demand_mw": values.reshape(-1)}).to_parquet(data/"demand.parquet", index=False)
    write_json(data/"manifest.json", {"parquet_sha256": file_hash(data/"demand.parquet")})
    config = {"folds": ["2012-03-01T00:00:00"], "test_days": 2, "validation_days": 20, "calibration_days": 10,
              "training_hours": 720, "training_stride": 6, "evaluation_stride": 6, "horizons": [1, 24],
              "quantiles": [.1, .5, .9], "candidates": [{"max_depth": 2, "n_estimators": 8}],
              "learning_rate": .07, "seed": 42, "seeds": [42], "threads": 1, "coverage": .8,
              "neural": {"history_hours": 336, "patch_hours": 24, "d_model": 8, "heads": 2, "layers": 1,
                         "dropout": .1, "epochs": 2, "batch_size": 32, "learning_rate": .001, "weight_decay": .0001}}
    out = root/"experiment"; run_experiment(data, config, out)
    return root, data, out, config


def test_experiment_replay_stress_pairing_resume_and_cli(experiment):
    root, data, out, config = experiment
    first = verify(data, out)
    assert run_experiment(data, config, out, resume=True) == first
    again = root/"repeat"; run_experiment(data, config, again)
    assert file_hash(again/"fit/predictions.parquet") == file_hash(out/"fit/predictions.parquet")
    assert file_hash(again/"audit/stress.parquet") == file_hash(out/"audit/stress.parquet")
    stress = pd.read_parquet(out/"audit/stress.parquet")
    clean = stress[stress.scenario == "clean"]
    fitted = pd.read_parquet(out/"fit/predictions.parquet")
    joined = clean.merge(fitted[fitted.method == "calibrated"], on=["fold", "origin", "family", "seed", "horizon", "node"], suffixes=("_s", "_f"), validate="one_to_one")
    assert len(joined) == len(clean)
    for column in ["actual", "lower", "point", "upper"]:
        np.testing.assert_allclose(joined[column+"_s"], joined[column+"_f"], rtol=2e-5, atol=.02)
    keys = ["fold", "origin", "family", "seed", "horizon", "node"]
    assert stress.groupby(keys).actual.nunique().max() == 1
    assert stress.groupby(keys).scenario.nunique().min() == 5
    command = [sys.executable, "-m", "grid_forecast.experiment_cli", "verify", "--data", str(data), "--output", str(out)]
    result = json.loads(subprocess.run(command, text=True, capture_output=True, check=True).stdout)
    assert result["version"] == first["version"]
    different = copy.deepcopy(config); different["neural"]["epochs"] += 1
    with pytest.raises(ValueError, match="identical"): run_experiment(data, different, out, resume=True)


def test_checksums_protect_models_postprocessing_and_dataset(experiment):
    _, data, out, _ = experiment
    for relative in ["fit/models/2012-03-01/tree/postprocess.json", "fit/models/2012-03-01/tree/h1/model.ubj", "protocol.json"]:
        path = out/relative; original = path.read_bytes()
        try:
            path.write_bytes(original+b" ")
            with pytest.raises(ValueError, match="checksum"): verify(data, out)
            with pytest.raises(ValueError): seal(data, out)
        finally:
            path.write_bytes(original)
    path = data/"manifest.json"; original = path.read_bytes()
    try:
        path.write_bytes(original+b" ")
        with pytest.raises(ValueError, match="Dataset version"): verify(data, out)
    finally:
        path.write_bytes(original)
    for unsafe in ["../data/manifest.json", str((out/"seal.json").resolve())]:
        with pytest.raises(ValueError): safe_file(out, unsafe)


def test_tracking_stays_local_and_persists_verified_metrics(experiment):
    pytest.importorskip("mlflow", reason="Install requirements/tracking.txt for local tracking")
    root, data, out, _ = experiment
    with pytest.raises(ValueError): track_local(data, out, out/"tracking")
    with pytest.raises(ValueError): track_local(data, out, "https://example.com/tracking")
    receipt = track_local(data, out, root/"tracking")
    assert receipt["status"] == "FINISHED" and len(receipt["metrics"]) == 6
    assert os.environ["MLFLOW_DISABLE_TELEMETRY"] == "true"
    assert (root/"tracking/experiments.db").is_file()
    from mlflow import MlflowClient
    client = MlflowClient(tracking_uri="sqlite:///"+(root/"tracking/experiments.db").as_posix())
    restored = client.get_run(receipt["run_id"])
    assert restored.data.params["data_sha256"] == verify(data, out)["data_sha256"]
    assert restored.info.artifact_uri.startswith("file:")
    assert any(a.path == "seal.json" for a in client.list_artifacts(receipt["run_id"]))
