import copy
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from grid_forecast.data import NODES
from grid_forecast.io import file_hash, write_json
from grid_forecast.probabilistic import run_probabilistic, saved_prediction, split_origins
from grid_forecast.temporal import inputs, fit_temporal, load_temporal, predict_temporal


@pytest.fixture
def prob_config(config):
    return {**config, "validation_days": 20, "calibration_days": 10, "coverage": .8, "seeds": [42],
            "neural": {"history_hours": 336, "patch_hours": 24, "d_model": 8, "heads": 2, "layers": 1,
                       "dropout": .1, "epochs": 2, "batch_size": 32, "learning_rate": .001, "weight_decay": .0001}}


def test_patch_inputs_never_access_unobserved_values(synthetic, prob_config):
    index, values = synthetic
    old = inputs(index, values, [1400], prob_config["horizons"], np.ones(6), prob_config["neural"])
    changed = values.copy(); changed[1401:] *= 100
    new = inputs(index, changed, [1400], prob_config["horizons"], np.ones(6), prob_config["neural"])
    for a, b in zip(old, new): torch.testing.assert_close(a, b, rtol=0, atol=0)
    changed[1400] += 10
    assert not torch.equal(old[0], inputs(index, changed, [1400], prob_config["horizons"], np.ones(6), prob_config["neural"])[0])


def test_checkpoint_resume_and_future_invariance(synthetic, prob_config, tmp_path):
    index, values = synthetic; cutoff, selection, calibration = split_origins(1800, prob_config)
    assert selection[-1]+max(prob_config["horizons"]) < min(x[0] for x in calibration.values())
    assert all(o[-1]+h < 1800 for h, o in calibration.items())
    a, ma = fit_temporal(index, values, cutoff, selection, prob_config, 42, tmp_path/"full")
    fit_temporal(index, values, cutoff, selection, prob_config, 42, tmp_path/"resume", stop_after=1)
    b, mb = fit_temporal(index, values, cutoff, selection, prob_config, 42, tmp_path/"resume")
    assert ma["history"] == mb["history"]
    for name, tensor in a.state_dict().items(): torch.testing.assert_close(tensor, b.state_dict()[name], rtol=0, atol=0)
    altered = values.copy(); altered[1800:] *= 5
    c, mc = fit_temporal(index, altered, cutoff, selection, prob_config, 42, tmp_path/"future")
    assert ma["fingerprint"] == mc["fingerprint"] and ma["history"] == mc["history"]
    for name, tensor in a.state_dict().items(): torch.testing.assert_close(tensor, c.state_dict()[name], rtol=0, atol=0)
    restored, meta = load_temporal(tmp_path/"full")
    np.testing.assert_array_equal(predict_temporal(a, ma, index, values, [1800]), predict_temporal(restored, meta, index, values, [1800]))
    changed_config = copy.deepcopy(prob_config); changed_config["neural"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="Checkpoint"):
        fit_temporal(index, values, cutoff, selection, changed_config, 42, tmp_path/"resume")
    (tmp_path/"full/weights.pt").write_bytes(b"altered")
    with pytest.raises(ValueError, match="checksum"): load_temporal(tmp_path/"full")


def test_probabilistic_run_reload_cli_and_resume(synthetic, prob_config, tmp_path):
    index, values = synthetic; data = tmp_path/"data"; data.mkdir()
    frame = pd.DataFrame({"timestamp": np.repeat(index, 6), "node": np.tile(NODES, len(index)), "demand_mw": values.reshape(-1)})
    frame.to_parquet(data/"demand.parquet", index=False)
    write_json(data/"manifest.json", {"parquet_sha256": file_hash(data/"demand.parquet")})
    out = tmp_path/"run"; report = run_probabilistic(data, prob_config, out)
    rows = pd.read_parquet(out/"predictions.parquet")
    assert report["prediction_rows"] == 8*2*6*3*2
    for family, path in [("tree", "tree"), ("transformer", "transformer-42")]:
        result = saved_prediction(data, out/"models/2012-03-01"/path, "2012-03-01", family)
        selected = rows[(rows.family==family)&(rows.method=="calibrated")&(rows.origin=="2012-03-01")]
        for horizon, output in result["forecasts"].items():
            actual = selected[selected.horizon==int(horizon)].set_index("node")
            for node in NODES: np.testing.assert_allclose(output["nodes"][node], actual.loc[node,["lower","point","upper"]].values.astype(float),rtol=1e-6,atol=1e-5)
        with pytest.raises(ValueError): saved_prediction(data, out/"models/2012-03-01"/path, "2012-02-01", family)
    command = [sys.executable, "-m", "grid_forecast.prob_cli", "predict", "--data", str(data), "--model", str(out/"models/2012-03-01/tree"), "--family", "tree", "--origin", "2012-03-01"]
    assert json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout)["columns"] == ["lower","point","upper"]
    for row in report["joint_metrics"]:
        if row["method"] != "raw": assert row["point_coherence_max_mw"] < 1e-8
    again = run_probabilistic(data, prob_config, out, resume=True)
    assert report["metrics"] == again["metrics"] and report["predictions_sha256"] == again["predictions_sha256"]
    altered = copy.deepcopy(prob_config); altered["seeds"] = [43]
    with pytest.raises(ValueError, match="identical"): run_probabilistic(data, altered, out, resume=True)
    future = tmp_path/"future-data"; future.mkdir()
    frame.loc[frame.timestamp >= pd.Timestamp("2012-03-01"), "demand_mw"] *= 7
    frame.to_parquet(future/"demand.parquet", index=False)
    write_json(future/"manifest.json", {"parquet_sha256": file_hash(future/"demand.parquet")})
    run_probabilistic(future, prob_config, tmp_path/"future-run")
    for path in ["tree", "transformer-42"]:
        before = json.loads((out/"models/2012-03-01"/path/"postprocess.json").read_text())
        after = json.loads((tmp_path/"future-run/models/2012-03-01"/path/"postprocess.json").read_text())
        assert before == after
