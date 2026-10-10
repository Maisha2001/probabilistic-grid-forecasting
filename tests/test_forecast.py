import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from grid_forecast.data import NODES, load_dataset
from grid_forecast.evaluation import backtest, summarize, validate_config
from grid_forecast.features import feature_matrix, pinball, training_origins, training_statistics
from grid_forecast.io import file_hash, read_json, write_json
from grid_forecast.model import load_model, predict_model, select_and_fit


@pytest.mark.parametrize("horizon", [1, 6, 24, 168])
def test_features_ignore_unobserved_future(synthetic, horizon):
    index, values = synthetic
    scale, _, _ = training_statistics(values, 1000, 720)
    before = feature_matrix(index, values, [1100], horizon, scale)
    altered = values.copy(); altered[1101:] *= 100
    np.testing.assert_array_equal(before, feature_matrix(index, altered, [1100], horizon, scale))
    altered[1100] *= 2
    assert not np.array_equal(before, feature_matrix(index, altered, [1100], horizon, scale))
    origins = training_origins(1000, horizon, 720, 6)
    assert (origins+horizon < 1000).all()


def test_future_changes_cannot_change_selection_or_fit(synthetic, config, tmp_path):
    index, values = synthetic
    altered = values.copy(); altered[1400:] = altered[1400:]*3+500
    model_a, meta_a, choices_a = select_and_fit(index, values, 1400, 24, config, tmp_path/"a")
    model_b, meta_b, choices_b = select_and_fit(index, altered, 1400, 24, config, tmp_path/"b")
    assert choices_a == choices_b and meta_a == meta_b
    assert file_hash(tmp_path/"a/model.ubj") == file_hash(tmp_path/"b/model.ubj")
    assert meta_a["last_training_target"] < 1400
    assert all(c["last_validation_target"] < 1400 for c in choices_a)


def test_saved_model_predictions_and_identity(synthetic, config, tmp_path):
    index, values = synthetic
    model, metadata, _ = select_and_fit(index, values, 1400, 24, config, tmp_path/"model")
    expected, _ = predict_model(model, metadata, index, values, [1450, 1460])
    restored, loaded = load_model(tmp_path/"model")
    actual, _ = predict_model(restored, loaded, index, values, [1450, 1460])
    np.testing.assert_array_equal(expected, actual)
    assert (np.diff(actual, axis=-1) >= 0).all() and (actual >= 0).all()
    (tmp_path/"model/model.ubj").write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity"): load_model(tmp_path/"model")


def dataset(path, synthetic):
    index, values = synthetic
    rows = pd.DataFrame({"timestamp": np.repeat(index, 6), "node": np.tile(NODES, len(index)),
                         "demand_mw": values.reshape(-1)})
    path.mkdir(); rows.to_parquet(path/"demand.parquet", index=False)
    write_json(path/"manifest.json", {"parquet_sha256": file_hash(path/"demand.parquet")})


def test_complete_run_and_cli_prediction(synthetic, config, tmp_path):
    dataset(tmp_path/"data", synthetic)
    report = backtest(tmp_path/"data", config, tmp_path/"run")
    assert report["prediction_rows"] == 8*6*2*2
    predictions = pd.read_parquet(tmp_path/"run/predictions.parquet")
    assert (predictions.target-predictions.origin == pd.to_timedelta(predictions.horizon, unit="h")).all()
    assert len(report["metrics"]) == 24 and len(report["median_coherence"]) == 4
    assert (tmp_path/"run/report.html").exists()
    command = [sys.executable, "-m", "grid_forecast", "predict", "--data", str(tmp_path/"data"),
               "--model", str(tmp_path/"run/models/2012-03-01/h24"), "--origin", "2012-03-01T00:00:00"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    actual = json.loads(result.stdout)
    subset = predictions[(predictions.method == "xgboost_quantile") & (predictions.horizon == 24)
                         & (predictions.origin == "2012-03-01T00:00:00")].set_index("node")
    for node in NODES:
        np.testing.assert_array_equal(actual["forecasts_mw"][node], subset.loc[node, ["q10", "q50", "q90"]].to_numpy(float))
    command[-1] = "2012-02-29T00:00:00"
    assert subprocess.run(command, capture_output=True).returncode == 2
    with pytest.raises(ValueError, match="exists"): backtest(tmp_path/"data", config, tmp_path/"run")
    (tmp_path/"data/demand.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"): load_dataset(tmp_path/"data")


def test_metrics_and_coherence_known_example():
    rows = []
    for node in NODES:
        rows.append({"fold": "test", "method": "model", "horizon": 1, "origin": "2014-01-01",
                     "node": node, "actual": 10., "q10": 8., "q50": 9., "q90": 12.})
    metrics, coherence = summarize(pd.DataFrame(rows), {"test": dict.fromkeys(NODES, 2.)})
    assert metrics[0]["mae_mw"] == 1 and metrics[0]["mase"] == .5
    assert metrics[0]["coverage_80"] == 1 and metrics[0]["interval_width_mw"] == 4
    assert coherence[0]["mean_absolute_mw"] == 36
    np.testing.assert_allclose(pinball([10.], np.array([[8., 9., 12.]])), [[.2, .5, .2]])


def test_overlapping_fold_targets_are_rejected(config):
    config["folds"] = ["2014-01-01", "2014-01-02"]
    with pytest.raises(ValueError, match="overlap"): validate_config(config)
