import copy

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient
from grid_forecast.data import NODES, load_dataset
from grid_forecast.service import create_app, ForecastStore
from scripts.serve_fixture import build_fixture


@pytest.fixture(scope="module")
def service_files(tmp_path_factory):
    return build_fixture(tmp_path_factory.mktemp("service"))


@pytest.fixture(scope="module")
def client(service_files):
    with TestClient(create_app(*service_files)) as connection:
        yield connection


def test_api_matches_saved_results_and_reconciles_every_endpoint(client, service_files):
    assert client.get("/api/health").json()["status"] == "ready"
    meta = client.get("/api/meta").json()
    assert meta["nodes"] == list(NODES) and "physical UTC" in meta["clock"]
    data, experiment = service_files
    rows = pd.read_parquet(experiment/"fit/predictions.parquet")
    for family in ["tree", "transformer"]:
        response = client.post("/api/forecast",json={"fold":"2012-03-01","family":family,"seed":42,"origin":"2012-03-01"})
        assert response.status_code == 200, response.text
        forecasts = pd.DataFrame(response.json()["forecasts"])
        for column in ["lower","point","upper"]:
            wide = forecasts.pivot(index="horizon",columns="node",values=column)
            np.testing.assert_allclose(wide.ALL_FIVE,wide[list(NODES[:5])].sum(axis=1),atol=1e-8)
        expected = rows[(rows.family==family)&(rows.origin==pd.Timestamp("2012-03-01"))&(rows.method=="calibrated")]
        matched = forecasts.merge(expected,on=["horizon","node"],suffixes=("_api","_saved"))
        for column in ["lower","point","upper","actual"]:
            np.testing.assert_allclose(matched[column+"_api"],matched[column+"_saved"],rtol=2e-5,atol=.02)
        result = client.get("/api/backtest",params={"fold":"2012-03-01","family":family,"seed":42,"node":"NSW","horizon":24})
        assert result.status_code == 200 and len(result.json()["rows"]) == 8
        assert 0 <= result.json()["joint_coverage"] <= 1
    assert len(client.get("/api/stress").json()["metrics"]) == 40


def test_custom_history_and_missing_policy(client, service_files):
    index, values, _ = load_dataset(service_files[0]); origin = index.get_loc(pd.Timestamp("2012-03-01"))
    history = values[origin-336:origin+1,:5].tolist()
    body = {"fold":"2012-03-01","origin":"2012-03-01","history":history}
    result = client.post("/api/forecast",json=body)
    assert result.status_code == 200
    custom = result.json(); baseline = client.post("/api/forecast",json={"fold":"2012-03-01","origin":"2012-03-01"}).json()
    for a,b in zip(custom["forecasts"],baseline["forecasts"]):
        assert a["actual"] is None
        np.testing.assert_allclose([a["lower"],a["point"],a["upper"]],[b["lower"],b["point"],b["upper"]],rtol=1e-7)
    damaged = copy.deepcopy(body)
    for row in damaged["history"][-24:]:row[0]=None
    response = client.post("/api/forecast",json=damaged)
    assert response.status_code == 200 and response.json()["repaired_counts"]["NSW"] == 24
    damaged["history"][-25][0]=None
    assert client.post("/api/forecast",json=damaged).status_code == 422
    body["origin"]="2012-05-01"
    assert any("outside" in w for w in client.post("/api/forecast",json=body).json()["warnings"])


@pytest.mark.parametrize("change", [
    {"origin":"2012-02-01"},{"origin":"2012-03-01T00:30:00"},{"origin":"2012-03-01T00:00:00Z"},
    {"origin":"not-a-date"},{"fold":"../outside"},{"family":"unknown"},{"seed":999},
    {"history":[[1,2,3,4,5]]},{"unexpected":True}])
def test_api_rejects_invalid_requests(client,change):
    body={"fold":"2012-03-01","origin":"2012-03-01",**change}
    assert client.post("/api/forecast",json=body).status_code == 422


def test_startup_rejects_corrupted_calibration(service_files):
    data,experiment=service_files;path=experiment/"fit/models/2012-03-01/tree/postprocess.json";before=path.read_bytes()
    try:
        path.write_bytes(before+b" ")
        with pytest.raises(ValueError,match="checksum"):ForecastStore(data,experiment)
    finally:path.write_bytes(before)
