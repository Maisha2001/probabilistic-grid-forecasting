import pytest

from grid_forecast.io import write_json
from grid_forecast import pipeline


def test_pipeline_recovers_after_fetch_failure_and_rejects_changed_protocol(tmp_path,monkeypatch):
    root=tmp_path/"run";calls=[]
    def failed_fetch(path):
        calls.append("fetch");raise OSError("offline")
    monkeypatch.setattr(pipeline,"fetch",failed_fetch)
    with pytest.raises(OSError):pipeline.reproduce(root,{"example":1})
    assert not (root/"pipeline.lock").exists()
    with pytest.raises(ValueError,match="identical"):pipeline.reproduce(root,{"example":2},resume=True)
    with pytest.raises(OSError):pipeline.reproduce(root,{"example":1},resume=True)
    assert calls == ["fetch","fetch"]
    (root/"pipeline.lock").write_text("active")
    with pytest.raises(ValueError,match="locked"):pipeline.reproduce(root,{"example":1},resume=True)
    assert (root/"pipeline.lock").read_text() == "active"
