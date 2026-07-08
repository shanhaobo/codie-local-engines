from fastapi.testclient import TestClient

from codie_asr_funasr.server import build_app


def _client(tmp_path):
    return TestClient(build_app(model_dir=str(tmp_path)))


def test_health_ok(tmp_path):
    r = _client(tmp_path).get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_list_models_reports_uncached(tmp_path):
    r = _client(tmp_path).get("/v1/models")
    assert r.status_code == 200
    models = r.json()["models"]
    assert [m["name"] for m in models] == ["paraformer-zh"]
    m = models[0]
    assert m["downloaded"] is False
    assert m["downloading"] is False
    assert m["bytes"] == 0
