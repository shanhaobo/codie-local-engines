from fastapi.testclient import TestClient

from codie_asr_sherpa.server import build_app


def _client(tmp_path):
    return TestClient(build_app(model_dir=str(tmp_path)))


def test_health_ok(tmp_path):
    r = _client(tmp_path).get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
