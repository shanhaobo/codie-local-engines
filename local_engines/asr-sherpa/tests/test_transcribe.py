import io
import wave

import numpy as np
from fastapi.testclient import TestClient

from codie_asr_sherpa import server


def _wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.zeros(16000, dtype="<i2").tobytes())
    return buf.getvalue()


class _FakeRecognizer:
    def transcribe(self, samples, language=None):  # matches server's call seam
        return "你好 hello"


def _client(tmp_path, monkeypatch):
    # Pretend the model is cached, and short-circuit both heavy seams.
    monkeypatch.setattr(server, "_is_cached", lambda md, name: True)
    monkeypatch.setattr(server.ModelCache, "get", lambda self, name: _FakeRecognizer())
    monkeypatch.setattr(server, "decode_to_16k_mono", lambda data: np.zeros(16000, dtype=np.float32))
    return TestClient(server.build_app(model_dir=str(tmp_path)))


def test_transcribe_returns_text(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", _wav_bytes(), "audio/wav")},
        data={"model": "sensevoice-small"},
    )
    assert r.status_code == 200
    assert r.json() == {"text": "你好 hello"}


def test_transcribe_unknown_model_400(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", _wav_bytes(), "audio/wav")},
        data={"model": "nope"},
    )
    assert r.status_code == 400


def test_transcribe_empty_file_400(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", b"", "audio/wav")},
        data={"model": "sensevoice-small"},
    )
    assert r.status_code == 400
