"""FastAPI app: OpenAI-compatible ASR over sherpa-onnx (SenseVoice).

Endpoints (identical to the faster-whisper wrapper's contract, so the Bridge's
generic sidecar path drives both without any adapter):
  GET  /health                      -> {"status": "ok"}   (cheap, no model load)
  GET  /v1/models                   -> per-variant download status
  POST /v1/models/{name}/download   -> background weight download
  POST /v1/audio/transcriptions     -> {"text": "..."}    (multipart: file, model)

Design notes:
- The heavy `sherpa_onnx` / `av` imports are LAZY (inside loaders), so app
  construction and /health stay instant — the Bridge's bounded health probe must
  pass long before any model is downloaded/loaded.
"""

from __future__ import annotations

import bz2
import io
import os
import shutil
import tarfile
import threading
import urllib.request

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

_CATALOG = ["sensevoice-small"]

# name -> where the weights come from. `dir` is the folder we extract into under
# model_dir; `files` are the required members that must exist for "downloaded".
_MODEL_SOURCES = {
    "sensevoice-small": {
        # sherpa-onnx official release tarball (int8). CODIE_SHERPA_MODEL_BASE
        # overrides the host for restricted networks / mirrors.
        "url_path": (
            "k2-fsa/sherpa-onnx/releases/download/asr-models/"
            "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2"
        ),
        "dir": "sensevoice-small",
        "files": ["model.int8.onnx", "tokens.txt"],
    },
}

_DEFAULT_BASE = "https://github.com"


def _model_dir_for(model_dir: str, name: str) -> str:
    return os.path.join(model_dir, _MODEL_SOURCES[name]["dir"])


def _is_cached(model_dir: str, name: str) -> bool:
    d = _model_dir_for(model_dir, name)
    return all(os.path.isfile(os.path.join(d, f)) for f in _MODEL_SOURCES[name]["files"])


def _dir_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _model_url(name: str) -> str:
    base = os.environ.get("CODIE_SHERPA_MODEL_BASE", _DEFAULT_BASE).rstrip("/")
    return f"{base}/{_MODEL_SOURCES[name]['url_path']}"


def _fetch_bytes(url: str) -> bytes:
    """Download the whole archive into memory. Seam for tests to monkeypatch."""
    with urllib.request.urlopen(url) as resp:  # noqa: S310 — trusted release host
        return resp.read()


def _extract_flatten(raw: bytes, dest_dir: str, required: list[str]) -> None:
    """Extract a .tar.bz2 whose members sit under one top-level dir, flattening
    the required files to `dest_dir/<basename>`."""
    os.makedirs(dest_dir, exist_ok=True)
    wanted = set(required)
    with tarfile.open(fileobj=bz2.BZ2File(io.BytesIO(raw)), mode="r:") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            base = os.path.basename(member.name)
            if base in wanted:
                src = tf.extractfile(member)
                if src is None:
                    continue
                with open(os.path.join(dest_dir, base), "wb") as out:
                    shutil.copyfileobj(src, out)
    missing = [f for f in required if not os.path.isfile(os.path.join(dest_dir, f))]
    if missing:
        raise RuntimeError(f"tarball missing required files: {missing}")


class Downloads:
    """Tracks in-flight background model downloads by name, thread-safely."""

    def __init__(self, model_dir: str) -> None:
        self._model_dir = model_dir
        self._lock = threading.Lock()
        self._active: dict[str, threading.Thread] = {}

    def is_downloading(self, name: str) -> bool:
        with self._lock:
            return name in self._active

    def start(self, name: str) -> str:
        if name not in _MODEL_SOURCES:
            raise KeyError(name)
        if _is_cached(self._model_dir, name):
            return "exists"
        with self._lock:
            if name in self._active:
                return "downloading"
            t = threading.Thread(target=self._run, args=(name,), name=f"dl-{name}", daemon=True)
            self._active[name] = t
        t.start()
        return "started"

    def _run(self, name: str) -> None:
        try:
            raw = _fetch_bytes(_model_url(name))
            _extract_flatten(raw, _model_dir_for(self._model_dir, name), _MODEL_SOURCES[name]["files"])
        except Exception:  # noqa: BLE001 — surfaced via status (downloaded stays false)
            pass
        finally:
            with self._lock:
                self._active.pop(name, None)

    def join_for_test(self, name: str, timeout: float = 30.0) -> None:
        """Test-only: block until the background download thread finishes."""
        with self._lock:
            t = self._active.get(name)
        if t is not None:
            t.join(timeout)


def build_app(*, model_dir: str) -> FastAPI:
    app = FastAPI(title="codie-asr-sherpa", version="0.1.0")
    downloads = Downloads(model_dir)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/v1/models")
    def list_models() -> dict:
        out = []
        for name in _CATALOG:
            cached = _is_cached(model_dir, name)
            d = _model_dir_for(model_dir, name)
            out.append(
                {
                    "name": name,
                    "downloaded": cached,
                    "downloading": downloads.is_downloading(name),
                    "bytes": _dir_bytes(d) if cached else 0,
                }
            )
        return {"models": out}

    @app.post("/v1/models/{name}/download")
    def download_model_route(name: str) -> dict:
        if name not in _MODEL_SOURCES:
            raise HTTPException(status_code=400, detail=f"unknown model {name!r}")
        return {"name": name, "status": downloads.start(name)}

    return app
