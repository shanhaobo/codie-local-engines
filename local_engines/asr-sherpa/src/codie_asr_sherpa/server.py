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

import os
import threading

from fastapi import FastAPI

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


class Downloads:
    def __init__(self, model_dir: str) -> None:
        self._model_dir = model_dir
        self._lock = threading.Lock()
        self._active: set[str] = set()

    def is_downloading(self, name: str) -> bool:
        with self._lock:
            return name in self._active


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

    return app
