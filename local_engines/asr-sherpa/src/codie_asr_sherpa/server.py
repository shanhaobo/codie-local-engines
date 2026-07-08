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

from fastapi import FastAPI

_CATALOG = ["sensevoice-small"]


def build_app(*, model_dir: str) -> FastAPI:
    app = FastAPI(title="codie-asr-sherpa", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    return app
