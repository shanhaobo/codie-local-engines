"""FastAPI app: OpenAI-compatible ASR over faster-whisper.

Endpoints (the sub-paths the Bridge's OpenAI provider already calls):
  GET  /health                      -> {"status": "ok"}  (cheap, no model load)
  POST /v1/audio/transcriptions     -> {"text": "..."}   (multipart: file, model)

Design notes:
- The heavy `faster_whisper` import is LAZY (inside the model loader), so app
  construction and /health stay instant — the Bridge's bounded health probe
  must pass long before any model is downloaded/loaded.
- Models are cached per normalized name. The first transcription for a given
  size downloads the CTranslate2 weights into `model_dir` (honoring HF_ENDPOINT
  for the mirror), which can take a while; subsequent calls are warm.
- Catalog model names carry a `whisper-` prefix (`whisper-small`); faster-whisper
  wants the bare size (`small`). `_normalize_model` bridges the two.
"""

from __future__ import annotations

import io
import os
import threading

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

# faster-whisper sizes we accept after stripping the catalog's `whisper-` prefix.
_KNOWN_SIZES = {
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v1",
    "large-v2",
    "large-v3",
    "large",
}


def _normalize_model(name: str) -> str:
    """`whisper-small` -> `small`; pass through bare faster-whisper sizes."""
    n = (name or "").strip()
    if n.startswith("whisper-"):
        n = n[len("whisper-") :]
    return n or "small"


class ModelCache:
    """Lazily loads and caches faster-whisper models by size, thread-safely."""

    def __init__(self, model_dir: str) -> None:
        self._model_dir = model_dir
        self._lock = threading.Lock()
        self._models: dict = {}

    def get(self, size: str):
        with self._lock:
            cached = self._models.get(size)
            if cached is not None:
                return cached
            # Lazy import: keeps app construction + /health free of the heavy
            # ctranslate2/av/onnxruntime import graph.
            from faster_whisper import WhisperModel

            # CPU by default: `device="auto"` makes ctranslate2 try to load CUDA
            # libs (cublas64_12.dll) on any machine where it detects a GPU, which
            # crashes hosts that have a GPU but no CUDA runtime installed — the
            # common desktop case. GPU users opt in explicitly. On CPU, int8 is
            # the fast, low-memory default.
            device = os.environ.get("CODIE_ASR_DEVICE", "cpu")
            compute_type = os.environ.get("CODIE_ASR_COMPUTE_TYPE") or (
                "int8" if device == "cpu" else "auto"
            )
            try:
                model = WhisperModel(
                    size,
                    device=device,
                    compute_type=compute_type,
                    download_root=self._model_dir,
                )
            except Exception:
                # A requested GPU device that can't actually load (missing CUDA
                # libs) should degrade to CPU rather than fail the request.
                if device == "cpu":
                    raise
                model = WhisperModel(
                    size,
                    device="cpu",
                    compute_type="int8",
                    download_root=self._model_dir,
                )
            self._models[size] = model
            return model


def build_app(*, model_dir: str) -> FastAPI:
    app = FastAPI(title="codie-asr-local", version="0.1.0")
    cache = ModelCache(model_dir)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/v1/audio/transcriptions")
    async def transcribe(
        file: UploadFile = File(...),
        model: str = Form("whisper-small"),
        language: str | None = Form(None),
    ) -> dict:
        size = _normalize_model(model)
        if size not in _KNOWN_SIZES:
            raise HTTPException(
                status_code=400,
                detail=f"unknown whisper model {model!r} (normalized {size!r})",
            )
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="empty audio file")

        try:
            whisper = cache.get(size)
            # faster-whisper decodes the container itself (PyAV); a BytesIO is a
            # valid `audio` argument, so no temp file / external ffmpeg needed.
            segments, _info = whisper.transcribe(
                io.BytesIO(data),
                language=language,
            )
            text = "".join(seg.text for seg in segments)
        except Exception as e:  # noqa: BLE001 — surface decode/transcribe errors
            raise HTTPException(status_code=500, detail=f"transcription failed: {e}")

        return {"text": text.strip()}

    return app
