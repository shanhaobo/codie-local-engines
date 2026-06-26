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

# Catalog whisper variants the Bridge AI 工具 page lists (and asks status for),
# in the `whisper-<size>` form the catalog uses. GET /v1/models reports each.
_CATALOG = [
    "whisper-tiny",
    "whisper-base",
    "whisper-small",
    "whisper-medium",
    "whisper-large-v3",
]

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


def _download_model_fn():
    """The faster-whisper download helper, across its module layouts (lazy)."""
    try:
        from faster_whisper import download_model  # type: ignore
    except ImportError:  # pragma: no cover - older/newer layout
        from faster_whisper.utils import download_model  # type: ignore
    return download_model


def _cached_path(model_dir: str, size: str):
    """Local cache dir for `size` if already downloaded, else None.

    Uses faster-whisper's own resolver with `local_files_only=True` so we don't
    have to guess the HF repo/cache-dir naming — it returns the path when cached
    and raises when not.
    """
    try:
        return _download_model_fn()(size, cache_dir=model_dir, local_files_only=True)
    except Exception:
        return None


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
    """Tracks in-flight background weight downloads by size, thread-safely."""

    def __init__(self, model_dir: str) -> None:
        self._model_dir = model_dir
        self._lock = threading.Lock()
        self._active: set[str] = set()

    def is_downloading(self, size: str) -> bool:
        with self._lock:
            return size in self._active

    def start(self, size: str) -> str:
        """Begin a background download for `size`. Returns a status string."""
        if _cached_path(self._model_dir, size) is not None:
            return "exists"
        with self._lock:
            if size in self._active:
                return "downloading"
            self._active.add(size)

        def _run() -> None:
            try:
                _download_model_fn()(size, cache_dir=self._model_dir)
            except Exception:  # noqa: BLE001 — surfaced via status (downloaded stays false)
                pass
            finally:
                with self._lock:
                    self._active.discard(size)

        threading.Thread(target=_run, name=f"dl-{size}", daemon=True).start()
        return "started"


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
    app = FastAPI(title="codie-asr-local", version="0.1.2")
    cache = ModelCache(model_dir)
    downloads = Downloads(model_dir)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/v1/models")
    def list_models() -> dict:
        """Per-variant download status, for the Bridge AI 工具 page to show
        which weights are already local (vs lazy-on-first-use)."""
        out = []
        for name in _CATALOG:
            size = _normalize_model(name)
            path = _cached_path(model_dir, size)
            out.append(
                {
                    "name": name,
                    "downloaded": path is not None,
                    "downloading": downloads.is_downloading(size),
                    "bytes": _dir_bytes(path) if path else 0,
                }
            )
        return {"models": out}

    @app.post("/v1/models/{name}/download")
    def download_model_route(name: str) -> dict:
        """Deliberately pre-fetch a variant's weights (background). Idempotent:
        returns `exists` when already cached, `downloading` when in flight."""
        size = _normalize_model(name)
        if size not in _KNOWN_SIZES:
            raise HTTPException(status_code=400, detail=f"unknown whisper model {name!r}")
        return {"name": name, "status": downloads.start(size)}

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
