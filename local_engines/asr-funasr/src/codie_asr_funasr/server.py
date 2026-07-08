"""FastAPI app: OpenAI-compatible ASR over FunASR (Paraformer-zh, torch).

Endpoints (identical contract to the faster-whisper / sherpa-onnx wrappers, so the
Bridge's generic sidecar path drives all three without any adapter):
  GET  /health                      -> {"status": "ok"}   (cheap, no model load)
  GET  /v1/models                   -> per-variant download status
  POST /v1/models/{name}/download   -> background weight download (ModelScope)
  POST /v1/audio/transcriptions     -> {"text": "..."}    (multipart: file, model)

Design notes:
- The heavy `funasr` (-> torch) / `modelscope` / `av` imports are LAZY (inside
  loaders/threads), so app construction and /health stay instant — the Bridge's
  bounded health probe must pass long before any model is downloaded/loaded.
"""

from __future__ import annotations

import os
import shutil
import threading

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from codie_asr_funasr.audio import decode_to_16k_mono

# name -> ModelScope coordinates. `dir` is the folder we download into under
# model_dir; `files` are the members that must exist for "downloaded".
_MODEL_SOURCES = {
    "paraformer-zh": {
        # The canonical model funasr's `model="paraformer-zh"` alias resolves to.
        "model_id": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "revision": "v2.0.4",
        "dir": "paraformer-zh",
        # model.pt = the weights (~840 MB); configuration.json = what funasr's
        # AutoModel reads when pointed at a local dir. Both present => downloaded.
        # (Filenames verified against the actual snapshot during the spike.)
        "files": ["model.pt", "configuration.json"],
    },
}

# The variants GET /v1/models reports, derived from the source table (single
# source of truth — never drift from _MODEL_SOURCES).
_CATALOG = list(_MODEL_SOURCES)


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


def _download_model(model_dir: str, name: str) -> None:
    """Download the ModelScope snapshot into <model_dir>/<dir>. Seam for tests.

    Atomic: snapshot into a sibling `<dest>.part` and only `os.replace` to `dest`
    once the required files are present, so `_is_cached` never observes a partial
    download (a crash mid-fetch leaves `dest` untouched -> still "not downloaded",
    so a later retry works instead of stranding a corrupt "downloaded" dir).
    """
    # Lazy import: modelscope pulls a heavy graph; keep it out of app construction.
    from modelscope import snapshot_download

    src = _MODEL_SOURCES[name]
    dest = _model_dir_for(model_dir, name)
    staging = dest + ".part"
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging, exist_ok=True)

    base = os.environ.get("CODIE_FUNASR_MODEL_BASE")  # optional mirror host
    if base:
        os.environ.setdefault("MODELSCOPE_DOMAIN", base)

    try:
        snapshot_download(src["model_id"], revision=src["revision"], local_dir=staging)
        missing = [f for f in src["files"] if not os.path.isfile(os.path.join(staging, f))]
        if missing:
            raise RuntimeError(f"snapshot missing required files: {missing}")
        shutil.rmtree(dest, ignore_errors=True)
        os.replace(staging, dest)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


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
            _download_model(self._model_dir, name)
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


class _Recognizer:
    """Thin wrapper over a funasr AutoModel with a `transcribe` seam."""

    def __init__(self, model) -> None:
        self._m = model

    def transcribe(self, samples, language=None) -> str:
        # `language` accepted for OpenAI-contract symmetry but ignored: the
        # paraformer-zh model is single-language (Chinese). funasr accepts a raw
        # 1-D float waveform via `input=` with the sample rate in `fs`.
        res = self._m.generate(input=samples, fs=16000)
        if not res:
            return ""
        return (res[0].get("text") or "").strip()


class ModelCache:
    """Lazily loads and caches the funasr AutoModel by model name."""

    def __init__(self, model_dir: str) -> None:
        self._model_dir = model_dir
        self._lock = threading.Lock()
        self._models: dict[str, _Recognizer] = {}

    def get(self, name: str) -> "_Recognizer":
        with self._lock:
            cached = self._models.get(name)
            if cached is not None:
                return cached
            # Lazy import: keeps app construction + /health free of the heavy
            # funasr -> torch import graph.
            from funasr import AutoModel

            d = _model_dir_for(self._model_dir, name)
            device = os.environ.get("CODIE_FUNASR_DEVICE", "cpu")
            model = AutoModel(model=d, disable_update=True, device=device)
            wrapped = _Recognizer(model)
            self._models[name] = wrapped
            return wrapped


def build_app(*, model_dir: str) -> FastAPI:
    app = FastAPI(title="codie-asr-funasr", version="0.1.0")
    downloads = Downloads(model_dir)
    cache = ModelCache(model_dir)

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

    @app.post("/v1/audio/transcriptions")
    async def transcribe(
        file: UploadFile = File(...),
        model: str = Form("paraformer-zh"),
        language: str | None = Form(None),
    ) -> dict:
        if model not in _MODEL_SOURCES:
            raise HTTPException(status_code=400, detail=f"unknown model {model!r}")
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="empty audio file")
        if not _is_cached(model_dir, model):
            raise HTTPException(
                status_code=409,
                detail=f"model {model!r} not downloaded — POST /v1/models/{model}/download first",
            )
        try:
            samples = decode_to_16k_mono(data)
            text = cache.get(model).transcribe(samples, language=language)
        except Exception as e:  # noqa: BLE001 — surface decode/transcribe errors
            raise HTTPException(status_code=500, detail=f"transcription failed: {e}")
        return {"text": text}

    return app
