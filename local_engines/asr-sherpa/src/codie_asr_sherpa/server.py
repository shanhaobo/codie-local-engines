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

import io
import os
import shutil
import tarfile
import threading
import urllib.request

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from codie_asr_sherpa.audio import decode_to_16k_mono

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

# The variants GET /v1/models reports, derived from the source table (single
# source of truth — never drift from _MODEL_SOURCES).
_CATALOG = list(_MODEL_SOURCES)

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
    the required files to `dest_dir/<basename>`.

    Atomic: files are staged into a sibling `<dest_dir>.part` dir and only
    published to `dest_dir` (via os.replace) once every required file is present.
    So `_is_cached` never observes a truncated/partial model — a crash mid-write
    or a missing member leaves `dest_dir` untouched (still "not downloaded"),
    which lets a later `Downloads.start` retry instead of stranding the user on a
    corrupt "downloaded" dir.
    """
    staging = dest_dir + ".part"
    shutil.rmtree(staging, ignore_errors=True)  # drop any stale partial
    os.makedirs(staging, exist_ok=True)
    wanted = set(required)
    try:
        # tarfile decompresses bz2 natively via `r:bz2` — no bz2.BZ2File wrapper.
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:bz2") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                base = os.path.basename(member.name)
                if base in wanted:
                    src = tf.extractfile(member)
                    if src is None:
                        continue
                    with open(os.path.join(staging, base), "wb") as out:
                        shutil.copyfileobj(src, out)
        missing = [f for f in required if not os.path.isfile(os.path.join(staging, f))]
        if missing:
            raise RuntimeError(f"tarball missing required files: {missing}")
        # Atomic publish: dest_dir is absent here (start() short-circuits on a
        # complete cache), so this rename exposes only the fully-staged dir.
        shutil.rmtree(dest_dir, ignore_errors=True)
        os.replace(staging, dest_dir)
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


class _Recognizer:
    """Thin wrapper over sherpa_onnx OfflineRecognizer with a `transcribe` seam."""

    def __init__(self, recognizer) -> None:
        self._r = recognizer

    def transcribe(self, samples, language=None) -> str:
        # `language` is accepted for OpenAI-contract symmetry but intentionally
        # ignored: SenseVoice auto-detects language (zh/en/ja/ko/yue), so there's
        # no per-request language knob to forward to the recognizer.
        s = self._r.create_stream()
        s.accept_waveform(16000, samples)
        self._r.decode_stream(s)
        return (s.result.text or "").strip()


class ModelCache:
    """Lazily loads and caches the sherpa-onnx recognizer by model name."""

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
            # sherpa_onnx import graph.
            import sherpa_onnx

            d = _model_dir_for(self._model_dir, name)
            provider = os.environ.get("CODIE_SHERPA_PROVIDER", "cpu")
            recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=os.path.join(d, "model.int8.onnx"),
                tokens=os.path.join(d, "tokens.txt"),
                num_threads=int(os.environ.get("CODIE_SHERPA_THREADS", "2")),
                use_itn=True,
                provider=provider,
                debug=False,
            )
            wrapped = _Recognizer(recognizer)
            self._models[name] = wrapped
            return wrapped


def build_app(*, model_dir: str) -> FastAPI:
    app = FastAPI(title="codie-asr-sherpa", version="0.1.0")
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
        model: str = Form("sensevoice-small"),
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
