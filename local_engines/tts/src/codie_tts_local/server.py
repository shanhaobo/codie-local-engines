"""FastAPI app: OpenAI-compatible TTS over Piper.

Endpoints:
  GET  /health                 -> {"status": "ok"}  (cheap, no voice load)
  POST /v1/audio/speech        -> audio/wav bytes
                                  body: {input, voice, response_format}

Only `wav` output is supported today (Piper emits 16-bit PCM WAV natively);
other `response_format` values are rejected rather than silently mis-served.

The heavy `piper` import is lazy (inside the voice loader) so /health and app
construction stay instant for the Bridge's bounded health probe. Voices are
downloaded on first use (honoring HF_ENDPOINT) and cached per voice id.
"""

from __future__ import annotations

import io
import os
import threading
import wave

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from codie_tts_local.voices import _repo_paths, ensure_voice, with_quality

# Catalog voices the Bridge AI 工具 page lists (and asks status for).
_CATALOG = ["en_US-amy", "en_US-ryan", "zh_CN-huayan"]


def _voice_files(voices_dir: str, voice: str) -> tuple[str, str]:
    """Local (onnx, onnx.json) paths for `voice` under `voices_dir`."""
    onnx_rel, json_rel = _repo_paths(with_quality(voice))
    return (
        os.path.join(voices_dir, onnx_rel.replace("/", os.sep)),
        os.path.join(voices_dir, json_rel.replace("/", os.sep)),
    )


def _is_cached(voices_dir: str, voice: str) -> bool:
    try:
        onnx, _json = _voice_files(voices_dir, voice)
        return os.path.exists(onnx)
    except Exception:
        return False


def _cached_bytes(voices_dir: str, voice: str) -> int:
    total = 0
    try:
        for p in _voice_files(voices_dir, voice):
            if os.path.exists(p):
                total += os.path.getsize(p)
    except Exception:
        pass
    return total


class Downloads:
    """Tracks in-flight background voice downloads by id, thread-safely."""

    def __init__(self, voices_dir: str) -> None:
        self._voices_dir = voices_dir
        self._lock = threading.Lock()
        self._active: set[str] = set()

    def is_downloading(self, voice: str) -> bool:
        with self._lock:
            return voice in self._active

    def start(self, voice: str) -> str:
        if _is_cached(self._voices_dir, voice):
            return "exists"
        with self._lock:
            if voice in self._active:
                return "downloading"
            self._active.add(voice)

        def _run() -> None:
            try:
                ensure_voice(voice, self._voices_dir)
            except Exception:  # noqa: BLE001 — surfaced via status (downloaded stays false)
                pass
            finally:
                with self._lock:
                    self._active.discard(voice)

        threading.Thread(target=_run, name=f"dl-{voice}", daemon=True).start()
        return "started"


class SpeechRequest(BaseModel):
    input: str
    voice: str = "en_US-amy"
    response_format: str = "wav"


class VoiceCache:
    """Lazily loads + caches Piper voices by id, thread-safely."""

    def __init__(self, voices_dir: str) -> None:
        self._voices_dir = voices_dir
        self._lock = threading.Lock()
        self._voices: dict = {}

    def get(self, voice_id: str):
        with self._lock:
            cached = self._voices.get(voice_id)
            if cached is not None:
                return cached
            from piper import PiperVoice

            onnx_path = ensure_voice(voice_id, self._voices_dir)
            voice = PiperVoice.load(onnx_path)
            self._voices[voice_id] = voice
            return voice


def _synthesize_wav(voice, text: str) -> bytes:
    """Render `text` to a complete in-memory WAV byte string.

    Verified against piper-tts 1.4.2: `synthesize_wav(text, wav_file)` writes a
    complete WAV (it sets channels/sampwidth/framerate itself). Falls back to the
    chunked `synthesize()` API (also 1.4.x) if `synthesize_wav` is unavailable.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        if hasattr(voice, "synthesize_wav"):
            voice.synthesize_wav(text, wav_file)
        else:  # pragma: no cover - alternate piper API surface
            chunks = list(voice.synthesize(text))
            first = chunks[0]
            wav_file.setnchannels(first.sample_channels)
            wav_file.setsampwidth(first.sample_width)
            wav_file.setframerate(first.sample_rate)
            for chunk in chunks:
                wav_file.writeframes(chunk.audio_int16_bytes)
    return buf.getvalue()


def build_app(*, voices_dir: str) -> FastAPI:
    app = FastAPI(title="codie-tts-local", version="0.1.2")
    cache = VoiceCache(voices_dir)
    downloads = Downloads(voices_dir)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/v1/models")
    def list_models() -> dict:
        """Per-voice download status for the Bridge AI 工具 page."""
        return {
            "models": [
                {
                    "name": v,
                    "downloaded": _is_cached(voices_dir, v),
                    "downloading": downloads.is_downloading(v),
                    "bytes": _cached_bytes(voices_dir, v),
                }
                for v in _CATALOG
            ]
        }

    @app.post("/v1/models/{name}/download")
    def download_voice_route(name: str) -> dict:
        """Deliberately pre-fetch a voice (background). Idempotent."""
        return {"name": name, "status": downloads.start(name)}

    @app.post("/v1/audio/speech")
    def speech(req: SpeechRequest) -> Response:
        if req.response_format.lower() != "wav":
            raise HTTPException(
                status_code=400,
                detail=f"unsupported response_format {req.response_format!r}; only 'wav'",
            )
        if not req.input.strip():
            raise HTTPException(status_code=400, detail="empty input text")

        try:
            voice = cache.get(req.voice)
            audio = _synthesize_wav(voice, req.input)
        except Exception as e:  # noqa: BLE001 — surface voice/synth errors
            raise HTTPException(status_code=500, detail=f"synthesis failed: {e}")

        return Response(content=audio, media_type="audio/wav")

    return app
