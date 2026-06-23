"""Entry point for the local TTS sidecar.

Launched by the Bridge LocalEngineManager exactly as:
    <binary> --host 0.0.0.0 --port 9100
so `--host` and `--port` are the only flags the manager relies on.

Voice storage defaults to `voices/` next to the binary, which for the Bridge
install layout `<dataDir>/ai-tools/codie-tts-local/<binary>` lands them under
`<dataDir>/ai-tools/codie-tts-local/voices`. Override with `--voices-dir` or
`CODIE_TTS_VOICES_DIR`.

HuggingFace source: voices come from the `rhasspy/piper-voices` repo via
huggingface_hub. We do NOT force a mirror — huggingface.co is the default;
operators on a restricted network can export `HF_ENDPOINT=https://<mirror>`
(verify the mirror serves `/resolve/` end to end — hf-mirror.com redirects some
repos back to upstream).
"""

from __future__ import annotations

import argparse
import os
import sys


def _binary_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _default_voices_dir() -> str:
    env = os.environ.get("CODIE_TTS_VOICES_DIR")
    if env:
        return env
    return os.path.join(_binary_dir(), "voices")


def main() -> int:
    parser = argparse.ArgumentParser(prog="codie-tts-local")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--voices-dir", default=_default_voices_dir())
    args = parser.parse_args()

    os.makedirs(args.voices_dir, exist_ok=True)

    import uvicorn

    from codie_tts_local.server import build_app

    app = build_app(voices_dir=args.voices_dir)
    print(
        f"[codie-tts-local] serving on {args.host}:{args.port} "
        f"(voices in {args.voices_dir}, "
        f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT', 'https://huggingface.co (default)')})",
        file=sys.stderr,
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
