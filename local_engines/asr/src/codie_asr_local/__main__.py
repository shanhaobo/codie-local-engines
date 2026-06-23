"""Entry point for the local ASR sidecar.

Launched by the Bridge LocalEngineManager exactly as:
    <binary> --host 0.0.0.0 --port 9000
so `--host` and `--port` are the only flags the manager relies on. Everything
else has a sensible default and an env override.

Model storage: defaults to `models/` next to the binary, which (for the
Bridge's install layout `<dataDir>/ai-tools/codie-asr-local/<binary>`) lands the
weights under `<dataDir>/ai-tools/codie-asr-local/models`. Override with
`--model-dir` or `CODIE_ASR_MODEL_DIR`.

HuggingFace source: faster-whisper pulls weights through huggingface_hub, which
honors `HF_ENDPOINT`. We do NOT force a mirror — the canonical huggingface.co is
the default. Operators on a restricted network can export
`HF_ENDPOINT=https://<working-mirror>` (the Bridge can inject it per host).
Note: hf-mirror.com 308-redirects `/resolve/` back to huggingface.co for some
repos, so it is NOT a safe blanket default — verify any mirror end to end first.
"""

from __future__ import annotations

import argparse
import os
import sys


def _binary_dir() -> str:
    # PyInstaller onedir: sys.executable is the extracted binary; in a plain
    # `python -m` run, anchor next to this source file instead.
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _default_model_dir() -> str:
    env = os.environ.get("CODIE_ASR_MODEL_DIR")
    if env:
        return env
    return os.path.join(_binary_dir(), "models")


def main() -> int:
    parser = argparse.ArgumentParser(prog="codie-asr-local")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--model-dir", default=_default_model_dir())
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    import uvicorn

    from codie_asr_local.server import build_app

    app = build_app(model_dir=args.model_dir)
    print(
        f"[codie-asr-local] serving on {args.host}:{args.port} "
        f"(models in {args.model_dir}, "
        f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT', 'https://huggingface.co (default)')})",
        file=sys.stderr,
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
