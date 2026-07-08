"""Entry point for the local sherpa-onnx ASR sidecar.

Launched by the Bridge LocalEngineManager exactly as:
    <binary> --host 0.0.0.0 --port 9001
so `--host` and `--port` are the only flags the manager relies on. Everything
else has a sensible default and an env override.

Model storage: defaults to `models/` next to the binary, which (for the Bridge's
install layout `<dataDir>/ai-tools/codie-asr-sherpa/<binary>`) lands weights
under `<dataDir>/ai-tools/codie-asr-sherpa/models`. Override with `--model-dir`
or `CODIE_SHERPA_MODEL_DIR`.

Model source: SenseVoice weights are downloaded on first use from the sherpa-onnx
official release tarball. Override the base with `CODIE_SHERPA_MODEL_BASE`.
"""

from __future__ import annotations

import argparse
import os
import sys


def _binary_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _default_model_dir() -> str:
    env = os.environ.get("CODIE_SHERPA_MODEL_DIR")
    if env:
        return env
    return os.path.join(_binary_dir(), "models")


def main() -> int:
    parser = argparse.ArgumentParser(prog="codie-asr-sherpa")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--model-dir", default=_default_model_dir())
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    import uvicorn

    from codie_asr_sherpa.server import build_app

    app = build_app(model_dir=args.model_dir)
    print(
        f"[codie-asr-sherpa] serving on {args.host}:{args.port} "
        f"(models in {args.model_dir})",
        file=sys.stderr,
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
