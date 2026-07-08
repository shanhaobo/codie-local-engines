"""Entry point for the local FunASR ASR sidecar.

Launched by the Bridge LocalEngineManager exactly as:
    <binary> --host 0.0.0.0 --port 9002
so `--host` and `--port` are the only flags the manager relies on. Everything
else has a sensible default and an env override.

Model storage: defaults to `models/` next to the binary, which (for the Bridge's
install layout `<dataDir>/ai-tools/codie-asr-funasr/<binary>`) lands weights
under `<dataDir>/ai-tools/codie-asr-funasr/models`. Override with `--model-dir`
or `CODIE_FUNASR_MODEL_DIR`.

Model source: Paraformer-zh weights are downloaded on first use from ModelScope.
Override the mirror host with `CODIE_FUNASR_MODEL_BASE`.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys


def _binary_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _default_model_dir() -> str:
    env = os.environ.get("CODIE_FUNASR_MODEL_DIR")
    if env:
        return env
    return os.path.join(_binary_dir(), "models")


def main() -> int:
    # MUST be the first call: in a PyInstaller-frozen build, funasr/torch spawn
    # multiprocessing children by re-exec'ing THIS binary with mp bootstrap args
    # (`-B -S -I -c "from multiprocessing.resource_tracker import main;..."`).
    # Without freeze_support() the child falls through to argparse, which errors
    # on those args and leaves broken/leaked worker processes. No-op when running
    # from source or as the parent process.
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(prog="codie-asr-funasr")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9002)
    parser.add_argument("--model-dir", default=_default_model_dir())
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    import uvicorn

    from codie_asr_funasr.server import build_app

    app = build_app(model_dir=args.model_dir)
    print(
        f"[codie-asr-funasr] serving on {args.host}:{args.port} "
        f"(models in {args.model_dir})",
        file=sys.stderr,
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
