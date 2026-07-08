#!/usr/bin/env bash
# Build the local FunASR ASR sidecar (our codie-asr-funasr wrapper around
# funasr's Paraformer-zh, torch) into a distributable archive. Runtime-download
# delivery (see build-asr-sidecar.sh header): produces
# dist/codie-asr-funasr-<os>-<arch>.tar.gz + prints its SHA-256, for an
# operator to upload and paste into LocalEngineSpec (id codie-asr-funasr,
# matching platform).
#
# macOS-arm64 ONLY for now — torch has shipped no macOS-x86_64 wheel since
# 2.2.2, so this script is only expected to run on Apple Silicon (see
# docs/plans/2026-07-08-funasr-asr-engine.md). Needs Python 3.11 on PATH (the
# funasr/torch/modelscope dep tree has no cp313 wheels as of the 2026-07-08
# spike) — CI's setup-python step pins this; for a local run, make sure
# `python3`/`python` resolves to 3.11 before invoking.
#
# The archive's root holds the launchable binary + _internal/, so the Bridge
# can extract it straight into <dataDir>/ai-tools/codie-asr-funasr/ and run
# ./codie-asr-funasr there.
#
# Usage:  bash scripts/build-asr-funasr-sidecar.sh
set -euo pipefail

cd "$(dirname "$0")/../local_engines/asr-funasr"
ID="codie-asr-funasr"

if [[ ! -d .venv ]]; then
  if command -v python3 >/dev/null 2>&1; then python3 -m venv .venv; else python -m venv .venv; fi
fi
if [[ -f .venv/Scripts/activate ]]; then source .venv/Scripts/activate; else source .venv/bin/activate; fi

case "$(uname -s)" in
  Darwin*) EXTRAS="dev,mac" ;;
  MINGW*|MSYS*|CYGWIN*) EXTRAS="dev,win" ;;
  *) EXTRAS="dev" ;;
esac
pip install -e ".[${EXTRAS}]" >/dev/null

rm -rf build dist
pyinstaller --clean --noconfirm codie_asr_funasr.spec

BUILT="dist/$ID"
[[ -x "$BUILT/$ID" ]] || { echo "pyinstaller did not produce $BUILT/$ID" >&2; exit 1; }

# Name the artifact by OS + arch.
case "$(uname -s)" in
  Darwin*) OS="macos" ;;
  *) OS="linux" ;;
esac
# Normalize to amd64 / arm64 (uname -m is x86_64 on Intel, arm64/aarch64 on ARM).
ARCH="$(uname -m)"
case "$ARCH" in x86_64|amd64) ARCH="amd64" ;; arm64|aarch64) ARCH="arm64" ;; esac
TARBALL="dist/${ID}-${OS}-${ARCH}.tar.gz"

# Archive the CONTENTS of dist/<id>/ (binary + _internal/) at the root, so
# extraction lands the binary directly at <dataDir>/ai-tools/<id>/<id>.
rm -f "$TARBALL"
tar -czf "$TARBALL" -C "$BUILT" .

if command -v shasum >/dev/null 2>&1; then
  SHA="$(shasum -a 256 "$TARBALL" | awk '{print $1}')"
else
  SHA="$(sha256sum "$TARBALL" | awk '{print $1}')"
fi

echo ""
echo "[build-asr-funasr-sidecar] done."
echo "  artifact : $(cd "$(dirname "$TARBALL")" && pwd)/$(basename "$TARBALL")"
echo "  sha256   : $SHA"
echo ""
echo "  Next: upload this archive to your release host, then set in"
echo "  lib/models/local_engine_spec.dart (id codie-asr-funasr, matching platform):"
echo "    url:    '<download URL of the uploaded archive>'"
echo "    sha256: '$SHA'"
