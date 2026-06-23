#!/usr/bin/env bash
# Build the local TTS sidecar (our codie-tts-local wrapper around Piper) into a
# distributable archive. Runtime-download delivery (see build-asr-sidecar.sh
# header): produces dist/codie-tts-local-<os>-<arch>.tar.gz + prints its SHA-256
# for an operator to upload and paste into LocalEngineSpec
# (id codie-tts-local, matching platform).
#
# Usage:  bash codie-agent-bridge/scripts/build-tts-sidecar.sh
set -euo pipefail

cd "$(dirname "$0")/../local_engines/tts"
ID="codie-tts-local"

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
pyinstaller --clean --noconfirm codie_tts_local.spec

BUILT="dist/$ID"
[[ -x "$BUILT/$ID" ]] || { echo "pyinstaller did not produce $BUILT/$ID" >&2; exit 1; }

case "$(uname -s)" in
  Darwin*) OS="macos" ;;
  *) OS="linux" ;;
esac
ARCH="$(uname -m)"
case "$ARCH" in x86_64) ARCH="x64" ;; aarch64) ARCH="arm64" ;; esac
TARBALL="dist/${ID}-${OS}-${ARCH}.tar.gz"

rm -f "$TARBALL"
tar -czf "$TARBALL" -C "$BUILT" .

if command -v shasum >/dev/null 2>&1; then
  SHA="$(shasum -a 256 "$TARBALL" | awk '{print $1}')"
else
  SHA="$(sha256sum "$TARBALL" | awk '{print $1}')"
fi

echo ""
echo "[build-tts-sidecar] done."
echo "  artifact : $(cd "$(dirname "$TARBALL")" && pwd)/$(basename "$TARBALL")"
echo "  sha256   : $SHA"
echo ""
echo "  Next: upload this archive to your release host, then set in"
echo "  lib/models/local_engine_spec.dart (id codie-tts-local, matching platform):"
echo "    url:    '<download URL of the uploaded archive>'"
echo "    sha256: '$SHA'"
