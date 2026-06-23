#!/usr/bin/env bash
# Build the local ASR sidecar (our codie-asr-local wrapper around faster-whisper)
# into a distributable archive.
#
# Unlike build-sidecar.sh (which bundles host_mcp into the app Resources), the
# ASR/TTS sidecars are delivered by RUNTIME DOWNLOAD: this script produces
# dist/codie-asr-local-<os>-<arch>.tar.gz + prints its SHA-256, for an operator
# to upload and paste into LocalEngineSpec (id codie-asr-local, matching platform).
#
# The archive's root holds the launchable binary + _internal/, so the Bridge
# can extract it straight into <dataDir>/ai-tools/codie-asr-local/ and run
# ./codie-asr-local there.
#
# Usage:  bash codie-agent-bridge/scripts/build-asr-sidecar.sh
set -euo pipefail

cd "$(dirname "$0")/../local_engines/asr"
ID="codie-asr-local"

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
pyinstaller --clean --noconfirm codie_asr_local.spec

BUILT="dist/$ID"
[[ -x "$BUILT/$ID" ]] || { echo "pyinstaller did not produce $BUILT/$ID" >&2; exit 1; }

# Name the artifact by OS + arch.
case "$(uname -s)" in
  Darwin*) OS="macos" ;;
  *) OS="linux" ;;
esac
ARCH="$(uname -m)"   # arm64 / x86_64
case "$ARCH" in x86_64) ARCH="x64" ;; aarch64) ARCH="arm64" ;; esac
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
echo "[build-asr-sidecar] done."
echo "  artifact : $(cd "$(dirname "$TARBALL")" && pwd)/$(basename "$TARBALL")"
echo "  sha256   : $SHA"
echo ""
echo "  Next: upload this archive to your release host, then set in"
echo "  lib/models/local_engine_spec.dart (id codie-asr-local, matching platform):"
echo "    url:    '<download URL of the uploaded archive>'"
echo "    sha256: '$SHA'"
