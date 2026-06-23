# codie_asr_local.spec — PyInstaller config for the local ASR sidecar.
#
# Onedir (EXE + COLLECT), same rationale as codie_host_mcp.spec: onefile would
# re-extract a very large bundle (ctranslate2 + av/ffmpeg + onnxruntime) to a
# temp dir on EVERY launch. Onedir starts fast and runs as one process.
#
# faster-whisper drags in heavy native packages whose data files / shared libs
# PyInstaller's static analysis misses; collect_all() grabs each one's
# binaries + datas + submodules wholesale.

from PyInstaller.utils.hooks import collect_all

block_cipher = None

datas = []
binaries = []
hiddenimports = [
    # uvicorn's auto-detected loop/protocol impls are imported by string.
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
]

# Heavy packages whose binaries + data must be collected wholesale.
for _pkg in (
    "faster_whisper",
    "ctranslate2",
    "av",
    "onnxruntime",
    "tokenizers",
    "huggingface_hub",
):
    _d, _b, _h = collect_all(_pkg)
    datas += _d
    binaries += _b
    hiddenimports += _h

a = Analysis(
    ["src/codie_asr_local/__main__.py"],
    pathex=["src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="codie-asr-local",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # Console bootloader so the Bridge supervisor can read stderr diagnostics.
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# Onedir output: dist/codie-asr-local/{codie-asr-local(.exe), _internal/}.
# The binary is named after our wrapper service (= the LocalEngineSpec id), not
# the embedded engine. The Bridge install layout expects the launchable binary
# at <dataDir>/ai-tools/codie-asr-local/codie-asr-local(.exe).
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="codie-asr-local",
)
