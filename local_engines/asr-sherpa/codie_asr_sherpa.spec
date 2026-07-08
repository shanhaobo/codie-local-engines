# codie_asr_sherpa.spec — PyInstaller config for the local sherpa-onnx ASR sidecar.
#
# Onedir (EXE + COLLECT), same rationale as codie_asr_local.spec: onefile would
# re-extract a large bundle (sherpa-onnx onnxruntime libs + av/ffmpeg) to a temp
# dir on EVERY launch. Onedir starts fast and runs as one process.
#
# sherpa-onnx ships native runtime shared libs + data files, and PyAV ships its
# own bundled ffmpeg shared libs; PyInstaller's static analysis misses both, so
# we collect their dynamic libs + data explicitly. (PyAV's ffmpeg libs are
# usually grabbed by pyinstaller-hooks-contrib, but we list `av` binaries +
# hiddenimport explicitly to be safe.)

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

block_cipher = None

# sherpa-onnx: native runtime shared libs + data. PyAV: bundled ffmpeg shared libs.
datas = collect_data_files("sherpa_onnx")
binaries = collect_dynamic_libs("sherpa_onnx") + collect_dynamic_libs("av")
hiddenimports = [
    # uvicorn's auto-detected loop/protocol impls are imported by string.
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
    # Heavy native packages loaded lazily inside the loaders — pin them so the
    # frozen bundle carries the Python extension modules, not just the libs above.
    "sherpa_onnx",
    "av",
]

a = Analysis(
    ["src/codie_asr_sherpa/__main__.py"],
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
    name="codie-asr-sherpa",
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

# Onedir output: dist/codie-asr-sherpa/{codie-asr-sherpa(.exe), _internal/}.
# The binary is named after our wrapper service (= the LocalEngineSpec id), not
# the embedded engine. The Bridge install layout expects the launchable binary
# at <dataDir>/ai-tools/codie-asr-sherpa/codie-asr-sherpa(.exe).
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="codie-asr-sherpa",
)
