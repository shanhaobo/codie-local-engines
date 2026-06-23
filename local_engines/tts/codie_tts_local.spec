# codie_tts_local.spec — PyInstaller config for the local TTS sidecar.
#
# Onedir (EXE + COLLECT), same rationale as the ASR/host_mcp specs. Piper pulls
# in onnxruntime + piper-phonemize native libs / espeak-ng data that static
# analysis misses; collect_all() grabs them wholesale.

from PyInstaller.utils.hooks import collect_all

block_cipher = None

datas = []
binaries = []
hiddenimports = [
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
]

for _pkg in (
    "piper",
    "onnxruntime",
    "huggingface_hub",
):
    _d, _b, _h = collect_all(_pkg)
    datas += _d
    binaries += _b
    hiddenimports += _h

a = Analysis(
    ["src/codie_tts_local/__main__.py"],
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
    name="codie-tts-local",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# Onedir output: dist/codie-tts-local/{codie-tts-local(.exe), _internal/}. The
# binary is named after our wrapper service (= the LocalEngineSpec id), not the
# embedded Piper engine. The Bridge expects the launchable binary at
# <dataDir>/ai-tools/codie-tts-local/codie-tts-local(.exe).
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="codie-tts-local",
)
