# codie_asr_funasr.spec — PyInstaller config for the local FunASR ASR sidecar.
#
# Onedir (EXE + COLLECT), same rationale as the other engines: onefile would
# re-extract a very large bundle (torch native libs + funasr/modelscope) to a
# temp dir on EVERY launch. Onedir starts fast and runs as one process.
#
# The hard part vs the sherpa wrapper: funasr + modelscope resolve model /
# pipeline classes through a *string* registry (funasr.register.tables and the
# modelscope registry), which PyInstaller's static import analysis cannot follow.
# So we collect_all() those (and transformers/jieba, which also load pieces
# dynamically + ship data), and pull torch/torchaudio native libs explicitly.

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_dynamic_libs,
)

block_cipher = None

datas = []
binaries = []
hiddenimports = []

# Registry-driven packages: collect everything (modules + data + libs).
for pkg in ("funasr", "modelscope", "transformers", "jieba"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# torch / torchaudio: native libs (libtorch, C-extensions) + data. The
# hooks-contrib torch hook covers most; be explicit so the frozen bundle
# carries the shared libs even if the hook under-collects. (torch's
# *.dist-info/licenses/** deep tree that blows Windows MAX_PATH is stripped from
# the FINAL data TOC after Analysis — see the a.datas filter below.)
for pkg in ("torch", "torchaudio"):
    binaries += collect_dynamic_libs(pkg)
    datas += collect_data_files(pkg)
    hiddenimports.append(pkg)

# librosa ships resource data; soundfile bundles libsndfile; PyAV bundles ffmpeg.
datas += collect_data_files("librosa")
binaries += collect_dynamic_libs("soundfile")
binaries += collect_dynamic_libs("av")

hiddenimports += [
    # uvicorn's auto-detected loop/protocol impls are imported by string.
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
    "av",
]

a = Analysis(
    ["src/codie_asr_funasr/__main__.py"],
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

# Windows MAX_PATH (260): several wheels (torch worst of all) ship a
# *.dist-info/licenses/** third-party license tree nested absurdly deep — torch
# reaches …/kineto/libkineto/…/prometheus-cpp/…/civetweb/…/duktape-1.5.2. Once
# prefixed with the onedir output dir it exceeds 260 chars and COLLECT dies
# copying it: [WinError 206] 文件名或扩展名太长. The PyInstaller torch hook re-adds
# the full dist-info regardless of any collect_* filtering we do above, so strip
# the license subtree from the FINAL data TOC here. Drop ONLY .dist-info/licenses/**
# — METADATA/RECORD stay so importlib.metadata.version (transformers' dep-version
# probes) still resolves. License texts are runtime-dead; harmless on macOS too.
a.datas = [
    entry
    for entry in a.datas
    if ".dist-info/licenses" not in entry[0].replace("\\", "/")
]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="codie-asr-funasr",
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

# Onedir output: dist/codie-asr-funasr/{codie-asr-funasr(.exe), _internal/}.
# The binary is named after our wrapper service (= the LocalEngineSpec id), not
# the embedded engine. The Bridge install layout expects the launchable binary
# at <dataDir>/ai-tools/codie-asr-funasr/codie-asr-funasr(.exe).
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="codie-asr-funasr",
)
