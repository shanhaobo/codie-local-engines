"""Resolve + download Piper voices from the HuggingFace `rhasspy/piper-voices` repo.

A Piper voice id is `<locale>-<name>-<quality>`, e.g. `en_US-amy-medium`. The
Bridge catalog uses the short form `en_US-amy` (no quality); `_with_quality`
appends a default. Each voice is two files in the repo, laid out as
`<lang>/<locale>/<name>/<quality>/<voice>.onnx(.json)`, e.g.
`en/en_US/amy/medium/en_US-amy-medium.onnx`.
"""

from __future__ import annotations

import os

_REPO_ID = "rhasspy/piper-voices"
_DEFAULT_QUALITY = "medium"


def with_quality(voice: str) -> str:
    """Append the default quality when the catalog short form omits it.

    `en_US-amy` -> `en_US-amy-medium`; `en_US-amy-high` passes through.
    """
    v = (voice or "").strip()
    parts = v.split("-")
    # `<locale>-<name>` has 2 parts; `<locale>-<name>-<quality>` has 3.
    if len(parts) == 2:
        return f"{v}-{_DEFAULT_QUALITY}"
    return v


def _repo_paths(full_voice: str) -> tuple[str, str]:
    """Map a full voice id to its (onnx, onnx.json) paths inside the HF repo."""
    parts = full_voice.split("-")
    if len(parts) < 3:
        raise ValueError(f"malformed voice id {full_voice!r} (want locale-name-quality)")
    locale, name, quality = parts[0], parts[1], parts[2]
    lang = locale.split("_")[0]
    base = f"{lang}/{locale}/{name}/{quality}/{full_voice}"
    return f"{base}.onnx", f"{base}.onnx.json"


def ensure_voice(voice: str, voices_dir: str) -> str:
    """Download the voice's .onnx + .onnx.json into `voices_dir` if missing.

    Returns the local path to the `.onnx` model. Honors HF_ENDPOINT (mirror)
    via huggingface_hub.
    """
    from huggingface_hub import hf_hub_download

    full = with_quality(voice)
    onnx_rel, json_rel = _repo_paths(full)
    os.makedirs(voices_dir, exist_ok=True)

    onnx_path = hf_hub_download(
        repo_id=_REPO_ID,
        filename=onnx_rel,
        local_dir=voices_dir,
    )
    hf_hub_download(
        repo_id=_REPO_ID,
        filename=json_rel,
        local_dir=voices_dir,
    )
    return onnx_path
