"""Decode arbitrary audio containers to 16 kHz mono float32, via PyAV.

FunASR's paraformer model wants 16 kHz mono float samples; we own the container
demux/resample here (rather than letting funasr's torchaudio/soundfile path try
to open an arbitrary upload format), so any container the client sends works.
"""

from __future__ import annotations

import io

import numpy as np


def decode_to_16k_mono(data: bytes) -> "np.ndarray":
    # Lazy import: keeps app construction + /health free of the av import graph.
    import av

    resampler = av.audio.resampler.AudioResampler(format="flt", layout="mono", rate=16000)
    chunks: list[np.ndarray] = []

    def _collect(frame) -> None:
        for rf in resampler.resample(frame):
            arr = rf.to_ndarray()  # mono "flt" -> shape (1, n)
            chunks.append(np.asarray(arr, dtype=np.float32).reshape(-1))

    # Context-managed so a decode error on garbage/truncated input still releases
    # the native container + BytesIO — otherwise a stream of bad clips leaks fds
    # in the long-lived sidecar.
    with av.open(io.BytesIO(data)) as container:
        for frame in container.decode(audio=0):
            _collect(frame)
        _collect(None)  # flush the resampler tail

    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)
