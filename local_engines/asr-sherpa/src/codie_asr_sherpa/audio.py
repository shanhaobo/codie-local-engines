"""Decode arbitrary audio containers to 16 kHz mono float32, via PyAV.

sherpa-onnx's recognizer only accepts raw float samples; unlike faster-whisper
it does no container demux/resample itself, so the wrapper owns decoding.
"""

from __future__ import annotations

import io

import numpy as np


def decode_to_16k_mono(data: bytes) -> "np.ndarray":
    # Lazy import: keeps app construction + /health free of the av import graph.
    import av

    container = av.open(io.BytesIO(data))
    resampler = av.audio.resampler.AudioResampler(format="flt", layout="mono", rate=16000)
    chunks: list[np.ndarray] = []

    def _collect(frame) -> None:
        for rf in resampler.resample(frame):
            arr = rf.to_ndarray()  # mono "flt" -> shape (1, n)
            chunks.append(np.asarray(arr, dtype=np.float32).reshape(-1))

    for frame in container.decode(audio=0):
        _collect(frame)
    _collect(None)  # flush the resampler tail
    container.close()

    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)
