import io
import wave

import numpy as np

from codie_asr_sherpa.audio import decode_to_16k_mono


def _sine_wav(seconds=0.5, rate=44100, freq=440) -> bytes:
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    pcm = (0.5 * np.sin(2 * np.pi * freq * t) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def test_decode_resamples_to_16k_mono_float():
    samples = decode_to_16k_mono(_sine_wav())
    assert samples.dtype == np.float32
    assert samples.ndim == 1
    # ~0.5s @ 16k ≈ 8000 samples (allow resampler edge slack).
    assert 7000 < len(samples) < 9000
    assert float(np.max(np.abs(samples))) <= 1.0
