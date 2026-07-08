import io
import os
import tarfile

from codie_asr_sherpa import server


def _fake_tarball_bytes() -> bytes:
    """A .tar.bz2 whose top-level dir mimics the sherpa release layout."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as tf:
        for member, data in [
            ("sherpa-onnx-sense-voice-xxx/model.int8.onnx", b"ONNX"),
            ("sherpa-onnx-sense-voice-xxx/tokens.txt", b"<blk>\n"),
            ("sherpa-onnx-sense-voice-xxx/test_wavs/zh.wav", b"junk"),
        ]:
            info = tarfile.TarInfo(member)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_download_extracts_and_flattens(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_fetch_bytes", lambda url: _fake_tarball_bytes())
    dl = server.Downloads(str(tmp_path))

    status = dl.start("sensevoice-small")
    assert status == "started"
    dl.join_for_test("sensevoice-small")  # deterministic wait (test-only helper)

    d = os.path.join(str(tmp_path), "sensevoice-small")
    assert os.path.isfile(os.path.join(d, "model.int8.onnx"))
    assert os.path.isfile(os.path.join(d, "tokens.txt"))
    assert server._is_cached(str(tmp_path), "sensevoice-small") is True

    # Idempotent: a second start sees it cached.
    assert dl.start("sensevoice-small") == "exists"
