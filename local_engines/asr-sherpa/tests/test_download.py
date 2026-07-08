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
    # Flatten keeps only the required members; the tarball's test_wavs/zh.wav
    # junk must NOT land in dest_dir.
    assert not os.path.isfile(os.path.join(d, "zh.wav"))
    assert server._is_cached(str(tmp_path), "sensevoice-small") is True

    # The staging dir is cleaned up after a successful atomic publish.
    assert not os.path.exists(d + ".part")

    # Idempotent: a second start sees it cached.
    assert dl.start("sensevoice-small") == "exists"


def _tarball_missing_tokens() -> bytes:
    """A .tar.bz2 that omits the required tokens.txt member."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as tf:
        info = tarfile.TarInfo("sherpa-onnx-sense-voice-xxx/model.int8.onnx")
        data = b"ONNX"
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_download_missing_required_leaves_uncached(tmp_path, monkeypatch):
    # A tarball missing a required file raises inside the worker; the error is
    # swallowed (surfaced via status) and dest_dir is NOT published, so the model
    # stays "not downloaded" and a retry is possible — no half-written strand.
    monkeypatch.setattr(server, "_fetch_bytes", lambda url: _tarball_missing_tokens())
    dl = server.Downloads(str(tmp_path))

    assert dl.start("sensevoice-small") == "started"
    dl.join_for_test("sensevoice-small")

    d = os.path.join(str(tmp_path), "sensevoice-small")
    assert server._is_cached(str(tmp_path), "sensevoice-small") is False
    assert not os.path.exists(d)  # partial never published
    assert not os.path.exists(d + ".part")  # staging cleaned up
    # Not stranded: a subsequent start re-attempts (isn't short-circuited to "exists").
    assert dl.start("sensevoice-small") == "started"
    dl.join_for_test("sensevoice-small")
