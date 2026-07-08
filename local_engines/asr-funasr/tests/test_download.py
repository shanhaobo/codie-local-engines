import os

from codie_asr_funasr import server


def _fake_snapshot(model_id, revision, dest_dir):
    """Mimics modelscope.snapshot_download writing a full model dir straight
    into dest_dir (no archive to extract — ModelScope hands you a directory)."""
    os.makedirs(dest_dir, exist_ok=True)
    with open(os.path.join(dest_dir, "model.pt"), "wb") as f:
        f.write(b"WEIGHTS")
    with open(os.path.join(dest_dir, "configuration.json"), "w") as f:
        f.write("{}")
    with open(os.path.join(dest_dir, "README.md"), "w") as f:
        f.write("not required")


def test_download_populates_and_publishes(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_snapshot_download", _fake_snapshot)
    dl = server.Downloads(str(tmp_path))

    status = dl.start("paraformer-zh")
    assert status == "started"
    dl.join_for_test("paraformer-zh")  # deterministic wait (test-only helper)

    d = os.path.join(str(tmp_path), "paraformer-zh")
    assert os.path.isfile(os.path.join(d, "model.pt"))
    assert os.path.isfile(os.path.join(d, "configuration.json"))
    # Non-required files ride along too (ModelScope hands over the whole repo) —
    # only _is_cached's required-files check gates "downloaded".
    assert os.path.isfile(os.path.join(d, "README.md"))
    assert server._is_cached(str(tmp_path), "paraformer-zh") is True

    # The staging dir is cleaned up after a successful atomic publish.
    assert not os.path.exists(d + ".part")

    # Idempotent: a second start sees it cached.
    assert dl.start("paraformer-zh") == "exists"


def _fake_snapshot_missing_weights(model_id, revision, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    with open(os.path.join(dest_dir, "configuration.json"), "w") as f:
        f.write("{}")
    # model.pt intentionally absent.


def test_download_missing_required_leaves_uncached(tmp_path, monkeypatch):
    # A snapshot missing a required file raises inside the worker; the error is
    # swallowed (surfaced via status) and dest_dir is NOT published, so the model
    # stays "not downloaded" and a retry is possible — no half-written strand.
    monkeypatch.setattr(server, "_snapshot_download", _fake_snapshot_missing_weights)
    dl = server.Downloads(str(tmp_path))

    assert dl.start("paraformer-zh") == "started"
    dl.join_for_test("paraformer-zh")

    d = os.path.join(str(tmp_path), "paraformer-zh")
    assert server._is_cached(str(tmp_path), "paraformer-zh") is False
    assert not os.path.exists(d)  # partial never published
    assert not os.path.exists(d + ".part")  # staging cleaned up
    # Not stranded: a subsequent start re-attempts (isn't short-circuited to "exists").
    assert dl.start("paraformer-zh") == "started"
    dl.join_for_test("paraformer-zh")
