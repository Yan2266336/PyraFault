import re

import numpy as np
import pytest
from PIL import Image

from pyrafault.features.registry import (
    ExtractorError,
    describe,
    get_extractor,
    resolve_local_revision,
)
from pyrafault.features.uni import assert_timm_model_matches_hf_repo

_SHA_RE = re.compile(r"[0-9a-f]{40}")


class Cfg:
    uni = {"hf_repo": "MahmoodLab/UNI", "timm_model": "hf-hub:MahmoodLab/UNI", "input_size": 224}
    dinov2 = {"hf_repo": "facebook/dinov2-base", "input_size": 224}
    batch_size = 8
    device = "cpu"


def _imgs(n=6, size=512, seed=0):
    rng = np.random.default_rng(seed)
    return [Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))
            for _ in range(n)]


def test_unknown_extractor_names_the_valid_ones():
    with pytest.raises(ExtractorError, match="handcrafted"):
        get_extractor("nope", Cfg)


def test_handcrafted_returns_l2_normalized_float32():
    ex = get_extractor("handcrafted", Cfg)
    out = ex.embed(_imgs())
    assert out.shape == (6, ex.dim) and out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_handcrafted_is_deterministic():
    ex = get_extractor("handcrafted", Cfg)
    imgs = _imgs()
    assert np.array_equal(ex.embed(imgs), ex.embed(imgs))


def test_handcrafted_separates_visually_different_patches():
    ex = get_extractor("handcrafted", Cfg)
    flat = [Image.fromarray(np.full((512, 512, 3), 200, np.uint8))]
    noisy = _imgs(1, seed=3)
    a, b = ex.embed(flat + noisy)
    assert float(a @ b) < 0.99


def test_describe_records_provenance():
    d = describe(get_extractor("handcrafted", Cfg))
    assert set(d) >= {"name", "revision", "dim", "input_size", "preprocessing"}
    assert d["name"] == "handcrafted"


def test_describe_records_hf_repo_and_timm_model():
    """I5: both fields must be present in the manifest, even when an extractor (like
    handcrafted) has neither -- getattr(..., None) keeps the manifest shape uniform
    across extractors instead of the key silently disappearing."""
    d = describe(get_extractor("handcrafted", Cfg))
    assert d["hf_repo"] is None and d["timm_model"] is None


def test_timm_model_matching_hf_repo_raises_nothing():
    assert_timm_model_matches_hf_repo("hf-hub:MahmoodLab/UNI", "MahmoodLab/UNI") is None


def test_timm_model_disagreeing_with_hf_repo_raises():
    """I5: the stale spec value `timm_model: vit_large_patch16_224` combined with
    `hf_repo: MahmoodLab/UNI` would load ImageNet weights under UNI's recorded
    revision -- exactly the silent-mismatch failure this guards against. Framed here
    against the hf-hub: form since that is what disagreement actually looks like."""
    with pytest.raises(ExtractorError, match="disagrees with"):
        assert_timm_model_matches_hf_repo("hf-hub:some/other-model", "MahmoodLab/UNI")


def test_non_hf_hub_timm_model_is_not_checked():
    """A bare timm architecture name (no hf-hub: prefix) carries no repo reference to
    validate against, so it must pass through rather than being rejected."""
    assert assert_timm_model_matches_hf_repo("vit_large_patch16_224", "MahmoodLab/UNI") is None


def test_batching_does_not_change_the_result():
    class Small(Cfg):
        batch_size = 2
    imgs = _imgs(7)
    a = get_extractor("handcrafted", Cfg).embed(imgs)
    b = get_extractor("handcrafted", Small).embed(imgs)
    assert np.allclose(a, b, atol=1e-6)


@pytest.mark.slow
def test_dinov2_downloads_and_embeds():
    ex = get_extractor("dinov2", Cfg)
    out = ex.embed(_imgs(4))
    assert out.shape == (4, ex.dim) and ex.dim >= 384
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-4)
    # A resolved commit sha, not "unknown" or a branch name like "main": len(...) >= 7
    # would pass identically whether resolution succeeded or silently degraded, since
    # "unknown" is itself 7 characters. Require the real 40-char lowercase hex form.
    assert _SHA_RE.fullmatch(ex.revision), ex.revision


@pytest.mark.slow
def test_uni_reports_an_actionable_error_when_gated():
    """MahmoodLab/UNI is `gated: auto`. Until the licence is accepted this must fail loudly."""
    try:
        ex = get_extractor("uni", Cfg)
    except ExtractorError as exc:
        assert "huggingface.co/MahmoodLab/UNI" in str(exc)
        pytest.skip("UNI access not yet granted — expected until the licence is accepted")
    else:
        out = ex.embed(_imgs(2))
        assert out.shape == (2, 1024)
        assert _SHA_RE.fullmatch(ex.revision), ex.revision


def test_revision_resolution_raises_rather_than_falling_back_on_download_failure(monkeypatch):
    """A network blip or offline node must never silently degrade to a placeholder sha."""
    def boom(*a, **k):
        raise RuntimeError("simulated network failure")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", boom)
    with pytest.raises(ExtractorError, match="could not resolve"):
        resolve_local_revision("facebook/dinov2-base")


def test_revision_resolution_rejects_a_non_sha_snapshot_dirname(monkeypatch):
    """Guards against a cache path whose parent directory is a ref name, not a commit sha."""
    def fake_download(repo, filename, token=None, local_files_only=None):
        return f"/fake/cache/models--x--y/snapshots/main/{filename}"

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    with pytest.raises(ExtractorError, match="not a 40-character commit sha"):
        resolve_local_revision("x/y")


def test_revision_resolution_never_makes_a_networked_hub_call(monkeypatch):
    """Pins the local-only contract: a networked default-revision lookup must not regress.

    `hf_hub_download`'s default `revision="main"` is a ref name, not a commit hash, so
    without `local_files_only=True` the call would silently become a live HTTP request to
    resolve "main" to the Hub's current HEAD -- exactly the unpinned, post-load query this
    function exists to eliminate. Assert the flag is actually passed, not just that some
    sha comes back.
    """
    captured = {}

    def fake_download(repo, filename, token=None, local_files_only=None):
        captured["local_files_only"] = local_files_only
        sha = "a" * 40
        return f"/fake/cache/models--x--y/snapshots/{sha}/{filename}"

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    sha = resolve_local_revision("x/y")
    assert captured["local_files_only"] is True
    assert sha == "a" * 40
