"""Memoising the extractor must not change a single number.

That is the whole claim. `CachingExtractor` exists to cut 720 GPU-hours off a full
seven-baseline run, and the only reason it is allowed to is that the matrix a baseline
receives is the matrix it would have computed anyway. So these tests compare against the
unwrapped extractor rather than against expectations about the cache: identical rows,
identical order, identical `Localisation` out the far end.

The interesting failure this guards is not "the cache returns garbage" -- that would be
obvious. It is that the cache returns *almost* the same thing: a row from a batch of one
where the uncached path embedded it in a batch of 962, differing in the last bits, moving
a top-1 window by one position on the items where two windows nearly tie. That would be
invisible in aggregate and would quietly make the numbers not the collaborator's. The
real-model test at the bottom is the one that can see it, which is why it exists and why
it is marked slow rather than skipped.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pyrafault.verify.embed_cache import (
    CachingExtractor, DEFAULT_MAX_ENTRIES, image_key,
)


class _Counting:
    """A deterministic stand-in: each image's embedding is a function of its bytes, so
    'same answer' is checkable without a GPU, and every call is counted."""

    name = "counting"
    revision = "7"
    dim = 4

    def __init__(self):
        self.calls = 0
        self.n_images = 0

    def embed(self, images):
        images = list(images)
        self.calls += 1
        self.n_images += len(images)
        out = []
        for im in images:
            a = np.frombuffer(im.tobytes(), dtype=np.uint8).astype(np.float64)
            out.append([a.sum() % 97, a.mean(), float(im.width), float(im.height)])
        return np.asarray(out, dtype=np.float32)


def _img(seed: int, size=(8, 8)) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (*size, 3), dtype=np.uint8))


class TestItReturnsExactlyWhatTheInnerExtractorWould:
    def test_one_call_is_unchanged(self):
        imgs = [_img(i) for i in range(5)]
        want = _Counting().embed(imgs)
        got = CachingExtractor(_Counting()).embed(imgs)
        assert np.array_equal(got, want)

    def test_a_second_call_on_the_same_images_is_unchanged(self):
        """The case the whole module is for: the next item re-asks for the same crops."""
        imgs = [_img(i) for i in range(5)]
        c = CachingExtractor(_Counting())
        first, second = c.embed(imgs), c.embed(imgs)
        assert np.array_equal(first, second)
        assert np.array_equal(second, _Counting().embed(imgs))

    def test_order_is_the_callers_order_not_the_caches(self):
        """Rows are looked up by key, so a reordered call must come back reordered --
        a baseline reads row 0 as the query and rows 1.. as its window grid."""
        imgs = [_img(i) for i in range(6)]
        c = CachingExtractor(_Counting())
        c.embed(imgs)
        shuffled = [imgs[i] for i in (4, 0, 5, 1, 3, 2)]
        assert np.array_equal(c.embed(shuffled), _Counting().embed(shuffled))

    def test_a_partial_overlap_is_unchanged(self):
        """What an actual run looks like: the windows hit, the query misses."""
        c = CachingExtractor(_Counting())
        c.embed([_img(i) for i in range(5)])
        mixed = [_img(99), *[_img(i) for i in range(5)]]
        assert np.array_equal(c.embed(mixed), _Counting().embed(mixed))

    def test_duplicates_inside_one_call_are_unchanged(self):
        imgs = [_img(1), _img(2), _img(1), _img(2), _img(1)]
        c = CachingExtractor(_Counting())
        assert np.array_equal(c.embed(imgs), _Counting().embed(imgs))

    def test_eviction_does_not_change_the_answer(self):
        """A cache too small to hold the working set must still be correct, only slower.
        Correctness that depends on capacity is not correctness."""
        imgs = [_img(i) for i in range(12)]
        c = CachingExtractor(_Counting(), max_entries=3)
        for _ in range(4):
            assert np.array_equal(c.embed(imgs), _Counting().embed(imgs))


class TestItActuallySavesTheWork:
    def test_the_second_pass_embeds_nothing(self):
        imgs = [_img(i) for i in range(50)]
        inner = _Counting()
        c = CachingExtractor(inner)
        c.embed(imgs)
        assert inner.n_images == 50
        c.embed(imgs)
        assert inner.n_images == 50, "a repeat call must not reach the model"
        assert c.n_hits == 50 and c.n_misses == 50

    def test_duplicates_inside_one_call_are_embedded_once(self):
        inner = _Counting()
        CachingExtractor(inner).embed([_img(1)] * 10)
        assert inner.n_images == 1

    def test_the_misses_go_in_one_batch(self):
        """Batched the way the uncached path would batch them, which is the point of
        collecting the misses before calling rather than calling per image."""
        inner = _Counting()
        CachingExtractor(inner).embed([_img(i) for i in range(30)])
        assert inner.calls == 1

    def test_the_run_shape_this_was_built_for(self):
        """One context's 40 items: 20 shared windows each, a fresh query each. Uncached
        that is 40 x 21 = 840 embeddings; cached it is 20 windows + 40 queries = 60, so
        780 of the 840 requests never reach the model."""
        windows = [_img(i) for i in range(20)]
        inner = _Counting()
        c = CachingExtractor(inner)
        for item in range(40):
            c.embed([_img(1000 + item), *windows])
        assert inner.n_images == 60
        assert c.hit_rate == pytest.approx(780 / 840, abs=1e-6)


class TestItDoesNotDisguiseTheModel:
    def test_identity_passes_through(self):
        """`verifier_revision` in a predictions file has to stay the model's, and a
        baseline reads `name`/`revision`/`dim` straight off the extractor."""
        c = CachingExtractor(_Counting())
        assert (c.name, c.revision, c.dim) == ("counting", "7", 4)

    def test_two_wrappers_do_not_share_rows(self):
        imgs = [_img(i) for i in range(4)]
        a, b = CachingExtractor(_Counting()), CachingExtractor(_Counting())
        a.embed(imgs)
        assert b.n_hits == 0

    def test_a_degenerate_capacity_is_refused(self):
        with pytest.raises(ValueError, match="at least 1"):
            CachingExtractor(_Counting(), max_entries=0)

    def test_the_default_capacity_holds_a_context_of_windows(self):
        """~961 windows per context, and `runner.shard_of` groups a shard by context."""
        assert DEFAULT_MAX_ENTRIES >= 961


class TestTheKeyIsTheImageContent:
    def test_different_pixels_different_key(self):
        assert image_key(_img(1)) != image_key(_img(2))

    def test_same_pixels_same_key(self):
        assert image_key(_img(1)) == image_key(_img(1))

    def test_shape_is_in_the_key(self):
        """`tobytes()` alone cannot tell 8x16 from 16x8, and a crop grid contains both
        shapes at the context's right and bottom edges."""
        a = Image.fromarray(np.zeros((8, 16, 3), dtype=np.uint8))
        b = Image.fromarray(np.zeros((16, 8, 3), dtype=np.uint8))
        assert image_key(a) != image_key(b)

    def test_mode_is_in_the_key(self):
        rgb = _img(3)
        assert image_key(rgb) != image_key(rgb.convert("L"))


class TestTheTwoCachedPluginsAreTheSameMethods:
    def test_they_keep_their_names(self):
        """Same name means stage 13 writes to the same directory and the identity check
        inside `_LazyPlugin._backend` still passes: this is the same method, computed
        once instead of 38 times, not a new one."""
        from pyrafault.verify.embed_cache import (
            CachedStatisticalUncertaintyPlugin, CachedSUEPlugin,
        )
        from pyrafault.verify.plugins import StatisticalUncertaintyPlugin, SUEPlugin
        assert CachedSUEPlugin.name == SUEPlugin.name == "sue_baseline"
        assert (CachedStatisticalUncertaintyPlugin.name
                == StatisticalUncertaintyPlugin.name
                == "statistical_uncertainty_baseline")

    def test_construction_still_loads_nothing(self):
        """`--from-localisations` constructs a verifier and only reads its revision, so
        a plugin that loaded weights on construction would make deciding expensive."""
        from pyrafault.verify.embed_cache import CachedSUEPlugin
        import os
        before = os.environ.get("PYRAFAULT_DINOV2_SNAPSHOT")
        os.environ.pop("PYRAFAULT_DINOV2_SNAPSHOT", None)
        try:
            p = CachedSUEPlugin()          # no path set: must still construct
            assert p._backend_instance is None
        finally:
            if before is not None:
                os.environ["PYRAFAULT_DINOV2_SNAPSHOT"] = before


# --------------------------------------------------------------------------------------
# The test that can actually see the failure mode. Needs the pinned DINOv2 and a GPU.
# --------------------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("plugin_name", ["sue", "statistical-uncertainty"])
def test_real_model_gives_the_identical_localisation(plugin_name):
    """Same items, cached and uncached, through the real frozen DINOv2.

    This is the claim the module rests on, and the only test that can refute it. The
    synthetic tests above prove the bookkeeping; they cannot prove that a row embedded
    in a batch of one equals the same row embedded in a batch of 962, which is exactly
    what differs between the two paths and exactly what would move a top-1 window on a
    near-tie without changing any aggregate.

    Asserted on the `Localisation` rather than on the embeddings, because that is what
    is written to a predictions file: the estimated box, the reliability, and the
    diagnostics a scorer may read.
    """
    import os
    if not os.environ.get("PYRAFAULT_DINOV2_SNAPSHOT"):
        pytest.skip("PYRAFAULT_DINOV2_SNAPSHOT not set")

    from pyrafault.config import load_config
    from pyrafault.dataset import PyraFaultDataset
    from pyrafault.verify.embed_cache import (
        CachedStatisticalUncertaintyPlugin, CachedSUEPlugin,
    )
    from pyrafault.verify.plugins import StatisticalUncertaintyPlugin, SUEPlugin

    plain_cls, cached_cls = (
        (SUEPlugin, CachedSUEPlugin) if plugin_name == "sue"
        else (StatisticalUncertaintyPlugin, CachedStatisticalUncertaintyPlugin))

    root = "data_lung_sample/01_benchmark"
    cfg = load_config("configs/lung_5x_20x.yaml")
    with PyraFaultDataset(root, cfg.source.root_dir, split="test",
                          materialize=True) as ds:
        # Items sharing one context are the case the cache is for: the first fills it,
        # the rest read it. Taken in the dataset's own order so they really do share.
        ids = list(ds.item_ids)[:12]
        inputs = [ds[i]["inputs"] for i in ids]

    plain, cached = plain_cls(), cached_cls()
    a = [plain.localise(x) for x in inputs]
    del plain
    b = [cached.localise(x) for x in inputs]

    assert cached.cache.n_hits > 0, (
        "no cache hits over 12 items, so this test proved nothing about the cache; "
        "the items probably did not share a context"
    )
    for i, (x, y) in enumerate(zip(a, b)):
        assert x.estimated_bbox == y.estimated_bbox, f"item {i}: box differs"
        assert x.reliability == y.reliability, (
            f"item {i}: reliability {x.reliability!r} != {y.reliability!r} -- the cache "
            "changed a number, which is the one thing it may not do")
