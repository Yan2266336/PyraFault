"""The localise cache must not collapse two degradations of one true box.

`runner.cache_key` keys the harness's dedupe on (slide_id, context bbox, true_slide_id,
true bbox). Degradation is drawn per *item*, so two items can share every one of those
and still be served different pixels -- a 5 degree rotation and a 15 degree one of the
same patch. Keyed without it, the second item receives the first's localisation, and the
run looks entirely healthy: one estimate per item, plausible reliabilities, no error.

Measured on the freezes on disk: 1,784 of 8,697 keys on `data_lung_sample` (the freeze the
reported comparison was computed on) and 37,922 of 185,388 on the degraded full freeze hold
more than one parameter set, up to five under a single key.

This is the same defect, in the same function, as the out-of-slide collision that put
`true_slide_id` in `target`: a harness cannot tell two items apart on information it is
never given. The fix is the same one -- `target` carries what the harness needs to key
its cache, while `inputs` still tells a verifier nothing about which band it got.
"""
from __future__ import annotations

from PIL import Image

from pyrafault.geom import BBoxL0
from pyrafault.verify.base import Localisation, Verifier
from pyrafault.verify.runner import cache_key, localise_items, run_verifier
from pyrafault.verify.thresholds import Thresholds


def _item(iid, digest, shade):
    return {
        "inputs": {
            "context": Image.new("RGB", (64, 64)),
            "query": Image.new("RGB", (16, 16), (shade,) * 3),
            "context_bbox": BBoxL0(0, 0, 1000, 1000),
            "claimed_bbox": BBoxL0(1000, 1000, 100, 100),
            "context_mpp": 2.0, "query_mpp": 0.5, "slide_id": "S1",
            "item_id": iid, "split": "test", "extraction_version": "v1",
        },
        "target": {
            "claim_correct": True, "true_bbox": BBoxL0(1000, 1000, 100, 100),
            "true_slide_id": "S1", "pair_id": iid, "degradation_digest": digest,
        },
    }


class TestTheKeyItself:
    def test_two_digests_of_one_box_are_different_keys(self):
        a = cache_key(_item("i0", "", 10))
        b = cache_key(_item("i1", "abc123", 10))
        assert a != b

    def test_the_same_digest_is_the_same_key(self):
        assert cache_key(_item("i0", "abc123", 10)) == cache_key(_item("i1", "abc123", 99))

    def test_an_undegraded_item_keys_on_the_empty_digest(self):
        """`none` is the identity transform and its digest is the empty string, so an
        undegraded freeze's keys are unchanged -- the dedupe it already had still holds."""
        assert cache_key(_item("i0", "", 10)) == cache_key(_item("i1", "", 10))

    def test_an_item_without_the_field_still_keys(self):
        """A third party's own dataset object, or a freeze predating the column, must not
        crash the harness -- it simply has one degradation, so one key."""
        item = _item("i0", "", 10)
        del item["target"]["degradation_digest"]
        assert cache_key(item) == cache_key(_item("i1", "", 10))


class _ByShade(Verifier):
    """Estimate follows the served pixels, as a real verifier's does."""
    name, revision = "by-shade", "1"

    def __init__(self):
        self.calls = 0

    def localise(self, inputs):
        self.calls += 1
        shade = inputs["query"].getpixel((0, 0))[0]
        return Localisation(BBoxL0(1000 + shade, 1000, 100, 100), shade / 255.0)


class _Bands:
    """Four items on one true box, one per band -- the shape `assign_bands` produces."""

    def __init__(self):
        self._items = {
            f"i{n}": _item(f"i{n}", d, 20 * (n + 1))
            for n, d in enumerate(("", "d_mild", "d_moderate", "d_severe"))
        }

    @property
    def item_ids(self):
        return list(self._items)

    def __getitem__(self, k):
        return self._items[k]


def _thresholds():
    return Thresholds(tau=0.0, delta=10.0, fitted_on="calibration", n_items=4,
                      residuals={"n": 4, "p50": 0.1, "p75": 0.2, "p90": 0.3,
                                 "p95": 0.4, "p99": 0.5})


class TestFourBandsOfOneBoxAreFourLocalisations:
    def test_localise_is_called_once_per_band(self):
        v = _ByShade()
        localise_items(v, _Bands())
        assert v.calls == 4, "bands of one box were collapsed into one localisation"

    def test_each_band_keeps_its_own_estimate(self):
        df = localise_items(_ByShade(), _Bands()).set_index("item_id")
        assert df["estimated_bbox_l0_x"].nunique() == 4

    def test_the_same_holds_through_run_verifier(self):
        v = _ByShade()
        df = run_verifier(v, _Bands(), _thresholds()).set_index("item_id")
        assert v.calls == 4
        assert df["reliability"].nunique() == 4

    def test_a_cross_band_collision_would_have_been_silent(self):
        """What the bug looked like: every item still gets a row, a plausible
        reliability and no error. Only the duplicated estimates give it away."""
        df = localise_items(_ByShade(), _Bands())
        assert len(df) == 4 and df["reliability"].notna().all()
        assert not df["estimated_bbox_l0_x"].duplicated().any()
