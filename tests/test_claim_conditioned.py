"""The claim-conditioned baseline, and the one property that is true before measurement.

The reviewer's question -- why not just compare `Q` against the crop at the claimed
location? -- has a structural half and an empirical half. These tests pin the structural
half: with `estimated_bbox == claimed_bbox`, `decide`'s centre distance is identically
zero, so the method can say `verified` or `unverifiable` and never `mismatched`, which
makes its conditional balanced accuracy exactly 0.5 on any data at any threshold. That
is the price of claim-conditioning, and it must not be possible to lose it in a
refactor.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pyrafault.geom import BBoxL0, iou
from pyrafault.verify.base import decide
from pyrafault.verify.claim_conditioned import (
    ClaimConditionedBaseline, query_side_in_context,
)
from pyrafault.verify.matchers import MatchResult
from pyrafault.verify.metrics import verification_score

CTX_PX = 256          # context image side, in its own pixels
CTX_L0 = 4096         # the same context, in level-0 pixels -- 16 L0 px per context px
SIDE = 32             # a query covers 32 context px, so 512 L0 px


class _Spy:
    """Records what it was asked to compare, and answers with a fixed score."""
    name = "spy"

    def __init__(self, score: float = 0.4):
        self.score, self.calls = score, []

    def match(self, query, window):
        self.calls.append((np.asarray(query).copy(), np.asarray(window).copy()))
        return MatchResult(self.score, 100, int(100 * self.score), np.eye(3))


def _inputs(claim_x_l0: int, *, ctx=None):
    """One item. 64 px of query at 0.25 mpp over a context at 0.5 mpp gives SIDE = 32."""
    context = ctx if ctx is not None else Image.new("RGB", (CTX_PX, CTX_PX))
    return {
        "context": context,
        "query": Image.new("RGB", (64, 64)),
        "context_bbox": BBoxL0(10_000, 20_000, CTX_L0, CTX_L0),
        "claimed_bbox": BBoxL0(claim_x_l0, 20_000, SIDE * 16, SIDE * 16),
        "context_mpp": 0.5, "query_mpp": 0.25, "slide_id": "S1",
        "item_id": "i0", "split": "test", "extraction_version": "v1",
    }


class TestTheGeometry:
    def test_the_side_comes_from_the_two_resolutions(self):
        assert query_side_in_context(_inputs(10_000)) == SIDE

    def test_the_crop_is_taken_at_the_claim(self):
        """Claim 64 context px into the context -> crop origin (64, 0)."""
        spy = _Spy()
        loc = ClaimConditionedBaseline(spy).localise(_inputs(10_000 + 64 * 16))
        assert loc.diagnostics["crop_origin"] == (64, 0)

    def test_it_is_the_claimed_pixels_that_are_compared(self):
        """A context that is black except for a white block at the claim: the window the
        matcher receives must be that block, which is what distinguishes this baseline
        from one that searches the context."""
        arr = np.zeros((CTX_PX, CTX_PX, 3), dtype=np.uint8)
        arr[0:SIDE, 64:64 + SIDE] = 255
        spy = _Spy()
        ClaimConditionedBaseline(spy).localise(
            _inputs(10_000 + 64 * 16, ctx=Image.fromarray(arr)))
        _, window = spy.calls[0]
        assert window.shape == (SIDE, SIDE) and (window == 255).all()

    def test_the_query_reaches_the_matcher_downsampled(self):
        spy = _Spy()
        ClaimConditionedBaseline(spy).localise(_inputs(10_000))
        q, w = spy.calls[0]
        assert q.shape == w.shape == (SIDE, SIDE)

    def test_one_match_per_item(self):
        """The whole cost argument: one match, against ZoomTrace's ten."""
        spy = _Spy()
        ClaimConditionedBaseline(spy).localise(_inputs(10_000))
        assert len(spy.calls) == 1

    def test_without_scale_alignment_the_query_is_not_resized(self):
        spy = _Spy()
        ClaimConditionedBaseline(spy, scale_alignment=False).localise(_inputs(10_000))
        assert spy.calls[0][0].shape == (64, 64)

    def test_a_claim_outside_the_context_raises(self):
        """Measured as impossible on the shipped freezes (0 of 503,320 test items), so it
        raises rather than clipping: a freeze where it happens has changed what a claim
        is, and the right crop is then its author's decision."""
        with pytest.raises(ValueError, match="not inside"):
            ClaimConditionedBaseline(_Spy()).localise(_inputs(10_000 + CTX_L0))

    def test_a_claim_that_is_not_query_sized_raises(self):
        inputs = _inputs(10_000)
        inputs["claimed_bbox"] = BBoxL0(10_000, 20_000, SIDE * 16 * 3, SIDE * 16 * 3)
        with pytest.raises(ValueError, match="nothing to crop"):
            ClaimConditionedBaseline(_Spy()).localise(inputs)


class TestTheEstimateIsTheClaim:
    def test_the_box_returned_is_the_claim(self):
        inputs = _inputs(10_000 + 64 * 16)
        loc = ClaimConditionedBaseline(_Spy()).localise(inputs)
        assert loc.estimated_bbox == inputs["claimed_bbox"]

    def test_the_reliability_is_the_match_score(self):
        loc = ClaimConditionedBaseline(_Spy(0.37)).localise(_inputs(10_000))
        assert loc.reliability == pytest.approx(0.37)

    def test_the_iou_is_exactly_one_so_the_metrics_read_the_raw_score(self):
        """`verification_score` is `rho * IoU(estimated, claimed)`; here the IoU factor
        is 1, which is what makes this row's AUPRC answer the reviewer's question
        directly rather than a product of two things."""
        inputs = _inputs(10_000)
        loc = ClaimConditionedBaseline(_Spy(0.37)).localise(inputs)
        assert iou(loc.estimated_bbox, inputs["claimed_bbox"]) == 1.0
        assert verification_score(loc.reliability, loc.estimated_bbox,
                                  inputs["claimed_bbox"]) == pytest.approx(0.37)


class TestItCanNeverSayMismatched:
    """The structural half of the answer, at every threshold rather than at one."""

    @pytest.mark.parametrize("score", [0.0, 0.01, 0.5, 0.99, 1.0])
    @pytest.mark.parametrize("tau", [0.0, 0.3, 0.7, 1.0])
    @pytest.mark.parametrize("delta", [1e-6, 0.375, 10.0])
    def test_verified_or_unverifiable_only(self, score, tau, delta):
        inputs = _inputs(10_000 + 64 * 16)
        loc = ClaimConditionedBaseline(_Spy(score)).localise(inputs)
        d = decide(loc, inputs["claimed_bbox"], query_side=SIDE * 16,
                   tau=tau, delta=delta)
        assert d in ("verified", "unverifiable")

    def test_the_distance_decide_tests_is_identically_zero(self):
        """Not "small": zero. `delta` could be fitted arbitrarily tight and the verdict
        above tau would still be `verified`."""
        inputs = _inputs(10_000 + 64 * 16)
        loc = ClaimConditionedBaseline(_Spy(1.0)).localise(inputs)
        assert (loc.estimated_bbox.cx, loc.estimated_bbox.cy) == (
            inputs["claimed_bbox"].cx, inputs["claimed_bbox"].cy)

    def test_conditional_balanced_accuracy_is_one_half(self):
        """Every covered item is `verified`, whatever the truth, so TPR = 1 and TNR = 0.
        This is the number the table reports, derived here rather than measured."""
        tau = 0.5
        truths, decisions = [], []
        for n, score in enumerate([0.9, 0.8, 0.7, 0.6]):
            inputs = _inputs(10_000)
            loc = ClaimConditionedBaseline(_Spy(score)).localise(inputs)
            decisions.append(decide(loc, inputs["claimed_bbox"], SIDE * 16, tau, 0.375))
            truths.append(n % 2 == 0)          # half the claims are wrong
        covered = [(t, d) for t, d in zip(truths, decisions) if d != "unverifiable"]
        assert covered and all(d == "verified" for _, d in covered)
        # Both classes are present among the covered items, and every one of them was
        # called `verified`: TPR = 1, TNR = 0, so the balanced accuracy is 0.5.
        assert {t for t, _ in covered} == {True, False}
        pos = [t for t, _ in covered if t]
        neg = [t for t, _ in covered if not t]
        tpr = sum(1 for _ in pos) / len(pos)
        tnr = sum(0 for _ in neg) / len(neg)
        assert (tpr + tnr) / 2 == 0.5


class TestTheDeclaration:
    def test_it_declares_that_it_reads_the_claim(self):
        assert ClaimConditionedBaseline.reads_claim is True

    def test_the_plugin_mirrors_the_declaration(self):
        """`load_plugin` returns the plugin, and the harness reads the flag off that --
        so a plugin that forgot it would key the cache without the claim."""
        from pyrafault.verify.plugins import ClaimConditionedPlugin
        p = ClaimConditionedPlugin()
        assert p.reads_claim is True and p.name == ClaimConditionedBaseline.name
