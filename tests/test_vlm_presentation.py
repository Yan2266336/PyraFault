"""How a (context, patch, claim) is shown to a multimodal model.

This is the part of the MLLM baseline most likely to decide its score, so it is built as
a switchable, tested unit rather than as prompt strings inline in a verifier. The reason
is a number: the patch occupies 128x128 of a 2048x2048 context -- 0.39% of its area --
and a model that downsamples the context to 1024 sees the target as 64x64 pixels, to 512
as 32x32. A naive "here are two images" presentation therefore measures the model's
input pipeline, not its perception, and a baseline weakened that way is worse than no
baseline: it is a strawman a reviewer will name.

So four presentations are implemented and all four are run on a pilot before the real
comparison, and the best one is reported. These pin what each presentation must produce;
the quality comparison is the experiment, not an assertion here.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pyrafault.geom import BBoxL0
from pyrafault.verify.vlm_presentation import (
    CLAIM_FREE, PRESENTATIONS, context_relative_box, expected_side_px, render,
)

CTX = BBoxL0(17616, 13212, 17617, 17617)      # a real context box from the lung freeze
CLAIM = BBoxL0(29433, 16652, 1101, 1101)      # and a real claimed box inside it


@pytest.fixture
def images():
    rng = np.random.default_rng(0)
    ctx = Image.fromarray(rng.integers(0, 255, (2048, 2048, 3), dtype=np.uint8))
    qry = Image.fromarray(rng.integers(0, 255, (512, 512, 3), dtype=np.uint8))
    return ctx, qry


class TestCoordinatesAreMadeRelativeFirst:
    """Level-0 coordinates are absolute slide pixels -- the claim sits at 29433, 16652
    while the context starts at 17616, 13212. Handing those to a model is handing it
    numbers with no frame of reference."""

    def test_the_claim_becomes_a_box_inside_the_context_image(self, images):
        ctx, _ = images
        b = context_relative_box(CLAIM, CTX, ctx.width)
        assert 0 <= b[0] < ctx.width and 0 <= b[1] < ctx.height

    def test_the_scale_factor_is_applied(self, images):
        ctx, _ = images
        b = context_relative_box(CLAIM, CTX, ctx.width)
        expected_side = round(CLAIM.w / CTX.w * ctx.width)
        assert abs((b[2] - b[0]) - expected_side) <= 1

    def test_a_claim_at_the_context_origin_maps_to_the_image_origin(self, images):
        ctx, _ = images
        b = context_relative_box(BBoxL0(CTX.x, CTX.y, 1101, 1101), CTX, ctx.width)
        assert b[0] == 0 and b[1] == 0

    def test_it_does_not_silently_accept_a_claim_outside_the_context(self):
        """Out-of-context fault types have exactly this: a claim whose box is not inside
        the context. The caller must handle it, so the helper says so."""
        far = BBoxL0(CTX.x + CTX.w * 3, CTX.y, 1101, 1101)
        with pytest.raises(ValueError, match="outside"):
            context_relative_box(far, CTX, 2048)


class TestEveryPresentationProducesSomething(object):
    @pytest.mark.parametrize("name", sorted(PRESENTATIONS))
    def test_it_returns_images_and_a_prompt(self, images, name):
        ctx, qry = images
        out = render(name, ctx, qry, CLAIM, CTX)
        assert out.images and all(isinstance(i, Image.Image) for i in out.images)
        assert isinstance(out.prompt, str) and out.prompt.strip()

    @pytest.mark.parametrize("name", sorted(PRESENTATIONS))
    def test_no_image_exceeds_what_a_model_will_accept(self, images, name):
        """Anything over ~1500 px gets downsampled by the model's own processor, which is
        the failure this module exists to avoid. Each presentation must do its own
        resizing deliberately rather than leaving it to the processor."""
        ctx, qry = images
        for im in render(name, ctx, qry, CLAIM, CTX).images:
            assert max(im.size) <= 1536, f"{name} emits {im.size}"

    @pytest.mark.parametrize("name", sorted(PRESENTATIONS))
    def test_the_prompt_never_leaks_the_answer(self, images, name):
        """`true_bbox` is the answer key. A prompt mentioning it would score the model on
        reading comprehension."""
        ctx, qry = images
        p = render(name, ctx, qry, CLAIM, CTX).prompt.lower()
        for banned in ("true_bbox", "answer is", "correct location is"):
            assert banned not in p

    def test_the_named_presentations_exist(self):
        assert set(PRESENTATIONS) == {"pair", "boxed", "sidebyside", "tiles",
                                      "locate", "locate_grid"}

    def test_exactly_the_locate_forms_are_claim_free(self):
        assert CLAIM_FREE == {"locate", "locate_grid"}


class TestTilesKeepsResolution:
    """`tiles` is the presentation designed for the 0.39% problem: the context is cut up
    so each piece reaches the model at native resolution instead of being shrunk."""

    def test_it_emits_more_than_one_context_image(self, images):
        ctx, qry = images
        out = render("tiles", ctx, qry, CLAIM, CTX)
        assert len(out.images) > 2

    def test_the_tiles_are_not_downsampled(self, images):
        ctx, qry = images
        out = render("tiles", ctx, qry, CLAIM, CTX)
        assert any(max(i.size) >= 512 for i in out.images)

    def test_the_prompt_tells_the_model_how_the_tiles_are_laid_out(self, images):
        ctx, qry = images
        p = render("tiles", ctx, qry, CLAIM, CTX).prompt.lower()
        assert "grid" in p or "tile" in p


class TestTheClaimFreeFormsNeverTouchTheClaim:
    """The locate presentations back the only MLLM row that may sit in the baseline
    table, and what earns them that is structural: `Verifier.localise` promises an
    estimate independent of the claim, `runner._assert_claim_independent` audits it, and
    `cache_key` excludes the claim on purpose -- so a presentation that reads the claim
    would also serve one cached answer to two items sharing a `(C, Q)`. These pin that
    the guarantee is enforced by the signature rather than left to care."""

    @pytest.mark.parametrize("name", sorted(CLAIM_FREE))
    def test_it_renders_with_no_claim_at_all(self, images, name):
        ctx, qry = images
        out = render(name, ctx, qry, None, CTX, context_mpp=2.0, query_mpp=0.5)
        assert out.images and out.prompt

    @pytest.mark.parametrize("name", sorted(set(PRESENTATIONS) - CLAIM_FREE))
    def test_a_claim_shown_form_refuses_to_invent_one(self, images, name):
        ctx, qry = images
        with pytest.raises(ValueError, match="needs one"):
            render(name, ctx, qry, None, CTX)

    @pytest.mark.parametrize("name", sorted(CLAIM_FREE))
    def test_the_claim_is_ignored_even_when_handed_over(self, images, name):
        """Same (C, Q), two very different claims, byte-identical prompt. This is the
        presentation-level analogue of the run harness's audit."""
        ctx, qry = images
        a = render(name, ctx, qry, CLAIM, CTX, 2.0, 0.5)
        far = BBoxL0(CLAIM.x + 9999, CLAIM.y + 9999, CLAIM.w, CLAIM.h)
        b = render(name, ctx, qry, far, CTX, 2.0, 0.5)
        assert a.prompt == b.prompt

    @pytest.mark.parametrize("name", sorted(CLAIM_FREE))
    def test_it_reports_the_frame_its_coordinates_are_in(self, images, name):
        """A model answers in the frame it was shown. Mapping that back without knowing
        the frame is the RomaMatcher failure -- right numbers, wrong frame, nothing
        raises -- so the frame travels with the render."""
        ctx, qry = images
        out = render(name, ctx, qry, None, CTX, 2.0, 0.5)
        assert out.frame_px == out.images[0].width

    @pytest.mark.parametrize("name", sorted(CLAIM_FREE))
    def test_it_says_the_patch_may_not_be_in_the_view(self, images, name):
        """Four of the eight fault types put the source outside the reference view. A
        model told to find the patch *somewhere* would be scored on a question that has
        no answer there."""
        ctx, qry = images
        p = render(name, ctx, qry, None, CTX, 2.0, 0.5).prompt
        assert "NONE" in p and "another slide" in p


class TestTheScaleHintIsDerivedNotGuessed:
    def test_the_patch_size_follows_from_the_two_resolutions(self, images):
        """512 query pixels at 0.5 um each is 256 um of slide; at 2.0 um per context
        pixel that is 128 of them -- the 0.39% of area this module exists for."""
        _ctx, qry = images
        assert expected_side_px(qry, context_mpp=2.0, query_mpp=0.5) == 128

    def test_it_is_claim_free_and_label_free(self, images):
        """It reads only `context_mpp` and `query_mpp`, which the loader exposes so a
        method can align magnifications, and which are constant across a freeze."""
        ctx, qry = images
        with_hint = render("locate", ctx, qry, None, CTX, 2.0, 0.5).prompt
        without = render("locate", ctx, qry, None, CTX).prompt
        assert "pixels" in with_hint and with_hint != without


class TestTheTilesPromptDescribesWhatWasActuallySent:
    def test_it_does_not_refer_to_images_that_do_not_exist(self, images):
        """`tiles` sends ten images, none of them named `IMAGE 1` or `IMAGE 2`. One
        substitution was missing and the prompt still referred to both -- caught only by
        printing the assembled text, not by any assertion about its parts."""
        ctx, qry = images
        out = render("tiles", ctx, qry, CLAIM, CTX)
        assert len(out.images) == 10
        assert "IMAGE 1" not in out.prompt
        assert "IMAGE 2" not in out.prompt
