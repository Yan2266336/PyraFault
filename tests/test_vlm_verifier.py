"""The multimodal-model baseline as a `Verifier`.

It answers the question a reviewer asks first -- "why not just show the two images to a
frontier model?" -- so it has to be a fair attempt, and it has to be scored by the same
code as every other method. That means the shared runner, the shared thresholds, the
shared scorer: only the localisation differs.

The model is not loaded here. What these pin is everything around it: how its text turns
into a `Localisation`, what happens when it answers unparseably, and the fact that the
verdict is converted into the same reliability-plus-estimate shape the protocol defines
rather than into a decision of its own.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pyrafault.geom import BBoxL0
from pyrafault.verify.base import Localisation, Verifier
from pyrafault.verify.runner import (
    _assert_claim_independent, _with_perturbed_claim,
)
from pyrafault.verify.vlm_baseline import (
    DEFAULT_CONF, BoxReply, VlmBaseline, VlmLocaliser, parse_box_reply,
    parse_reply,
)


class TestParsingTheReply:
    """One number: P(the patch really came from the claimed place).

    The first version asked for a verdict plus an undefined "confidence". The model read
    it as the probability the claim was right and answered `no` with `0.0`; the code read
    it as certainty in the verdict and computed `1 - 0.0 = 1.0`, so the most confident
    rejection scored as the highest reliability. The readings agree on `yes` and invert
    on `no`, so half the cases looked fine. These pin the single definition.
    """

    def test_the_requested_form(self):
        assert parse_reply("PROBABILITY: 0.9") == (pytest.approx(0.9), False)

    def test_certain_rejection_is_near_zero_not_near_one(self):
        """The regression. A model certain the patch is NOT there says 0.0, and that has
        to stay 0.0 all the way to `reliability`."""
        p, _ = parse_reply("PROBABILITY: 0.0")
        assert p == pytest.approx(0.0)

    def test_case_and_whitespace_do_not_matter(self):
        assert parse_reply("  probability : 0.75 ")[0] == pytest.approx(0.75)

    def test_surrounding_chatter_is_tolerated(self):
        p, _ = parse_reply("Looking at the tissue...\nPROBABILITY: 0.35\nHope that helps!")
        assert p == pytest.approx(0.35)

    def test_a_bare_number_is_accepted(self):
        assert parse_reply("0.42")[0] == pytest.approx(0.42)

    def test_a_bare_verdict_falls_back_and_says_so(self):
        """A model that gave no number expressed no certainty, so the fallback is
        deliberately unsharp -- recording it as 1.0 would hand the baseline a sharper
        score than it earned. The flag lets the run report how often this happened."""
        p, fb = parse_reply("VERDICT: no")
        assert p == pytest.approx(0.25) and fb is True
        p, fb = parse_reply("yes")
        assert p == pytest.approx(0.75) and fb is True

    def test_a_trailing_full_stop_does_not_destroy_the_answer(self):
        """`0.9.` -- the number with a period after it. The regex was anchored at `$`
        with only whitespace allowed, so this made 100 of 100 replies unparseable and
        the run reported a 32B model as having produced no answer at all. One
        character."""
        assert parse_reply("0.9.") == (0.9, False)
        assert parse_reply("0.0.") == (0.0, False)
        assert parse_reply("0.7 .") == (0.7, False)

    def test_an_unparseable_reply_is_reported_not_guessed(self):
        assert parse_reply("I cannot tell from these images.") == (None, False)

    def test_out_of_range_numbers_are_clamped(self):
        assert parse_reply("PROBABILITY: 1.7")[0] == pytest.approx(1.0)
        assert parse_reply("PROBABILITY: -2")[0] == pytest.approx(0.0)


class _Scripted(VlmBaseline):
    """A VlmBaseline whose model is a list of canned replies, so the plumbing can be
    tested without a GPU."""

    def __init__(self, replies, **kw):
        super().__init__(model_id="scripted", presentation="boxed", _defer_load=True, **kw)
        self._replies = list(replies)
        self.seen = []

    def _ask(self, images, prompt):
        self.seen.append((images, prompt))
        return self._replies.pop(0)


def _inputs():
    return {
        "context": Image.new("RGB", (2048, 2048)),
        "query": Image.new("RGB", (512, 512)),
        "context_bbox": BBoxL0(17616, 13212, 17617, 17617),
        "claimed_bbox": BBoxL0(29433, 16652, 1101, 1101),
        "context_mpp": 2.0, "query_mpp": 0.5, "slide_id": "S1",
        "item_id": "i0", "split": "test", "extraction_version": "v1",
    }


class TestItHonoursTheVerifierContract:
    def test_it_is_a_verifier(self):
        assert isinstance(_Scripted(["PROBABILITY: 0.8"]), Verifier)

    def test_it_returns_a_localisation(self):
        r = _Scripted(["PROBABILITY: 0.8"]).localise(_inputs())
        assert isinstance(r, Localisation)

    def test_reliability_is_in_range(self):
        r = _Scripted(["PROBABILITY: 0.8"]).localise(_inputs())
        assert 0.0 <= r.reliability <= 1.0

    def test_the_name_says_which_model_and_presentation(self):
        """Output lands in `06_verify/<name>/`, and the presentation is part of the
        method exactly as the matcher is -- two presentations must not share a
        directory or overwrite each other's thresholds."""
        v = _Scripted(["PROBABILITY: 0.8"])
        assert "scripted" in v.name and "boxed" in v.name


class TestTheVerdictBecomesAnEstimateHonestly:
    def test_yes_estimates_the_claimed_box(self):
        """The model is asked to judge, not to localise -- so a `yes` means "the source
        is where the claim says", and the estimate is the claimed box. Nothing is
        invented."""
        r = _Scripted(["PROBABILITY: 0.9"]).localise(_inputs())
        assert r.estimated_bbox == _inputs()["claimed_bbox"]

    def test_no_returns_no_estimate(self):
        """A `no` says the patch is not there and offers no alternative, which is exactly
        `estimated_bbox=None` -- the protocol's confident rejection."""
        r = _Scripted(["PROBABILITY: 0.1"]).localise(_inputs())
        assert r.estimated_bbox is None

    def test_an_unparseable_reply_abstains_rather_than_guessing(self):
        r = _Scripted(["no idea"]).localise(_inputs())
        assert r.estimated_bbox is None and r.reliability == 0.0

    def test_unparseable_replies_are_counted(self):
        v = _Scripted(["no idea", "PROBABILITY: 0.6"])
        v.localise(_inputs()); v.localise(_inputs())
        assert v.n_unparsed == 1

    def test_reliability_is_the_probability_the_claim_is_right(self):
        """`decide` thresholds reliability and reads high as "trust this". A model
        certain the patch is NOT there must therefore come out LOW -- the inversion that
        the first prompt caused."""
        yes = _Scripted(["PROBABILITY: 0.9"]).localise(_inputs())
        no = _Scripted(["PROBABILITY: 0.1"]).localise(_inputs())
        assert yes.reliability > no.reliability


class TestItNeverSeesTheAnswer:
    def test_the_prompt_carries_no_target_information(self):
        v = _Scripted(["PROBABILITY: 0.55"])
        v.localise(_inputs())
        _, prompt = v.seen[0]
        assert "true" not in prompt.lower().replace("true?", "")


# --------------------------------------------------------------------------------------
# The claim-free localiser: the only MLLM form that may sit in the baseline table.
# --------------------------------------------------------------------------------------

class _ScriptedLoc(VlmLocaliser):
    """A VlmLocaliser whose model is a list of canned replies."""

    def __init__(self, replies, **kw):
        super().__init__(model_id="scripted", _defer_load=True, **kw)
        self._replies = list(replies)
        self.seen = []

    def _ask(self, images, prompt):
        self.seen.append((images, prompt))
        return self._replies.pop(0)


class TestItAsksTheTaskTheProtocolDefines:
    def test_it_refuses_a_presentation_that_shows_the_claim(self):
        """Not a style preference. A claim-shown presentation breaks the contract
        `localise` promises, so the constructor is where it has to stop."""
        with pytest.raises(ValueError, match="claim-independent"):
            _ScriptedLoc(["BOX: NONE"], presentation="boxed")

    def test_it_passes_the_claim_independence_audit(self):
        """The audit the run harness performs: same (C, Q), a claim shifted far away,
        bit-identical Localisation."""
        v = _ScriptedLoc(["BOX: 300 300 428 428\nCONFIDENCE: 0.8"] * 2)
        inp = _inputs()
        loc = v.localise(inp)
        _assert_claim_independent(v, v.localise(_with_perturbed_claim(inp)), loc)

    def test_the_judge_form_fails_that_same_audit(self):
        """Why the judge form cannot be a baseline row, stated as a test rather than as
        a caveat in prose. It is not that the comparison would be unflattering -- the
        shared runner would reject it, and its cache would mix two claims' answers."""
        v = _Scripted(["PROBABILITY: 0.8"] * 2)
        inp = _inputs()
        loc = v.localise(inp)
        with pytest.raises(AssertionError, match="must not read"):
            _assert_claim_independent(v, v.localise(_with_perturbed_claim(inp)), loc)

    def test_its_prompt_carries_no_claim_and_no_target(self):
        v = _ScriptedLoc(["BOX: NONE\nCONFIDENCE: 0.5"])
        v.localise(_inputs())
        p = v.seen[0][1].lower()
        for leak in ("29433", "16652", "clean", "fault", "claim"):
            assert leak not in p

    def test_its_name_separates_it_from_the_judge_diagnostic(self):
        """Two different questions must not write predictions into one directory."""
        v = _ScriptedLoc(["BOX: NONE"])
        assert v.name == "vlmloc_scripted_locate_grid"
        assert v.name != _Scripted(["PROBABILITY: 0.5"]).name


class TestItMapsTheAnswerBackIntoTheSlideFrame:
    def test_a_correct_box_in_the_shown_frame_recovers_the_true_place(self):
        """The model answers in the frame it was shown, which may have been shrunk from
        the context image, which is itself a window on the slide. Two conversions, and
        getting either wrong produces plausible numbers that nothing would flag."""
        inp = _inputs()
        claimed = inp["claimed_bbox"]
        # Build the frame exactly the way localise does, then answer correctly in it.
        from pyrafault.verify.vlm_presentation import context_relative_box, render
        r = render("locate_grid", inp["context"], inp["query"], None,
                   inp["context_bbox"], inp["context_mpp"], inp["query_mpp"])
        x0, y0, x1, y1 = context_relative_box(claimed, inp["context_bbox"],
                                              inp["context"].width)
        s = r.frame_px / inp["context"].width
        box = " ".join(str(round(v_ * s)) for v_ in (x0, y0, x1, y1))

        got = _ScriptedLoc([f"BOX: {box}\nCONFIDENCE: 0.9"]).localise(inp)
        assert got.estimated_bbox is not None
        assert abs(got.estimated_bbox.cx - claimed.cx) < 0.05 * claimed.w
        assert abs(got.estimated_bbox.cy - claimed.cy) < 0.05 * claimed.w

    def test_the_estimate_is_one_query_side_wide(self):
        """ZoomTrace and NCC search fixed-size windows, so their box is exactly one
        query side by construction and never loses overlap to a mis-sized guess. Snapping
        the model's centre to the same side puts it on that footing; the raw reply stays
        in diagnostics so the un-snapped version can be re-scored without a rerun."""
        inp = _inputs()
        got = _ScriptedLoc(["BOX: 300 300 900 900\nCONFIDENCE: 0.9"]).localise(inp)
        assert got.estimated_bbox.w == pytest.approx(inp["claimed_bbox"].w, rel=0.01)
        assert got.diagnostics["raw_box"] == (300, 300, 900, 900)

    def test_without_snapping_the_models_own_size_is_kept(self):
        inp = _inputs()
        got = _ScriptedLoc(["BOX: 300 300 900 900\nCONFIDENCE: 0.9"],
                           snap_to_query_side=False).localise(inp)
        assert got.estimated_bbox.w > 2 * inp["claimed_bbox"].w

    def test_the_centre_survives_snapping(self):
        inp = _inputs()
        a = _ScriptedLoc(["BOX: 300 300 900 900\nCONFIDENCE: 0.9"]).localise(inp)
        b = _ScriptedLoc(["BOX: 300 300 900 900\nCONFIDENCE: 0.9"],
                         snap_to_query_side=False).localise(inp)
        assert abs(a.estimated_bbox.cx - b.estimated_bbox.cx) < 2
        assert abs(a.estimated_bbox.cy - b.estimated_bbox.cy) < 2


class TestNoneIsARejectionNotAnAbstention:
    def test_a_confident_none_keeps_its_confidence(self):
        """`decide` reads a None estimate above tau as `mismatched` -- the correct answer
        for the four out-of-view fault types. Recording a confident NONE as low
        reliability would turn every one of those right answers into an abstention."""
        got = _ScriptedLoc(["BOX: NONE\nCONFIDENCE: 0.95"]).localise(_inputs())
        assert got.estimated_bbox is None
        assert got.reliability == pytest.approx(0.95)
        assert got.diagnostics["said_none"] is True

    def test_an_unparseable_reply_abstains_at_zero_and_is_counted(self):
        v = _ScriptedLoc(["the tissue looks similar in places"])
        got = v.localise(_inputs())
        assert (got.estimated_bbox, got.reliability) == (None, 0.0)
        assert v.n_unparsed == 1
        assert v.n_degenerate == 0

    def test_a_meaningless_box_is_the_models_failure_not_the_parsers(self):
        """`BOX: 1536 1536 1536 1536` -- a zero-area box in the corner. The model
        answered, meaninglessly, and that measurement stands. A reply the parser could
        not read means the measurement is worthless instead. One counter for both cannot
        tell which case a run is looking at, and the two call for opposite responses."""
        v = _ScriptedLoc(["BOX: 1536 1536 1536 1536\nCONFIDENCE: 1.0"])
        got = v.localise(_inputs())
        assert got.estimated_bbox is None
        assert (v.n_degenerate, v.n_unparsed) == (1, 0)
        assert got.diagnostics["degenerate"] is True

    def test_a_box_with_no_confidence_is_not_recorded_as_certain(self):
        v = _ScriptedLoc(["BOX: 300 300 428 428"])
        got = v.localise(_inputs())
        assert got.reliability == pytest.approx(DEFAULT_CONF)
        assert v.n_default_conf == 1


class TestTheBoxParserIsLenientButNeverInvents:
    def test_the_requested_form(self):
        assert parse_box_reply("BOX: 10 20 30 40\nCONFIDENCE: 0.5") == BoxReply(
            (10, 20, 30, 40), False, 0.5)

    def test_commas_parens_and_case_do_not_matter(self):
        assert parse_box_reply("box: (10, 20, 30, 40)").box == (10, 20, 30, 40)

    def test_a_reversed_box_is_normalised(self):
        assert parse_box_reply("BOX: 30 40 10 20").box == (10, 20, 30, 40)

    def test_a_degenerate_box_is_not_an_estimate(self):
        """Expanding a zero-area box to something usable would turn a formatting failure
        into a confident wrong answer."""
        assert parse_box_reply("BOX: 10 20 10 20").box is None

    def test_none_wins_over_coordinates_that_follow_it(self):
        """A model that says NONE and then explains what it ruled out must not have that
        explanation read back as its estimate."""
        r = parse_box_reply("BOX: NONE\nI considered 100 100 228 228 but the glands "
                            "do not match.")
        assert r.said_none and r.box is None

    def test_prose_with_no_answer_is_unparsed_not_a_rejection(self):
        """The `no idea` trap again: a refusal to answer is not a confident rejection,
        and that is the one direction that must never be guessed."""
        r = parse_box_reply("I have no idea where this came from.")
        assert r.box is None and r.said_none is False

    def test_four_bare_numbers_on_their_own_line_are_accepted(self):
        assert parse_box_reply("512 300 640 428").box == (512, 300, 640, 428)

    def test_an_unlabelled_confidence_on_its_own_line_is_read(self):
        """The model drops the `CONFIDENCE:` label and just writes the number on the
        next line. Recording that as "no confidence given" would charge the model a
        default it did not ask for."""
        r = parse_box_reply("BOX: 1152 1152 1248 1248\n0.95")
        assert r.box == (1152, 1152, 1248, 1248)
        assert r.confidence == pytest.approx(0.95)

    def test_a_box_with_a_trailing_full_stop_is_read(self):
        r = parse_box_reply("BOX: 1088 1047 1184 1143.\nCONFIDENCE: 0.9.")
        assert r.box == (1088, 1047, 1184, 1143)
        assert r.confidence == pytest.approx(0.9)

    def test_four_numbers_inside_prose_are_not(self):
        r = parse_box_reply("Slide 12 has 40 glands across 8 fields of 5 mm.")
        assert r.box is None
