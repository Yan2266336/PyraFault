"""The two diagnostics that are deliberately not baselines.

`VlmLocaliser` produces the comparable number. When that number lands near chance it
does not say why, and three explanations fit it: the model cannot tell the tissue apart,
it can but cannot express a coordinate, or the question was asked badly. Attributing the
failure needs strictly easier versions of the same question -- which is what `judge` and
`choice` are for, and why neither may appear in the baseline table: one reads the claim,
the other is handed a shortlist containing the answer.

These pin the properties that make the choice question honest. The interesting one is
that the answer letter must not be predictable: build the option set in placement order
and the truth is always `A`, and a model that noticed would score perfectly without
looking at a single cell.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pyrafault.geom import BBoxL0
from pyrafault.verify.vlm_probe import (
    MIN_SEPARATION, N_OPTIONS, build_choice, chance_rate, parse_choice_reply,
    render_choice,
)

CTX = BBoxL0(17616, 13212, 17617, 17617)
TRUE = BBoxL0(29433, 16652, 1101, 1101)
SIDE = 1101


@pytest.fixture
def images():
    rng = np.random.default_rng(0)
    return (Image.fromarray(rng.integers(0, 255, (2048, 2048, 3), dtype=np.uint8)),
            Image.fromarray(rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)))


class TestTheOptionSetContainsTheAnswerWhenThereIsOne:
    def test_the_true_source_is_one_of_the_options(self):
        it = build_choice(TRUE, CTX, SIDE, True, seed=3)
        assert it.answer is not None
        box = it.candidates[it.answer]
        assert (box.x, box.y) == (TRUE.x, TRUE.y)

    def test_an_out_of_view_source_makes_none_the_right_answer(self):
        """The four out-of-view fault types have no answer among the options, and that
        is not a gap in the question -- it is the question. `NONE` is offered as an
        option precisely so the correct answer is always expressible."""
        it = build_choice(TRUE, CTX, SIDE, False, seed=3)
        assert it.answer is None
        assert len(it.candidates) == N_OPTIONS

    def test_there_are_always_the_same_number_of_options(self):
        """A question with three options is not the question being scored, so a context
        that cannot hold four separated boxes raises rather than quietly shrinking."""
        for within in (True, False):
            for seed in range(20):
                it = build_choice(TRUE, CTX, SIDE, within, seed=seed)
                assert len(it.candidates) == N_OPTIONS

    def test_a_context_too_small_to_lay_them_out_raises(self):
        tiny = BBoxL0(0, 0, SIDE + 10, SIDE + 10)
        with pytest.raises(ValueError):
            build_choice(TRUE, tiny, SIDE, False, seed=0)


class TestNothingButTheTissueIdentifiesTheAnswer:
    def test_the_answer_letter_is_not_fixed(self):
        """Built in placement order the truth would be `A` every time. Over many items a
        model could learn that and never look at the tissue, and the probe would report
        a perception it had not measured."""
        seen = {build_choice(TRUE, CTX, SIDE, True, seed=s).answer
                for s in range(200)}
        assert seen == set("ABCD")

    def test_the_letters_are_roughly_evenly_used(self):
        counts: dict[str, int] = {}
        for s in range(400):
            a = build_choice(TRUE, CTX, SIDE, True, seed=s).answer
            counts[a] = counts.get(a, 0) + 1
        assert min(counts.values()) > 0.5 * max(counts.values())

    def test_the_options_do_not_overlap(self):
        """A model that picked a neighbour of the truth would otherwise be scored as
        right, and `adjacent_drift` is a fault type -- near-misses are exactly what this
        benchmark is about."""
        it = build_choice(TRUE, CTX, SIDE, True, seed=11)
        boxes = list(it.candidates.values())
        for i, a in enumerate(boxes):
            for b in boxes[i + 1:]:
                assert max(abs(a.x - b.x), abs(a.y - b.y)) >= MIN_SEPARATION * SIDE

    def test_every_option_lies_inside_the_reference_view(self):
        it = build_choice(TRUE, CTX, SIDE, False, seed=5)
        for box in it.candidates.values():
            assert CTX.x <= box.x and box.x + box.w <= CTX.x + CTX.w
            assert CTX.y <= box.y and box.y + box.h <= CTX.y + CTX.h

    def test_the_same_seed_asks_the_same_question(self):
        a = build_choice(TRUE, CTX, SIDE, True, seed=42)
        b = build_choice(TRUE, CTX, SIDE, True, seed=42)
        assert a == b


class TestTheRenderedQuestion:
    def test_it_shows_the_context_the_patch_and_the_letters(self, images):
        ctx, qry = images
        it = build_choice(TRUE, CTX, SIDE, True, seed=1)
        out = render_choice(ctx, qry, it, CTX)
        assert len(out.images) == 2
        assert all("/".join(sorted(it.candidates)) in out.prompt for _ in (0,))
        assert "NONE" in out.prompt

    def test_the_prompt_does_not_say_which_letter_is_right(self, images):
        ctx, qry = images
        it = build_choice(TRUE, CTX, SIDE, True, seed=1)
        out = render_choice(ctx, qry, it, CTX).prompt
        assert str(TRUE.x) not in out and str(TRUE.y) not in out
        assert f"ANSWER: {it.answer}" not in out

    def test_no_image_exceeds_what_a_model_will_accept(self, images):
        ctx, qry = images
        it = build_choice(TRUE, CTX, SIDE, True, seed=1)
        for im in render_choice(ctx, qry, it, CTX).images:
            assert max(im.size) <= 1536


class TestTheReplyParser:
    def test_a_letter(self):
        assert parse_choice_reply("ANSWER: C\nCONFIDENCE: 0.7",
                                  list("ABCD")) == ("C", False, 0.7)

    def test_none(self):
        assert parse_choice_reply("ANSWER: NONE", list("ABCD")) == (None, True, None)

    def test_a_bare_letter(self):
        assert parse_choice_reply("B", list("ABCD"))[0] == "B"

    def test_the_shape_the_model_actually_answers_in(self):
        """`B\n0.95` -- the letter on its own line, the confidence unlabelled on the
        next. The first parser anchored against the whole stripped reply, saw two lines,
        matched nothing, and discarded 9 of 10 real answers as unparseable. Throwing
        away a usable answer over layout is the exact failure the probe exists to rule
        out, so this is pinned with the strings the model produced."""
        assert parse_choice_reply("B\n0.95", list("ABCD")) == ("B", False, 0.95)
        assert parse_choice_reply("NONE\n0.0", list("ABCD")) == (None, True, 0.0)
        assert parse_choice_reply("A\n0.95", list("ABCD")) == ("A", False, 0.95)

    def test_a_letter_inside_prose_is_not_an_answer(self):
        """Single letters occur in prose, and a wrong pick is worse than an abstention.
        `none of them` gets the opposite treatment because it is unambiguous."""
        assert parse_choice_reply("Region A and region C both have glands, so it is "
                                  "hard to say.", list("ABCD"))[0] is None

    def test_a_letter_that_was_not_offered_is_unparsed(self):
        """Mapping `F` to the nearest option would record an answer the model did not
        give."""
        assert parse_choice_reply("ANSWER: F", list("ABCD")) == (None, False, None)

    def test_prose_with_no_answer_is_unparsed_not_a_rejection(self):
        assert parse_choice_reply("I cannot tell.", list("ABCD")) == (None, False, None)

    def test_none_spelled_out_in_prose_counts(self):
        assert parse_choice_reply("It came from none of them.",
                                  list("ABCD"))[1] is True


def test_chance_counts_the_none_option():
    """Four letters plus NONE is five answers, so a blind guess scores 20% -- and a
    model that always answers NONE scores whatever share of items are out-of-view,
    which is why the probe reports accuracy split by that."""
    assert chance_rate() == pytest.approx(1 / (N_OPTIONS + 1))
