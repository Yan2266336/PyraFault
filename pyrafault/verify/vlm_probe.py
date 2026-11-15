"""Diagnostics that are deliberately NOT baselines.

`VlmLocaliser` asks a multimodal model the task the protocol defines and its number may
appear in the baseline table. When that number comes out near chance, it does not say
why, and three explanations fit it equally well: the model cannot tell the tissue apart,
the model can but cannot express a coordinate, or the question was asked badly. A
baseline reported on an unexamined prompt failure is a strawman, and a reviewer will say
so -- so the failure has to be attributed, and attributing it needs strictly easier
versions of the same question.

Two are implemented here, in decreasing difficulty:

* **judge** -- the claim is shown and the model only has to rate it. One bit of work
  instead of a search over the whole reference view.
* **choice** -- four candidate regions are drawn and lettered, one of them the true
  source when the true source is inside the view at all, and the model picks a letter or
  NONE. Retrieval is handed over; only the comparison is left.

Neither can be a `Verifier`. `judge` reads the claim, which `Verifier.localise` forbids
and `runner._assert_claim_independent` catches; `choice` needs `true_bbox` to build its
option set, which lives in `target` and is the answer key. Both therefore run through
this module, against `dataset[item]["target"]`, and are reported in their own table under
their own heading. Mixing either into the baseline comparison would be comparing a method
that had to find the answer with one that was handed a shortlist containing it.

What a result here means:

* easier forms also near chance  -> the model cannot perceive the difference. The
  baseline number stands as measured, and the prompt is exonerated.
* easier forms clearly better    -> the model can perceive it and the free-form output
  is the bottleneck. The baseline number stands but must be reported with that caveat,
  and the gap is itself a finding about how these models have to be used here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from pyrafault.geom import BBoxL0
from pyrafault.verify.vlm_presentation import (
    Rendered, _fit, context_relative_box,
)

#: Lettered options per item, not counting NONE. Four plus NONE keeps a blind guess at
#: 20% and the prompt short enough that the option list is not itself the hard part.
N_OPTIONS = 4

#: Candidate boxes are kept this many query sides apart, so no two options overlap and
#: "the model picked a neighbour of the truth" is not scored as a hit.
MIN_SEPARATION = 1.5


@dataclass(frozen=True)
class ChoiceItem:
    """One multiple-choice question and its answer key.

    `answer` is the letter of the true source, or `None` when the true source is not
    inside the reference view -- which is the correct answer for the four out-of-view
    fault types and is offered as the explicit NONE option.
    """
    candidates: dict[str, BBoxL0]
    answer: str | None


def _letters(n: int) -> list[str]:
    return [chr(ord("A") + i) for i in range(n)]


def build_choice(true_bbox: BBoxL0, context_bbox: BBoxL0, side_l0: int,
                 true_within_context: bool, seed: int,
                 n_options: int = N_OPTIONS) -> ChoiceItem:
    """Lay out `n_options` query-sized candidate boxes inside the context.

    The true source is one of them when it is inside the context; otherwise all of them
    are distractors and the answer is NONE. Distractors are drawn from a seeded RNG so
    the question is identical on a rerun, and placed at least `MIN_SEPARATION` query
    sides from the truth and from each other.

    The letters are assigned by the same RNG rather than in placement order: otherwise
    the truth would always be `A` on every in-view item, and a model that noticed would
    score perfectly without looking at the tissue.
    """
    rng = np.random.default_rng(seed)
    lo_x, hi_x = context_bbox.x, context_bbox.x + context_bbox.w - side_l0
    lo_y, hi_y = context_bbox.y, context_bbox.y + context_bbox.h - side_l0
    if hi_x <= lo_x or hi_y <= lo_y:
        raise ValueError(
            f"context {context_bbox.w}x{context_bbox.h} cannot hold a {side_l0} box"
        )

    placed: list[BBoxL0] = []
    if true_within_context:
        placed.append(BBoxL0(true_bbox.x, true_bbox.y, side_l0, side_l0))

    def far_enough(x: int, y: int) -> bool:
        return all(max(abs(x - p.x), abs(y - p.y)) >= MIN_SEPARATION * side_l0
                   for p in placed)

    # Rejection sampling, with a bounded number of tries: a context is 16x the query
    # side per axis, so four separated boxes fit comfortably and failing to place one
    # means the geometry is not what this assumed. Raise rather than quietly return
    # fewer options -- a question with three options is not the question being scored.
    for _ in range(n_options - len(placed)):
        for _attempt in range(2000):
            x = int(rng.integers(lo_x, hi_x + 1))
            y = int(rng.integers(lo_y, hi_y + 1))
            if far_enough(x, y):
                placed.append(BBoxL0(x, y, side_l0, side_l0))
                break
        else:
            raise ValueError(
                f"could not place {n_options} boxes of {side_l0} px at least "
                f"{MIN_SEPARATION} sides apart in a {context_bbox.w} px context"
            )

    order = list(rng.permutation(len(placed)))
    letters = _letters(len(placed))
    candidates = {letters[slot]: placed[idx] for slot, idx in enumerate(order)}
    answer = None
    if true_within_context:
        # Index 0 is the truth by construction above; find where the permutation put it.
        answer = letters[order.index(0)]
    return ChoiceItem(candidates=candidates, answer=answer)


_CHOICE_TASK = (
    "You are tracing where a pathology image patch came from.\n"
    "IMAGE 1 is a low-magnification reference view of a region of a slide, with "
    "{n} candidate regions outlined and labelled {labels}.\n"
    "IMAGE 2 is a small patch, more magnified, showing tissue in much finer detail.\n"
    "The patch was taken from exactly one of the labelled regions, OR from somewhere "
    "else entirely -- a different part of the slide, or another slide.\n"
)
_CHOICE_ASK = (
    "\nAnswer with exactly two lines and nothing else:\n"
    "ANSWER: a single letter {labels} for the region the patch came from, or NONE if "
    "it came from none of them.\n"
    "CONFIDENCE: a number from 0.0 to 1.0 saying how sure you are.\n"
)


def render_choice(context: Image.Image, query: Image.Image,
                  item: ChoiceItem, context_bbox: BBoxL0) -> Rendered:
    """Draw the lettered candidates on the context and ask for a letter.

    Every candidate is drawn identically -- same colour, same line width, same label
    style -- so nothing but the tissue distinguishes the true one.
    """
    marked = _fit(context).convert("RGB").copy()
    scale = marked.width / context.width
    d = ImageDraw.Draw(marked)
    width = max(2, marked.width // 300)
    for letter, box in sorted(item.candidates.items()):
        x0, y0, x1, y1 = context_relative_box(box, context_bbox, context.width)
        x0, y0, x1, y1 = (round(v * scale) for v in (x0, y0, x1, y1))
        d.rectangle((x0, y0, x1, y1), outline=(255, 0, 0), width=width)
        d.text((x0 + width + 2, max(0, y0 - 14)), letter, fill=(255, 0, 0))
    labels = "/".join(sorted(item.candidates))
    return Rendered(
        [marked, _fit(query)],
        _CHOICE_TASK.format(n=len(item.candidates), labels=labels)
        + _CHOICE_ASK.format(labels=labels),
        frame_px=marked.width,
    )


_ANSWER = re.compile(r"answer\s*:\s*\(?\s*([a-z]+)", re.I)
#: A letter, or NONE, alone on a line. `re.M` rather than a match against the whole
#: stripped reply: the model answers `B\n0.95` -- the letter on its own line and the
#: confidence, unlabelled, on the next. An anchored whole-string match sees two lines,
#: fails, and the answer is thrown away as unparseable. That is the one failure this
#: whole module exists to rule out, so the anchors are per-line.
_LETTER_LINE = re.compile(r"^\s*\(?([a-z])\)?[.!]?\s*$", re.I | re.M)
_NONE_LINE = re.compile(r"^\s*(?:answer\s*:\s*)?none[.!]?\s*$", re.I | re.M)
_CONF = re.compile(r"confidence\s*:\s*(-?\d*\.?\d+)", re.I)
#: The same leniency for the number: a bare 0-1 value on a line of its own, which is how
#: the model actually spells it when it drops the label.
_CONF_LINE = re.compile(r"^\s*(0?\.\d+|[01](?:\.0+)?)\s*[.!,;:]?\s*$", re.M)


def parse_choice_reply(text: str, letters: list[str]
                       ) -> tuple[str | None, bool, float | None]:
    """Return `(letter, said_none, confidence)`.

    `(None, False, _)` means nothing usable was found. A letter outside the offered set
    is treated as unparsed rather than mapped to the nearest one: the model did not
    answer the question that was asked.
    """
    text = text or ""
    conf = None
    m = _CONF.search(text) or _CONF_LINE.search(text)
    if m:
        conf = min(1.0, max(0.0, float(m.group(1))))

    # A labelled answer first, then a letter or NONE on a line of its own. `NONE` is
    # checked before the bare-letter sweep so a reply that says NONE and then names the
    # options it ruled out is not read as picking one of them.
    m = _ANSWER.search(text)
    tok = m.group(1).upper() if m else None
    if tok is not None:
        if tok.startswith("NONE"):
            return None, True, conf
        if len(tok) == 1 and tok in letters:
            return tok, False, conf
        # A letter outside the offered set. Mapping it to the nearest option would
        # record an answer the model did not give.
        return None, False, conf
    if _NONE_LINE.search(text):
        return None, True, conf
    m = _LETTER_LINE.search(text)
    if m and m.group(1).upper() in letters:
        return m.group(1).upper(), False, conf
    # Last resort: `none` inside a sentence. A bare letter inside a sentence gets no
    # such fallback -- single letters occur in prose and a wrong pick is worse than an
    # abstention, while "none of them" is unambiguous.
    if re.search(r"\bnone\b", text, re.I):
        return None, True, conf
    return None, False, conf


def chance_rate(n_options: int = N_OPTIONS) -> float:
    """What a blind guess scores: `n_options` letters plus NONE, uniformly."""
    return 1.0 / (n_options + 1)
