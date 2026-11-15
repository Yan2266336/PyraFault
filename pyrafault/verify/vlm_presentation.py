"""How a (context, patch, claim) is shown to a multimodal model.

Separate from the verifier because this, not the model choice, is the most likely thing
to decide the baseline's score. The number that forces the issue: the patch covers
128x128 of a 2048x2048 context -- 0.39% of its area. A model whose processor shrinks the
context to 1024 sees the target as 64x64 pixels; at 512 it is 32x32. Hand over two raw
images and the result measures the input pipeline rather than the model, and a baseline
weakened like that is worse than none -- it is a strawman, and a reviewer will say so.

So several presentations are implemented, all are run on a pilot, and the best one is
what the comparison reports. Everything else about the call is held fixed, which is the
same rule the matcher ablation follows: only the thing under test may vary.

Two families, and the difference between them is not cosmetic:

* **`CLAIM_FREE`** (`locate`, `locate_grid`) ask the model the task the protocol
  actually defines -- *where did this patch come from* -- and never mention the claim.
  Only these can back a row in the baseline table, because `Verifier.localise` promises
  an estimate independent of the claim and `runner._assert_claim_independent` audits it
  by re-running under a shifted claim. A presentation that shows the claim fails that
  audit, and worse: `cache_key` deliberately excludes the claim, so two items sharing
  one `(C, Q)` under different claims would be served a single cached answer -- the same
  class of silent cross-contamination as the degradation cache-key bug.
* **The claim-shown forms** (`pair`, `boxed`, `sidebyside`, `tiles`) ask a strictly
  easier question: *is this particular claim right*. They are diagnostics, not
  baselines. Their job is to separate "the model cannot see the difference" from "the
  model cannot express an answer in the requested format" -- without which a near-chance
  score on the real task cannot be attributed, and a baseline reported on an unexamined
  prompt failure is the strawman this module exists to avoid. They must be run through
  `vlm_probe`, never through the shared runner.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PIL import Image, ImageDraw

from pyrafault.geom import BBoxL0

#: Above roughly this, a model's own processor starts downsampling and the presentation
#: loses control of what the model actually sees. Each presentation resizes deliberately
#: to stay under it rather than leaving the decision to the processor.
MAX_SIDE = 1536

#: The context is cut into this many tiles per axis for the `tiles` presentation.
TILE_GRID = 3

#: Gridlines per axis for the `locate_grid` presentation's coordinate overlay.
GRID_LINES = 8


@dataclass(frozen=True)
class Rendered:
    """What one presentation hands the model: images in order, and the text.

    `frame_px` is the width, in the pixels the model actually sees, of the context as
    this presentation showed it. A presentation that asks the model for coordinates has
    to report it: the model answers in the frame it was given, and mapping that back to
    level-0 without knowing the frame is the `RomaMatcher` failure again -- plausible
    numbers in the wrong frame, and nothing raises. `None` for presentations that never
    ask for a coordinate.
    """
    images: list[Image.Image]
    prompt: str
    frame_px: int | None = None


def context_relative_box(claim: BBoxL0, context: BBoxL0, context_px: int
                         ) -> tuple[int, int, int, int]:
    """A level-0 claim box as pixel coordinates inside the context image.

    Level-0 coordinates are absolute slide pixels: a claim sits at (29433, 16652) while
    its context starts at (17616, 13212) and is 17617 wide. Those numbers mean nothing to
    a model that can only see the context image, so they are converted before being
    written into any prompt. Getting this wrong is the same failure as in `RomaMatcher`:
    plausible numbers in the wrong frame, and nothing raises.

    Raises when the claim is not inside the context. That is a real case -- the
    out-of-context fault types are exactly it -- so the caller has to decide what to do
    rather than be handed a silently clamped box.
    """
    scale = context_px / context.w
    x0 = round((claim.x - context.x) * scale)
    y0 = round((claim.y - context.y) * scale)
    side = round(claim.w * scale)
    if not (0 <= x0 <= context_px and 0 <= y0 <= context_px):
        raise ValueError(
            f"claim ({claim.x}, {claim.y}) is outside context "
            f"({context.x}, {context.y}, {context.w}) -- an out-of-context item; the "
            "caller must decide how to present it"
        )
    return x0, y0, x0 + side, y0 + side


def _fit(img: Image.Image, cap: int = MAX_SIDE) -> Image.Image:
    if max(img.size) <= cap:
        return img
    s = cap / max(img.size)
    return img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)


_TASK = (
    "You are checking a provenance claim about a pathology slide.\n"
    "IMAGE 1 is a low-magnification reference view of a region of the slide.\n"
    "IMAGE 2 is a high-magnification patch, 4x more magnified, taken from somewhere.\n"
    "A tool claims IMAGE 2 came from a specific place inside IMAGE 1.\n"
)
#: One number, defined once, in the direction the decision rule reads it.
#:
#: The first version asked for a VERDICT plus a CONFIDENCE "from 0.0 to 1.0" without
#: saying confidence in WHAT. The model read it the sensible way -- the probability the
#: patch came from the claimed place -- and answered `no` with `0.0`, meaning "certainly
#: not there". The code read it the other way, as certainty in its own verdict, and
#: computed `1 - 0.0 = 1.0`: the most confident rejection came out as the highest
#: reliability. The two readings agree on `yes` and invert on `no`, so the bug was
#: invisible in half the cases.
#:
#: Asking for the probability alone removes the ambiguity rather than papering over it:
#: it is one quantity, it points the same way as `reliability` in the protocol (high
#: means "trust the claim"), and there is no second number to disagree with it.
_ASK = (
    "\nAnswer with exactly one line and nothing else:\n"
    "PROBABILITY: a number from 0.0 to 1.0 giving the probability that the patch "
    "really came from the claimed place. Use 1.0 if you are certain it did, 0.0 if "
    "you are certain it did not, and 0.5 if you cannot tell.\n"
)


def _pair(ctx, qry, claim_box, _ctx_box, _side_px) -> Rendered:
    """The naive presentation: two images and the claim as numbers.

    Kept because it is what anyone would try first, and because the comparison needs the
    floor as well as the ceiling -- if a better presentation wins, that is itself a
    finding about how these models have to be used on gigapixel imagery.
    """
    x0, y0, x1, y1 = claim_box
    return Rendered(
        [_fit(ctx), _fit(qry)],
        _TASK + f"\nThe claimed place is the box ({x0}, {y0}) to ({x1}, {y1}) in "
                f"IMAGE 1's pixel coordinates, where (0, 0) is its top-left corner "
                f"and it is {ctx.width} pixels wide." + _ASK,
    )


def _boxed(ctx, qry, claim_box, _ctx_box, _side_px) -> Rendered:
    """The claim drawn on the context, so finding it is not part of the task.

    This separates two abilities the naive presentation confounds: reading a coordinate
    into an image, and judging whether two tissue regions are the same place.
    """
    marked = ctx.convert("RGB").copy()
    d = ImageDraw.Draw(marked)
    d.rectangle(claim_box, outline=(255, 0, 0), width=max(3, ctx.width // 256))
    return Rendered(
        [_fit(marked), _fit(qry)],
        _TASK + "\nThe claimed place is outlined with a red box in IMAGE 1." + _ASK,
    )


def _sidebyside(ctx, qry, claim_box, _ctx_box, _side_px) -> Rendered:
    """Both in one image, so the model never has to hold two encodings in register."""
    marked = ctx.convert("RGB").copy()
    d = ImageDraw.Draw(marked)
    d.rectangle(claim_box, outline=(255, 0, 0), width=max(3, ctx.width // 256))
    h = min(marked.height, MAX_SIDE)
    left = _fit(marked, h)
    right = qry.convert("RGB").resize((h, h), Image.LANCZOS)
    canvas = Image.new("RGB", (left.width + right.width + 16, h), (255, 255, 255))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + 16, 0))
    return Rendered(
        [_fit(canvas)],
        _TASK.replace("IMAGE 1", "the LEFT image").replace("IMAGE 2", "the RIGHT image")
        + "\nBoth are in one picture: the reference view on the left with the claimed "
          "place outlined in red, the high-magnification patch on the right." + _ASK,
    )


def _tiles(ctx, qry, claim_box, _ctx_box, _side_px) -> Rendered:
    """The context cut into a grid, each piece sent at its own resolution.

    This is the presentation built for the 0.39% problem. Shrinking a 2048 context to
    1536 costs a quarter of the linear detail in the one region that matters; nine tiles
    of ~683 px each arrive whole. The claimed tile is marked and named so the model is
    not also being asked to do the indexing.
    """
    n = TILE_GRID
    w, h = ctx.size
    tw, th = w // n, h // n
    cx = (claim_box[0] + claim_box[2]) // 2
    cy = (claim_box[1] + claim_box[3]) // 2
    ci, cj = min(cx // tw, n - 1), min(cy // th, n - 1)
    images, marked_name = [], f"tile ({cj + 1}, {ci + 1})"
    for j in range(n):
        for i in range(n):
            t = ctx.crop((i * tw, j * th, (i + 1) * tw, (j + 1) * th)).convert("RGB")
            if (i, j) == (ci, cj):
                d = ImageDraw.Draw(t)
                d.rectangle(
                    (claim_box[0] - i * tw, claim_box[1] - j * th,
                     claim_box[2] - i * tw, claim_box[3] - j * th),
                    outline=(255, 0, 0), width=max(3, tw // 128))
            images.append(_fit(t))
    images.append(_fit(qry))
    return Rendered(
        images,
        _TASK.replace(
            "IMAGE 1 is a low-magnification reference view of a region of the slide.",
            f"The reference view is given as a {n}x{n} grid of tiles, in reading order "
            f"(row 1 left to right, then row 2, and so on). The LAST image is the "
            f"high-magnification patch.",
        ).replace("IMAGE 2 is a high-magnification patch, 4x more magnified, taken from "
                  "somewhere.\n", "")
        # This third substitution was missing, so the prompt still referred to "IMAGE 1"
        # and "IMAGE 2" in a presentation that has neither -- ten images, none of them
        # called that. Visible only once the assembled text was printed, which is the
        # argument for printing it before trusting any number computed from it.
        .replace("A tool claims IMAGE 2 came from a specific place inside IMAGE 1.",
                 "A tool claims the patch came from a specific place inside the "
                 "reference view.")
        + f"\nThe claimed place is outlined in red inside {marked_name}." + _ASK,
    )


#: The task as the protocol defines it, with no claim in it. The wording states plainly
#: that the patch may not be in the reference view at all: four of the eight fault types
#: are exactly that case, and a model told to find the patch somewhere would be scored on
#: a question with no answer.
_LOCATE_TASK = (
    "You are tracing where a pathology image patch came from.\n"
    "IMAGE 1 is a low-magnification reference view of a region of a slide.\n"
    "IMAGE 2 is a small patch, more magnified, showing tissue in much finer detail.\n"
    "The patch may come from somewhere inside the region shown in IMAGE 1, or it may "
    "come from a completely different part of the slide, or from another slide "
    "entirely.\n"
)
#: `CONFIDENCE` here means confidence in the model's OWN box, which is what the protocol
#: calls reliability and what the scorer multiplies by the overlap with the claim. That
#: is a different quantity from the claim-shown forms' `PROBABILITY`, and deliberately
#: so: those forms are handed a hypothesis to rate, this one has to produce the answer.
_LOCATE_ASK = (
    "\nAnswer with exactly two lines and nothing else:\n"
    "BOX: x0 y0 x1 y1 -- the square region of IMAGE 1 the patch was taken from, in "
    "IMAGE 1's pixel coordinates, where (0, 0) is its top-left corner. Write "
    "BOX: NONE instead if the patch did not come from anywhere inside IMAGE 1.\n"
    "CONFIDENCE: a number from 0.0 to 1.0 saying how sure you are of that answer.\n"
)


def expected_side_px(query: Image.Image,
                     context_mpp: float, query_mpp: float) -> int:
    """How many pixels of the shown context one query patch spans.

    Derived from `context_mpp` and `query_mpp`, which the loader puts in `inputs`
    precisely so a method can align the two magnifications -- so this is claim-free and
    label-free, and it is the same constant every method is handed. Telling the model the
    size is not a hint about the answer: it is the scale of the task, and withholding it
    would score the model on guessing the magnification rather than on finding the
    tissue.
    """
    physical_um = query.width * query_mpp          # the patch's true width on the slide
    return max(1, round(physical_um / context_mpp))


def _scale_note(shown: Image.Image, ctx: Image.Image, side_px: int | None) -> str:
    """The sentence stating the frame's size and the patch's size within it.

    It goes in the task description, not after the answer format. Appended to the last
    line of the format block it reads as part of what CONFIDENCE means -- which is how
    the first version of this prompt shipped, and a model reading it that way is being
    asked something other than the question intended.
    """
    note = (f"IMAGE 1 is {shown.width} pixels wide and {shown.height} pixels tall.")
    if side_px is not None:
        k = round(side_px * shown.width / ctx.width)
        note += f" The region you are looking for is about {k} by {k} pixels in it."
    return "\n" + note + "\n"


def _locate(ctx, qry, _claim_box, _ctx_box, side_px) -> Rendered:
    """The protocol's own question, asked plainly. The comparable form."""
    shown = _fit(ctx)
    return Rendered([shown, _fit(qry)],
                    _LOCATE_TASK + _scale_note(shown, ctx, side_px) + _LOCATE_ASK,
                    frame_px=shown.width)


def _locate_grid(ctx, qry, _claim_box, _ctx_box, side_px) -> Rendered:
    """The same question with a labelled coordinate grid drawn over the reference view.

    Reading a pixel coordinate off an unmarked image is a separate skill from finding
    the tissue, and failing at the first would be scored here as failing at the second.
    The grid removes that confound without giving anything away -- it is drawn from the
    image's own dimensions and is identical for every item, clean or faulty.
    """
    shown = _fit(ctx).convert("RGB").copy()
    d = ImageDraw.Draw(shown)
    step = shown.width // GRID_LINES
    for k in range(1, GRID_LINES):
        p = k * step
        d.line([(p, 0), (p, shown.height)], fill=(0, 160, 255), width=1)
        d.line([(0, p), (shown.width, p)], fill=(0, 160, 255), width=1)
        d.text((p + 3, 3), str(p), fill=(0, 100, 200))
        d.text((3, p + 3), str(p), fill=(0, 100, 200))
    return Rendered(
        [shown, _fit(qry)],
        _LOCATE_TASK
        + _scale_note(shown, ctx, side_px).rstrip("\n")
        + f" A blue coordinate grid is drawn over IMAGE 1, with gridlines every {step} "
          f"pixels labelled along the top and left edges, to help you read off "
          f"coordinates.\n"
        + _LOCATE_ASK,
        frame_px=shown.width,
    )


PRESENTATIONS: dict[str, Callable[..., Rendered]] = {
    "pair": _pair,
    "boxed": _boxed,
    "sidebyside": _sidebyside,
    "tiles": _tiles,
    "locate": _locate,
    "locate_grid": _locate_grid,
}

#: The presentations that never mention the claim, and so are the only ones a
#: `Verifier` may use -- see this module's docstring for why that is a hard line and
#: not a preference.
CLAIM_FREE = frozenset({"locate", "locate_grid"})


def render(name: str, context: Image.Image, query: Image.Image,
           claim: BBoxL0 | None, context_bbox: BBoxL0,
           context_mpp: float | None = None,
           query_mpp: float | None = None) -> Rendered:
    """Build one presentation. `claim` and `context_bbox` are level-0; the conversion to
    image coordinates happens here so no caller has to repeat it.

    `claim` may be `None` only for a `CLAIM_FREE` presentation, and a claim-free
    presentation is never handed one. A caller that wants the audited guarantee passes
    `None`, which makes reading the claim impossible rather than merely discouraged.
    """
    if name not in PRESENTATIONS:
        raise ValueError(f"unknown presentation {name!r}; known: {sorted(PRESENTATIONS)}")
    side = (None if context_mpp is None or query_mpp is None else
            expected_side_px(query, context_mpp, query_mpp))
    if name in CLAIM_FREE:
        box = None
    elif claim is None:
        raise ValueError(f"presentation {name!r} shows the claim, so it needs one")
    else:
        box = context_relative_box(claim, context_bbox, context.width)
    return PRESENTATIONS[name](context, query, box, context_bbox, side)
