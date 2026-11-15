"""A multimodal language model as a CMPV verifier, for the baseline table.

This exists to answer the question a reviewer asks before any other: why a purpose-built
verifier, when a frontier multimodal model can look at two images? Answering it needs a
number, and the number is only worth having if the model got a fair attempt -- so the
presentation is chosen by ablation (`vlm_presentation`), the model is asked the easiest
form of the question, and its reply is parsed leniently.

What it does NOT do is score itself. It returns a `Localisation` like every other
verifier and is thresholded by the shared `decide`, fitted by the shared `fit_thresholds`
and scored by stage 14. A baseline that brought its own metric could not be compared.
"""
from __future__ import annotations

import os
import re
from typing import Any, Mapping

from dataclasses import dataclass

from pyrafault.geom import BBoxL0
from pyrafault.verify.base import Localisation, Verifier
from pyrafault.verify.vlm_presentation import (
    CLAIM_FREE, expected_side_px, render,
)
from pyrafault.verify.zoomtrace import context_to_level0

#: The model is asked for one number: P(the patch really came from the claimed place).
#: `verdict` is kept as a fallback spelling because models sometimes volunteer it, but
#: the probability is what the score uses.
_PROB = re.compile(r"probability\s*:\s*(-?\d*\.?\d+)", re.I)
#: `_TAIL` is why this is not a one-character detail. One model answers `0.9.` -- the
#: number with a full stop after it. Anchored at `$` with only whitespace allowed, that
#: period made every one of 100 replies unparseable, and the run reported a 32B model as
#: having produced no answer at all. A trailing period is not a failure to answer, and
#: reading it as one is the exact mistake this module was written to prevent.
_TAIL = r"\s*[.!,;:]?\s*$"
_BARE = re.compile(r"(?:^|\s)(0?\.\d+|[01](?:\.0+)?)" + _TAIL)
#: A verdict counts only when it is labelled, or when it is the entire reply. A loose
#: `\b(yes|no)\b` matched the "no" inside "no idea" -- turning a refusal to answer into
#: a confident rejection, which is the one direction that must never be guessed.
_VERDICT_LABELLED = re.compile(r"verdict\s*:\s*(yes|no)\b", re.I)
_VERDICT_ALONE = re.compile(r"^(yes|no)[.!]?$", re.I)

#: What a bare `yes`/`no` with no number is worth. Deliberately not 1.0/0.0: a model that
#: skipped the number did not express certainty, and recording it as certain would hand
#: the baseline a sharper score than it earned. Runs report how often this was needed.
FALLBACK_YES, FALLBACK_NO = 0.75, 0.25


def parse_reply(text: str) -> tuple[float | None, bool]:
    """Pull P(claim is right) out of a model's reply.

    Returns `(probability, used_fallback)`; `(None, False)` when nothing usable is
    there. Lenient about formatting, because discarding a usable answer over layout
    would understate the baseline -- the failure this module exists to avoid -- but
    never inventing a number, because a guess would turn a formatting failure into a
    wrong answer and contaminate the score silently.

    Three spellings, in order of trust: the requested `PROBABILITY: x`; a bare number on
    its own; and a bare yes/no, which is mapped to a deliberately unsharp 0.75/0.25
    since a model that gave no number expressed no certainty.
    """
    text = text or ""
    m = _PROB.search(text)
    if m:
        return min(1.0, max(0.0, float(m.group(1)))), False
    m = _BARE.search(text.strip())
    if m:
        return min(1.0, max(0.0, float(m.group(1)))), False
    m = _VERDICT_LABELLED.search(text) or _VERDICT_ALONE.match(text.strip())
    if m:
        return (FALLBACK_YES if m.group(1).lower() == "yes" else FALLBACK_NO), True
    return None, False


_JUDGE_NOTE = """Ask a multimodal model whether the patch came from the claimed place.

    The model is asked to JUDGE, not to localise. Free-form coordinate regression is the
    thing these models are worst at, and a baseline built on their weakest output would
    be a strawman; judging is the easiest honest form of the task. The consequence is
    that a `yes` estimates the claimed box (the model asserts the claim is right) and a
    `no` estimates nothing -- which is the protocol's confident rejection, not an
    abstention.

    `reliability` is P(the claim is right), which is what `decide` thresholds: high means
    "trust this claim". The model is asked for exactly that number, in exactly that
    direction, so no conversion stands between its answer and the score. An earlier
    version asked for a verdict plus a "confidence" it never defined, and the two readings
    of that word inverted on every `no` -- see `_ASK` in `vlm_presentation`.

    NOT A BASELINE. It reads `claimed_bbox`, which `Verifier.localise` forbids, so
    `runner._assert_claim_independent` fails on it and `cache_key` -- which excludes the
    claim on purpose -- would serve one cached answer to two items that share a
    `(C, Q)` under different claims. Run it only through `vlm_probe`, and report it only
    as the diagnostic it is: evidence about whether the model can perceive the
    difference at all, next to the real number from `VlmLocaliser`.
"""

class _HfVlm:
    """Loading and calling one Hugging Face image-text model.

    Shared by the baseline and the diagnostic so that the only difference between them
    is the question asked -- the same rule the matcher ablation follows.
    """

    #: Models whose chat template opens the assistant turn with `<think>`, so the reply
    #: begins inside a reasoning block whatever the prompt asks for. Matched on the id
    #: because that is what a caller has before the template is loaded; `_load` then
    #: checks the rendered template and widens this set at runtime, so a model not
    #: listed here is still handled.
    THINKS = ("qwen3.8", "qwen3-next", "deepseek-r1", "qwq")

    def __init__(self, model_id: str, max_new_tokens: int = 64,
                 device: str | None = None, _defer_load: bool = False,
                 no_thinking: bool | None = None) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        #: Switch the reasoning block off for a template that gates it on
        #: `enable_thinking`. Default `None` means "decide from the model", which is the
        #: only safe default: Qwen3.8-27B's template opens every assistant turn with
        #: `<think>\n`, so with 64 tokens it produced 100 unparsed replies on all seven
        #: ladder presentations and its whole row read as AUROC 0.49-0.50 -- the value a
        #: constant score gives, indistinguishable from a model that cannot do the task.
        #: It was in fact describing the tissue correctly and never reaching an answer.
        #: The same trap had already been found and fixed in the agent loop
        #: (`agent.responder.ask_chat`) without the fix reaching here, which is why the
        #: switch now lives on the shared base rather than at each call site.
        self._no_thinking = no_thinking
        #: `"auto"` shards the weights across whatever GPUs the job was given. Needed
        #: because the largest medical model in the comparison is 32B -- about 64 GB in
        #: bf16 -- and the GPUs this cluster grants without PI approval hold 48 GB. The
        #: alternative was quantising it, which would have made its row answer a
        #: different question from the others'.
        self._device = device or os.environ.get("PYRAFAULT_VLM_DEVICE") or None
        self._model = self._proc = None
        if not _defer_load:
            self._load()

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._proc = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map=self._device).eval()
        if self._no_thinking is None:
            self._no_thinking = self._template_opens_thinking()

    def _template_opens_thinking(self) -> bool:
        """Whether this model's own template starts the reply inside a reasoning block.

        Asked of the rendered template rather than of the id alone, because the id list
        can only name the models known when it was written and a reasoning template is
        the direction the field is moving.
        """
        if any(k in self.model_id.lower() for k in self.THINKS):
            return True
        try:
            rendered = self._proc.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
                add_generation_prompt=True, tokenize=False)
        except Exception:
            return False
        return "<think>" in str(rendered)[-200:]

    def _ask(self, images: list, prompt: str) -> str:
        """One call. Overridden in tests with canned replies."""
        import torch
        msg = [{"role": "user", "content":
                [{"type": "image", "image": im} for im in images]
                + [{"type": "text", "text": prompt}]}]
        kw = {"enable_thinking": False} if self._no_thinking else {}
        try:
            inputs = self._proc.apply_chat_template(
                msg, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt", **kw).to(self._model.device)
        except TypeError:
            # The template takes no `enable_thinking`. Falling back rather than raising:
            # a model with no reasoning mode needs nothing switched off, and a template
            # that opens `<think>` unconditionally is reported by `n_unparsed` instead
            # of silently costing a whole row.
            inputs = self._proc.apply_chat_template(
                msg, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                       do_sample=False)
        return self._proc.decode(out[0][inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True)


class VlmBaseline(_HfVlm, Verifier):
    """DIAGNOSTIC ONLY -- see the class docstring below and `vlm_probe`."""

    revision = "1"

    def __init__(self, model_id: str, presentation: str = "boxed", **kw) -> None:
        super().__init__(model_id, **kw)
        self.presentation = presentation
        #: Both are part of the method: two presentations reach different score
        #: distributions, so their thresholds are not interchangeable and their
        #: predictions must not land in one directory.
        self.name = f"vlm_{_slug(model_id)}_{presentation}"
        self.n_unparsed = 0
        self.n_default_conf = 0

    def localise(self, inputs: Mapping[str, Any]) -> Localisation:
        claimed = inputs["claimed_bbox"]
        try:
            shown = render(self.presentation, inputs["context"], inputs["query"],
                           claimed, inputs["context_bbox"])
        except ValueError:
            # The claim is outside the context -- the out-of-context fault types. There
            # is nothing to point at inside the reference view, which is itself the
            # answer, so it is reported rather than forced into a presentation.
            return Localisation(estimated_bbox=None, reliability=0.0)

        p, used_fallback = parse_reply(self._ask(shown.images, shown.prompt))
        if p is None:
            self.n_unparsed += 1
            return Localisation(estimated_bbox=None, reliability=0.0)
        self.n_default_conf += int(used_fallback)
        # Above the midpoint the model is asserting the claim, so the estimate IS the
        # claimed box; below it there is no alternative on offer, which is the protocol's
        # confident rejection. `reliability` is the probability either way -- `decide`
        # applies tau, and where that line falls is fitted on calibration, not here.
        if p >= 0.5:
            return Localisation(estimated_bbox=claimed, reliability=p)
        return Localisation(estimated_bbox=None, reliability=p)


def _slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", model_id.split("/")[-1]).strip("_").lower()


#: What a box with no stated confidence is worth. Same reasoning as `FALLBACK_YES`: a
#: model that skipped the number expressed no certainty, and recording it as certain
#: would hand the baseline a sharper score than it earned. Runs report how often it was
#: needed.
DEFAULT_CONF = 0.5

_BOX_NONE = re.compile(r"box\s*:\s*(none|n/a|unknown|not\s+(?:in|present|visible))",
                       re.I)
_BOX_NUMS = re.compile(
    r"box\s*:\s*\(?\s*(-?\d+)\s*[,\s]\s*(-?\d+)\s*[,\s]\s*"
    r"(-?\d+)\s*[,\s]\s*(-?\d+)", re.I)
_CONF = re.compile(r"confidence\s*:\s*(-?\d*\.?\d+)", re.I)
#: The same number without its label, alone on a line -- which is how the model spells
#: it when it drops the label. Per-line anchors, for the reason `_LETTER_LINE` in
#: `vlm_probe` gives: a whole-string anchor sees the box line too and matches nothing.
_CONF_LINE = re.compile(r"^\s*(0?\.\d+|[01](?:\.0+)?)" + _TAIL, re.M)
#: A last resort for a model that gives four numbers without the label. Anchored to a
#: line of its own so it cannot pick up four digits out of a sentence of prose.
_FOUR_BARE = re.compile(r"^\(?\s*(-?\d+)\s*[,\s]\s*(-?\d+)\s*[,\s]\s*"
                        r"(-?\d+)\s*[,\s]\s*(-?\d+)\s*\)?" + _TAIL, re.M)


@dataclass(frozen=True)
class BoxReply:
    """A parsed answer to the locate question.

    `box` and `said_none` are the two usable outcomes and they are different answers,
    not two spellings of one: a box is an estimate, `NONE` is the protocol's confident
    rejection. Both `None`/`False` means nothing usable was found, which is counted and
    abstained on rather than guessed -- the same rule `parse_reply` follows.

    `degenerate` separates two failures a single `unparsed` counter would conflate, and
    they call for opposite responses. One model answered `BOX: 1536 1536 1536 1536` --
    a zero-area box in the corner. That is the model giving a meaningless answer, and
    the measurement stands. Another answered `0.9.` and was discarded over the full
    stop: that was the parser's fault, and the measurement was worthless. A run that
    reports one number for both cannot tell which it is looking at.
    """
    box: tuple[int, int, int, int] | None
    said_none: bool
    confidence: float | None
    degenerate: bool = False


def parse_box_reply(text: str) -> BoxReply:
    """Pull a box (in the frame the model was shown) and a confidence out of a reply."""
    text = text or ""
    conf = None
    m = _CONF.search(text) or _CONF_LINE.search(text)
    if m:
        conf = min(1.0, max(0.0, float(m.group(1))))
    # `NONE` is checked first: a reply that says `BOX: NONE` and then explains itself
    # with coordinates of what it ruled out must not be read as an estimate.
    if _BOX_NONE.search(text):
        return BoxReply(None, True, conf)
    m = _BOX_NUMS.search(text) or _FOUR_BARE.search(text)
    if not m:
        return BoxReply(None, False, conf)
    x0, y0, x1, y1 = (int(g) for g in m.groups())
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    if x1 <= x0 or y1 <= y0:
        # A degenerate box is not an estimate. Expanding it to something usable would
        # turn the model's nonsense into a confident wrong answer -- but it IS an
        # answer, so it is flagged rather than filed under "could not be read".
        return BoxReply(None, False, conf, degenerate=True)
    return BoxReply((x0, y0, x1, y1), False, conf)


class VlmLocaliser(_HfVlm, Verifier):
    """Ask a multimodal model the task the protocol actually defines.

    This is the form that may appear in the baseline table, and the only one: it is
    handed no claim, so its estimate is independent of the claim by construction rather
    than by promise, and `runner._assert_claim_independent` passes. Every other method in
    the table -- ZoomTrace, NCC, SIFT, RoMa -- searches the reference view for the patch
    and reports where it landed plus how much it trusts that. So does this.

    `reliability` is the model's confidence in its OWN box, which is what `decide`
    thresholds and what the verification score multiplies by the overlap with the claim.
    A `BOX: NONE` with high confidence is therefore a confident rejection, exactly as a
    `None` estimate from ZoomTrace is.
    """

    revision = "1"

    def __init__(self, model_id: str, presentation: str = "locate_grid",
                 snap_to_query_side: bool = True, **kw) -> None:
        if presentation not in CLAIM_FREE:
            raise ValueError(
                f"presentation {presentation!r} shows the claim, so a Verifier cannot "
                f"use it: localise must be claim-independent and the run harness "
                f"audits it. Claim-free presentations: {sorted(CLAIM_FREE)}. For the "
                "claim-shown diagnostics use pyrafault.verify.vlm_probe."
            )
        super().__init__(model_id, **kw)
        self.presentation = presentation
        self.snap_to_query_side = snap_to_query_side
        self.name = f"vlmloc_{_slug(model_id)}_{presentation}"
        self.n_unparsed = 0
        self.n_default_conf = 0
        self.n_said_none = 0
        #: A zero-area or inverted box: the model answered, meaninglessly. Counted apart
        #: from `n_unparsed`, which is the parser failing to read a real answer.
        self.n_degenerate = 0

    def localise(self, inputs: Mapping[str, Any]) -> Localisation:
        ctx, qry = inputs["context"], inputs["query"]
        ctx_box = inputs["context_bbox"]
        shown = render(self.presentation, ctx, qry, None, ctx_box,
                       inputs["context_mpp"], inputs["query_mpp"])
        reply = parse_box_reply(self._ask(shown.images, shown.prompt))

        if reply.box is None and not reply.said_none:
            self.n_degenerate += int(reply.degenerate)
            self.n_unparsed += int(not reply.degenerate)
            return Localisation(estimated_bbox=None, reliability=0.0,
                                diagnostics={"degenerate": reply.degenerate})
        self.n_default_conf += int(reply.confidence is None)
        rho = DEFAULT_CONF if reply.confidence is None else reply.confidence
        if reply.said_none:
            self.n_said_none += 1
            return Localisation(estimated_bbox=None, reliability=rho,
                                diagnostics={"said_none": True, "raw_box": None})

        # The model answered in the frame it was shown, which `_fit` may have shrunk.
        k = ctx.width / shown.frame_px
        x0, y0, x1, y1 = reply.box
        box_ctx = BBoxL0(round(x0 * k), round(y0 * k),
                         round((x1 - x0) * k), round((y1 - y0) * k))
        side_ctx = expected_side_px(qry, inputs["context_mpp"], inputs["query_mpp"])
        if self.snap_to_query_side:
            # Keep the centre, take the side from the magnification ratio. Not a
            # correction of the model's answer but the same footing every other method
            # stands on: ZoomTrace and NCC search fixed-size windows, so their estimate
            # is exactly one query side wide by construction and never loses IoU to a
            # mis-sized box. The patch's size is a constant of the task, stated in the
            # prompt, and carries no information about where it came from. The raw reply
            # is kept in diagnostics so the un-snapped version can be re-scored without
            # another run.
            cx, cy = box_ctx.x + box_ctx.w / 2.0, box_ctx.y + box_ctx.h / 2.0
            box_ctx = BBoxL0(round(cx - side_ctx / 2.0), round(cy - side_ctx / 2.0),
                             side_ctx, side_ctx)
        return Localisation(
            estimated_bbox=context_to_level0(box_ctx, ctx_box, ctx.width),
            reliability=rho,
            diagnostics={"said_none": False, "raw_box": reply.box,
                         "frame_px": shown.frame_px, "expected_side_px": side_ctx},
        )
