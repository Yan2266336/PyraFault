"""The same MLLM baseline, asked over the Anthropic API instead of local weights.

A reviewer's objection to the five open models in SS6.5 is that none of them is the model
anyone would actually reach for. This answers it without changing the question: the
presentation (`vlm_presentation.render`), the parser (`parse_box_reply`), the snapping,
the shared `decide` and stage 14 are all inherited unchanged from `VlmLocaliser`, and
the single overridden method is `_ask`. The row is therefore comparable with the open
rows cell for cell, and the only difference between them is which model answered.

Two things are deliberately NOT done here.

It does not load any weights -- `_defer_load=True` skips `_HfVlm._load`, so the class
runs on a CPU node and never touches a GPU. Every observation it makes costs money
instead of GPU time, which is why the item sample is fixed by seed and reused rather
than resampled per model.

It does not turn an API failure into an answer. A transport error read as an unparseable
reply becomes `reliability=0.0` -- a confident abstention -- and would pull the row's
AUPRC down for reasons that have nothing to do with the model's perception. Errors are
retried, and an item that still fails raises, because a silently dropped row is the one
failure mode that cannot be detected after the fact.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import random
import threading
import time
from pathlib import Path

from pyrafault.verify.vlm_baseline import VlmLocaliser, _slug

#: Per-image cap. Anthropic downscales anything over 1568px on the long edge, which
#: would silently put the model in a different coordinate frame from the one the prompt
#: states (`localise` scales the reply by `ctx.width / shown.frame_px`). The renderer's
#: own `MAX_SIDE` is 1536, so nothing is resized -- this is asserted, not assumed.
_API_MAX_SIDE = 1568

#: Retried statuses. Overload and rate limit are expected on a 4,000-call run; a 400 is
#: a bug in the request and must surface immediately rather than after eight sleeps.
_RETRY_AFTER = 8


class AnthropicVlmLocaliser(VlmLocaliser):
    """`VlmLocaliser` with the generation step served by the Messages API."""

    revision = "1"

    def __init__(self, model_id: str = "claude-haiku-5-5",
                 presentation: str = "locate_grid",
                 max_new_tokens: int = 512, thinking: bool = False,
                 replicate: int = 0,
                 cache: "str | Path | None" = None, **kw) -> None:
        kw.pop("_defer_load", None)
        super().__init__(model_id, presentation=presentation,
                         max_new_tokens=max_new_tokens, _defer_load=True, **kw)
        # Overwritten so a thinking run cannot land its predictions in the same
        # directory as a non-thinking one: the two reach different score distributions,
        # so their fitted thresholds are not interchangeable.
        self.name = f"vlmapi_{_slug(model_id)}_{presentation}" + ("_think" if thinking else "")
        self.thinking = thinking
        #: These models cannot be decoded greedily: `temperature` and `top_p` are both
        #: rejected outright ("`temperature` is deprecated for this model"), so the
        #: sampling the open-weight rows switch off with `do_sample=False` cannot be
        #: switched off here. That is a real and unavoidable difference between the API
        #: rows and the open rows, and it showed up as two runs of the same cell on the
        #: same 2,000 items reporting 0.5289 and 0.5266 AUROC.
        #:
        #: It is handled by measurement rather than by assertion: `replicate` is part of
        #: the cache key, so the same cell can be run K times and the run-to-run spread
        #: reported next to the item-resampling interval. Hiding an uncontrollable
        #: variance component would make every digit in the row overstated.
        self.replicate = replicate
        self.n_api_retry = 0
        self.n_cache_hit = 0
        self.in_tokens = self.out_tokens = 0
        #: Replies on disk, so a run that dies partway does not pay for the same items
        #: twice. The first attempt at this row stopped mid-presentation when the
        #: account's credit ran out and ~600 answered items went with it, which is money
        #: spent for nothing -- a cost no amount of retry logic addresses.
        #:
        #: The key is content-addressed over the model, the reasoning mode, the prompt
        #: and the image bytes. Not the item id: the id would have to be threaded down
        #: through `localise`, which is inherited and shared with the open-weight rows,
        #: and a key that is wrong about which pixels it describes is worse than no
        #: cache. Hashing what was actually sent cannot mismatch.
        self._cache_path = Path(cache) if cache else None
        self._cache: dict[str, str] = {}
        self._cache_lock = threading.Lock()
        if self._cache_path and self._cache_path.exists():
            for line in self._cache_path.read_text().splitlines():
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._cache[d["key"]] = d["text"]
        import anthropic
        # The key is read from the environment by the SDK itself and never passed as an
        # argument, so it cannot reach a process listing or a traceback.
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY not in the environment")
        self._client = anthropic.Anthropic(max_retries=0)
        self._err = (anthropic.APIStatusError, anthropic.APIConnectionError)

    def _block(self, im) -> dict:
        if max(im.size) > _API_MAX_SIDE:
            raise ValueError(
                f"{im.size} exceeds {_API_MAX_SIDE}px, so the API would downscale it "
                "and the coordinates the prompt asks for would be in a frame the model "
                "never saw. Lower vlm_presentation.MAX_SIDE instead of resizing here."
            )
        buf = io.BytesIO()
        # PNG, not JPEG: the open rows are handed the raw PIL image, and compression
        # artefacts at the magnifications this task turns on are not a difference worth
        # introducing to save bandwidth.
        im.convert("RGB").save(buf, format="PNG")
        return {"type": "image", "source": {
            "type": "base64", "media_type": "image/png",
            "data": base64.standard_b64encode(buf.getvalue()).decode()}}

    def _ask(self, images: list, prompt: str) -> str:
        blocks = [self._block(im) for im in images]
        # Every parameter that can change the reply is in the key. `temperature` was
        # not, and that was a correctness bug waiting to happen: a greedy rerun would
        # have been served the sampled replies already on disk and reported them as
        # greedy. A cache that can answer the wrong question is worse than none.
        data = [b["source"]["data"] for b in blocks]
        key = hashlib.sha256(json.dumps(
            [self.model_id, self.thinking, self.max_new_tokens, self.replicate,
             prompt, data], sort_keys=True).encode()).hexdigest()
        # Replies cached before `replicate` entered the key are still valid for
        # replicate 0: they ARE a single unreplicated run of this cell, which is what
        # replicate 0 means. Consulted only for replicate 0, so a later replicate can
        # never be served the first one's reply and collapse the spread to zero -- which
        # would silently turn the variance this field exists to measure into 0.
        legacy = hashlib.sha256(json.dumps(
            [self.model_id, self.thinking, self.max_new_tokens, prompt, data],
            sort_keys=True).encode()).hexdigest() if self.replicate == 0 else None
        if self._cache_path is not None:
            with self._cache_lock:
                hit = self._cache.get(key)
                if hit is None and legacy is not None:
                    hit = self._cache.get(legacy)
            if hit is not None:
                self.n_cache_hit += 1
                return hit
        content = blocks + [{"type": "text", "text": prompt}]
        kw = {}
        if self.thinking:
            # Adaptive, not a token budget: `budget_tokens` is rejected outright by the
            # 5-series models. Reported as a separate diagnostic row, never mixed into
            # the main table, because the open models were asked with reasoning off.
            kw["thinking"] = {"type": "adaptive"}
        for attempt in range(_RETRY_AFTER):
            try:
                r = self._client.messages.create(
                    model=self.model_id, max_tokens=self.max_new_tokens,
                    messages=[{"role": "user", "content": content}], **kw)
            except self._err as e:
                status = getattr(e, "status_code", None)
                if status is not None and status not in (408, 409, 429, 500, 502, 503, 529):
                    raise
                self.n_api_retry += 1
                if attempt == _RETRY_AFTER - 1:
                    raise
                time.sleep(min(60.0, 2.0 ** attempt) * (0.5 + random.random()))
                continue
            self.in_tokens += r.usage.input_tokens
            self.out_tokens += r.usage.output_tokens
            # Thinking blocks are skipped: the parser is the same one the open rows use
            # and must see the same thing, the model's final text.
            text = "\n".join(b.text for b in r.content if b.type == "text")
            if self._cache_path is not None:
                # Appended before returning, under the lock, so the reply survives a
                # crash on the very next item. One line per call, so a truncated final
                # line is dropped on load rather than corrupting the file.
                with self._cache_lock:
                    self._cache[key] = text
                    with self._cache_path.open("a") as fh:
                        fh.write(json.dumps({"key": key, "text": text}) + "\n")
            return text
        raise AssertionError("unreachable")
