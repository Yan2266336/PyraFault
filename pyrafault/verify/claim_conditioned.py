"""The claim-conditioned baseline: score `Q` against the crop at the claimed location.

A reviewer's question, and a fair one: the context `C` is trusted, so why localise at
all? Downsample `Q` to the context's physical resolution, crop `C` at the claimed box
`l`, match the two, and report the inlier fraction. One match per item instead of the
ten ZoomTrace's retrieval leaves it, no window bank, no refinement, no decoys.

This baseline exists to measure what the independence buys, and it answers the question
in two parts.

The first part is structural, and holds before any number is measured. The method has no
estimate of where `Q` came from -- it only scores a claim -- so the faithful way to put it
in the harness's shape is `estimated_bbox = claimed_bbox` with the match score as
`reliability`. `decide` then has `d = 0` for every item, always below `delta *
query_side`, so the method can return `verified` or `unverifiable` and **never
`mismatched`**. Its conditional balanced accuracy is therefore exactly 0.5 by
construction, on any data, at any threshold. A verifier that cannot localise
independently cannot name a wrong claim as wrong; it can only decline to confirm it.
That is the cost of claim-conditioning, and it is not an empirical finding about this
benchmark.

(The alternative mapping -- return `estimated_bbox = None` when the score is low, so that
`decide` yields `mismatched` -- is not taken. It needs a second cut, on the same score
`tau` is already fitted against, and the harness fits one threshold per method. Reporting
a method under a free parameter the protocol does not fit would make its row
incomparable with every other row in the table.)

The second part is the measured one: `verification_score` is `rho * IoU(estimated,
claimed)`, and with `estimated == claimed` the IoU is exactly 1, so the ranking metrics
read the raw match score. AUPRC here is precisely "how well does the LoFTR inlier
fraction at the claimed box separate correct claims from incorrect ones", which is the
reviewer's question with nothing else mixed in -- same matcher as ZoomTrace, same
scale alignment, same decision rule, the single difference being where the patch
compared against `Q` comes from.

`reads_claim = True`, which is what keeps the harness honest about it: see
`Verifier.reads_claim`, without which two items differing only in their claim would
share one cached answer.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from pyrafault.verify.base import Localisation, Verifier
from pyrafault.verify.matchers import Matcher
from pyrafault.verify.zoomtrace import downsample_query, normalise_colour


def query_side_in_context(inputs: Mapping) -> int:
    """How many context pixels `Q` occupies, from the two resolutions alone.

    The same quantity `bank.window_side` derives for ZoomTrace's windows, recomputed
    here from `inputs` because this method has no bank to read it off. Deriving it the
    same way is what makes the crop directly comparable to a ZoomTrace candidate.
    """
    return round(inputs["query"].size[0] * inputs["query_mpp"] / inputs["context_mpp"])


class ClaimConditionedBaseline(Verifier):
    """One match, at the claimed location, scored exactly as ZoomTrace scores a window."""

    name = "claim_conditioned"
    revision = "1"
    reads_claim = True

    def __init__(self, matcher: Matcher, *, use_colour_normalisation: bool = False,
                 scale_alignment: bool = True) -> None:
        self._matcher = matcher
        self._use_colour = use_colour_normalisation
        self._scale_alignment = scale_alignment

    def localise(self, inputs: Mapping) -> Localisation:
        context, claimed = inputs["context"], inputs["claimed_bbox"]
        ctx_bbox = inputs["context_bbox"]
        side = query_side_in_context(inputs)

        # L0 back to the context image's own pixels -- the inverse of
        # `zoomtrace.context_to_level0`, which the estimate does not need here because
        # the estimate *is* the claim, already in L0.
        scale = ctx_bbox.w / context.width
        x = int(round((claimed.x - ctx_bbox.x) / scale))
        y = int(round((claimed.y - ctx_bbox.y) / scale))
        w = int(round(claimed.w / scale))

        if abs(w - side) > 1:
            raise ValueError(
                f"{self.name}: the claimed box is {w} context px wide but a query covers "
                f"{side}; the claim is not a query-sized box and there is nothing to crop"
            )
        if not (0 <= x <= context.width - side and 0 <= y <= context.height - side):
            # Measured on the shipped five-cohort test split: 0 of 503,320 items, across
            # all eight fault conditions -- the claim is always a box inside the context,
            # because a claim outside it would be refutable without looking at pixels.
            # Raising rather than clipping: a freeze where this happens has changed what a
            # claim is, and the right crop is then its author's call, not this file's.
            raise ValueError(
                f"{self.name}: claimed box at ({x}, {y}) is not inside the "
                f"{context.width}x{context.height} context; see this method's docstring"
            )

        q_img = downsample_query(inputs["query"], side) if self._scale_alignment else inputs["query"]
        crop = context.crop((x, y, x + side, y + side))
        if self._use_colour:
            q_img, crop = normalise_colour(q_img), normalise_colour(crop)
        res = self._matcher.match(
            np.asarray(q_img.convert("L"), dtype=np.uint8),
            np.asarray(crop.convert("L"), dtype=np.uint8),
        )

        return Localisation(
            # The claim itself, not a refined position. `refine_bbox` would move the box
            # off the claim by the correspondences' translation, which is a localisation
            # -- the one thing this baseline is defined not to do. Keeping the box at the
            # claim is what makes the IoU exactly 1 and the ranking metrics read the raw
            # score.
            estimated_bbox=claimed,
            reliability=float(res.score),
            diagnostics={
                "s_claim": float(res.score),
                "n_matches": int(res.n_matches),
                "n_inliers": int(res.n_inliers),
                "crop_origin": (x, y),
                "side": int(side),
                "scale_alignment": self._scale_alignment,
                # Recorded so the structural claim in the docstring is checkable from the
                # parquet alone rather than only from the source.
                "estimate_is_claim": True,
            },
        )
