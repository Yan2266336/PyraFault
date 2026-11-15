"""Provenance fault construction (guideline §9.1–§9.3).

A fault changes only the *claimed* location. The query pixels and their true source
bbox are never touched — that is what makes one query reusable across four pairs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from pyrafault.geom import BBoxL0, center_distance, iou


class FaultError(AssertionError):
    """A constructed pair violates the benchmark's definition."""


@dataclass(frozen=True)
class FaultResult:
    claimed: BBoxL0
    fault_type: str
    severity: str | None
    normalized_offset: float
    offset_frac_context: float
    iou_claim_true: float


def _measure(claimed: BBoxL0, true: BBoxL0, context: BBoxL0) -> tuple[float, float, float]:
    d = center_distance(claimed, true)
    return d / true.w, d / context.w, iou(claimed, true)


def assign_severities(
    query_ids: Sequence[str], bands: Sequence[str], seed: int
) -> dict[str, str]:
    """Deal queries round-robin into severity bands over a seeded permutation.

    Round-robin rather than independent sampling so the three bands are balanced to
    within one record without any rejection loop.
    """
    ordered = sorted(query_ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ordered))
    return {ordered[int(p)]: bands[i % len(bands)] for i, p in enumerate(perm)}


def make_clean(true_bbox: BBoxL0) -> FaultResult:
    return FaultResult(
        claimed=true_bbox, fault_type="clean", severity=None,
        normalized_offset=0.0, offset_frac_context=0.0, iou_claim_true=1.0,
    )


def make_drift(
    true_bbox: BBoxL0, context: BBoxL0, severity: str, cfg, rng: np.random.Generator
) -> FaultResult:
    """Displace the claim by a distance drawn from the severity band, staying inside C.

    The direction is resampled rather than the distance, so the recorded
    `normalized_offset` stays faithful to the band. Only if every direction fails does the
    radius shrink toward the band's lower edge.
    """
    lo, hi = cfg.severity_bands[severity]
    side = true_bbox.w

    for _ in range(cfg.max_theta_retries):
        r = float(rng.uniform(lo, hi)) * side
        theta = float(rng.uniform(0.0, 2.0 * math.pi))
        cand = BBoxL0(
            int(round(true_bbox.x + r * math.cos(theta))),
            int(round(true_bbox.y + r * math.sin(theta))),
            side, side,
        )
        if context.contains(cand) and cand != true_bbox:
            n, f, i = _measure(cand, true_bbox, context)
            if lo <= n < hi:
                return FaultResult(cand, "adjacent_drift", severity, n, f, i)

    # Fall back to the direction with the most room, at the smallest in-band radius.
    room = {
        0.0: context.x1 - true_bbox.x1,
        math.pi: true_bbox.x - context.x,
        math.pi / 2: context.y1 - true_bbox.y1,
        3 * math.pi / 2: true_bbox.y - context.y,
    }
    theta = max(room, key=room.get)
    r = lo * side
    while r < hi * side:
        cand = BBoxL0(
            int(round(true_bbox.x + r * math.cos(theta))),
            int(round(true_bbox.y + r * math.sin(theta))),
            side, side,
        )
        if context.contains(cand) and cand != true_bbox:
            n, f, i = _measure(cand, true_bbox, context)
            if lo <= n < hi:
                return FaultResult(cand, "adjacent_drift", severity, n, f, i)
        r += 0.05 * side

    raise FaultError(
        f"no {severity} drift fits inside the context for true bbox {true_bbox} "
        f"in context {context}; band=[{lo}, {hi}) query widths"
    )


def make_random(
    true_bbox: BBoxL0,
    context: BBoxL0,
    candidates: Sequence[BBoxL0],
    cfg,
    rng: np.random.Generator,
) -> FaultResult | None:
    """Claim a different eligible region of the same context, chosen uniformly."""
    side = true_bbox.w
    pool = [
        c for c in candidates
        if context.contains(c)
        and iou(c, true_bbox) <= cfg.random_max_iou
        and center_distance(c, true_bbox) >= cfg.random_min_center_dist * side
    ]
    if not pool:
        return None
    pick = pool[int(rng.integers(len(pool)))]
    n, f, i = _measure(pick, true_bbox, context)
    return FaultResult(pick, "random_same_context", None, n, f, i)


#: Every `fault_type` the pipeline can emit. Single source of truth -- `qc.report` and
#: the tests import this rather than each keeping a hand-maintained copy that can drift
#: apart silently (they did, before `out_of_context` was added).
FAULT_TYPES = (
    "clean",
    "adjacent_drift",
    "random_same_context",
    "hard_mismatch",
    "out_of_context",
    "out_of_context_hard",
    "out_of_slide",
    "out_of_slide_hard",
)

#: Types whose true source lies outside the parent context but on the same slide
#: (proposal table 3 rows 4-5).
OUT_OF_CONTEXT_TYPES = ("out_of_context", "out_of_context_hard")

#: Types whose true source is on a DIFFERENT slide. `true_slide_id != slide_id` for
#: exactly these, and containment against the parent context is not merely false but
#: meaningless -- the boxes live in another slide's coordinate frame.
OUT_OF_SLIDE_TYPES = ("out_of_slide", "out_of_slide_hard")

#: Everything whose patch did not come from inside the parent context. These are the
#: types with `true_within_context = False` and `local_recovery_possible = False`.
SUBSTITUTED_TYPES = OUT_OF_CONTEXT_TYPES + OUT_OF_SLIDE_TYPES


def assert_pair_invariants(
    claimed: BBoxL0, true: BBoxL0, context: BBoxL0, fault_type: str
) -> None:
    """Guideline §13.1. Any violation fails the stage rather than writing a bad record."""
    if not context.contains(claimed):
        raise FaultError(f"claimed bbox {claimed} is not inside context {context}")
    # `claimed` is inside `C` for every fault type without exception -- it is the agent's
    # own zoom request. `true` is inside `C` for every type EXCEPT the out-of-context
    # pair, whose whole point is a patch served from elsewhere; for those the assertion
    # inverts rather than lapsing, so "outside" stays as strictly enforced as "inside".
    if fault_type in OUT_OF_SLIDE_TYPES:
        # `true` indexes another slide; comparing it to this slide's context box would
        # be a category error, so the caller asserts `true_slide_id != slide_id` and
        # there is nothing geometric to check here.
        pass
    elif fault_type in OUT_OF_CONTEXT_TYPES:
        if context.contains(true):
            raise FaultError(
                f"{fault_type} pair must have its true bbox outside the context, "
                f"but {true} is inside {context}"
            )
    elif not context.contains(true):
        raise FaultError(f"true bbox {true} is not inside context {context}")
    if fault_type == "clean":
        if claimed != true:
            raise FaultError(f"clean pair must claim the true bbox: {claimed} != {true}")
    elif claimed == true:
        raise FaultError(f"{fault_type} pair has identical claimed and true bbox {claimed}")


def add_provenance_columns(pairs, cfg):
    """Fill the record-level provenance columns every pair carries, for any builder.

    Shared by stages 06 and 07 rather than written out in each: these values must be
    identical across fault types, and the schema can only check that a column exists,
    not that two stages agreed on what to put in it.

    - `true_slide_id` defaults to `slide_id`; only an out-of-slide builder sets it.
    - `same_slide` is derived from that, then cross-checked against `fault_type`, so a
      row that forgot to set `true_slide_id` fails here instead of silently claiming to
      be same-slide.
    - `context_mpp`/`query_mpp`/`scale_ratio` record the magnification combination, so
      several freezes can be concatenated and stratified. Ratio alone is not a key:
      5x->20x and 10x->40x share ratio 4.0.
    """
    import pandas as pd

    pairs = pairs.copy()
    if "true_slide_id" not in pairs.columns:
        pairs["true_slide_id"] = pd.NA
    pairs["true_slide_id"] = pairs["true_slide_id"].fillna(pairs["slide_id"])
    pairs["same_slide"] = pairs["true_slide_id"] == pairs["slide_id"]

    expected_cross = pairs["fault_type"].isin(OUT_OF_SLIDE_TYPES)
    bad = pairs.loc[expected_cross == pairs["same_slide"], "pair_id"]
    if len(bad):
        raise FaultError(
            f"{len(bad)} pair(s) whose same_slide disagrees with fault_type, e.g. "
            f"{sorted(bad)[:5]}"
        )

    pairs["context_mpp"] = float(cfg.context.target_mpp)
    pairs["query_mpp"] = float(cfg.query.target_mpp)
    pairs["scale_ratio"] = float(cfg.context.target_mpp) / float(cfg.query.target_mpp)
    return pairs
