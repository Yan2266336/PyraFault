"""Hard-negative mining (guideline §9.4, spec decision D2).

The claimed location is the visually most similar *other* region of the same context.
Restricting the pool to the parent context is deliberate: a claim outside `C` could be
rejected by a bbox containment check alone, which would make the hardest fault type the
easiest one to detect.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from pyrafault.geom import BBoxL0, center_distance, iou


@dataclass(frozen=True)
class HardMatch:
    index: int
    bbox: BBoxL0
    similarity: float
    rank: int


def mine_hard(
    query_emb: np.ndarray,
    pool_emb: np.ndarray,
    pool_boxes: Sequence[BBoxL0],
    true_bbox: BBoxL0,
    context: BBoxL0,
    cfg,
    top_k: int = 1,
) -> list[HardMatch]:
    """Top-`k` eligible candidates ranked by cosine similarity to the query."""
    side = true_bbox.w
    eligible = [
        i for i, b in enumerate(pool_boxes)
        if context.contains(b)
        and b != true_bbox
        and iou(b, true_bbox) <= cfg.hard_max_iou
        and center_distance(b, true_bbox) >= cfg.hard_min_center_dist * side
    ]
    if not eligible:
        return []

    sims = pool_emb[eligible] @ np.asarray(query_emb, dtype=np.float32)
    # Sort by descending similarity; ties broken by pool index so the result is stable.
    order = sorted(range(len(eligible)), key=lambda j: (-float(sims[j]), eligible[j]))

    return [
        HardMatch(
            index=eligible[j],
            bbox=pool_boxes[eligible[j]],
            similarity=float(sims[j]),
            rank=rank + 1,
        )
        for rank, j in enumerate(order[:top_k])
    ]


@dataclass(frozen=True)
class OutOfContextPick:
    index: int
    bbox: BBoxL0
    similarity: float
    policy: str


def _outside(box: BBoxL0, context: BBoxL0, min_gap_px: float) -> bool:
    """`box` shares no area with `context`, and clears it by `min_gap_px`.

    Overlap is tested geometrically rather than by comparing `context_id`s: contexts
    are sampled on a 0.25-stride grid and 914 of the pilot's 2196 same-slide context
    pairs overlap, so "drawn from another context" does not imply "outside this one".
    """
    gap_x = max(context.x - (box.x + box.w), box.x - (context.x + context.w))
    gap_y = max(context.y - (box.y + box.h), box.y - (context.y + context.h))
    return max(gap_x, gap_y) >= min_gap_px


def select_out_of_context(
    query_emb: np.ndarray,
    pool_emb: np.ndarray,
    pool_boxes: Sequence[BBoxL0],
    context: BBoxL0,
    policy: str,
    min_gap_px: float,
    rng: np.random.Generator | None = None,
) -> OutOfContextPick | None:
    """Choose the true source of a patch that did NOT come from the parent context.

    The claim is handled by the caller and always stays inside `context`; this picks
    only `ℓ*`, which must lie wholly outside it. `policy` is the difference between the
    proposal's two out-of-context rows:

    - `"random"` — a uniformly drawn eligible candidate. The fault is a parent-level
      provenance failure with no visual conspiracy.
    - `"hard"` — the candidate whose frozen embedding is closest to the query's own, i.e.
      the tissue elsewhere on the slide that looks most like what the agent asked for.
      This is same-slide hard substitution: rejecting it requires deciding that similar
      tissue is not the *same* tissue.

    Returns `None` when no candidate on the slide clears the geometry, which is a real
    outcome for a slide whose contexts blanket its tissue -- the caller records the query
    as unpaired rather than relaxing the constraint.
    """
    eligible = [i for i, b in enumerate(pool_boxes) if _outside(b, context, min_gap_px)]
    if not eligible:
        return None

    sims = pool_emb[eligible] @ np.asarray(query_emb, dtype=np.float32)
    if policy == "hard":
        # Ties broken by pool index, matching mine_hard, so the pick is reproducible.
        j = min(range(len(eligible)), key=lambda k: (-float(sims[k]), eligible[k]))
    elif policy == "random":
        if rng is None:
            raise ValueError("policy='random' needs a seeded rng")
        j = int(rng.integers(len(eligible)))
    else:
        raise ValueError(f"unknown out-of-context policy {policy!r}")

    return OutOfContextPick(
        index=eligible[j], bbox=pool_boxes[eligible[j]],
        similarity=float(sims[j]), policy=policy,
    )


@dataclass(frozen=True)
class OutOfSlidePick:
    slide_id: str
    bbox: BBoxL0
    similarity: float
    policy: str


def select_out_of_slide(
    query_emb: np.ndarray,
    pools: dict[str, tuple[np.ndarray, Sequence[BBoxL0]]],
    exclude_slide: str,
    policy: str,
    rng: np.random.Generator | None = None,
) -> OutOfSlidePick | None:
    """Choose the true source of a patch served from a DIFFERENT slide.

    `pools` maps `slide_id -> (embeddings, boxes)` and must already be restricted to
    slides the caller considers admissible. That restriction is the caller's job because
    it is a *split* question, not a geometry one: drawing a substitute from another
    split would put dev pixels inside a test item, undoing the case-level split the
    benchmark is built around. This function only refuses to draw from `exclude_slide`.

    `policy` mirrors `select_out_of_context`: `random` is the realistic cache-mix
    failure, `hard` picks the most similar tissue on any other slide. Note that
    out-of-slide faults may carry a cue no same-slide fault does -- staining and scanner
    differences between slides -- so a verifier may reject them on colour statistics
    without any spatial reasoning. That is a property of the fault, not a defect in the
    construction, but it means out-of-slide accuracy should not be pooled with
    same-slide accuracy without saying so.

    Returns `None` when no admissible slide other than `exclude_slide` has candidates --
    a one-slide split, for instance.
    """
    admissible = [s for s in sorted(pools) if s != exclude_slide and len(pools[s][1])]
    if not admissible:
        return None

    if policy == "random":
        if rng is None:
            raise ValueError("policy='random' needs a seeded rng")
        slide_id = admissible[int(rng.integers(len(admissible)))]
        emb, boxes = pools[slide_id]
        i = int(rng.integers(len(boxes)))
        sim = float(emb[i] @ np.asarray(query_emb, dtype=np.float32))
        return OutOfSlidePick(slide_id, boxes[i], sim, policy)

    if policy == "hard":
        best = None
        for slide_id in admissible:
            emb, boxes = pools[slide_id]
            sims = emb @ np.asarray(query_emb, dtype=np.float32)
            i = int(np.argmax(sims))
            if best is None or float(sims[i]) > best[2]:
                best = (slide_id, boxes[i], float(sims[i]))
        return OutOfSlidePick(best[0], best[1], best[2], policy)

    raise ValueError(f"unknown out-of-slide policy {policy!r}")
