"""Context candidate pool and spatially spread selection (guideline §7.1)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from pyrafault.geom import BBoxL0, fov_l0, grid_positions
from pyrafault.tissue import TissueMask


@dataclass(frozen=True)
class ContextCandidate:
    candidate_id: str
    slide_id: str
    bbox: BBoxL0
    tissue_fraction: float


def candidate_grid(
    slide_id: str,
    width_l0: int,
    height_l0: int,
    mpp_l0: float,
    mask: TissueMask,
    cfg,
) -> list[ContextCandidate]:
    """All in-bounds grid positions whose tissue fraction clears the threshold."""
    fov = fov_l0(cfg.size_px, cfg.target_mpp, mpp_l0)
    stride = max(1, int(round(fov * cfg.stride_frac)))

    out: list[ContextCandidate] = []
    for box in grid_positions(width_l0, height_l0, fov, stride):
        frac = mask.fraction(box)
        if frac < cfg.min_tissue_fraction:
            continue
        out.append(
            ContextCandidate(
                candidate_id=f"{slide_id}__cc{len(out):04d}",
                slide_id=slide_id,
                bbox=box,
                tissue_fraction=frac,
            )
        )
    return out


def farthest_point_select(candidates: Sequence[ContextCandidate], n: int) -> list[int]:
    """Indices of `n` candidates spread as far apart as the pool allows.

    Deterministic: the seed is the highest tissue fraction with raster order breaking
    ties, and every later pick maximises the minimum distance to the already-chosen set,
    again with raster order breaking ties. No RNG is involved, so the frozen pool does
    not depend on the seed.
    """
    if n <= 0 or not candidates:
        return []
    n = min(n, len(candidates))

    order = sorted(
        range(len(candidates)),
        key=lambda i: (-candidates[i].tissue_fraction,
                       candidates[i].bbox.y, candidates[i].bbox.x),
    )
    chosen = [order[0]]
    chosen_set = {order[0]}
    best = [
        math.hypot(c.bbox.cx - candidates[chosen[0]].bbox.cx,
                   c.bbox.cy - candidates[chosen[0]].bbox.cy)
        for c in candidates
    ]

    while len(chosen) < n:
        pick = max(
            (i for i in range(len(candidates)) if i not in chosen_set),
            key=lambda i: (best[i], -candidates[i].bbox.y, -candidates[i].bbox.x),
        )
        chosen.append(pick)
        chosen_set.add(pick)
        px, py = candidates[pick].bbox.cx, candidates[pick].bbox.cy
        for i, c in enumerate(candidates):
            d = math.hypot(c.bbox.cx - px, c.bbox.cy - py)
            if d < best[i]:
                best[i] = d
    return chosen


def select_with_quality(candidates, n: int, evaluate_fn):
    """Walk the farthest-point ordering, dropping regions that fail image validity.

    `evaluate_fn(index) -> QualityResult` is called lazily so a slide with 320 candidates
    costs ~25 pixel reads, not 320. Returns `[(index, QualityResult), ...]` in selection
    order, possibly shorter than `n` if the pool is exhausted.
    """
    if not candidates or n <= 0:
        return []
    order = farthest_point_select(candidates, len(candidates))
    out = []
    for idx in order:
        if len(out) >= n:
            break
        result = evaluate_fn(idx)
        if result.hard_fail:
            continue
        out.append((idx, result))
    return out
