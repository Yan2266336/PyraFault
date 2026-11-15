"""Query candidate pool inside a frozen context (guideline §7.2).

Each candidate's bbox is the true source location of the query it will become. It is
recorded because it is what gets read, never because anyone labelled it.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from pyrafault.geom import BBoxL0, fov_l0, subgrid
from pyrafault.tissue import TissueMask


@dataclass(frozen=True)
class QueryCandidate:
    candidate_id: str
    context_id: str
    slide_id: str
    bbox: BBoxL0
    tissue_fraction: float


def candidate_pool(
    context_id: str,
    slide_id: str,
    context_bbox: BBoxL0,
    mpp_l0: float,
    mask: TissueMask,
    cfg,
    seed: int,
) -> list[QueryCandidate]:
    """Non-overlapping, boundary-inset, tissue-rich candidate positions inside a context.

    The grid origin is jittered per context (`seed`, `context_id`) rather than centred,
    so it cannot be reconstructed from the context bbox and slide MPP alone -- see spec
    15.1. `subgrid` is used only for this query grid; the context grid uses
    `grid_positions`, which this jitter does not touch.
    """
    fov = fov_l0(cfg.size_px, cfg.target_mpp, mpp_l0)
    rings = int(round(cfg.boundary_margin_frac))
    boxes = subgrid(context_bbox, fov, margin_rings=rings, seed=seed, context_id=context_id)

    out: list[QueryCandidate] = []
    for box in boxes:
        frac = mask.fraction(box)
        if frac < cfg.min_tissue_fraction:
            continue
        out.append(
            QueryCandidate(
                candidate_id=f"{context_id}__qc{len(out):04d}",
                context_id=context_id,
                slide_id=slide_id,
                bbox=box,
                tissue_fraction=frac,
            )
        )
    return out


def context_completeness(
    context_ids: Iterable[str], candidate_context_ids: Iterable[str]
) -> dict[str, int]:
    """Per-context candidate counts, indexed off every known context id — including zero.

    A context that produced zero candidates has no rows among `candidate_context_ids`, so
    tallying only the candidate rows (e.g. `groupby("context_id").size()` over the candidate
    table alone) is blind to exactly the failure this check exists to catch: a context that
    cannot produce even one query, let alone `num_per_context`.
    """
    counts = Counter(candidate_context_ids)
    return {cid: counts.get(cid, 0) for cid in context_ids}
