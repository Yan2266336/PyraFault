import numpy as np
import pytest

from pyrafault.geom import BBoxL0
from pyrafault.sampling.queries import QueryCandidate, candidate_pool, context_completeness
from pyrafault.tissue import TissueMask

CTX_FOV = 16254
Q_FOV = 1016


class Q:
    target_mpp = 0.5
    size_px = 512
    stride_frac = 1.0
    boundary_margin_frac = 1.0
    min_tissue_fraction = 0.80
    num_per_context = 4


def _mask(w=40000, h=40000, mh=800, mw=800, filled=True):
    return TissueMask(np.ones((mh, mw), bool) if filled else np.zeros((mh, mw), bool), w, h)


SEED = 12345


def test_pool_size_for_the_common_mpp():
    ctx = BBoxL0(1000, 2000, CTX_FOV, CTX_FOV)
    pool = candidate_pool("C0", "S", ctx, 0.2520, _mask(), Q, SEED)
    # 16254 // 1016 == 15, remainder 1014 < 1016 so n decrements to 14 (spec 15.1);
    # a 14x14 grid minus one ring is 12x12 == 144.
    assert len(pool) == 144, "14x14 grid minus one ring, after the n decrement"


def test_pool_size_for_the_other_mpp():
    ctx = BBoxL0(0, 0, 16209, 16209)
    pool = candidate_pool("C0", "S", ctx, 0.2527, _mask(), Q, SEED)
    # 16209 // 1013 == 16, remainder 1 < 1013 so n decrements to 15; 15x15 minus one
    # ring is 13x13 == 169.
    assert len(pool) == 169, "15x15 grid minus one ring, after the n decrement"


def test_every_candidate_is_fully_inside_the_parent_context():
    ctx = BBoxL0(1000, 2000, CTX_FOV, CTX_FOV)
    for c in candidate_pool("C0", "S", ctx, 0.2520, _mask(), Q, SEED):
        assert ctx.contains(c.bbox)


def test_candidates_do_not_touch_the_context_boundary():
    """One margin ring keeps every candidate at least one full FOV away from the true
    context boundary -- this bound holds regardless of the per-context jitter offset
    (see test_subgrid_jittered_candidates_stay_inside_parent_and_do_not_overlap), unlike
    the exact centred positions the old version of this test hard-coded."""
    ctx = BBoxL0(0, 0, CTX_FOV, CTX_FOV)
    pool = candidate_pool("C0", "S", ctx, 0.2520, _mask(), Q, SEED)
    xs = [c.bbox.x for c in pool]
    ys = [c.bbox.y for c in pool]
    assert min(xs) >= Q_FOV and min(ys) >= Q_FOV
    assert max(xs) + Q_FOV <= CTX_FOV - Q_FOV
    assert max(ys) + Q_FOV <= CTX_FOV - Q_FOV


def test_candidates_do_not_overlap_each_other():
    from pyrafault.geom import iou
    ctx = BBoxL0(0, 0, CTX_FOV, CTX_FOV)
    pool = candidate_pool("C0", "S", ctx, 0.2520, _mask(), Q, SEED)
    for a in range(0, 30):
        for b in range(a + 1, 30):
            assert iou(pool[a].bbox, pool[b].bbox) == 0.0


def test_ids_are_unique_and_namespaced_by_context():
    ctx = BBoxL0(0, 0, CTX_FOV, CTX_FOV)
    pool = candidate_pool("S__c0007", "S", ctx, 0.2520, _mask(), Q, SEED)
    assert pool[0].candidate_id == "S__c0007__qc0000"
    assert len({c.candidate_id for c in pool}) == len(pool)
    assert all(c.context_id == "S__c0007" for c in pool)


def test_low_tissue_candidates_are_dropped():
    ctx = BBoxL0(0, 0, CTX_FOV, CTX_FOV)
    assert candidate_pool("C0", "S", ctx, 0.2520, _mask(filled=False), Q, SEED) == []


def test_candidate_bbox_is_the_true_source_box():
    ctx = BBoxL0(500, 700, CTX_FOV, CTX_FOV)
    c = candidate_pool("C0", "S", ctx, 0.2520, _mask(), Q, SEED)[0]
    assert isinstance(c, QueryCandidate)
    assert c.bbox.w == Q_FOV and c.bbox.h == Q_FOV
    assert c.bbox.x >= ctx.x and c.bbox.y >= ctx.y


def test_pool_is_jittered_per_context_not_centered():
    """Two contexts at the same offset from their own origin, given different
    context_ids, must land the grid at different absolute positions -- otherwise the
    jitter isn't doing anything and the grid origin is still derivable (spec 15.1)."""
    ctx = BBoxL0(1000, 2000, CTX_FOV, CTX_FOV)
    pool_a = candidate_pool("S__c0000", "S", ctx, 0.2520, _mask(), Q, SEED)
    pool_b = candidate_pool("S__c0001", "S", ctx, 0.2520, _mask(), Q, SEED)
    assert (pool_a[0].bbox.x, pool_a[0].bbox.y) != (pool_b[0].bbox.x, pool_b[0].bbox.y)


def test_pool_is_deterministic_for_the_same_seed_and_context_id():
    ctx = BBoxL0(1000, 2000, CTX_FOV, CTX_FOV)
    pool_a = candidate_pool("S__c0000", "S", ctx, 0.2520, _mask(), Q, SEED)
    pool_b = candidate_pool("S__c0000", "S", ctx, 0.2520, _mask(), Q, SEED)
    assert [(c.bbox.x, c.bbox.y) for c in pool_a] == [(c.bbox.x, c.bbox.y) for c in pool_b]


def test_a_context_with_zero_candidates_is_not_invisible_to_completeness_counting():
    """A context that produced no candidates has no rows to tally, so the completeness
    check must be indexed off the full context list, not off the candidate rows — otherwise
    the exact failure it exists to catch (a context that cannot produce even one query)
    silently disappears instead of being reported."""
    context_ids = ["S__c0000", "S__c0001", "S__c0002"]
    candidate_context_ids = ["S__c0000"] * 6 + ["S__c0002"] * 6  # c0001 produced nothing

    per_ctx = context_completeness(context_ids, candidate_context_ids)

    assert per_ctx == {"S__c0000": 6, "S__c0001": 0, "S__c0002": 6}
    assert set(per_ctx) == set(context_ids), "every known context must appear, even at zero"

    threshold = 5
    thin = {cid: n for cid, n in per_ctx.items() if n < threshold}
    assert thin == {"S__c0001": 0}, "the zero-candidate context must trip the thin-context check"
