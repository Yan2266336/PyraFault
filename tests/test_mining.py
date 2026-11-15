import numpy as np
import pytest

from pyrafault.config import FaultsCfg
from pyrafault.geom import BBoxL0, center_distance, iou
from pyrafault.mining import HardMatch, mine_hard

Q_FOV = 1016
CTX = BBoxL0(0, 0, 16254, 16254)
CFG = FaultsCfg(
    clean=True, drift_enabled=True, drift_normalize_by="query_side",
    severity_bands={"small": (0.25, 0.75), "medium": (0.75, 2.0), "large": (2.0, 6.0)},
    drift_balance="stratified", max_theta_retries=32,
    random_enabled=True, random_max_iou=0.0, random_min_center_dist=1.5,
    hard_enabled=True, hard_pool="context", hard_max_iou=0.0,
    hard_min_center_dist=1.5, hard_top_k=1, out_of_context_enabled=False,
)


def _boxes(n=12):
    return [BBoxL0((i % 6 + 2) * Q_FOV, (i // 6 + 2) * Q_FOV, Q_FOV, Q_FOV) for i in range(n)]


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_returns_the_most_similar_eligible_candidate():
    boxes = _boxes()
    true = boxes[0]
    # Default pool direction is orthogonal to the query, so untouched candidates score
    # 0.0 and cannot tie with the deliberately-crafted near matches below. Note: index 7
    # (a diagonal neighbour of `true`, center-distance ratio sqrt(2) =~ 1.4142) is NOT
    # used here because it falls below hard_min_center_dist=1.5 and would be excluded as
    # ineligible before similarity is ever considered; index 2 (ratio 2.0) is eligible.
    emb = np.zeros((len(boxes), 4), dtype=np.float32)
    emb[:, 1] = 1.0
    emb[2] = _unit([0.99, 0.14, 0, 0])       # most similar to the query, and eligible
    emb[1] = _unit([0.999, 0.04, 0, 0])      # even more similar, but too close to true
    q = _unit([1, 0, 0, 0])

    got = mine_hard(q, emb, boxes, true, CTX, CFG, top_k=1)
    assert len(got) == 1 and isinstance(got[0], HardMatch)
    assert got[0].index == 2 and got[0].rank == 1
    assert got[0].similarity == pytest.approx(float(q @ emb[2]), rel=1e-6)


def test_excludes_the_true_box_itself():
    boxes = _boxes()
    emb = np.tile(_unit([1, 0, 0, 0]), (len(boxes), 1))
    got = mine_hard(emb[0], emb, boxes, boxes[0], CTX, CFG, top_k=1)
    assert got and got[0].bbox != boxes[0]


def test_excludes_overlapping_and_too_close_candidates():
    boxes = _boxes()
    true = boxes[0]
    emb = np.tile(_unit([1, 0, 0, 0]), (len(boxes), 1))
    got = mine_hard(emb[0], emb, boxes, true, CTX, CFG, top_k=5)
    for m in got:
        assert iou(m.bbox, true) <= CFG.hard_max_iou
        assert center_distance(m.bbox, true) >= CFG.hard_min_center_dist * Q_FOV


def test_claimed_location_is_always_inside_the_context():
    boxes = _boxes() + [BBoxL0(CTX.x1 + 100, 0, Q_FOV, Q_FOV)]   # outside C
    emb = np.tile(_unit([1, 0, 0, 0]), (len(boxes), 1))
    emb[-1] = _unit([1, 0, 0, 0])
    got = mine_hard(emb[0], emb, boxes, boxes[0], CTX, CFG, top_k=3)
    assert all(CTX.contains(m.bbox) for m in got)


def test_ranks_are_one_based_and_ordered_by_similarity():
    boxes = _boxes()
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(len(boxes), 8)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    got = mine_hard(emb[0], emb, boxes, boxes[0], CTX, CFG, top_k=4)
    assert [m.rank for m in got] == [1, 2, 3, 4]
    assert all(got[i].similarity >= got[i + 1].similarity for i in range(len(got) - 1))


def test_returns_empty_when_nothing_qualifies():
    true = BBoxL0(0, 0, Q_FOV, Q_FOV)
    emb = np.tile(_unit([1, 0, 0, 0]), (1, 1))
    assert mine_hard(emb[0], emb, [true], true, CTX, CFG, top_k=1) == []


def test_is_deterministic():
    boxes = _boxes()
    rng = np.random.default_rng(3)
    emb = rng.normal(size=(len(boxes), 8)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    a = mine_hard(emb[0], emb, boxes, boxes[0], CTX, CFG, 2)
    b = mine_hard(emb[0], emb, boxes, boxes[0], CTX, CFG, 2)
    assert a == b
