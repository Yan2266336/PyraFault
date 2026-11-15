import collections

import numpy as np
import pytest

from pyrafault.config import FaultsCfg
from pyrafault.faults import (
    FaultError, assert_pair_invariants, assign_severities,
    make_clean, make_drift, make_random,
)
from pyrafault.geom import BBoxL0, center_distance, iou

CTX_FOV, Q_FOV = 16254, 1016
CTX = BBoxL0(1000, 2000, CTX_FOV, CTX_FOV)
BANDS = {"small": (0.25, 0.75), "medium": (0.75, 2.0), "large": (2.0, 6.0)}

CFG = FaultsCfg(
    clean=True, drift_enabled=True, drift_normalize_by="query_side",
    severity_bands=BANDS, drift_balance="stratified", max_theta_retries=32,
    random_enabled=True, random_max_iou=0.0, random_min_center_dist=1.5,
    hard_enabled=True, hard_pool="context", hard_max_iou=0.0,
    hard_min_center_dist=1.5, hard_top_k=1, out_of_context_enabled=False,
)


def _centre_true():
    """A query near the middle of the context, where every band is reachable."""
    return BBoxL0(CTX.x + 7 * Q_FOV, CTX.y + 7 * Q_FOV, Q_FOV, Q_FOV)


def _edge_true():
    """A query on the inner ring — the hardest case for a large drift."""
    return BBoxL0(CTX.x + Q_FOV, CTX.y + Q_FOV, Q_FOV, Q_FOV)


def test_clean_claims_exactly_the_true_box():
    t = _centre_true()
    r = make_clean(t)
    assert r.claimed == t
    assert r.fault_type == "clean" and r.severity is None
    assert r.normalized_offset == 0.0 and r.iou_claim_true == 1.0


def test_severity_assignment_is_balanced_and_deterministic():
    ids = [f"q{i:04d}" for i in range(935)]
    a = assign_severities(ids, list(BANDS), seed=20260812)
    b = assign_severities(ids, list(BANDS), seed=20260812)
    assert a == b
    counts = collections.Counter(a.values())
    assert set(counts) == set(BANDS)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_severity_assignment_changes_with_seed():
    ids = [f"q{i:04d}" for i in range(99)]
    assert assign_severities(ids, list(BANDS), 1) != assign_severities(ids, list(BANDS), 2)


@pytest.mark.parametrize("severity", ["small", "medium", "large"])
def test_drift_lands_inside_its_band_and_inside_the_context(severity):
    t = _centre_true()
    rng = np.random.default_rng(0)
    for _ in range(50):
        r = make_drift(t, CTX, severity, CFG, rng)
        lo, hi = BANDS[severity]
        assert lo <= r.normalized_offset < hi, f"{r.normalized_offset} outside [{lo},{hi})"
        assert CTX.contains(r.claimed)
        assert r.fault_type == "adjacent_drift" and r.severity == severity
        assert r.claimed != t


def test_drift_from_the_inner_ring_still_converges():
    """A 6-query-width displacement must remain reachable in at least one direction."""
    t = _edge_true()
    rng = np.random.default_rng(1)
    for _ in range(50):
        r = make_drift(t, CTX, "large", CFG, rng)
        assert CTX.contains(r.claimed)
        assert 2.0 <= r.normalized_offset < 6.0


def test_small_drift_still_overlaps_the_true_box():
    t = _centre_true()
    rng = np.random.default_rng(2)
    overlaps = [make_drift(t, CTX, "small", CFG, rng).iou_claim_true for _ in range(30)]
    assert min(overlaps) > 0.0, "a sub-one-width offset must leave the boxes overlapping"


def test_large_drift_never_overlaps():
    t = _centre_true()
    rng = np.random.default_rng(3)
    assert all(make_drift(t, CTX, "large", CFG, rng).iou_claim_true == 0.0 for _ in range(30))


def test_offset_normalization_uses_the_query_side():
    t = _centre_true()
    rng = np.random.default_rng(4)
    r = make_drift(t, CTX, "medium", CFG, rng)
    d = center_distance(r.claimed, t)
    assert r.normalized_offset == pytest.approx(d / Q_FOV, rel=1e-6)
    assert r.offset_frac_context == pytest.approx(d / CTX_FOV, rel=1e-6)


def test_drift_is_reproducible_from_the_same_rng_state():
    t = _centre_true()
    a = make_drift(t, CTX, "medium", CFG, np.random.default_rng(9))
    b = make_drift(t, CTX, "medium", CFG, np.random.default_rng(9))
    assert a == b


def _pool(n=40):
    return [BBoxL0(CTX.x + (i % 10 + 2) * Q_FOV, CTX.y + (i // 10 + 2) * Q_FOV, Q_FOV, Q_FOV)
            for i in range(n)]


def test_random_picks_a_disjoint_candidate():
    t = _pool()[0]
    r = make_random(t, CTX, _pool(), CFG, np.random.default_rng(0))
    assert r is not None
    assert r.fault_type == "random_same_context"
    assert iou(r.claimed, t) == 0.0
    assert center_distance(r.claimed, t) >= 1.5 * Q_FOV
    assert CTX.contains(r.claimed)


def test_random_returns_none_when_no_candidate_qualifies():
    t = _pool()[0]
    assert make_random(t, CTX, [t], CFG, np.random.default_rng(0)) is None


def test_random_is_deterministic_for_a_given_rng():
    t = _pool()[0]
    a = make_random(t, CTX, _pool(), CFG, np.random.default_rng(5))
    b = make_random(t, CTX, _pool(), CFG, np.random.default_rng(5))
    assert a == b


def test_invariants_accept_a_valid_clean_pair():
    t = _centre_true()
    assert_pair_invariants(t, t, CTX, "clean")


def test_invariants_reject_a_clean_pair_whose_boxes_differ():
    t = _centre_true()
    other = BBoxL0(t.x + Q_FOV, t.y, Q_FOV, Q_FOV)
    with pytest.raises(FaultError, match="clean"):
        assert_pair_invariants(other, t, CTX, "clean")


def test_invariants_reject_a_fault_pair_whose_boxes_coincide():
    t = _centre_true()
    with pytest.raises(FaultError, match="identical"):
        assert_pair_invariants(t, t, CTX, "adjacent_drift")


def test_invariants_reject_a_claim_outside_the_context():
    t = _centre_true()
    outside = BBoxL0(CTX.x1 + 10, CTX.y, Q_FOV, Q_FOV)
    with pytest.raises(FaultError, match="claimed"):
        assert_pair_invariants(outside, t, CTX, "random_same_context")


def test_invariants_reject_a_true_box_outside_the_context():
    outside = BBoxL0(CTX.x1 + 10, CTX.y, Q_FOV, Q_FOV)
    inside = _centre_true()
    with pytest.raises(FaultError, match="true"):
        assert_pair_invariants(inside, outside, CTX, "random_same_context")
