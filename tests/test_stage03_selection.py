import pytest

from pyrafault.geom import BBoxL0
from pyrafault.quality import QualityResult
from pyrafault.sampling.contexts import ContextCandidate, select_with_quality


def _cands(n):
    return [
        ContextCandidate(f"S__cc{i:04d}", "S", BBoxL0(i * 5000, 0, 100, 100), 0.9)
        for i in range(n)
    ]


def test_selects_exactly_n_when_everything_passes():
    calls = []

    def ev(i):
        calls.append(i)
        return QualityResult(flags=[], hard_fail=False)

    got = select_with_quality(_cands(20), 5, ev)
    assert len(got) == 5
    assert len(calls) == 5, "must evaluate lazily, not the whole pool"


def test_hard_failures_are_skipped_and_replaced():
    bad = {0, 3}

    def ev(i):
        return QualityResult(flags=["blur"] if i in bad else [], hard_fail=i in bad)

    got = select_with_quality(_cands(20), 4, ev)
    assert len(got) == 4
    assert not (bad & {i for i, _ in got})


def test_returns_fewer_than_n_when_the_pool_runs_out():
    def ev(i):
        return QualityResult(flags=[], hard_fail=i % 2 == 0)

    got = select_with_quality(_cands(6), 5, ev)
    assert len(got) == 3


def test_quality_result_is_returned_alongside_the_index():
    def ev(i):
        return QualityResult(flags=["pen_mark"], blur_var=42.0, hard_fail=False)

    got = select_with_quality(_cands(3), 2, ev)
    assert all(q.blur_var == 42.0 and q.flags == ["pen_mark"] for _, q in got)


def test_empty_pool_is_empty():
    assert select_with_quality([], 5, lambda i: QualityResult()) == []


def _cands_2d(n_side):
    """A 2-D grid pool where raster order and FPS order are distinguishable (C1)."""
    out = []
    i = 0
    for y in range(n_side):
        for x in range(n_side):
            out.append(ContextCandidate(f"S__cc{i:04d}", "S", BBoxL0(x * 1000, y * 1000, 100, 100), 0.9))
            i += 1
    return out


def test_selection_is_not_first_n_in_raster_order():
    """C1 regression: select_with_quality must run genuine spatial FPS, not raster order."""
    cands = _cands_2d(6)  # 36 candidates on a 6x6 grid

    def ev(i):
        return QualityResult(flags=[], hard_fail=False)

    got = select_with_quality(cands, 6, ev)
    picked_idx = [i for i, _ in got]
    raster_first_n = list(range(6))
    assert picked_idx != raster_first_n, (
        "selection must not be the first N candidates in raster (candidate_id) order"
    )
