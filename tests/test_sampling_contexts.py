import numpy as np
import pytest

from pyrafault.geom import BBoxL0, center_distance
from pyrafault.sampling.contexts import (
    ContextCandidate, candidate_grid, farthest_point_select,
)
from pyrafault.tissue import TissueMask


class Ctx:
    target_mpp = 2.0
    size_px = 2048
    stride_frac = 0.25
    min_tissue_fraction = 0.50
    num_per_slide = 25
    selection = "spatial_fps"


def _full_mask(w, h, mh=400, mw=400):
    return TissueMask(np.ones((mh, mw), bool), w, h)


@pytest.mark.parametrize(
    "w,h,expected",
    [(27818, 23967, 6), (35492, 31380, 20), (44125, 27646, 21), (110313, 22108, 48)],
)
def test_candidate_counts_match_spec_d1_table(w, h, expected):
    cands = candidate_grid("S", w, h, 0.2520, _full_mask(w, h), Ctx)
    assert len(cands) == expected


def test_candidates_carry_ids_bboxes_and_fractions():
    cands = candidate_grid("TCGA-55-8094-01Z-00-DX1", 35492, 31380, 0.2520,
                           _full_mask(35492, 31380), Ctx)
    c = cands[0]
    assert isinstance(c, ContextCandidate)
    assert c.candidate_id == "TCGA-55-8094-01Z-00-DX1__cc0000"
    assert c.bbox.w == 16254 and c.bbox.h == 16254
    assert c.tissue_fraction == pytest.approx(1.0)
    assert len({x.candidate_id for x in cands}) == len(cands)


def test_low_tissue_candidates_are_dropped():
    w = h = 40000
    mask = np.zeros((400, 400), bool)
    mask[:200, :200] = True                     # tissue only in the top-left quadrant
    cands = candidate_grid("S", w, h, 0.2520, TissueMask(mask, w, h), Ctx)
    assert cands, "some top-left windows must survive"
    assert all(c.tissue_fraction >= 0.50 for c in cands)
    assert all(c.bbox.x < w // 2 and c.bbox.y < h // 2 for c in cands)


def test_out_of_bounds_positions_never_appear():
    w, h = 35492, 31380
    for c in candidate_grid("S", w, h, 0.2520, _full_mask(w, h), Ctx):
        assert c.bbox.within_bounds(w, h)


def test_slide_too_small_yields_no_candidates():
    assert candidate_grid("S", 10000, 10000, 0.2520, _full_mask(10000, 10000), Ctx) == []


def _cands(points, fractions=None):
    fractions = fractions or [0.9] * len(points)
    return [
        ContextCandidate(f"S__cc{i:04d}", "S", BBoxL0(x, y, 100, 100), f)
        for i, ((x, y), f) in enumerate(zip(points, fractions))
    ]


def test_fps_starts_from_highest_tissue_fraction():
    c = _cands([(0, 0), (1000, 0), (2000, 0)], [0.6, 0.99, 0.7])
    assert farthest_point_select(c, 1) == [1]


def test_fps_spreads_points_out():
    pts = [(0, 0), (10, 0), (20, 0), (5000, 0), (5010, 0), (0, 5000)]
    order = farthest_point_select(_cands(pts), 3)
    chosen = [pts[i] for i in order]
    # the three picks must be mutually far apart, not the three clustered near the origin
    for a in range(3):
        for b in range(a + 1, 3):
            dx = chosen[a][0] - chosen[b][0]
            dy = chosen[a][1] - chosen[b][1]
            assert (dx * dx + dy * dy) ** 0.5 > 1000


def test_fps_is_deterministic():
    pts = [(i * 37 % 900, i * 53 % 700) for i in range(60)]
    c = _cands(pts)
    assert farthest_point_select(c, 12) == farthest_point_select(c, 12)


def test_fps_returns_everything_when_n_exceeds_pool():
    c = _cands([(0, 0), (500, 500)])
    assert sorted(farthest_point_select(c, 25)) == [0, 1]


def test_fps_of_zero_is_empty():
    assert farthest_point_select(_cands([(0, 0)]), 0) == []


def test_fps_spreads_points_out_on_a_2d_grid():
    """A straight-line layout can't distinguish FPS from raster order (C1 regression).

    Lay candidates on a 6x6 grid instead: raster order visits (0,0),(1,0),(2,0)... which
    are adjacent (distance 1 grid step apart). A genuine FPS ordering must instead pick
    corners/far points early, not the first few positions in the raster scan.
    """
    pts = [(x * 1000, y * 1000) for y in range(6) for x in range(6)]
    fractions = [0.9] * len(pts)
    # bump one point's tissue fraction so the seed is deterministic and not (0,0)
    fractions[0] = 0.99
    cands = _cands(pts, fractions)
    order = farthest_point_select(cands, 4)
    chosen = [pts[i] for i in order]

    raster_first_4 = pts[:4]
    assert chosen != raster_first_4, "FPS must not degrade to raster order"

    # the first 4 picks must be mutually far apart (>= half the grid extent)
    for a in range(4):
        for b in range(a + 1, 4):
            dx = chosen[a][0] - chosen[b][0]
            dy = chosen[a][1] - chosen[b][1]
            dist = (dx * dx + dy * dy) ** 0.5
            assert dist >= 2000, (
                f"picks {chosen[a]} and {chosen[b]} are only {dist} apart; "
                "raster-order adjacency would give 1000"
            )
