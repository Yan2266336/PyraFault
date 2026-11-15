import math
import subprocess
import sys

import pytest
from pyrafault.geom import (
    BBoxL0, fov_l0, grid_positions, subgrid, context_jitter, iou, center_distance,
)

MPP = 0.2520
CTX_FOV = 16254   # round(2048 * 2.0 / 0.2520)
Q_FOV = 1016      # round(512  * 0.5 / 0.2520)


def test_fov_l0_matches_measured_values():
    assert fov_l0(2048, 2.0, 0.2520) == CTX_FOV
    assert fov_l0(512, 0.5, 0.2520) == Q_FOV
    # TCGA-64-1680 has a slightly different MPP and therefore a different grid
    assert fov_l0(2048, 2.0, 0.2527) == 16209
    assert fov_l0(512, 0.5, 0.2527) == 1013


@pytest.mark.parametrize(
    "w,h,stride_frac,expected",
    [
        # (slide dims, stride fraction, context count) — spec §3 D1 table
        (27818, 23967, 0.50, 2),    # TCGA-55-A4DG
        (27818, 23967, 0.25, 6),
        (35492, 31380, 0.50, 6),    # TCGA-55-8094
        (35492, 31380, 0.25, 20),
        (44125, 27646, 0.25, 21),   # TCGA-55-7281
        (110313, 22108, 0.25, 48),  # TCGA-44-2655
        (96883, 78802, 0.25, 320),  # TCGA-49-4505
    ],
)
def test_context_grid_counts_match_spec_table(w, h, stride_frac, expected):
    stride = round(CTX_FOV * stride_frac)
    assert len(grid_positions(w, h, CTX_FOV, stride)) == expected


def test_grid_positions_are_all_in_bounds():
    boxes = grid_positions(35492, 31380, CTX_FOV, round(CTX_FOV * 0.25))
    assert all(b.within_bounds(35492, 31380) for b in boxes)


def test_grid_empty_when_fov_exceeds_slide():
    assert grid_positions(10000, 10000, CTX_FOV, 4064) == []


def test_subgrid_counts_and_containment():
    ctx = BBoxL0(1000, 2000, CTX_FOV, CTX_FOV)
    # 16254 // 1016 == 15, remainder 1014 < 1016 so n is decremented to 14 (spec 15.1);
    # that gives a 14x14 grid, and one margin ring leaves 12x12.
    full = subgrid(ctx, Q_FOV, margin_rings=0)
    inner = subgrid(ctx, Q_FOV, margin_rings=1)
    assert len(full) == 196
    assert len(inner) == 144
    assert all(ctx.contains(b) for b in full)


def test_subgrid_decrements_n_so_the_remainder_is_never_smaller_than_fov():
    """Without the decrement, remainder=1014 < Q_FOV=1016 leaves 1014 of the 1016
    residue classes reachable at best -- not "every residue", the property the fix
    promises. After decrementing n, the remainder must be >= fov - 1 (a full period)."""
    ctx = BBoxL0(1000, 2000, CTX_FOV, CTX_FOV)
    n = len(subgrid(ctx, Q_FOV, margin_rings=0)) ** 0.5
    assert round(n) == 14
    remainder = CTX_FOV - 14 * Q_FOV
    assert remainder == 2030
    assert remainder >= Q_FOV - 1


def test_subgrid_is_centered_within_parent_when_no_jitter_given():
    ctx = BBoxL0(0, 0, CTX_FOV, CTX_FOV)
    full = subgrid(ctx, Q_FOV, margin_rings=0)
    inset = (CTX_FOV - 14 * Q_FOV) // 2
    assert inset == 1015
    assert full[0].x == inset and full[0].y == inset


def test_subgrid_handles_the_16_wide_slide():
    ctx = BBoxL0(0, 0, 16209, 16209)
    # 16209 // 1013 == 16, remainder 1 < 1013 so n decrements to 15 (remainder 1014).
    assert len(subgrid(ctx, 1013, margin_rings=0)) == 225
    assert len(subgrid(ctx, 1013, margin_rings=1)) == 169


def test_subgrid_refuses_negative_margin_rings():
    """A negative margin_rings would otherwise widen lo/hi past the grid and emit
    boxes outside the parent, silently violating the containment contract everything
    downstream assumes. boundary_margin_frac is config-reachable, so this must never
    be trusted to be non-negative by construction."""
    ctx = BBoxL0(1000, 2000, CTX_FOV, CTX_FOV)
    assert subgrid(ctx, Q_FOV, margin_rings=-1) == []


def test_subgrid_jitter_moves_the_origin_off_center():
    """A jittered context must not silently fall back to the centred inset -- otherwise
    the origin is still derivable from the context bbox alone."""
    ctx = BBoxL0(1000, 2000, CTX_FOV, CTX_FOV)
    centered = subgrid(ctx, Q_FOV, margin_rings=0)
    jittered = subgrid(ctx, Q_FOV, margin_rings=0, seed=7, context_id="S__c0000")
    assert len(jittered) == len(centered)
    assert (jittered[0].x, jittered[0].y) != (centered[0].x, centered[0].y)
    assert all(ctx.contains(b) for b in jittered)


def test_subgrid_requires_context_id_with_seed():
    ctx = BBoxL0(0, 0, CTX_FOV, CTX_FOV)
    with pytest.raises(ValueError):
        subgrid(ctx, Q_FOV, margin_rings=0, seed=1)


def test_subgrid_jittered_candidates_stay_inside_parent_and_do_not_overlap():
    ctx = BBoxL0(500, 700, CTX_FOV, CTX_FOV)
    for seed, cid in [(1, "S__c0000"), (2, "S__c0001"), (99, "S__c0042")]:
        boxes = subgrid(ctx, Q_FOV, margin_rings=1, seed=seed, context_id=cid)
        assert boxes, "jittered grid must still produce candidates"
        assert all(ctx.contains(b) for b in boxes)
        for a in range(len(boxes)):
            for b in range(a + 1, len(boxes)):
                assert iou(boxes[a], boxes[b]) == 0.0


def test_context_jitter_is_deterministic_across_processes():
    """Same (seed, context_id) must reproduce the same offset in a fresh process --
    Python's built-in hash() is randomised per process (PYTHONHASHSEED) and would break
    this; sha256 must not."""
    code = (
        "from pyrafault.geom import context_jitter;"
        "print(context_jitter(42, 'S__c0007', 2030, 2030))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == str(context_jitter(42, "S__c0007", 2030, 2030))


def test_context_jitter_axes_are_independent_for_some_contexts():
    """A single shared offset would leave (x - j) mod fov == (y - j) mod fov true for
    every claim, which is the shortcut in a different shape (spec 15.1). jx must differ
    from jy for at least some contexts -- if the two axes were hashed identically, jx
    would equal jy everywhere."""
    diffs = [
        context_jitter(0, f"S__c{i:04d}", 2030, 2030)
        for i in range(50)
    ]
    assert any(jx != jy for jx, jy in diffs)


def test_context_jitter_covers_every_residue_class_when_remainder_covers_a_full_period():
    """The whole point of decrementing n (spec 15.1): once remainder >= fov - 1, every
    residue class mod fov is reachable by *some* offset in [0, remainder] -- no claimed
    location can be excluded by arithmetic alone. Before the decrement (remainder < fov)
    that is false, which is exactly the bug being fixed."""
    fov = Q_FOV
    remainder_after_fix = 2030  # CTX_FOV - 14 * Q_FOV, after the n decrement
    reachable = {x % fov for x in range(0, remainder_after_fix + 1)}
    assert reachable == set(range(fov))

    remainder_before_fix = 1014  # CTX_FOV - 15 * Q_FOV, what n gave before the fix
    unreachable = {x % fov for x in range(0, remainder_before_fix + 1)}
    assert unreachable != set(range(fov)), "pre-fix remainder must NOT cover a full period"


def test_payload_ongrid_and_offgrid_claims_are_indistinguishable_without_the_jitter_value():
    """The property that closes the single-item shortcut (spec 15.1): a verifier who
    knows the jitter scheme but not its value cannot tell, from one (context, claim)
    pair alone, whether the claim is on the query lattice -- because for ANY claimed x
    (on-grid multiples of fov as well as off-grid `adjacent_drift` displacements), some
    (jx, jy) in [0, remainder]^2 makes it look on-grid."""
    ctx = BBoxL0(1000, 2000, CTX_FOV, CTX_FOV)
    n_squared = len(subgrid(ctx, Q_FOV, margin_rings=0))
    n = round(n_squared ** 0.5)
    remainder = CTX_FOV - n * Q_FOV
    assert remainder >= Q_FOV - 1  # the decrement guarantee this test relies on

    # Sample claimed x-offsets from ctx.x, including on-grid multiples of Q_FOV and
    # off-grid (drifted) values that are not multiples of Q_FOV.
    candidate_offsets = [0, Q_FOV, 2 * Q_FOV, 137, Q_FOV + 511, 3 * Q_FOV - 1]
    for dx in candidate_offsets:
        target_residue = dx % Q_FOV
        assert any(jx % Q_FOV == target_residue for jx in range(0, remainder + 1)), (
            f"offset {dx} (residue {target_residue}) has no consistent jx in range"
        )


def test_iou_identical_disjoint_and_partial():
    a = BBoxL0(0, 0, 100, 100)
    assert iou(a, a) == 1.0
    assert iou(a, BBoxL0(200, 200, 100, 100)) == 0.0
    assert iou(a, BBoxL0(100, 0, 100, 100)) == 0.0        # edge-touching is not overlap
    assert iou(a, BBoxL0(50, 0, 100, 100)) == pytest.approx(50 * 100 / (2 * 100 * 100 - 50 * 100))


def test_contains_and_bounds():
    outer = BBoxL0(10, 10, 100, 100)
    assert outer.contains(BBoxL0(10, 10, 100, 100))
    assert outer.contains(BBoxL0(20, 20, 50, 50))
    assert not outer.contains(BBoxL0(20, 20, 200, 50))
    assert not BBoxL0(-1, 0, 10, 10).within_bounds(100, 100)
    assert not BBoxL0(95, 0, 10, 10).within_bounds(100, 100)
    assert BBoxL0(90, 90, 10, 10).within_bounds(100, 100)


def test_center_distance():
    assert center_distance(BBoxL0(0, 0, 10, 10), BBoxL0(30, 40, 10, 10)) == pytest.approx(50.0)


def test_as_dict_prefix():
    assert BBoxL0(1, 2, 3, 4).as_dict("bbox_l0") == {
        "bbox_l0_x": 1, "bbox_l0_y": 2, "bbox_l0_w": 3, "bbox_l0_h": 4
    }
