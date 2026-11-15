"""Local correspondence and the score built from it.

The score is the inlier RATIO, never the match count. Measured on real items, match
count against windows offset by 0, 8, 16, 24, 32 px is 144, 121, 100, 81, 64 -- which
is 12², 11², 10², 9², 8², the overlapping area. A count-based score would measure
geometric overlap, and overlap differs systematically between clean (on a lattice
point) and adjacent_drift (necessarily between grid positions), reintroducing exactly
the artifact the refined-bbox design removes.
"""
import math

import numpy as np
import pytest

from pyrafault.verify.matchers import (
    DEFAULT_GEOMETRIC_CONSISTENCY_SCALE,
    MatchResult, Matcher, SiftRansacMatcher, refine_bbox,
    geometric_consistency_score, structure_layout_score,
)


def _textured(seed, size=128):
    """Structured, not uniform noise: SIFT needs corners to find."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (size // 8, size // 8), dtype=np.uint8)
    return np.kron(small, np.ones((8, 8), dtype=np.uint8))


class TestSiftRansacMatcher:
    def test_identical_images_score_high(self):
        img = _textured(0)
        assert SiftRansacMatcher().match(img, img).score > 0.5

    def test_unrelated_images_score_low(self):
        m = SiftRansacMatcher()
        assert m.match(_textured(0), _textured(999)).score < 0.2

    def test_score_is_a_ratio_in_the_unit_interval(self):
        r = SiftRansacMatcher().match(_textured(1), _textured(1))
        assert 0.0 <= r.score <= 1.0
        assert r.n_inliers <= r.n_matches

    def test_a_featureless_image_scores_zero_rather_than_raising(self):
        flat = np.full((128, 128), 128, dtype=np.uint8)
        r = SiftRansacMatcher().match(flat, flat)
        assert r.score == 0.0 and r.transform is None


class TestRefineBbox:
    def test_identity_transform_returns_the_window_itself(self):
        box = refine_bbox(origin=(100, 200), side=128, transform=np.eye(3))
        assert (box.x, box.y, box.w, box.h) == (100, 200, 128, 128)

    def test_a_translation_shifts_the_estimate_off_the_grid(self):
        """The whole point: the estimate is continuous, so grid alignment cannot make
        clean and adjacent_drift score differently for a geometric reason."""
        t = np.eye(3); t[0, 2], t[1, 2] = 5.0, -3.0
        box = refine_bbox(origin=(100, 200), side=128, transform=t)
        assert (box.x, box.y) == (95, 203)

    def test_the_estimate_beats_the_unrefined_window_against_known_truth(self):
        """The test above cannot fail on a sign error: it hands a synthetic transform
        straight in and checks only that the arithmetic is self-consistent. This one
        plants a patch at a known offset and runs a real matcher, so a mirrored estimate
        shows up as being further from the truth than not refining at all."""
        rng = np.random.default_rng(3)
        canvas = np.kron(rng.integers(0, 255, (48, 48), dtype=np.uint8),
                         np.ones((8, 8), dtype=np.uint8))
        side, true_xy, win_xy = 128, (110, 95), (100, 100)
        query = canvas[true_xy[1]:true_xy[1] + side, true_xy[0]:true_xy[0] + side]
        window = canvas[win_xy[1]:win_xy[1] + side, win_xy[0]:win_xy[0] + side]

        res = SiftRansacMatcher().match(query, window)
        box = refine_bbox(win_xy, side, res.transform)
        refined = math.hypot(box.x - true_xy[0], box.y - true_xy[1])
        unrefined = math.hypot(win_xy[0] - true_xy[0], win_xy[1] - true_xy[1])
        assert refined < unrefined, (
            f"refinement moved the estimate {refined:.1f}px from truth, worse than the "
            f"window box's {unrefined:.1f}px -- the transform is being applied backwards"
        )

    def test_no_transform_yields_no_estimate(self):
        assert refine_bbox(origin=(0, 0), side=128, transform=None) is None


def test_matcher_is_an_interface_others_can_implement():
    import inspect
    assert inspect.isabstract(Matcher)
    assert list(inspect.signature(Matcher.match).parameters) == ["self", "query", "window"]


def _linear_part(sigma1: float, sigma2: float) -> np.ndarray:
    """A 3x3 homography whose 2x2 linear part has exactly these singular values
    (built via a trivial SVD with identity rotations, i.e. diag(sigma1, sigma2)), plus
    an arbitrary translation -- geometric consistency must ignore the translation and
    respond only to the linear part."""
    h = np.eye(3)
    h[0, 0], h[1, 1] = sigma1, sigma2
    h[0, 2], h[1, 2] = 17.0, -42.0
    return h


class TestGeometricConsistencyScore:
    """Measured on real LoFTR correspondences (see matchers.py): the maximum deviation
    of the linear part's singular values from 1 is 0.0023 (median) at the true
    location and 0.9124 at a random one -- non-overlapping ranges. A constant
    replacement for this function could not reproduce that separation."""

    def test_near_identity_transform_scores_high(self):
        """0.0023 is the measured median true-location deviation."""
        h = _linear_part(1.0 + 0.0023, 1.0)
        assert geometric_consistency_score(h) > 0.99

    def test_strongly_non_rigid_transform_scores_low(self):
        """0.9124 is the measured random-location deviation."""
        h = _linear_part(1.0 + 0.9124, 1.0)
        assert geometric_consistency_score(h) < 0.15

    def test_the_measured_true_and_random_deviations_separate_cleanly(self):
        true_h = _linear_part(1.0 + 0.0023, 1.0)
        random_h = _linear_part(1.0 + 0.9124, 1.0)
        assert geometric_consistency_score(true_h) - geometric_consistency_score(random_h) > 0.5

    def test_pure_translation_scores_the_maximum(self):
        assert geometric_consistency_score(np.eye(3)) == pytest.approx(1.0)

    def test_no_transform_is_neutral_not_penalising(self):
        assert geometric_consistency_score(None) == 1.0

    def test_score_stays_in_the_unit_interval_even_far_from_identity(self):
        h = _linear_part(50.0, 0.001)
        s = geometric_consistency_score(h)
        assert 0.0 <= s <= 1.0

    def test_scale_parameter_controls_the_falloff_steepness(self):
        """A tighter scale must punish the same deviation harder -- otherwise `scale`
        is decorative rather than parameterising the falloff as the spec asks."""
        h = _linear_part(1.3, 1.0)
        loose = geometric_consistency_score(h, scale=2.0)
        tight = geometric_consistency_score(h, scale=0.1)
        assert tight < loose

    def test_default_scale_constant_matches_the_documented_derivation(self):
        assert DEFAULT_GEOMETRIC_CONSISTENCY_SCALE == pytest.approx(0.4)


class TestStructureLayoutScore:
    """A constant replacement for this function could not distinguish a dispersed
    inlier set from a clustered one -- these tests would catch that."""

    def test_inliers_spread_across_the_patch_score_high(self):
        side = 128
        corners = np.array([[2, 2], [126, 2], [126, 126], [2, 126]], dtype=np.float32)
        assert structure_layout_score(corners, side) > 0.9

    def test_inliers_clustered_in_one_corner_score_low(self):
        # < 0.05, not < 0.01 as before the layout-score fix: even a tight cluster now
        # gets a small non-zero credit from `extent` (its own bounding-box diagonal is
        # not literally 0), but it stays far below any dispersed point set.
        side = 128
        cluster = np.array([[2, 2], [4, 3], [3, 5], [5, 4]], dtype=np.float32)
        assert structure_layout_score(cluster, side) < 0.05

    def test_dispersed_inliers_score_higher_than_clustered_ones(self):
        side = 128
        corners = np.array([[2, 2], [126, 2], [126, 126], [2, 126]], dtype=np.float32)
        cluster = np.array([[60, 60], [62, 61], [61, 63], [63, 62]], dtype=np.float32)
        assert (structure_layout_score(corners, side)
                - structure_layout_score(cluster, side)) > 0.5

    def test_fewer_than_three_points_scores_zero(self):
        assert structure_layout_score(np.array([[10.0, 10.0]]), 128) == 0.0
        assert structure_layout_score(np.zeros((0, 2)), 128) == 0.0

    def test_collinear_points_confined_to_a_corner_score_low(self):
        """Collinear points that are also bunched up in a small region are exactly as
        uninformative as a non-collinear tight cluster: neither hull area nor
        bounding-box extent has anything to credit here."""
        collinear_cornered = np.array(
            [[2, 2], [6, 6], [10, 10], [14, 14]], dtype=np.float32)
        assert structure_layout_score(collinear_cornered, 128) < 0.15

    def test_collinear_points_spanning_the_patch_score_higher_than_a_tight_cluster(self):
        """A line of correspondences running the length of the patch -- the pattern a
        real match along a thin structure (vessel, duct, edge) produces -- is spread
        evidence a hull-area-only measure would zero out just because it's a line.
        Same point count as the cluster, very different amount of real estate covered."""
        side = 128
        collinear_spanning = np.array(
            [[2, 2], [42, 42], [84, 84], [126, 126]], dtype=np.float32)
        tight_cluster = np.array(
            [[60, 60], [61, 60.5], [62, 61], [60.5, 61.5]], dtype=np.float32)
        assert (structure_layout_score(collinear_spanning, side)
                - structure_layout_score(tight_cluster, side)) > 0.5

    def test_collinear_points_spanning_the_patch_score_higher_than_collinear_cornered(self):
        side = 128
        collinear_spanning = np.array(
            [[2, 2], [42, 42], [84, 84], [126, 126]], dtype=np.float32)
        collinear_cornered = np.array(
            [[2, 2], [6, 6], [10, 10], [14, 14]], dtype=np.float32)
        assert (structure_layout_score(collinear_spanning, side)
                - structure_layout_score(collinear_cornered, side)) > 0.5

    def test_score_never_exceeds_one_even_if_points_lie_outside_the_patch(self):
        side = 128
        far = np.array([[-500, -500], [500, -500], [500, 500], [-500, 500]], dtype=np.float32)
        assert structure_layout_score(far, side) == 1.0


class TestComponentsAreIndependentlySwitchable:
    """The ablation the experiment plan needs: geometry (and layout) must be removable
    without touching the other components, and a disabled component must be a neutral
    1.0 factor in the composite -- not merely unreported."""

    def _images(self):
        return _textured(0), _textured(0)  # identical -> near-identity transform

    def test_default_matcher_score_equals_the_plain_inlier_ratio(self):
        """Both new components default OFF: the composite must reduce EXACTLY to the
        pre-existing inlier-ratio-only score, so shipped thresholds.json files are not
        silently invalidated."""
        query, window = self._images()
        r = SiftRansacMatcher().match(query, window)
        assert r.n_matches > 0
        assert r.score == pytest.approx(r.n_inliers / r.n_matches)

    def test_enabling_geometric_consistency_changes_the_score_for_a_non_rigid_case(self):
        """Same correspondences, only the switch differs -- if geometric consistency
        were wired in but not actually gating anything, this would be a no-op."""
        query, window = self._images()
        off = SiftRansacMatcher().match(query, window)
        on = SiftRansacMatcher(use_geometric_consistency=True).match(query, window)
        assert on.transform is not None and off.transform is not None
        assert on.score == pytest.approx(off.score * on.geometric_consistency)

    def test_enabling_structure_layout_changes_the_score(self):
        query, window = self._images()
        off = SiftRansacMatcher().match(query, window)
        on = SiftRansacMatcher(use_structure_layout=True).match(query, window)
        assert on.score == pytest.approx(off.score * on.structure_layout)

    def test_component_values_are_reported_even_when_the_switch_is_off(self):
        """So a run can be re-scored under a different ablation setting from its own
        diagnostics, without re-matching."""
        query, window = self._images()
        r = SiftRansacMatcher().match(query, window)
        assert 0.0 <= r.geometric_consistency <= 1.0
        assert 0.0 <= r.structure_layout <= 1.0

    def test_composite_stays_in_the_unit_interval_with_all_components_on(self):
        query, window = self._images()
        r = SiftRansacMatcher(use_geometric_consistency=True,
                              use_structure_layout=True).match(query, window)
        assert 0.0 <= r.score <= 1.0

    def test_match_result_defaults_are_neutral_for_hand_built_stubs(self):
        """Existing test doubles across the repo build `MatchResult` positionally
        without the two new fields (e.g. `_ranked_matcher` in test_verify_zoomtrace.py)
        -- they must come out neutral, not zero, or those stubs' arithmetic changes."""
        r = MatchResult(0.7, 10, 7, np.eye(3))
        assert r.geometric_consistency == 1.0
        assert r.structure_layout == 1.0
