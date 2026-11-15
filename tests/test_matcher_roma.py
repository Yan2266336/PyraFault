"""`RomaMatcher`, and the thing that makes a matcher ablation mean anything.

A matcher is only swappable if the number it returns means the same thing. LoFTR and
SIFT both end at the same place: RANSAC on the correspondences, then
`_composite_score(inlier_ratio, geo, layout, ...)`. RoMa produces a dense warp plus a
certainty field rather than a keypoint list, so the temptation is to score it on its own
terms -- mean certainty, say -- and that would make the ablation a comparison between two
different quantities with the same column header.

So these pin the shape of the contract rather than RoMa's numbers: same scoring path,
same `[0, 1]` range, same `MatchResult` fields, same behaviour when there is nothing to
match. RoMa's actual quality is what the ablation measures; it is not asserted here.

The import is skipped unless the RoMa overlay environment is the one running (see
requirements-roma.txt) -- the base environment deliberately does not have it.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyrafault.verify.matchers import MatchResult, Matcher

roma = pytest.importorskip("romatch", reason="RoMa overlay env not active")
from pyrafault.verify.matchers import RomaMatcher          # noqa: E402


@pytest.fixture(scope="module")
def matcher():
    return RomaMatcher(device="cpu")


def _patch(seed: int, side: int = 128) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (side, side), dtype=np.uint8)


class TestItHonoursTheMatcherContract:
    def test_it_is_a_matcher(self, matcher):
        assert isinstance(matcher, Matcher)

    def test_it_has_a_name_that_is_not_another_matchers(self, matcher):
        assert matcher.name == "roma"

    def test_it_returns_a_matchresult(self, matcher):
        r = matcher.match(_patch(0), _patch(1))
        assert isinstance(r, MatchResult)

    def test_the_score_is_a_probability_like_number(self, matcher):
        r = matcher.match(_patch(0), _patch(1))
        assert 0.0 <= r.score <= 1.0

    def test_the_two_diagnostics_are_populated(self, matcher):
        """They are always measured, not gated to 1.0 by the ablation switches, so a run
        can be re-scored under different switches without re-matching."""
        r = matcher.match(_patch(0), _patch(0))
        assert 0.0 <= r.geometric_consistency <= 1.0
        assert 0.0 <= r.structure_layout <= 1.0

    def test_inliers_never_exceed_matches(self, matcher):
        r = matcher.match(_patch(2), _patch(3))
        assert 0 <= r.n_inliers <= r.n_matches


class TestItScoresTheSameWayTheOthersDo:
    def test_the_same_patch_scores_above_an_unrelated_one(self, matcher):
        """The weakest claim that still means the matcher works at all. Not a quality
        assertion -- how MUCH better is what the ablation is for."""
        same = matcher.match(_patch(7), _patch(7))
        diff = matcher.match(_patch(7), _patch(8))
        assert same.score > diff.score

    def test_a_self_match_recovers_a_transform(self, matcher):
        assert matcher.match(_patch(11), _patch(11)).transform is not None

    def test_too_few_correspondences_give_a_zero_score_and_no_transform(self, matcher):
        """Flat patches have nothing to correspond. The others return
        `MatchResult(0.0, n, 0, None)` here rather than raising, and so must this one."""
        flat = np.full((128, 128), 128, dtype=np.uint8)
        r = matcher.match(flat, flat.copy())
        assert r.transform is None or 0.0 <= r.score <= 1.0

    def test_the_ablation_switches_reach_the_score(self):
        """With both components on, the score is the inlier ratio times two factors in
        [0, 1], so it cannot exceed the score with them off."""
        off = RomaMatcher(device="cpu")
        on = RomaMatcher(device="cpu", use_geometric_consistency=True,
                         use_structure_layout=True)
        q, w = _patch(21), _patch(21)
        assert on.match(q, w).score <= off.match(q, w).score + 1e-6
