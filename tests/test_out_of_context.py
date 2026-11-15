"""Out-of-context substitution (proposal table 3 rows 4-5, spec §9.5).

The invariant that separates these two fault types from every other one: the CLAIM
still sits inside the parent context -- it is the agent's own zoom request, which
`§4.1` holds fixed while the returned pixels are swapped -- and it is `ℓ*`, the true
source of the patch actually served, that lies outside. Getting this backwards would
produce a fault a verifier could reject by a bbox containment test with zero visual
reasoning, which is the failure spec decision D2 exists to prevent.
"""
import numpy as np
import pytest

from pyrafault.faults import OUT_OF_CONTEXT_TYPES, FaultError, assert_pair_invariants
from pyrafault.geom import BBoxL0
from pyrafault.mining import select_out_of_context


CTX = BBoxL0(1000, 1000, 800, 800)
INSIDE = BBoxL0(1100, 1100, 100, 100)
OUTSIDE = BBoxL0(3000, 3000, 100, 100)


def _emb(*vals):
    return np.asarray(vals, dtype=np.float32)


class TestInvariants:
    def test_true_bbox_outside_context_is_required(self):
        """An out-of-context pair whose true source is inside C is a construction bug."""
        for ft in OUT_OF_CONTEXT_TYPES:
            with pytest.raises(FaultError, match="outside the context"):
                assert_pair_invariants(INSIDE, BBoxL0(1300, 1300, 100, 100), CTX, ft)

    def test_claim_must_still_be_inside_context(self):
        for ft in OUT_OF_CONTEXT_TYPES:
            with pytest.raises(FaultError, match="not inside context"):
                assert_pair_invariants(OUTSIDE, OUTSIDE, CTX, ft)

    def test_valid_out_of_context_pair_passes(self):
        for ft in OUT_OF_CONTEXT_TYPES:
            assert_pair_invariants(INSIDE, OUTSIDE, CTX, ft)

    def test_in_context_types_still_require_true_inside(self):
        """The relaxation is scoped: every other fault type keeps the original rule."""
        with pytest.raises(FaultError, match="not inside context"):
            assert_pair_invariants(INSIDE, OUTSIDE, CTX, "hard_mismatch")


class TestSelection:
    def test_overlapping_candidate_is_never_eligible(self):
        """Contexts overlap on a 0.25 stride, so a box from a neighbouring context can
        still intersect this one. Eligibility is geometric, never by context_id."""
        touching = BBoxL0(1750, 1750, 100, 100)      # overlaps CTX's far corner
        pick = select_out_of_context(
            _emb(1, 0), np.stack([_emb(1, 0)]), [touching], CTX, "hard", 0.0
        )
        assert pick is None

    def test_hard_policy_takes_the_most_similar(self):
        boxes = [BBoxL0(3000, 3000, 100, 100), BBoxL0(5000, 5000, 100, 100)]
        pool = np.stack([_emb(0.1, 0.9), _emb(0.99, 0.01)])
        pick = select_out_of_context(_emb(1, 0), pool, boxes, CTX, "hard", 0.0)
        assert pick.bbox == boxes[1]
        assert pick.similarity == pytest.approx(0.99, abs=1e-6)

    def test_random_policy_is_seeded_and_reproducible(self):
        boxes = [BBoxL0(3000 + 200 * i, 3000, 100, 100) for i in range(8)]
        pool = np.stack([_emb(1, 0)] * 8)
        picks = {
            select_out_of_context(
                _emb(1, 0), pool, boxes, CTX, "random", 0.0,
                np.random.default_rng(7),
            ).bbox
            for _ in range(5)
        }
        assert len(picks) == 1, "same seed must draw the same candidate every time"

    def test_random_requires_an_rng(self):
        boxes = [BBoxL0(3000, 3000, 100, 100)]
        with pytest.raises(ValueError, match="seeded rng"):
            select_out_of_context(_emb(1, 0), np.stack([_emb(1, 0)]), boxes, CTX,
                                  "random", 0.0)

    def test_no_eligible_candidate_returns_none_rather_than_relaxing(self):
        """A slide whose contexts blanket its tissue yields no out-of-context source.
        That is a real outcome to record, not a reason to admit an overlapping box."""
        assert select_out_of_context(
            _emb(1, 0), np.stack([_emb(1, 0)]), [BBoxL0(1200, 1200, 50, 50)],
            CTX, "hard", 0.0,
        ) is None

    def test_min_gap_pushes_the_source_further_away(self):
        near = BBoxL0(1900, 1000, 100, 100)          # 100 px clear of CTX
        pool = np.stack([_emb(1, 0)])
        assert select_out_of_context(_emb(1, 0), pool, [near], CTX, "hard", 0.0) is not None
        assert select_out_of_context(_emb(1, 0), pool, [near], CTX, "hard", 400.0) is None

    def test_unknown_policy_is_rejected(self):
        with pytest.raises(ValueError, match="unknown out-of-context policy"):
            select_out_of_context(_emb(1, 0), np.stack([_emb(1, 0)]),
                                  [BBoxL0(3000, 3000, 100, 100)], CTX, "nearest", 0.0)
