"""Out-of-slide substitution: the tool hands back another slide's pixels entirely.

Two things separate this from every other fault type and both are load-bearing. The
served patch's box indexes a *different slide's* coordinate frame, so any comparison
against the parent context -- containment, IoU, centre distance -- is a category error
rather than merely a false statement. And the substitute must come from the same split:
drawing a dev slide's pixels into a test item would undo the case-level split as surely
as putting the slide itself in both.
"""
import numpy as np
import pytest

from pyrafault.faults import (
    OUT_OF_SLIDE_TYPES, FaultError, add_provenance_columns, assert_pair_invariants,
)
from pyrafault.geom import BBoxL0
from pyrafault.mining import select_out_of_slide

CTX = BBoxL0(1000, 1000, 800, 800)
INSIDE = BBoxL0(1100, 1100, 100, 100)


def _emb(*v):
    return np.asarray(v, dtype=np.float32)


def _pool(*vecs):
    return (np.stack([_emb(*v) for v in vecs]),
            [BBoxL0(100 * i, 100 * i, 50, 50) for i in range(len(vecs))])


class TestSelection:
    def test_never_draws_its_own_slide(self):
        pools = {"A": _pool((1, 0)), "B": _pool((0, 1))}
        for _ in range(10):
            pick = select_out_of_slide(_emb(1, 0), pools, "A", "random",
                                       np.random.default_rng(_))
            assert pick.slide_id == "B"

    def test_hard_takes_the_most_similar_across_all_other_slides(self):
        pools = {"A": _pool((1, 0)), "B": _pool((0.2, 0.98)), "C": _pool((0.9, 0.44))}
        pick = select_out_of_slide(_emb(1, 0), pools, "A", "hard")
        assert pick.slide_id == "C"
        assert pick.similarity == pytest.approx(0.9, abs=1e-6)

    def test_returns_none_when_the_split_holds_only_this_slide(self):
        """A one-slide split cannot produce an out-of-slide fault. That is a real
        outcome to record, not a reason to reach into another split."""
        assert select_out_of_slide(_emb(1, 0), {"A": _pool((1, 0))}, "A", "hard") is None

    def test_random_is_seeded(self):
        pools = {c: _pool((1, 0), (0, 1)) for c in "BCDEF"}
        picks = {(p.slide_id, p.bbox) for p in (
            select_out_of_slide(_emb(1, 0), pools, "A", "random",
                                np.random.default_rng(11)) for _ in range(5))}
        assert len(picks) == 1

    def test_random_requires_an_rng(self):
        with pytest.raises(ValueError, match="seeded rng"):
            select_out_of_slide(_emb(1, 0), {"B": _pool((1, 0))}, "A", "random")

    def test_unknown_policy_rejected(self):
        with pytest.raises(ValueError, match="unknown out-of-slide policy"):
            select_out_of_slide(_emb(1, 0), {"B": _pool((1, 0))}, "A", "nearest")


class TestInvariants:
    def test_claim_must_still_be_inside_the_context(self):
        outside = BBoxL0(9000, 9000, 100, 100)
        for ft in OUT_OF_SLIDE_TYPES:
            with pytest.raises(FaultError, match="not inside context"):
                assert_pair_invariants(outside, BBoxL0(0, 0, 100, 100), CTX, ft)

    def test_true_bbox_is_not_compared_to_the_context(self):
        """Numerically inside, but on another slide -- and therefore not a violation.
        A check that rejected this would be comparing two different coordinate frames."""
        for ft in OUT_OF_SLIDE_TYPES:
            assert_pair_invariants(INSIDE, BBoxL0(1300, 1300, 100, 100), CTX, ft)


class TestProvenanceColumns:
    def _frame(self, **over):
        import pandas as pd
        row = {"pair_id": "p1", "slide_id": "A", "fault_type": "clean"}
        row.update(over)
        return pd.DataFrame([row])

    class _Cfg:
        class context: target_mpp = 2.0
        class query: target_mpp = 0.5

    def test_same_slide_types_default_to_their_own_slide(self):
        out = add_provenance_columns(self._frame(), self._Cfg())
        assert out["true_slide_id"].iloc[0] == "A"
        assert bool(out["same_slide"].iloc[0]) is True
        assert out["scale_ratio"].iloc[0] == pytest.approx(4.0)

    def test_out_of_slide_row_without_a_source_slide_is_caught(self):
        """The failure this cross-check exists for: forget to set true_slide_id and both
        it and same_slide are wrong in the same direction, so comparing them agrees."""
        with pytest.raises(FaultError, match="same_slide disagrees"):
            add_provenance_columns(self._frame(fault_type="out_of_slide"), self._Cfg())

    def test_same_slide_type_claiming_a_different_slide_is_caught(self):
        with pytest.raises(FaultError, match="same_slide disagrees"):
            add_provenance_columns(
                self._frame(fault_type="clean", true_slide_id="B"), self._Cfg())

    def test_a_correct_out_of_slide_row_passes(self):
        out = add_provenance_columns(
            self._frame(fault_type="out_of_slide", true_slide_id="B"), self._Cfg())
        assert bool(out["same_slide"].iloc[0]) is False


class TestQCCatchesBadCrossSlidePairs:
    """Each guard is asserted to FIRE on a deliberately broken record. A guard that has
    never been seen to fail is indistinguishable from one that cannot."""

    def _tables(self, tmp_path, **over):
        import pandas as pd
        ctx = pd.DataFrame([{
            "context_id": "A_c0", "slide_id": "A", "split": "test",
            "bbox_l0_x": 1000, "bbox_l0_y": 1000, "bbox_l0_w": 800, "bbox_l0_h": 800,
            "target_mpp": 2.0, "output_size": 100, "extraction_version": "v1",
            "image_sha256": None, "materialized_path": None,
        }])
        qry = pd.DataFrame([{
            "query_id": "A_q0", "context_id": "A_c0", "slide_id": "A", "split": "test",
            "true_bbox_l0_x": 1100, "true_bbox_l0_y": 1100,
            "true_bbox_l0_w": 100, "true_bbox_l0_h": 100, "target_mpp": 0.5,
            "output_size": 50, "extraction_version": "v1",
            "image_sha256": None, "materialized_path": None,
        }])
        row = {
            "pair_id": "A_q0__out_of_slide", "slide_id": "A", "context_id": "A_c0",
            "query_id": "A_q0", "fault_type": "out_of_slide",
            "claimed_bbox_l0_x": 1100, "claimed_bbox_l0_y": 1100,
            "claimed_bbox_l0_w": 100, "claimed_bbox_l0_h": 100,
            "true_bbox_l0_x": 40, "true_bbox_l0_y": 40,
            "true_bbox_l0_w": 100, "true_bbox_l0_h": 100,
            "claim_correct": False, "fault_severity": None,
            "normalized_offset": None, "offset_frac_context": None,
            "iou_claim_true": 0.0, "true_within_context": False,
            "local_recovery_possible": False, "split": "test",
            "true_slide_id": "B", "same_slide": False,
        }
        row.update(over)
        pairs = pd.DataFrame([row])
        splits = pd.DataFrame([
            {"case_id": "cA", "slide_id": "A", "split": "test"},
            {"case_id": "cB", "slide_id": "B", "split": "test"},
            {"case_id": "cD", "slide_id": "D", "split": "dev"},
        ])
        slides = pd.DataFrame([
            {"slide_id": s, "mpp_x": 0.25, "mpp_y": 0.25, "source_relpath": f"{s}.svs",
             "width_l0": 20000, "height_l0": 20000, "qc_status": "ok",
             "sha256_local": "0" * 64, "case_id": f"c{s}"}
            for s in ("A", "B", "D")])
        return slides, ctx, qry, pairs, splits

    def _checks(self, tmp_path, **over):
        from pyrafault.qc.report import check_invariants
        s, c, q, p, sp = self._tables(tmp_path, **over)
        for slide in ("A", "B", "D"):
            (tmp_path / f"{slide}.svs").write_bytes(b"")   # existence only
        return {v.check for v in check_invariants(s, c, q, p, sp, source_root=tmp_path)}

    def test_a_correct_cross_slide_pair_is_clean(self, tmp_path):
        assert self._checks(tmp_path) == set()

    def test_substitution_from_another_split_is_caught(self, tmp_path):
        """The one that matters most: a dev slide's pixels inside a test item."""
        assert "cross_split_substitution" in self._checks(tmp_path, true_slide_id="D")

    def test_true_slide_equal_to_slide_is_caught(self, tmp_path):
        got = self._checks(tmp_path, true_slide_id="A", same_slide=True)
        assert "out_of_slide_same_slide" in got

    def test_nonzero_iou_across_slides_is_caught(self, tmp_path):
        assert "iou_mismatch" in self._checks(tmp_path, iou_claim_true=0.4)

    def test_a_distance_between_two_slides_frames_is_caught(self, tmp_path):
        got = self._checks(tmp_path, normalized_offset=3.2)
        assert "normalized_offset_across_slides" in got

    def test_true_within_context_true_on_a_cross_slide_pair_is_caught(self, tmp_path):
        assert "true_within_context" in self._checks(tmp_path, true_within_context=True)

    def test_claim_outside_the_context_is_still_caught(self, tmp_path):
        got = self._checks(tmp_path, claimed_bbox_l0_x=9000, claimed_bbox_l0_y=9000)
        assert "claim_outside_context" in got


def test_every_pairs_pixel_read_resolves_the_source_slide():
    """A cross-slide box read against the wrong slide either escapes the canvas or --
    the dangerous case -- lands inside it and returns a plausible image of unrelated
    tissue. dataset.py was fixed for this; 09_run_qc.py's overlay renderer was missed
    and only surfaced as an OutOfBoundsError during a rebuild. This scans the source
    for the pattern rather than trusting that both sites stay fixed.

    Reads keyed off `contexts` or `queries` are exempt: those tables only ever describe
    boxes on their own slide. Only reads that take a bbox from `pairs` need resolving.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    offenders = []
    for path in list((root / "scripts").glob("*.py")) + list((root / "pyrafault").rglob("*.py")):
        text = path.read_text()
        for m in re.finditer(r"read_at_mpp\((.{0,200}?)\)", text, re.S):
            call = m.group(1)
            if "true_bbox_l0" not in call:
                continue
            # The handle passed as the first argument must not be a bare slide handle
            # opened from the pair's own slide_id; the fixed sites either compare
            # true_slide_id first or open the source slide explicitly.
            window = text[max(0, m.start() - 700):m.start()]
            if "true_slide_id" not in window:
                line = text[:m.start()].count("\n") + 1
                offenders.append(f"{path.relative_to(root)}:{line}")
    assert not offenders, (
        "these read a pairs-table true_bbox without resolving true_slide_id first: "
        f"{offenders}"
    )
