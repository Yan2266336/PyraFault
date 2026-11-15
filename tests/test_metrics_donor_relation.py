"""Where the served patch came from, as a stratum of its own.

The five-cohort freeze's stage 07 filtered `out_of_slide` donors by split only, so 47.5%
of them landed in a different cancer and 9.44% of the test split became "notice that
brain tissue is not endometrium". Those items are realistic -- a pipeline that mixes up
slides does not restrict itself to one organ -- so they stay. What cannot stay is pooling
them, because a stratum every method gets right lifts every method's headline together
and stops the comparison discriminating.

Measured on the lung freeze, which has only one organ, no method distinguished a
same-cancer donor from a different-cancer-same-organ one (ZoomTrace 0.0084 against
0.0079; the LoFTR family 0.0000 against 0.0000). That is the baseline the cross-organ
stratum has to be read against, and it is why the middle level exists at all.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from pyrafault.verify.metrics import score_predictions

# The stage script is not an importable module (it is `14_score_verifier.py`), and the
# derivation lives there on purpose: it needs `slides.parquet`, which `score_predictions`
# never sees.
_spec = importlib.util.spec_from_file_location(
    "stage14", Path(__file__).resolve().parents[1] / "scripts" / "14_score_verifier.py")
stage14 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage14)


def _slides(tmp_path: Path, mapping: dict) -> Path:
    p = tmp_path / "slides.parquet"
    pd.DataFrame([{"slide_id": s, "project_id": proj}
                  for s, proj in mapping.items()]).to_parquet(p, index=False)
    return p


def _pairs(rows) -> pd.DataFrame:
    return pd.DataFrame([
        {"item_id": f"i{i}", "slide_id": a, "true_slide_id": b,
         "claim_correct": a == b, "fault_type": "clean" if a == b else "out_of_slide",
         "true_bbox_l0_x": 0, "true_bbox_l0_y": 0,
         "true_bbox_l0_w": 100, "true_bbox_l0_h": 100,
         "claimed_bbox_l0_x": 0, "claimed_bbox_l0_y": 0,
         "claimed_bbox_l0_w": 100, "claimed_bbox_l0_h": 100,
         "true_within_context": True, "scale_ratio": 4.0, "context_mpp": 2.0}
        for i, (a, b) in enumerate(rows)])


def _preds(n, decision="verified") -> pd.DataFrame:
    return pd.DataFrame([
        {"item_id": f"i{i}", "reliability": 0.9, "decision": decision,
         "estimated_bbox_l0_x": 0, "estimated_bbox_l0_y": 0,
         "estimated_bbox_l0_w": 100, "estimated_bbox_l0_h": 100}
        for i in range(n)])


MAP = {"luad1": "TCGA-LUAD", "luad2": "TCGA-LUAD",
       "lusc1": "TCGA-LUSC", "lgg1": "TCGA-LGG"}


class TestTheFourLevelsAreOrderedByHowDifferentTheTissueLooks:
    def test_each_level_is_assigned(self, tmp_path):
        pairs = _pairs([("luad1", "luad1"),   # not out_of_slide at all
                        ("luad1", "luad2"),   # same cancer
                        ("luad1", "lusc1"),   # same organ, other cancer
                        ("luad1", "lgg1")])   # different organ
        out = stage14._with_donor_relation(pairs, _slides(tmp_path, MAP))
        assert list(out["donor_relation"]) == [
            "same_slide", "out_of_slide__same_project",
            "out_of_slide__same_organ", "out_of_slide__cross_organ"]

    def test_an_unmapped_cohort_is_named_not_called_cross_organ(self, tmp_path):
        """The one that would corrupt the headline. A cohort missing from ORGAN would
        otherwise fall into the easy stratum and inflate the very gap this table reports.
        """
        m = {**MAP, "mystery1": "TCGA-NEWCOHORT"}
        out = stage14._with_donor_relation(_pairs([("luad1", "mystery1")]),
                                           _slides(tmp_path, m))
        assert list(out["donor_relation"]) == ["out_of_slide__unknown_organ"]

    def test_lung_subtypes_are_the_middle_level_not_cross_organ(self, tmp_path):
        """LUAD and LUSC are both lung, so a patch from the other one is a hard negative
        and not a different tissue. The lung freeze is entirely this level."""
        out = stage14._with_donor_relation(_pairs([("lusc1", "luad1")]),
                                           _slides(tmp_path, MAP))
        assert list(out["donor_relation"]) == ["out_of_slide__same_organ"]


class TestAFreezeThatCannotSupportTheCutGetsNoCut:
    def test_no_true_slide_id_means_no_column(self, tmp_path):
        pairs = _pairs([("luad1", "luad2")]).drop(columns=["true_slide_id"])
        out = stage14._with_donor_relation(pairs, _slides(tmp_path, MAP))
        assert "donor_relation" not in out.columns

    def test_no_slides_table_means_no_column(self, tmp_path):
        out = stage14._with_donor_relation(_pairs([("luad1", "luad2")]),
                                           tmp_path / "absent.parquet")
        assert "donor_relation" not in out.columns

    def test_the_scorer_then_reports_no_such_stratum(self):
        """Absent rather than empty-but-present: a reader must not see a stratum table
        that silently describes nothing."""
        m = score_predictions(_preds(1), _pairs([("luad1", "luad2")]))
        assert m["by_donor_relation"] == {}


class TestTheRateIsOnlyPrintedWhereItMeansOneThing:
    def test_a_pure_fault_stratum_is_flagged_all_faults(self, tmp_path):
        pairs = stage14._with_donor_relation(
            _pairs([("luad1", "lgg1"), ("luad1", "lgg1")]), _slides(tmp_path, MAP))
        d = score_predictions(_preds(2), pairs)["by_donor_relation"]
        v = d["out_of_slide__cross_organ"]
        assert v["all_faults"] is True
        assert v["false_accept_rate"] == 1.0, "both were faults and both were accepted"

    def test_the_same_slide_stratum_is_not(self, tmp_path):
        """It mixes clean items with on-slide faults, so one rate would mean two things.
        `all_faults` is what the report keys on to leave it out."""
        pairs = stage14._with_donor_relation(
            _pairs([("luad1", "luad1"), ("luad1", "lgg1")]), _slides(tmp_path, MAP))
        d = score_predictions(_preds(2), pairs)["by_donor_relation"]
        assert d["same_slide"]["all_faults"] is False

    def test_the_markdown_omits_a_mixed_stratum_and_keeps_a_pure_one(self, tmp_path):
        pairs = stage14._with_donor_relation(
            _pairs([("luad1", "luad1"), ("luad1", "lgg1"), ("luad1", "luad2")]),
            _slides(tmp_path, MAP))
        md = stage14._as_markdown(score_predictions(_preds(3), pairs), "test")
        assert "out_of_slide__cross_organ" in md
        assert "| `same_slide` |" not in md

    def test_abstaining_everywhere_leaves_the_rate_undefined_not_zero(self, tmp_path):
        """NCC abstained on every out_of_slide item in the lung freeze. A 0.000 there
        would read as a perfect method rather than as a method that never answered."""
        pairs = stage14._with_donor_relation(
            _pairs([("luad1", "lgg1")]), _slides(tmp_path, MAP))
        d = score_predictions(_preds(1, decision="unverifiable"),
                              pairs)["by_donor_relation"]
        v = d["out_of_slide__cross_organ"]
        assert v["coverage"] == 0.0 and v["false_accept_rate"] is None
