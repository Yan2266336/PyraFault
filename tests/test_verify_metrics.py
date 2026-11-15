"""The shared scorer.

Two decisions here are what make two methods comparable, and both would be invisible
if each author made them privately: the continuous score AUROC is computed on, and
what happens to abstentions.
"""
import numpy as np
import pandas as pd
import pytest

from pyrafault.geom import BBoxL0
from pyrafault.verify.metrics import _auprc, _auroc, score_predictions, verification_score

BOX = BBoxL0(1000, 1000, 100, 100)


class TestVerificationScore:
    def test_a_confident_agreeing_estimate_scores_high(self):
        assert verification_score(0.9, BOX, BOX) == pytest.approx(0.9)

    def test_a_confident_disagreeing_estimate_scores_zero(self):
        far = BBoxL0(9000, 9000, 100, 100)
        assert verification_score(0.9, far, BOX) == pytest.approx(0.0)

    def test_reliability_alone_is_not_the_score(self):
        """rho is high when localisation is discriminative, which says nothing about
        whether it agrees with the claim."""
        far = BBoxL0(9000, 9000, 100, 100)
        assert verification_score(0.9, far, BOX) < verification_score(0.2, BOX, BOX)

    def test_declining_to_localise_scores_zero(self):
        assert verification_score(0.9, None, BOX) == 0.0


def _frames(decisions, clean, rho=None):
    n = len(decisions)
    rho = rho or [0.9] * n
    preds = pd.DataFrame({
        "item_id": [f"i{i}" for i in range(n)],
        "estimated_bbox_l0_x": [1000] * n, "estimated_bbox_l0_y": [1000] * n,
        "estimated_bbox_l0_w": [100] * n, "estimated_bbox_l0_h": [100] * n,
        "reliability": rho, "decision": decisions,
    })
    pairs = pd.DataFrame({
        "item_id": [f"i{i}" for i in range(n)],
        "claim_correct": clean,
        "fault_type": ["clean" if c else "adjacent_drift" for c in clean],
        "true_within_context": [True] * n,
        "true_bbox_l0_x": [1000] * n, "true_bbox_l0_y": [1000] * n,
        "true_bbox_l0_w": [100] * n, "true_bbox_l0_h": [100] * n,
        "claimed_bbox_l0_x": [1000] * n, "claimed_bbox_l0_y": [1000] * n,
        "claimed_bbox_l0_w": [100] * n, "claimed_bbox_l0_h": [100] * n,
        "scale_ratio": [4.0] * n, "hardness_score": [0.5] * n,
    })
    return preds, pairs


class TestAbstention:
    def test_abstentions_are_excluded_from_accuracy_and_counted_in_coverage(self):
        """Both classes must survive the abstention, or balanced accuracy is undefined
        rather than perfect -- which is the next test."""
        preds, pairs = _frames(["verified", "mismatched", "unverifiable", "unverifiable"],
                               [True, False, True, False])
        m = score_predictions(preds, pairs)
        assert m["coverage"] == pytest.approx(0.5)
        assert m["balanced_accuracy"] == pytest.approx(1.0)

    def test_accuracy_is_undefined_rather_than_perfect_when_one_class_is_all_abstained(self):
        preds, pairs = _frames(["verified", "unverifiable"], [True, False])
        m = score_predictions(preds, pairs)
        assert m["coverage"] == pytest.approx(0.5)
        assert m["balanced_accuracy"] is None

    def test_abstaining_on_everything_gives_zero_coverage_not_an_error(self):
        preds, pairs = _frames(["unverifiable"] * 4, [True, False, True, False])
        m = score_predictions(preds, pairs)
        assert m["coverage"] == 0.0
        assert m["balanced_accuracy"] is None


class TestVerificationMetrics:
    def test_a_perfect_verifier_scores_one(self):
        preds, pairs = _frames(["verified", "mismatched"], [True, False])
        m = score_predictions(preds, pairs)
        assert m["balanced_accuracy"] == pytest.approx(1.0)

    def test_an_inverted_verifier_scores_zero(self):
        preds, pairs = _frames(["mismatched", "verified"], [True, False])
        m = score_predictions(preds, pairs)
        assert m["balanced_accuracy"] == pytest.approx(0.0)


class TestLocalisation:
    def test_centre_error_is_reported_only_where_the_source_is_in_context(self):
        preds, pairs = _frames(["verified"] * 2, [True, True])
        pairs.loc[1, "true_within_context"] = False
        m = score_predictions(preds, pairs)
        assert m["localisation"]["n"] == 1


class TestStratification:
    def test_metrics_are_broken_out_by_fault_type(self):
        preds, pairs = _frames(["verified", "mismatched"], [True, False])
        m = score_predictions(preds, pairs)
        assert set(m["by_fault_type"]) == {"clean", "adjacent_drift"}

    def test_by_scale_ratio_keys_on_context_mpp_too(self):
        """Finding: 5x->20x and 10x->40x share scale_ratio=4.0 but differ in
        context_mpp -- keying on ratio alone would silently pool two distinct strata
        the spec keeps apart."""
        preds, pairs = _frames(["verified", "verified"], [True, True])
        pairs["context_mpp"] = [2.0, 1.0]  # same ratio (4.0), different context_mpp
        m = score_predictions(preds, pairs)
        assert len(m["by_scale_ratio"]) == 2, (
            "two different (scale_ratio, context_mpp) strata were pooled into one"
        )

    def test_by_scale_ratio_falls_back_to_ratio_alone_without_context_mpp(self):
        preds, pairs = _frames(["verified", "verified"], [True, True])
        m = score_predictions(preds, pairs)
        assert set(m["by_scale_ratio"]) == {"4.000"}


class TestDegradationBands:
    """The spec's acceptance criterion for the imaging-degradation dimension is "the
    curve is monotone in band", which is a statement about per-band numbers. Reporting
    them is the scorer's job: a band is not in `inputs` and not in `target`, so a method
    cannot report them for itself, and every author computing the breakdown privately is
    exactly the situation `by_fault_type` exists to avoid."""

    def test_metrics_are_broken_out_by_band(self):
        preds, pairs = _frames(["verified", "mismatched", "verified", "mismatched"],
                               [True, False, True, False])
        pairs["degradation_band"] = ["none", "none", "severe", "severe"]

        m = score_predictions(preds, pairs)

        assert set(m["by_degradation_band"]) == {"none", "severe"}
        assert m["by_degradation_band"]["none"]["n"] == 2
        assert m["by_degradation_band"]["severe"]["balanced_accuracy"] == pytest.approx(1.0)

    def test_a_band_where_one_class_is_absent_reports_no_accuracy_rather_than_perfect(self):
        preds, pairs = _frames(["verified", "mismatched"], [True, False])
        pairs["degradation_band"] = ["none", "severe"]

        m = score_predictions(preds, pairs)

        assert m["by_degradation_band"]["none"]["balanced_accuracy"] is None
        assert m["by_degradation_band"]["none"]["verified_rate"] == pytest.approx(1.0)

    def test_a_freeze_without_the_column_reports_an_empty_breakdown(self):
        """Every freeze shipped today is entirely `none`; a pre-Task-2 table has no
        column at all and must still score."""
        preds, pairs = _frames(["verified", "mismatched"], [True, False])

        assert score_predictions(preds, pairs)["by_degradation_band"] == {}


class TestDiscrimination:
    """Substantive 6: AUROC/AUPRC over the per-item `verification_score` against
    `claim_correct` -- the threshold-free measurement neither the three-state
    decision nor the two means (mean_clean/mean_fault) provide."""

    def test_auroc_of_a_perfectly_separable_score_is_one(self):
        assert _auroc([0.9, 0.8, 0.3, 0.1], [True, True, False, False]) == pytest.approx(1.0)

    def test_auroc_of_an_inverted_score_is_zero(self):
        assert _auroc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == pytest.approx(0.0)

    def test_auroc_of_ties_between_classes_is_a_half(self):
        assert _auroc([0.5, 0.5, 0.5, 0.5], [True, False, True, False]) == pytest.approx(0.5)

    def test_auroc_is_none_for_a_single_class(self):
        assert _auroc([0.1, 0.2, 0.3], [True, True, True]) is None
        assert _auroc([0.1, 0.2, 0.3], [False, False, False]) is None

    def test_auprc_of_a_perfectly_separable_score_is_one(self):
        assert _auprc([0.9, 0.8, 0.3, 0.1], [True, True, False, False]) == pytest.approx(1.0)

    def test_auprc_matches_the_hand_computed_value_for_an_imperfect_ranking(self):
        # sorted by score: True, False, True, False -- AP = 1.0*0.5 + (2/3)*0.5
        auprc = _auprc([0.9, 0.7, 0.5, 0.3], [True, False, True, False])
        assert auprc == pytest.approx(1.0 * 0.5 + (2 / 3) * 0.5, abs=1e-9)

    def test_auprc_is_none_for_a_single_class(self):
        assert _auprc([0.1, 0.2, 0.3], [True, True, True]) is None
        assert _auprc([0.1, 0.2, 0.3], [False, False, False]) is None

    def test_auprc_of_a_tie_is_order_invariant_and_equals_a_half(self):
        """The tied-score case the review flagged: with one clean and one faulty item
        scoring identically, AUPRC must not depend on which one happens to come first
        in the arrays -- both orderings pool to the same threshold and the same
        answer, 0.5 (one true positive and one false positive tied at the only
        threshold: precision 0.5 at full recall)."""
        assert _auprc([0.5, 0.5], [True, False]) == pytest.approx(0.5)
        assert _auprc([0.5, 0.5], [False, True]) == pytest.approx(0.5)

    def test_auprc_of_a_shuffled_tied_dataset_matches_the_unshuffled_one(self):
        """The property that actually matters on this pipeline: `empirical_reliability`
        quantises to at most 129 distinct values, so real score arrays are full of
        ties. AUPRC must be a function of the (score, label) multiset, not of row
        order -- checked over several permutations with a fixed seed."""
        rng = np.random.default_rng(20260830)
        n = 60
        # Only a handful of distinct scores, guaranteeing heavy ties, mirroring the
        # coarse quantisation `empirical_reliability` produces on real data.
        scores = rng.choice([0.1, 0.3, 0.3, 0.6, 0.6, 0.6, 0.9], size=n).tolist()
        labels = rng.choice([True, False], size=n).tolist()
        baseline = _auprc(scores, labels)
        assert baseline is not None
        for _ in range(20):
            perm = rng.permutation(n)
            shuffled_scores = [scores[i] for i in perm]
            shuffled_labels = [labels[i] for i in perm]
            assert _auprc(shuffled_scores, shuffled_labels) == pytest.approx(baseline)

    def test_auprc_matches_sklearn_on_random_tied_data(self):
        average_precision_score = pytest.importorskip("sklearn.metrics").average_precision_score

        rng = np.random.default_rng(20260830)
        for _ in range(10):
            n = rng.integers(10, 40)
            scores = rng.choice(np.round(rng.uniform(0, 1, size=5), 2), size=n)
            labels = rng.choice([True, False], size=n)
            if labels.all() or not labels.any():
                continue
            assert _auprc(scores.tolist(), labels.tolist()) == pytest.approx(
                float(average_precision_score(labels, scores))
            )

    def test_score_predictions_reports_auroc_and_auprc(self):
        """The continuous score can discriminate the label even when every decision
        collapses to one class -- rho tuned so estimated/claimed/true coincide for
        every row (fixed by `_frames`) but rho itself separates clean from faulty."""
        preds, pairs = _frames(["verified"] * 4, [True, True, False, False],
                               rho=[0.9, 0.8, 0.3, 0.1])
        m = score_predictions(preds, pairs)
        assert m["verification_score"]["auroc"] == pytest.approx(1.0)
        assert m["verification_score"]["auprc"] == pytest.approx(1.0)

    def test_score_predictions_reports_none_when_the_split_has_one_class(self):
        preds, pairs = _frames(["verified", "verified"], [True, True])
        m = score_predictions(preds, pairs)
        assert m["verification_score"]["auroc"] is None
        assert m["verification_score"]["auprc"] is None


class TestItemCoverage:
    """A predictions file that only covers a subset of the benchmark must not merge
    cleanly: `how="inner"` plus `validate="one_to_one"` would otherwise let a submission
    that omits the items it does worst on score as if it were complete."""

    def test_missing_items_raise_and_name_a_missing_id(self):
        preds, pairs = _frames(["verified", "mismatched", "verified"],
                               [True, False, True])
        preds = preds[preds["item_id"] != "i1"]
        with pytest.raises(ValueError, match="i1"):
            score_predictions(preds, pairs)

    def test_unexpected_id_raises_and_names_it(self):
        preds, pairs = _frames(["verified", "mismatched"], [True, False])
        extra = preds.iloc[[0]].copy()
        extra["item_id"] = "stray_item"
        preds = pd.concat([preds, extra], ignore_index=True)
        with pytest.raises(ValueError, match="stray_item"):
            score_predictions(preds, pairs)

    def test_exact_match_still_scores_normally(self):
        preds, pairs = _frames(["verified", "mismatched"], [True, False])
        m = score_predictions(preds, pairs)
        assert m["n_items"] == 2
        assert m["coverage"] == pytest.approx(1.0)

    def test_explicit_item_ids_narrower_than_pairs_scores_the_subset(self):
        preds, pairs = _frames(["verified", "mismatched", "verified"],
                               [True, False, True])
        preds = preds[preds["item_id"] != "i2"]
        m = score_predictions(preds, pairs, item_ids=["i0", "i1"])
        assert m["n_items"] == 2
