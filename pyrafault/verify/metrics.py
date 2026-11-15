"""The shared scorer: predictions plus the benchmark tables in, metrics out.

Reads no verifier code, so a method can be scored by submitting a conforming
predictions file without entering this repository.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pyrafault.geom import BBoxL0, center_distance, iou


def verification_score(rho: float, estimated: BBoxL0 | None, claimed: BBoxL0) -> float:
    """The continuous quantity AUROC and AUPRC are computed on.

    `rho` alone will not do: it is high whenever localisation is discriminative,
    which says nothing about whether the estimate agrees with the claim. Parameter-free
    and monotone in both quantities that should matter.
    """
    if estimated is None:
        return 0.0
    return float(rho) * float(iou(estimated, claimed))


def _rankdata(a: np.ndarray) -> np.ndarray:
    """1-based average ranks, numpy only -- `scipy.stats.rankdata(method="average")`
    without the dependency. Ties share the mean of the ranks they'd occupy sorted."""
    order = np.argsort(a, kind="mergesort")
    sorted_a = a[order]
    ranks = np.empty(len(a), dtype=float)
    i, n = 0, len(a)
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _auroc(scores, labels) -> float | None:
    """AUROC via the rank-sum (Mann-Whitney U) identity:

        AUC = (sum_of_positive_ranks - n_pos*(n_pos+1)/2) / (n_pos * n_neg)

    equivalent to the probability a random positive outscores a random negative, with
    ties (via `_rankdata`'s average ranks) counted as half a win each -- the same
    definition `sklearn.metrics.roc_auc_score` and a full ROC-curve integration give,
    without adding either dependency. `labels` positive is `claim_correct`: a correct
    claim's estimate (independent of the claim) should coincide with it, so
    `verification_score` should read high on clean items and low on faulty ones.

    Degenerate when every label is the same class -- AUROC is then undefined (there is
    no negative, or no positive, to rank against), so this returns `None` rather than a
    number that would read as a real measurement.
    """
    scores = np.asarray(list(scores), dtype=float)
    labels = np.asarray(list(labels), dtype=bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _rankdata(scores)
    sum_ranks_pos = ranks[labels].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _auprc(scores, labels) -> float | None:
    """AUPRC (average precision) via the step-wise precision-recall sum, evaluated at
    each *distinct score threshold* rather than at each item: sort by score descending,
    group tied scores together, and at each group boundary accumulate precision-so-far
    (over everything scoring at or above the threshold) times the recall that whole
    group adds. This is the exact area under the step function the precision-recall
    curve traces -- no interpolation, matching `sklearn.metrics.average_precision_score`'s
    definition, without the dependency.

    Grouping by threshold (rather than by item, as a naive per-item cumulative sum
    would) is what makes this order-invariant on ties: `empirical_reliability`
    quantises to at most 129 distinct values, so on real data most rows share a score
    with many others, and a per-item sum's answer would otherwise depend on the
    arbitrary order those tied rows arrive in.

    Degenerate when every label is the same class (no positives to recall, or no
    negatives to ever misrank ahead of a positive) -- returns `None`, not a number.
    """
    scores = np.asarray(list(scores), dtype=float)
    labels = np.asarray(list(labels), dtype=bool)
    n_pos = int(labels.sum())
    if n_pos == 0 or n_pos == len(labels):
        return None
    order = np.argsort(-scores, kind="mergesort")
    scores_sorted = scores[order]
    labels_sorted = labels[order]
    tp_cum = np.cumsum(labels_sorted)
    fp_cum = np.cumsum(~labels_sorted)
    # True at the last index of each run of equal scores (and always at the last
    # element) -- i.e. exactly where tp_cum/fp_cum reflect the whole tied group, not
    # some arbitrary prefix of it.
    group_end = np.concatenate((np.diff(scores_sorted) != 0, [True]))
    tp = tp_cum[group_end]
    fp = fp_cum[group_end]
    precision = tp / (tp + fp)
    recall = tp / n_pos
    delta_recall = np.diff(np.concatenate(([0.0], recall)))
    return float(np.sum(precision * delta_recall))


def _box(row, prefix) -> BBoxL0 | None:
    if pd.isna(row[f"{prefix}_x"]):
        return None
    return BBoxL0(int(row[f"{prefix}_x"]), int(row[f"{prefix}_y"]),
                  int(row[f"{prefix}_w"]), int(row[f"{prefix}_h"]))


def _balanced_accuracy(pred_positive, actual_positive) -> float | None:
    p = np.asarray(actual_positive, dtype=bool)
    q = np.asarray(pred_positive, dtype=bool)
    if p.sum() == 0 or (~p).sum() == 0:
        return None
    tpr = (q & p).sum() / p.sum()
    tnr = ((~q) & (~p)).sum() / (~p).sum()
    return float((tpr + tnr) / 2)


def _bounded_sample(ids, cap=10) -> list:
    ids = sorted(ids)
    return ids[:cap]


def _localisation(df: pd.DataFrame) -> dict:
    """Median IoU and median normalised centre error over the items that have an answer.

    Restricted to `true_within_context`: an out-of-context pair has no in-context source
    to recover, so including it would measure a question that was not asked. An item
    whose `estimated_bbox` is None is skipped rather than counted as a zero -- the method
    declined to localise it, which `coverage` already records -- and `n_localised` is how
    many items are behind these two numbers, which is not the group's size.

    Factored out so the per-band cut is this same measurement rather than a second
    definition of it.
    """
    inside = df[df["true_within_context"].astype(bool)]
    errs, ious = [], []
    for row in inside.to_dict("records"):
        est = _box(row, "estimated_bbox_l0")
        true = _box(row, "true_bbox_l0")
        if est is None or true is None:
            continue
        errs.append(center_distance(est, true) / true.w)
        ious.append(iou(est, true))
    return {
        "n": len(errs),
        "median_normalised_centre_error": float(np.median(errs)) if errs else None,
        "median_iou": float(np.median(ious)) if ious else None,
    }


def score_predictions(predictions: pd.DataFrame, pairs: pd.DataFrame, item_ids=None) -> dict:
    """Score a predictions file against the benchmark's pairs table.

    `item_ids`, when given, is the authoritative expected item set (the deliberate-subset
    path: score only these items, whether or not `pairs` holds more). When omitted, the
    expected set is `pairs["item_id"]`.

    The merge below is `how="inner"`, and `validate="one_to_one"` only rejects a duplicated
    key -- it says nothing about the two frames' key sets being equal. Left unchecked, a
    predictions file that omits the items a method does worst on merges cleanly, and
    `n_items`/`coverage`/`balanced_accuracy` all come out looking like a complete run:
    coverage already reads as 1.0, so no field in the returned dict would tell a caller
    to look closer. This scorer exists to grade submissions from people who never see this
    repository, which is exactly when a missing-items failure has to be unavoidable rather
    than advisory -- so a mismatched key set raises instead of being folded into the metrics.
    """
    expected = set(pairs["item_id"]) if item_ids is None else set(item_ids)
    submitted = set(predictions["item_id"])
    missing = expected - submitted
    unexpected = submitted - expected
    if missing or unexpected:
        raise ValueError(
            f"predictions item_id set does not match the expected item_id set: "
            f"{len(missing)} missing (e.g. {_bounded_sample(missing)}), "
            f"{len(unexpected)} unexpected (e.g. {_bounded_sample(unexpected)})"
        )

    df = predictions.merge(pairs, on="item_id", how="inner", validate="one_to_one")

    covered = df[df["decision"] != "unverifiable"]
    metrics: dict = {
        "n_items": int(len(df)),
        "coverage": float(len(covered) / len(df)) if len(df) else 0.0,
        "decisions": df["decision"].value_counts().to_dict(),
    }

    metrics["balanced_accuracy"] = (
        _balanced_accuracy(covered["decision"] == "verified", covered["claim_correct"])
        if len(covered) else None
    )

    scores, labels = [], []
    for row in df.to_dict("records"):
        scores.append(verification_score(row["reliability"],
                                         _box(row, "estimated_bbox_l0"),
                                         _box(row, "claimed_bbox_l0")))
        labels.append(bool(row["claim_correct"]))
    metrics["verification_score"] = {
        "mean_clean": float(np.mean([s for s, l in zip(scores, labels) if l])) if any(labels) else None,
        "mean_fault": float(np.mean([s for s, l in zip(scores, labels) if not l])) if not all(labels) else None,
        # The threshold-free view of the same per-item scores: whether the continuous
        # verification_score discriminates claim_correct at all, independent of any
        # tau/delta. Spec S5.2; None in the degenerate single-class case rather than a
        # number that would misread as a real measurement.
        "auroc": _auroc(scores, labels),
        "auprc": _auprc(scores, labels),
    }

    metrics["localisation"] = _localisation(df)

    metrics["by_fault_type"] = {
        ft: {
            "n": int(len(g)),
            "coverage": float((g["decision"] != "unverifiable").mean()),
            "verified_rate": float((g["decision"] == "verified").mean()),
        }
        for ft, g in df.groupby("fault_type")
    }
    # The imaging-degradation dimension. Its acceptance criterion is a statement about
    # per-band numbers ("the curve is monotone in band"), and a method cannot produce
    # them for itself: `degradation_band` is in neither `inputs` (a verifier is never
    # told which band it got) nor `target`, so only the scorer, holding `pairs`, can cut
    # the results this way. Empty for a freeze built before the column existed, and a
    # single-band `{"none": ...}` for the freezes shipped today, all of whose rows are
    # the identity transform. `balanced_accuracy` is None rather than 1.0 for a band
    # holding only one class, the same rule the headline number follows.
    if "degradation_band" in df.columns:
        metrics["by_degradation_band"] = {
            str(band): {
                "n": int(len(g)),
                "coverage": float((g["decision"] != "unverifiable").mean()),
                "verified_rate": float((g["decision"] == "verified").mean()),
                "balanced_accuracy": _balanced_accuracy(
                    g.loc[g["decision"] != "unverifiable", "decision"] == "verified",
                    g.loc[g["decision"] != "unverifiable", "claim_correct"]),
                # Localisation per band, because the pooled median is not comparable
                # between methods when one of them is bimodal. On the degraded sample
                # freeze the NCC control's pooled median IoU (0.905) reads HIGHER than
                # ZoomTrace's (0.801) while its balanced accuracy is 0.500 against 0.988:
                # NCC either nails a patch to the pixel or misses it entirely, and a
                # median over two humps reports the taller hump rather than the method.
                **{f"{k}" if k != "n" else "n_localised": v
                   for k, v in _localisation(g).items()},
            }
            for band, g in df.groupby("degradation_band")
        }
    else:
        metrics["by_degradation_band"] = {}

    # Where the served patch came from, relative to the claim's own slide. Added for the
    # five-cohort freeze, where stage 07 filtered out_of_slide donors by split only and
    # 47.5% of them landed in a different cancer -- 9.44% of the test split is therefore
    # "notice that brain tissue is not endometrium". That is a realistic failure (a
    # pipeline that mixes up slides does not restrict itself to one organ), so the items
    # stay; what cannot stay is pooling them with the rest, because a stratum any method
    # gets right lifts every method's headline together and stops the comparison
    # discriminating.
    #
    # Reported as a false-accept rate rather than a balanced accuracy: every item in the
    # out_of_slide family is a fault, so a stratum holds one class and balanced accuracy
    # and AUROC are both undefined there. `verified_rate` on one class IS the error rate,
    # and the mean verification_score is defined too, so both are given.
    if "donor_relation" in df.columns:
        df = df.assign(_vscore=scores)
        metrics["by_donor_relation"] = {
            str(rel): {
                "n": int(len(g)),
                "coverage": float((g["decision"] != "unverifiable").mean()),
                # Among the items it was willing to decide: how often it wrongly accepted.
                "false_accept_rate": (
                    float((g.loc[g["decision"] != "unverifiable", "decision"]
                           == "verified").mean())
                    if (g["decision"] != "unverifiable").any() else None),
                "mean_verification_score": float(g["_vscore"].mean()),
                "all_faults": bool(not g["claim_correct"].any()),
            }
            for rel, g in df.groupby("donor_relation")
        }
    else:
        metrics["by_donor_relation"] = {}

    # Keyed on (scale_ratio, context_mpp), not scale_ratio alone: two different
    # magnification pairs can share a ratio -- 5x->20x and 10x->40x are both 4.0 -- and
    # collapsing them into one group would average together strata the spec says to
    # keep apart.
    if "scale_ratio" in df.columns and "context_mpp" in df.columns:
        metrics["by_scale_ratio"] = {
            f"{r:.3f}@{mpp:.4f}": {"n": int(len(g)),
                                   "verified_rate": float((g["decision"] == "verified").mean())}
            for (r, mpp), g in df.groupby(["scale_ratio", "context_mpp"])
        }
    elif "scale_ratio" in df.columns:
        metrics["by_scale_ratio"] = {
            f"{r:.3f}": {"n": int(len(g)),
                        "verified_rate": float((g["decision"] == "verified").mean())}
            for r, g in df.groupby("scale_ratio")
        }
    else:
        metrics["by_scale_ratio"] = {}

    return metrics
