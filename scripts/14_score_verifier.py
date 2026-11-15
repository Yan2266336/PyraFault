#!/usr/bin/env python
"""Stage 14 — score a predictions file against the benchmark.

Reads only predictions and the benchmark tables; it imports no verifier, so any method
can be scored by submitting a conforming file.

    python scripts/14_score_verifier.py --predictions data/06_verify/<verifier>/predictions_test.parquet

Predictions are produced one split at a time (stage 13 writes `predictions_<split>.parquet`),
while the archived `pairs.parquet` holds every split. `score_predictions` raises when the
submitted `item_id` set does not equal the expected set, so scoring a per-split file against
the full, unfiltered pairs table raises. Pass `--split` to restrict the expected set to that
split before scoring.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pyrafault.dataset import make_item_id
from pyrafault.verify.metrics import score_predictions


#: TCGA project to organ. Two lung projects, so LUAD and LUSC are the same organ: a
#: patch from the other one is a genuinely hard negative, not a different tissue, and
#: lumping it with brain-vs-endometrium would hide the stratum that actually matters.
ORGAN = {
    "TCGA-LUAD": "lung", "TCGA-LUSC": "lung", "TCGA-LGG": "brain",
    "TCGA-BLCA": "bladder", "TCGA-UCEC": "uterus",
}


def _with_donor_relation(pairs: pd.DataFrame, slides_path) -> pd.DataFrame:
    """Add `donor_relation` so the scorer can cut out_of_slide by the donor's organ.

    Derived here rather than stored in `pairs`: it is a fact about the two slides, and a
    verifier is told neither, so only the scorer can produce it -- the same reason
    `degradation_band` is cut here. A freeze without `slides.parquet` or without
    `true_slide_id` simply gets no column, and `score_predictions` then reports no such
    stratum rather than guessing one.
    """
    if "true_slide_id" not in pairs.columns or not slides_path.exists():
        return pairs
    slides = pd.read_parquet(slides_path, columns=["slide_id", "project_id"])
    proj = {r.slide_id: r.project_id for r in slides.itertuples()}
    pa, pb = pairs["slide_id"].map(proj), pairs["true_slide_id"].map(proj)
    oa, ob = pa.map(ORGAN), pb.map(ORGAN)

    # Four levels, ordered by how different the donor tissue looks, because that is what
    # makes a negative easy. The middle level is not a refinement for its own sake: in
    # the lung freeze every out_of_slide donor is lung, 37% of them from the OTHER lung
    # subtype, so without it that freeze has one stratum and says nothing -- while with
    # it, its LUAD<->LUSC rate becomes the baseline the five-cohort cross-organ rate is
    # read against.
    rel = pd.Series("same_slide", index=pairs.index, dtype=object)
    off = pairs["slide_id"] != pairs["true_slide_id"]
    rel[off & (pa == pb)] = "out_of_slide__same_project"
    rel[off & (pa != pb) & (oa == ob)] = "out_of_slide__same_organ"
    rel[off & (oa != ob)] = "out_of_slide__cross_organ"
    # An unmapped project is named, not folded into cross_organ: a cohort added to
    # ORGAN's blind side would otherwise read as the easy stratum and inflate the gap
    # this table exists to report.
    rel[off & (pa.isna() | pb.isna() | oa.isna() | ob.isna())] = \
        "out_of_slide__unknown_organ"
    return pairs.assign(donor_relation=rel)


def _as_markdown(m: dict, split: str | None) -> str:
    lines = ["# Verifier metrics", "",
             f"- split: {split or 'unrestricted (--split not given)'}",
             f"- items: {m['n_items']}",
             f"- coverage: {m['coverage']:.3f}",
             f"- balanced accuracy: {m['balanced_accuracy']}",
             "", "## By fault type", "",
             "| fault_type | n | coverage | verified rate |", "|---|---|---|---|"]
    for ft, v in sorted(m["by_fault_type"].items()):
        lines.append(f"| `{ft}` | {v['n']} | {v['coverage']:.3f} | {v['verified_rate']:.3f} |")
    bands = m.get("by_degradation_band") or {}
    if bands:
        # Localisation belongs in this table and not only in the headline, because a
        # pooled median IoU is not comparable between methods when one of them is
        # bimodal: the NCC control's pooled median reads higher than ZoomTrace's while
        # its balanced accuracy is 0.500 against 0.988. Band order is the severity
        # order, not alphabetical, since the acceptance criterion is monotonicity in it.
        order = [b for b in ("none", "mild", "moderate", "severe") if b in bands]
        order += [b for b in sorted(bands) if b not in order]
        lines += ["", "## By degradation band", "",
                  "| band | n | coverage | balanced accuracy | median IoU | "
                  "median centre error | n localised |",
                  "|---|---|---|---|---|---|---|"]
        for band in order:
            v = bands[band]
            lines.append(
                f"| `{band}` | {v['n']} | {v['coverage']:.3f} | {_num(v['balanced_accuracy'])} "
                f"| {_num(v.get('median_iou'))} "
                f"| {_num(v.get('median_normalised_centre_error'))} "
                f"| {v.get('n_localised', '—')} |"
            )
    donor = m.get("by_donor_relation") or {}
    cross = donor.get("out_of_slide__cross_organ")
    # The hardest out-of-slide stratum present, as the thing to compare against: in a
    # single-organ freeze that is the same-organ row, and in neither case is it
    # `same_slide`, which is not an out_of_slide item at all.
    same = (donor.get("out_of_slide__same_project")
            or donor.get("out_of_slide__same_organ"))
    if cross or same:
        lines += ["", "## By where the served patch came from", "",
                  "Only the out_of_slide strata. Every row is a fault, so one stratum "
                  "holds one class, a balanced accuracy would be undefined there, and "
                  "the rate at which the method accepted anyway is the number to read. "
                  "The `same_slide` rows are left out on purpose: they mix clean items "
                  "with on-slide faults, where accepting is sometimes correct, so one "
                  "rate would mean two things. `By fault type` above covers them.", "",
                  "| donor | n | coverage | false-accept rate | mean verification score |",
                  "|---|---|---|---|---|"]
        order = ["out_of_slide__same_project", "out_of_slide__same_organ",
                 "out_of_slide__cross_organ", "out_of_slide__unknown_organ"]
        order += [k for k in sorted(donor) if k not in order]
        for k in order:
            v = donor.get(k)
            # `all_faults` is why the rate is printable: it is the scorer saying this
            # stratum has no clean item in it, so `verified_rate` IS the error rate.
            if not v or not v.get("all_faults"):
                continue
            lines.append(
                f"| `{k}` | {v['n']} | {v['coverage']:.3f} "
                f"| {_num(v['false_accept_rate'])} "
                f"| {_num(v['mean_verification_score'])} |")
        if cross and same and cross["false_accept_rate"] is not None \
                and same["false_accept_rate"] is not None:
            # Stated as a sentence because it is the reason the table exists: if the
            # cross-organ stratum is much easier, the pooled headline is lifted by a
            # stratum every method gets right, and the gap says by how much.
            lines += ["", f"Cross-organ is "
                      f"{same['false_accept_rate'] - cross['false_accept_rate']:+.3f} "
                      f"easier in false-accept rate than same-organ "
                      f"({cross['n']} of {cross['n'] + same['n']} out_of_slide items "
                      f"are cross-organ)."]
    return "\n".join(lines) + "\n"


def _num(v) -> str:
    """An em dash for a quantity that does not exist, never a zero.

    A band whose every estimate was `None` has no localisation to report; printing 0.000
    would read as "localised perfectly badly", which is a different claim.
    """
    return "—" if v is None else f"{float(v):.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--pairs", default=None,
                    help="pairs table already carrying item_id; else --data-root")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--split", default=None,
                    help="restrict the expected item_id set to this split of `pairs` "
                         "before scoring. Predictions are produced per split "
                         "(predictions_<split>.parquet); pairs.parquet holds every "
                         "split, and score_predictions raises on a mismatched "
                         "item_id set, so a per-split predictions file needs this "
                         "unless --pairs already names only that split's rows.")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    preds = pd.read_parquet(args.predictions)
    if args.pairs:
        pairs = pd.read_parquet(args.pairs)
    else:
        root = Path(args.data_root)
        pairs = pd.read_parquet(root / "01_benchmark" / "pairs.parquet")
        cfg_seed = args.seed
        if cfg_seed is None:
            from pyrafault.config import load_config
            cfg_seed = load_config(root / "01_benchmark" / "config.yaml").seed
        pairs["item_id"] = [make_item_id(cfg_seed, pid) for pid in pairs["pair_id"]]

    if args.split is not None:
        if "split" not in pairs.columns:
            raise SystemExit(
                f"--split {args.split!r} was given, but the pairs table has no "
                "'split' column to filter by, so it silently would not have "
                "restricted the expected item_id set; pass --pairs pointing at a "
                "table that carries a 'split' column, or drop --split and "
                "pre-filter --pairs yourself"
            )
        pairs = pairs[pairs["split"] == args.split]

    slides_path = (Path(args.pairs).parent if args.pairs
                   else Path(args.data_root) / "01_benchmark") / "slides.parquet"
    pairs = _with_donor_relation(pairs, slides_path)

    metrics = score_predictions(preds, pairs)
    # Named per split. Scoring two splits of one run into the same directory used to
    # overwrite a single metrics.json, and nothing inside it said which split it
    # described -- so the surviving file was whichever was scored last, and reading it
    # as the other one is a silent misreport, not an error. `scored_split` records the
    # same fact inside the file, for a copy that gets moved away from its name.
    metrics = {**metrics, "scored_split": args.split}
    out = Path(args.out_dir or Path(args.predictions).parent)
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.split}" if args.split else ""
    (out / f"metrics{suffix}.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    (out / f"metrics{suffix}.md").write_text(_as_markdown(metrics, args.split))
    print(json.dumps({k: metrics[k] for k in ("n_items", "coverage", "balanced_accuracy")},
                     indent=2))


if __name__ == "__main__":
    main()
