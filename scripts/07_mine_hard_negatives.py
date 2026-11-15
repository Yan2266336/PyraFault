#!/usr/bin/env python
"""Stage 07 — one hard-mismatch pair per query, appended to pairs.parquet."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pyrafault.config import load_config
from pyrafault.degrade_assign import assign_bands
from pyrafault.faults import (
    OUT_OF_CONTEXT_TYPES, OUT_OF_SLIDE_TYPES, SUBSTITUTED_TYPES,
    add_provenance_columns, assert_pair_invariants,
)
from pyrafault.geom import BBoxL0, center_distance, iou
from pyrafault.mining import mine_hard, select_out_of_context, select_out_of_slide
from pyrafault.schema import read_table, write_table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/example10.yaml")
    ap.add_argument("--data-root", default="data")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(args.data_root)

    # `hard_mismatch` pools the parent context's own candidates and nothing else, and
    # that is a decision rather than a gap. Widening this pool to the slide would let the
    # CLAIM land outside `C`, and a claim outside `C` is refutable by a bbox containment
    # test with zero visual reasoning -- the hardest fault type would become the easiest
    # (spec decision D2).
    #
    # Slide-wide hard negatives are still built, but on the other side of the record:
    # `faults.out_of_context.selection: hard` holds the claim inside `C` and draws the
    # SERVED patch from the most similar tissue anywhere on the slide. That is the
    # proposal's "same-slide hard substitution", and it is `out_of_context_hard`.
    if cfg.faults.hard_pool != "context":
        raise NotImplementedError(
            f"faults.hard_same_slide.pool={cfg.faults.hard_pool!r} would place the "
            "claimed bbox outside its context, which a verifier can reject without "
            "looking at a single pixel (spec D2). For slide-wide hard negatives set "
            "faults.out_of_context.enabled: true with selection: hard — that keeps the "
            "claim inside the context and substitutes the served patch instead."
        )
    hard_pool_used = "context"

    contexts = read_table(root / "01_benchmark" / "contexts.parquet", "contexts")
    queries = read_table(root / "01_benchmark" / "queries.parquet", "queries")
    cands = read_table(root / "01_benchmark" / "query_candidates.parquet", "query_candidates")
    pairs = read_table(root / "01_benchmark" / "pairs.parquet", "pairs")
    manifest = json.loads((root / "04_features" / "feature_manifest.json").read_text())

    ctx_box = {
        c.context_id: BBoxL0(c.bbox_l0_x, c.bbox_l0_y, c.bbox_l0_w, c.bbox_l0_h)
        for c in contexts.itertuples()
    }

    # candidate_id -> embedding, loaded per slide. I4: z["valid"] is discarded no
    # longer -- a hard-failed candidate (stage 05) gets an all-zero embedding row,
    # which would otherwise win the cosine ranking whenever every genuine similarity
    # is negative. Excluding invalid candidates from the lookup means every downstream
    # `candidate_id in lut` filter (below, and in the hardness backfill) naturally
    # excludes them from both the hard-mismatch pool and the random-pair join.
    emb_by_slide = {}
    for slide_id in queries["slide_id"].unique():
        z = np.load(root / "04_features" / f"{slide_id}.npz", allow_pickle=True)
        valid_mask = z["valid"]
        emb_by_slide[slide_id] = dict(
            zip(z["candidate_id"][valid_mask], z["embedding"][valid_mask])
        )

    by_ctx: dict[str, list] = {}
    for c in cands.itertuples():
        by_ctx.setdefault(c.context_id, []).append(c)

    rows, missing = [], []
    for q in queries.itertuples():
        ctx = ctx_box[q.context_id]
        true = BBoxL0(q.true_bbox_l0_x, q.true_bbox_l0_y, q.true_bbox_l0_w, q.true_bbox_l0_h)
        members = by_ctx.get(q.context_id, [])
        lut = emb_by_slide[q.slide_id]
        members = [m for m in members if m.candidate_id in lut]
        if not members:
            missing.append(q.query_id)
            continue

        boxes = [BBoxL0(m.bbox_l0_x, m.bbox_l0_y, m.bbox_l0_w, m.bbox_l0_h) for m in members]
        pool_emb = np.stack([lut[m.candidate_id] for m in members])
        hits = mine_hard(lut[q.candidate_id], pool_emb, boxes, true, ctx,
                         cfg.faults, cfg.faults.hard_top_k)
        if not hits:
            missing.append(q.query_id)
            continue

        h = hits[0]
        assert_pair_invariants(h.bbox, true, ctx, "hard_mismatch")
        d = center_distance(h.bbox, true)
        rows.append({
            "pair_id": f"{q.query_id}__hard_mismatch",
            "slide_id": q.slide_id, "context_id": q.context_id, "query_id": q.query_id,
            **h.bbox.as_dict("claimed_bbox_l0"),
            "true_bbox_l0_x": true.x, "true_bbox_l0_y": true.y,
            "true_bbox_l0_w": true.w, "true_bbox_l0_h": true.h,
            "claim_correct": False, "fault_type": "hard_mismatch", "fault_severity": None,
            "normalized_offset": d / true.w, "offset_frac_context": d / ctx.w,
            "iou_claim_true": iou(h.bbox, true),
            # Computed, not a literal True -- see scripts/06_build_pairs.py's _row for
            # why: round 1 always has true inside context (D2, assert_pair_invariants
            # enforces it above), but the column should describe the geometry, not
            # assume it, so check_invariants' true_within_context check is meaningful.
            "true_within_context": ctx.contains(true), "local_recovery_possible": ctx.contains(true),
            "hardness_score": h.similarity, "hard_negative_miner": "cosine_topk",
            "miner_model": manifest["name"], "miner_revision": manifest["revision"],
            "candidate_rank": h.rank, "hard_pool": hard_pool_used,
            "construction_seed": cfg.seed, "pipeline_version": cfg.pipeline_version,
            "split": q.split, "qc_status": "pending",
        })

    # Spec §10: hardness_score is defined for hard_mismatch AND random_same_context —
    # both land on grid candidates with stored embeddings, so both can carry a genuine
    # query<->claimed cosine similarity. random_same_context rows were written null by
    # stage 06 (make_random has no embedding access there); backfill them here, where
    # the embeddings are already loaded. The join is a coordinate round-trip: a
    # random_same_context row's claimed_bbox_l0_{x,y,w,h} plus context_id identifies
    # exactly one query_candidates row (verified unique above), which carries the
    # candidate_id needed to look up its embedding.
    # Slide-wide candidate pools, built once and shared by both substitution families:
    # out-of-context filters this slide's pool by geometry, out-of-slide draws from other
    # slides' pools entirely. Building them per family would mean stacking a hundred
    # thousand embeddings twice, and -- worse -- would leave one family's pool defined
    # only when the other happened to be enabled.
    pool_by_slide: dict[str, tuple] = {}
    if cfg.faults.out_of_context_enabled or cfg.faults.out_of_slide_enabled:
        grouped: dict[str, list] = {}
        for c in cands.itertuples():
            grouped.setdefault(c.slide_id, []).append(c)
        for slide_id, members in grouped.items():
            lut = emb_by_slide.get(slide_id, {})
            members = [m for m in members if m.candidate_id in lut]
            if not members:
                continue
            pool_by_slide[slide_id] = (
                np.stack([lut[m.candidate_id] for m in members]),
                [BBoxL0(m.bbox_l0_x, m.bbox_l0_y, m.bbox_l0_w, m.bbox_l0_h)
                 for m in members],
            )

    # ---- out-of-context pairs (proposal table 3 rows 4-5) -------------------------
    # The claim never leaves `C`: it is the query's own requested location, exactly as
    # `clean` claims it. What changes is the patch actually served, whose true source is
    # drawn from the whole slide's candidate pool minus everything touching `C`. The two
    # selection policies are two different sample types, not two settings of one:
    # `random` measures parent-level provenance failure, `hard` measures whether tissue
    # self-similarity across the slide can be told apart from identity.
    ooc_rows, ooc_missing = [], []
    if cfg.faults.out_of_context_enabled:
        policies = {"random": "out_of_context", "hard": "out_of_context_hard"}
        wanted = (["random", "hard"] if cfg.faults.out_of_context_selection == "both"
                  else [cfg.faults.out_of_context_selection])

        for q in queries.itertuples():
            entry = pool_by_slide.get(q.slide_id)
            if entry is None:
                continue
            pool_emb, boxes = entry
            ctx = ctx_box[q.context_id]
            true_q = BBoxL0(q.true_bbox_l0_x, q.true_bbox_l0_y,
                            q.true_bbox_l0_w, q.true_bbox_l0_h)
            min_gap_px = cfg.faults.out_of_context_min_gap * ctx.w
            lut = emb_by_slide[q.slide_id]
            q_emb = lut.get(q.candidate_id)
            if q_emb is None:
                ooc_missing.append(q.query_id)
                continue
            for policy in wanted:
                fault_type = policies[policy]
                # Seeded per (query, policy) so a rerun reproduces the same draw and the
                # two policies never collapse onto the same candidate by accident.
                # sha256, never Python's hash(): hash() is salted per process, so the
                # same query would draw a different candidate on every rerun and the
                # benchmark would not be reproducible (same rule as geom.context_jitter).
                rng = np.random.default_rng(
                    int.from_bytes(
                        hashlib.sha256(
                            f"{cfg.seed}|{q.query_id}|{policy}".encode()
                        ).digest()[:8], "big"
                    )
                ) if policy == "random" else None
                pick = select_out_of_context(
                    q_emb, pool_emb, boxes, ctx, policy, min_gap_px, rng
                )
                if pick is None:
                    ooc_missing.append(f"{q.query_id}/{policy}")
                    continue
                # claimed = the request (the query's own bbox); true = what was served.
                assert_pair_invariants(true_q, pick.bbox, ctx, fault_type)
                d = center_distance(true_q, pick.bbox)
                ooc_rows.append({
                    "pair_id": f"{q.query_id}__{fault_type}",
                    "slide_id": q.slide_id, "context_id": q.context_id,
                    "query_id": q.query_id,
                    **true_q.as_dict("claimed_bbox_l0"),
                    "true_bbox_l0_x": pick.bbox.x, "true_bbox_l0_y": pick.bbox.y,
                    "true_bbox_l0_w": pick.bbox.w, "true_bbox_l0_h": pick.bbox.h,
                    "claim_correct": False, "fault_type": fault_type,
                    "fault_severity": None,
                    "normalized_offset": d / pick.bbox.w,
                    "offset_frac_context": d / ctx.w,
                    "iou_claim_true": iou(true_q, pick.bbox),
                    # Computed from geometry like every other row, not written as a
                    # literal False -- so check_invariants is testing the pipeline
                    # rather than testing that a constant equals itself.
                    "true_within_context": ctx.contains(pick.bbox),
                    "local_recovery_possible": ctx.contains(pick.bbox),
                    "hardness_score": pick.similarity,
                    "hard_negative_miner": "cosine_top1" if policy == "hard" else "uniform",
                    "miner_model": manifest["name"], "miner_revision": manifest["revision"],
                    "candidate_rank": 1 if policy == "hard" else None,
                    "hard_pool": "slide",
                    "construction_seed": cfg.seed, "pipeline_version": cfg.pipeline_version,
                    "split": q.split, "qc_status": "pending",
                })

    # ---- out-of-slide pairs --------------------------------------------------------
    # The tool hands back another slide's pixels entirely -- the cache/concurrency
    # failure the proposal's introduction names. Claim still inside `C`; the true source
    # is on a different slide, so `true_slide_id != slide_id` and `true_bbox_l0` indexes
    # that other slide's frame.
    oos_rows, oos_missing = [], []
    if cfg.faults.out_of_slide_enabled:
        policies = {"random": "out_of_slide", "hard": "out_of_slide_hard"}
        wanted = (["random", "hard"] if cfg.faults.out_of_slide_selection == "both"
                  else [cfg.faults.out_of_slide_selection])

        # Pools keyed by slide, and the split each slide belongs to. Substitutes are
        # drawn only from slides in the SAME split: pulling a dev slide's pixels into a
        # test item would undo the case-level split, which no fault type is licensed to
        # do (spec 8).
        slide_split = dict(zip(queries["slide_id"], queries["split"]))

        for q in queries.itertuples():
            ctx = ctx_box[q.context_id]
            true_q = BBoxL0(q.true_bbox_l0_x, q.true_bbox_l0_y,
                            q.true_bbox_l0_w, q.true_bbox_l0_h)
            q_emb = emb_by_slide.get(q.slide_id, {}).get(q.candidate_id)
            if q_emb is None:
                oos_missing.append(q.query_id)
                continue
            same_split = {s: v for s, v in pool_by_slide.items()
                          if slide_split.get(s) == q.split}
            for policy in wanted:
                fault_type = policies[policy]
                rng = np.random.default_rng(
                    int.from_bytes(
                        hashlib.sha256(
                            f"{cfg.seed}|{q.query_id}|{fault_type}".encode()
                        ).digest()[:8], "big"
                    )
                ) if policy == "random" else None
                pick = select_out_of_slide(q_emb, same_split, q.slide_id, policy, rng)
                if pick is None:
                    oos_missing.append(f"{q.query_id}/{policy}")
                    continue
                assert pick.slide_id != q.slide_id, "out-of-slide drew its own slide"
                assert_pair_invariants(true_q, pick.bbox, ctx, fault_type)
                oos_rows.append({
                    "pair_id": f"{q.query_id}__{fault_type}",
                    "slide_id": q.slide_id, "context_id": q.context_id,
                    "query_id": q.query_id,
                    **true_q.as_dict("claimed_bbox_l0"),
                    "true_bbox_l0_x": pick.bbox.x, "true_bbox_l0_y": pick.bbox.y,
                    "true_bbox_l0_w": pick.bbox.w, "true_bbox_l0_h": pick.bbox.h,
                    "claim_correct": False, "fault_type": fault_type,
                    "fault_severity": None,
                    # Offsets compare boxes in one slide's frame; across slides the
                    # difference has no spatial meaning, so it is null rather than a
                    # number that invites being averaged with the same-slide ones.
                    "normalized_offset": None, "offset_frac_context": None,
                    "iou_claim_true": 0.0,
                    "true_within_context": False, "local_recovery_possible": False,
                    "hardness_score": pick.similarity,
                    "hard_negative_miner": "cosine_top1" if policy == "hard" else "uniform",
                    "miner_model": manifest["name"], "miner_revision": manifest["revision"],
                    "candidate_rank": 1 if policy == "hard" else None,
                    "hard_pool": "cohort",
                    "true_slide_id": pick.slide_id,
                    "construction_seed": cfg.seed, "pipeline_version": cfg.pipeline_version,
                    "split": q.split, "qc_status": "pending",
                })

    cand_by_bbox = {
        (c.context_id, c.bbox_l0_x, c.bbox_l0_y, c.bbox_l0_w, c.bbox_l0_h): c.candidate_id
        for c in cands.itertuples()
    }
    # (context_id, bbox) is one candidate per grid cell per context by construction in
    # stage 04's subgrid; this was previously assumed, never asserted. A future change
    # that broke it would silently drop rows out of the dict without any error.
    assert len(cand_by_bbox) == len(cands), (
        f"(context_id, bbox) join key is not unique: {len(cand_by_bbox)} distinct keys "
        f"for {len(cands)} query_candidates rows"
    )
    query_lut = {
        q.query_id: (q.slide_id, q.candidate_id) for q in queries.itertuples()
    }

    pairs = pairs.copy()
    hardness_backfilled = pairs["hardness_score"].to_dict()
    unmatched_random = []
    for r in pairs.itertuples():
        if r.fault_type != "random_same_context":
            continue
        q_slide, q_cand = query_lut.get(r.query_id, (None, None))
        key = (
            r.context_id, r.claimed_bbox_l0_x, r.claimed_bbox_l0_y,
            r.claimed_bbox_l0_w, r.claimed_bbox_l0_h,
        )
        claimed_cand = cand_by_bbox.get(key)
        q_emb = emb_by_slide.get(q_slide, {}).get(q_cand) if q_slide else None
        c_emb = emb_by_slide.get(q_slide, {}).get(claimed_cand) if q_slide else None
        if claimed_cand is None or q_emb is None or c_emb is None:
            unmatched_random.append(r.pair_id)
            continue
        hardness_backfilled[r.Index] = float(np.asarray(q_emb) @ np.asarray(c_emb))
    pairs["hardness_score"] = pd.Series(hardness_backfilled)

    # Drop every type this stage owns before re-appending, so a rerun replaces its own
    # output instead of duplicating it.
    owned = ("hard_mismatch",) + SUBSTITUTED_TYPES
    keep = pairs[~pairs["fault_type"].isin(owned)]
    new_rows = pd.concat(
        [pd.DataFrame(rows)]
        + ([pd.DataFrame(ooc_rows)] if ooc_rows else [])
        + ([pd.DataFrame(oos_rows)] if oos_rows else []),
        ignore_index=True,
    )
    if len(new_rows):
        # Stage 06 already assigned bands to `keep` (frozen there, on write) --
        # `assign_bands` balances within each (split, fault_type) cell, and every
        # fault type this stage owns is disjoint from stage 06's, so assigning only to
        # `new_rows` here keeps each of those cells balanced on its own without
        # touching a band stage 06 already froze. Re-assigning `keep` on every stage 07
        # run would redraw the parameters behind pixel hashes stage 09 may already have
        # sampled against the first assignment -- the same hazard
        # migrate_pairs_degradation.py refuses to risk.
        new_rows = assign_bands(new_rows, cfg.seed)
        new_rows["served_pixel_sha256"] = ""  # stage 09 fills what it actually samples
    combined = pd.concat(
        [keep, new_rows], ignore_index=True
    ).sort_values("pair_id").reset_index(drop=True)
    combined = add_provenance_columns(combined, cfg)

    write_table(combined, root / "01_benchmark" / "pairs.parquet", "pairs")

    print(combined.groupby("fault_type").size().to_string())
    hs = combined[combined.fault_type == "hard_mismatch"]["hardness_score"]
    print(f"hardness_score: min={hs.min():.3f} median={hs.median():.3f} max={hs.max():.3f}")
    print(f"total {len(combined)} pairs")
    rs = combined[combined.fault_type == "random_same_context"]["hardness_score"]
    print(
        f"random_same_context hardness_score: min={rs.min():.3f} "
        f"median={rs.median():.3f} max={rs.max():.3f}"
    )
    for ft in OUT_OF_CONTEXT_TYPES:
        s = combined[combined.fault_type == ft]["hardness_score"]
        if len(s):
            print(f"{ft} hardness_score: min={s.min():.3f} median={s.median():.3f} "
                  f"max={s.max():.3f}  (n={len(s)})")
    if missing:
        print(f"WARNING: {len(missing)} queries had no eligible hard candidate")
    if ooc_missing:
        print(f"WARNING: {len(ooc_missing)} out-of-context draws found no eligible "
              f"candidate outside the context: {ooc_missing[:5]}"
              f"{' ...' if len(ooc_missing) > 5 else ''}")
    if unmatched_random:
        for pid in unmatched_random:
            print(f"WARNING: random_same_context pair {pid} has no matching candidate_id/embedding")
        print(f"WARNING: {len(unmatched_random)} random_same_context pairs left with null hardness_score")


if __name__ == "__main__":
    main()
