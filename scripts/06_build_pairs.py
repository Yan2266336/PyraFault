#!/usr/bin/env python
"""Stage 06 — clean, adjacent-drift and random-same-context pairs."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from pyrafault.config import load_config
from pyrafault.degrade_assign import assign_bands
from pyrafault.faults import (
    add_provenance_columns, assert_pair_invariants, assign_severities, make_clean,
    make_drift, make_random,
)
from pyrafault.geom import BBoxL0
from pyrafault.schema import read_table, write_table


def _row(cfg, ctx, q, res, extra=None):
    # Computed, not a literal True: round 1 always has true inside context (D2), but
    # writing the literal instead of the geometry it claims to describe means a future
    # out_of_context pair (round 2, spec §16) would inherit a wrong value with nothing
    # to catch it -- and check_invariants' true_within_context check exists precisely
    # to verify this column against real geometry, which a hardcoded value defeats.
    true = BBoxL0(q.true_bbox_l0_x, q.true_bbox_l0_y, q.true_bbox_l0_w, q.true_bbox_l0_h)
    within = ctx.contains(true)
    base = {
        "pair_id": f"{q.query_id}__{res.fault_type}",
        "slide_id": q.slide_id, "context_id": q.context_id, "query_id": q.query_id,
        **res.claimed.as_dict("claimed_bbox_l0"),
        "true_bbox_l0_x": q.true_bbox_l0_x, "true_bbox_l0_y": q.true_bbox_l0_y,
        "true_bbox_l0_w": q.true_bbox_l0_w, "true_bbox_l0_h": q.true_bbox_l0_h,
        "claim_correct": res.fault_type == "clean",
        "fault_type": res.fault_type, "fault_severity": res.severity,
        "normalized_offset": res.normalized_offset,
        "offset_frac_context": res.offset_frac_context,
        "iou_claim_true": res.iou_claim_true,
        "true_within_context": within, "local_recovery_possible": within,
        "hardness_score": None, "hard_negative_miner": None,
        "miner_model": None, "miner_revision": None, "candidate_rank": None,
        "hard_pool": None,
        "construction_seed": cfg.seed, "pipeline_version": cfg.pipeline_version,
        "split": q.split, "qc_status": "pending",
    }
    base.update(extra or {})
    return base


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/example10.yaml")
    ap.add_argument("--data-root", default="data")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(args.data_root)

    contexts = read_table(root / "01_benchmark" / "contexts.parquet", "contexts")
    queries = read_table(root / "01_benchmark" / "queries.parquet", "queries")
    cands = read_table(root / "01_benchmark" / "query_candidates.parquet", "query_candidates")

    ctx_box = {
        c.context_id: BBoxL0(c.bbox_l0_x, c.bbox_l0_y, c.bbox_l0_w, c.bbox_l0_h)
        for c in contexts.itertuples()
    }

    # I4: exclude candidates stage 05 hard-failed (valid=False, all-zero embedding row)
    # from the random_same_context pool. An untracked exclusion here is latent today
    # (0 hard failures in the pilot) but a hard-failed candidate is not a legitimate
    # "different plausible region" to claim as a mismatch.
    valid_candidate_ids: set[str] = set()
    for slide_id in cands["slide_id"].unique():
        z = np.load(root / "04_features" / f"{slide_id}.npz", allow_pickle=True)
        valid_candidate_ids.update(
            str(cid) for cid, ok in zip(z["candidate_id"], z["valid"]) if ok
        )

    pool: dict[str, list[BBoxL0]] = {}
    for c in cands.itertuples():
        if c.candidate_id not in valid_candidate_ids:
            continue
        pool.setdefault(c.context_id, []).append(
            BBoxL0(c.bbox_l0_x, c.bbox_l0_y, c.bbox_l0_w, c.bbox_l0_h)
        )

    severities = assign_severities(
        list(queries["query_id"]), list(cfg.faults.severity_bands), cfg.seed
    )

    rows, skipped = [], []
    for q in queries.itertuples():
        ctx = ctx_box[q.context_id]
        true = BBoxL0(q.true_bbox_l0_x, q.true_bbox_l0_y, q.true_bbox_l0_w, q.true_bbox_l0_h)
        # One RNG per query, seeded from the global seed and the stable query id via a
        # sha256 digest (not Python's str hash, which is randomised per process unless
        # PYTHONHASHSEED is fixed), so a re-run of a single query reproduces exactly the
        # same fault regardless of process hash-seed randomisation.
        seed_int = int.from_bytes(
            hashlib.sha256(f"{cfg.seed}|{q.query_id}".encode()).digest()[:8], "big"
        )
        rng = np.random.default_rng(seed_int)

        results = [make_clean(true)]
        if cfg.faults.drift_enabled:
            results.append(make_drift(true, ctx, severities[q.query_id], cfg.faults, rng))
        if cfg.faults.random_enabled:
            r = make_random(true, ctx, pool.get(q.context_id, []), cfg.faults, rng)
            if r is None:
                skipped.append(q.query_id)
            else:
                results.append(r)

        for res in results:
            assert_pair_invariants(res.claimed, true, ctx, res.fault_type)
            rows.append(_row(cfg, ctx, q, res))

    df = pd.DataFrame(rows)
    df = add_provenance_columns(df, cfg)
    # Assigned here rather than per row: balance is a property of the cell, not of a
    # row, so it cannot be decided while building one.
    df = assign_bands(df, cfg.seed)
    df["served_pixel_sha256"] = ""  # stage 09 fills what it actually samples
    write_table(df, root / "01_benchmark" / "pairs.parquet", "pairs")

    print(df.groupby("fault_type").size().to_string())
    print(df[df.fault_type == "adjacent_drift"].groupby("fault_severity").size().to_string())
    print(f"total {len(df)} pairs from {len(queries)} queries")
    if skipped:
        print(f"WARNING: {len(skipped)} queries had no valid random-mismatch candidate")


if __name__ == "__main__":
    main()
