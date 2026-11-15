#!/usr/bin/env python
"""Stage 03 — freeze the context coordinate pool, then assign case-level splits.

Runs in two modes:
  --slides ...    per-slide array phase: writes 05_qc/stage03/{slide_id}.parquet shards
  --gather        merges the shards into contexts.parquet + context_candidates.parquet,
                  then computes splits.parquet (which needs global context counts)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pyrafault.config import load_config
from pyrafault.io.read import read_at_mpp
from pyrafault.io.slide import open_slide
from pyrafault.quality import evaluate
from pyrafault.sampling.contexts import candidate_grid, select_with_quality
from pyrafault.sampling.splits import assign_splits
from pyrafault.schema import read_table, write_table
from pyrafault.tissue import TissueMask


def _per_slide(cfg, root: Path, slides: pd.DataFrame) -> None:
    shard_dir = root / "05_qc" / "stage03"
    shard_dir.mkdir(parents=True, exist_ok=True)
    gdc = Path(cfg.source.root_dir)

    for row in slides.itertuples():
        mask = TissueMask.load(root / "05_qc" / "masks" / f"{row.slide_id}.npz")
        cands = candidate_grid(
            row.slide_id, int(row.width_l0), int(row.height_l0), float(row.mpp_x),
            mask, cfg.context,
        )

        with open_slide(gdc / row.source_relpath, row.slide_id) as slide:
            cache: dict[int, object] = {}

            def ev(i: int):
                img, rec = read_at_mpp(
                    slide, cands[i].bbox, cfg.context.target_mpp,
                    cfg.context.size_px, cfg.extraction_version,
                )
                cache[i] = rec
                return evaluate(np.asarray(img), cfg.quality)

            picks = select_with_quality(cands, cfg.context.num_per_slide, ev)

        cand_rows = [{
            "candidate_id": c.candidate_id, "slide_id": c.slide_id,
            **c.bbox.as_dict("bbox_l0"),
            "tissue_fraction": c.tissue_fraction, "quality_flags": [],
            "blur_var": None, "mean_intensity": None,
            "selected": False, "selection_rank": None,
        } for c in cands]

        ctx_rows = []
        for rank, (idx, q) in enumerate(picks):
            c = cands[idx]
            rec = cache[idx]
            cand_rows[idx].update({
                "quality_flags": q.flags, "blur_var": q.blur_var,
                "mean_intensity": q.mean_intensity,
                "selected": True, "selection_rank": rank,
            })
            ctx_rows.append({
                "context_id": f"{row.slide_id}__c{rank:04d}", "slide_id": row.slide_id,
                **c.bbox.as_dict("bbox_l0"),
                "target_mpp": cfg.context.target_mpp, "output_size": cfg.context.size_px,
                "native_level": rec.native_level, "native_downsample": rec.native_downsample,
                "resize_required": rec.resize_required,
                "extraction_version": cfg.extraction_version,
                "tissue_fraction": c.tissue_fraction, "quality_flags": q.flags,
                "blur_var": q.blur_var, "mean_intensity": q.mean_intensity,
                "morphology_summary": None, "self_similarity": None,
                "materialized_path": None, "image_sha256": None,
                "split": None,
            })

        pd.DataFrame(cand_rows).to_parquet(shard_dir / f"{row.slide_id}__cand.parquet")
        pd.DataFrame(ctx_rows).to_parquet(shard_dir / f"{row.slide_id}__ctx.parquet")
        print(f"{row.slide_id}: {len(cands)} eligible -> {len(ctx_rows)} frozen contexts")


def _gather(cfg, root: Path, slides: pd.DataFrame) -> None:
    shard_dir = root / "05_qc" / "stage03"
    cands = pd.concat([pd.read_parquet(p) for p in sorted(shard_dir.glob("*__cand.parquet"))])
    ctxs = pd.concat([pd.read_parquet(p) for p in sorted(shard_dir.glob("*__ctx.parquet"))])

    counts = ctxs.groupby("slide_id").size().to_dict()
    case_to_slides: dict[str, list[str]] = {}
    for row in slides.itertuples():
        case_to_slides.setdefault(row.case_id, []).append(row.slide_id)

    assignment = assign_splits(case_to_slides, counts, cfg.splits, cfg.seed)
    slide_split = {
        sid: assignment[case] for case, sids in case_to_slides.items() for sid in sids
    }
    ctxs["split"] = ctxs["slide_id"].map(slide_split)

    write_table(cands, root / "01_benchmark" / "context_candidates.parquet", "context_candidates")
    write_table(ctxs, root / "01_benchmark" / "contexts.parquet", "contexts")
    write_table(
        pd.DataFrame([
            {"case_id": row.case_id, "slide_id": row.slide_id,
             "split": assignment[row.case_id]}
            for row in slides.itertuples()
        ]),
        root / "01_benchmark" / "splits.parquet", "splits",
    )

    for split in ("dev", "calibration", "test"):
        members = [c for c, s in assignment.items() if s == split]
        total = sum(counts.get(sid, 0) for c in members for sid in case_to_slides[c])
        print(f"{split:12s} {len(members)} cases, {total} contexts: {sorted(members)}")
    print(f"TOTAL {len(ctxs)} contexts from {len(cands)} candidates")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/example10.yaml")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--slides", nargs="*", default=None)
    ap.add_argument("--gather", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(args.data_root)
    slides = read_table(root / "01_benchmark" / "slides.parquet", "slides")
    slides = slides[slides["qc_status"] == "ok"]

    if args.gather:
        _gather(cfg, root, slides)
    else:
        if args.slides:
            slides = slides[slides["slide_id"].isin(args.slides)]
        _per_slide(cfg, root, slides)


if __name__ == "__main__":
    main()
