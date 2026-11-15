#!/usr/bin/env python
"""M1 gate — render frozen contexts with their query candidates to verify coordinates."""
from __future__ import annotations

import argparse
from pathlib import Path

from pyrafault.config import load_config
from pyrafault.geom import BBoxL0
from pyrafault.io.read import read_at_mpp
from pyrafault.io.slide import open_slide
from pyrafault.qc.overlay import bbox_to_context_px, render_pair
from pyrafault.schema import read_table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/example10.yaml")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--slide", required=True)
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(args.data_root)

    slides = read_table(root / "01_benchmark" / "slides.parquet", "slides")
    row = slides[slides["slide_id"] == args.slide].iloc[0]
    contexts = read_table(root / "01_benchmark" / "contexts.parquet", "contexts")
    contexts = contexts[contexts["slide_id"] == args.slide].head(args.n)
    cands = read_table(root / "01_benchmark" / "query_candidates.parquet", "query_candidates")

    out_dir = root / "05_qc" / "overlays_m1"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open_slide(Path(cfg.source.root_dir) / row["source_relpath"], args.slide) as slide:
        for ctx in contexts.itertuples():
            cbox = BBoxL0(ctx.bbox_l0_x, ctx.bbox_l0_y, ctx.bbox_l0_w, ctx.bbox_l0_h)
            pool = cands[cands["context_id"] == ctx.context_id]
            if pool.empty:
                print(f"{ctx.context_id}: no candidates, skipped")
                continue
            # pick a candidate away from the centre so a transposition would be obvious
            q = pool.iloc[len(pool) // 3]
            qbox = BBoxL0(q.bbox_l0_x, q.bbox_l0_y, q.bbox_l0_w, q.bbox_l0_h)
            assert cbox.contains(qbox), f"{q.candidate_id} escapes its context"

            cimg, crec = read_at_mpp(slide, cbox, cfg.context.target_mpp,
                                     cfg.context.size_px, cfg.extraction_version)
            qimg, qrec = read_at_mpp(slide, qbox, cfg.query.target_mpp,
                                     cfg.query.size_px, cfg.extraction_version)
            px = bbox_to_context_px(qbox, cbox, cfg.context.size_px)
            cap = (f"{q.candidate_id}  ctx L{crec.native_level}/ds{crec.native_downsample:g}  "
                   f"qry L{qrec.native_level}/ds{qrec.native_downsample:g}  "
                   f"footprint={px[2]-px[0]}px  tissue={q.tissue_fraction:.2f}")
            render_pair(cimg, qimg, px, px, cap).save(out_dir / f"{q.candidate_id}.png")
            print(cap)


if __name__ == "__main__":
    main()
