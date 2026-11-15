#!/usr/bin/env python
"""Stage 04 — query candidate pool with true source bboxes. Pure geometry, no pixel reads."""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import pandas as pd

from pyrafault.config import load_config
from pyrafault.geom import BBoxL0
from pyrafault.sampling.queries import candidate_pool, context_completeness
from pyrafault.schema import read_table, write_table
from pyrafault.tissue import TissueMask


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/example10.yaml")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--slides", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(args.data_root)

    slides = read_table(root / "01_benchmark" / "slides.parquet", "slides")
    mpp = dict(zip(slides["slide_id"], slides["mpp_x"]))
    contexts = read_table(root / "01_benchmark" / "contexts.parquet", "contexts")
    if args.slides:
        contexts = contexts[contexts["slide_id"].isin(args.slides)]

    masks: dict[str, TissueMask] = {}
    rows = []
    for ctx in contexts.itertuples():
        if ctx.slide_id not in masks:
            masks[ctx.slide_id] = TissueMask.load(
                root / "05_qc" / "masks" / f"{ctx.slide_id}.npz"
            )
        box = BBoxL0(ctx.bbox_l0_x, ctx.bbox_l0_y, ctx.bbox_l0_w, ctx.bbox_l0_h)
        pool = candidate_pool(
            ctx.context_id, ctx.slide_id, box, float(mpp[ctx.slide_id]),
            masks[ctx.slide_id], cfg.query, cfg.seed,
        )
        for c in pool:
            rows.append({
                "candidate_id": c.candidate_id, "context_id": c.context_id,
                "slide_id": c.slide_id, **c.bbox.as_dict("bbox_l0"),
                "tissue_fraction": c.tissue_fraction, "quality_flags": [],
                "blur_var": None, "mean_intensity": None,
                "morphology_cluster": None, "selected": False,
            })

    df = pd.DataFrame(rows)
    write_table(df, root / "01_benchmark" / "query_candidates.parquet", "query_candidates")

    # Indexed off every FROZEN context, not off the candidate rows: a context that produced
    # zero candidates has no rows in `df`, so counting `df` alone is blind to exactly the
    # failure this check exists to catch.
    per_ctx = context_completeness(contexts["context_id"], df["context_id"])
    counts = list(per_ctx.values())
    print(f"{len(df)} candidates across {len(per_ctx)} contexts "
          f"(min {min(counts)}, median {int(statistics.median(counts))}, max {max(counts)})")
    thin = {cid: n for cid, n in per_ctx.items() if n < cfg.query.num_per_context}
    if thin:
        print(f"WARNING: {len(thin)} contexts have fewer than "
              f"{cfg.query.num_per_context} candidates: {sorted(thin)}")


if __name__ == "__main__":
    main()
