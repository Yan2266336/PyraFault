#!/usr/bin/env python
"""Stage 12 — build the per-context retrieval window bank.

Windows are cropped from the materialised context images, so this needs no WSI access.
One bank serves every query in its context; across the three freezes that is 674 banks
for 3,370 queries.

    python scripts/12_build_window_bank.py --config configs/example10.yaml --data-root data

At full scale this is the expensive stage: one bank per context, dominated by reading the
context out of the slide (about eight seconds each), so a freeze with 7,469 evaluated
contexts is roughly 17 hours in one process. `--splits` and `--shard` are what make that
a SLURM array instead: each shard writes its own files, so the shards need no
coordination, and the build resumes rather than restarting if one hits its wall clock.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow  # noqa: F401  (must precede kornia; see pyrafault/verify/__init__.py)

from pyrafault.config import load_config
from pyrafault.features.registry import get_extractor
from pyrafault.io.read import read_at_mpp
from pyrafault.io.slide import open_slide
from pyrafault.schema import read_table
from pyrafault.verify.bank import bank_dir, window_side, write_bank_manifest
from scripts._bank_builder_entry import build


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/example10.yaml")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--extractor", default=None,
                    help="Which feature extractor to embed the windows with; defaults "
                         "to the config's `visual_sampling.extractor`. Banks live in a "
                         "per-extractor directory and carry a manifest naming it, so "
                         "three extractors' banks coexist and stage 13 refuses to "
                         "retrieve against the wrong set.")
    ap.add_argument("--splits", nargs="*", default=["calibration", "test"],
                    help="Which splits' contexts to build. `dev` is excluded by default: "
                         "it is two thirds of the contexts and no evaluation reads it.")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--no-resume", action="store_true",
                    help="Rebuild banks that already exist at this window side.")
    args = ap.parse_args()
    if not 0 <= args.shard < args.n_shards:
        raise SystemExit(f"--shard {args.shard} out of range for --n-shards {args.n_shards}")

    cfg = load_config(args.config)
    root = Path(args.data_root)
    contexts = read_table(root / "01_benchmark" / "contexts.parquet", "contexts")
    slides = read_table(root / "01_benchmark" / "slides.parquet", "slides")
    relpath = dict(zip(slides["slide_id"], slides["source_relpath"]))
    source_root = Path(cfg.source.root_dir)

    if args.splits:
        if "split" not in contexts.columns:
            raise SystemExit("contexts.parquet has no 'split' column to filter by")
        contexts = contexts[contexts["split"].isin(args.splits)]
    # Sorted before sharding so which context lands in which shard depends on the freeze
    # alone, not on row order -- a shard rerun after a failure must cover the same set.
    contexts = contexts.sort_values("context_id").reset_index(drop=True)
    total = len(contexts)
    if args.n_shards > 1:
        contexts = contexts.iloc[args.shard::args.n_shards]
    print(f"splits={args.splits} shard {args.shard}/{args.n_shards}: "
          f"{len(contexts)} of {total} contexts")

    # Indexed by context_id rather than scanned per call: the linear scan this replaces
    # cost one pass over every row for each context built, which at 24,113 rows and 7,469
    # builds is 180 million comparisons for a lookup a dict does once.
    by_id = {r.context_id: r for r in contexts.itertuples()}

    handles: dict = {}

    def load_image(context_id):
        row = by_id[context_id]
        sid = row.slide_id
        if sid not in handles:
            handles[sid] = open_slide(source_root / relpath[sid], sid)
        from pyrafault.geom import BBoxL0
        img, _ = read_at_mpp(handles[sid],
                             BBoxL0(int(row.bbox_l0_x), int(row.bbox_l0_y),
                                    int(row.bbox_l0_w), int(row.bbox_l0_h)),
                             float(row.target_mpp), int(row.output_size),
                             str(row.extraction_version))
        return img

    extractor_name = args.extractor or cfg.visual_sampling.extractor
    extractor = get_extractor(extractor_name, cfg.features)
    side = window_side(cfg)
    out = bank_dir(root, extractor_name)
    # `try/finally`: this stage has no resume, so a build that dies partway through
    # (a bad slide, an OOM on a large context) must not leak open slide handles --
    # closing them is unconditional, whether `build` returns or raises.
    try:
        n = build(contexts, load_image, extractor, side, out,
                  resume=not args.no_resume)
    finally:
        for h in handles.values():
            h.close()
    # Written after the build, counting what is actually on disk rather than what this
    # shard added: with `--n-shards` the directory is finished by many jobs, and a count
    # of this job's own output would make the last writer's number the manifest's.
    write_bank_manifest(out, extractor=extractor_name,
                        n_banks=len(list(out.glob("*.npz"))))
    print(f"built {n} window banks (side={side}) with extractor {extractor_name} "
          f"under {out}")


if __name__ == "__main__":
    main()
