#!/usr/bin/env python
"""Stage 02 — thumbnail and tissue mask per slide."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from pyrafault.config import load_config
from pyrafault.io.slide import open_slide
from pyrafault.schema import read_table
from pyrafault.tissue import TissueMask, build_mask, thumbnail


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/example10.yaml")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--slides", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(args.data_root)
    slides = read_table(root / "01_benchmark" / "slides.parquet", "slides")
    slides = slides[slides["qc_status"] == "ok"]
    if args.slides:
        slides = slides[slides["slide_id"].isin(args.slides)]

    out_dir = root / "05_qc" / "masks"
    thumb_dir = root / "05_qc" / "thumbnails"
    gdc = Path(cfg.source.root_dir)

    for row in slides.itertuples():
        with open_slide(gdc / row.source_relpath, row.slide_id) as s:
            arr, mpp = thumbnail(s, cfg.thumbnail.target_mpp, cfg.extraction_version)
            mask = build_mask(arr, mpp, cfg.tissue)
            tm = TissueMask(mask, s.meta.width_l0, s.meta.height_l0)
        tm.save(out_dir / f"{row.slide_id}.npz")

        thumb_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(thumb_dir / f"{row.slide_id}_thumb.png")
        overlay = arr.copy()
        overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        Image.fromarray(overlay).save(thumb_dir / f"{row.slide_id}_mask.png")

        print(f"{row.slide_id}: thumb {arr.shape[1]}x{arr.shape[0]} @ {mpp} um/px, "
              f"tissue_fraction={mask.mean():.3f}")


if __name__ == "__main__":
    main()
