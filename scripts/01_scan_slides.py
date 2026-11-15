#!/usr/bin/env python
"""Stage 01 — read WSI pyramid metadata into slides.parquet."""
from __future__ import annotations

import argparse
from pathlib import Path

from pyrafault.config import load_config
from pyrafault.io.slide import open_slide, scan_status
from pyrafault.schema import read_table, write_table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/example10.yaml")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--slides", nargs="*", default=None,
                    help="restrict to these slide_ids (M1 single-slide debugging)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(args.data_root)
    path = root / "01_benchmark" / "slides.parquet"
    df = read_table(path, "slides")

    targets = set(args.slides) if args.slides else set(df["slide_id"])
    gdc = Path(cfg.source.root_dir)

    for i, row in df.iterrows():
        if row["slide_id"] not in targets:
            continue
        with open_slide(gdc / row["source_relpath"], row["slide_id"]) as s:
            m = s.meta
        status, reason = scan_status(m, cfg.slide.require_mpp, cfg.slide.max_mpp_l0)
        df.at[i, "mpp_x"] = m.mpp_x
        df.at[i, "mpp_y"] = m.mpp_y
        df.at[i, "objective_power"] = m.objective_power
        df.at[i, "vendor"] = m.vendor
        df.at[i, "level_count"] = m.level_count
        df.at[i, "level_dimensions"] = [list(d) for d in m.level_dimensions]
        df.at[i, "level_downsamples"] = list(m.level_downsamples)
        df.at[i, "width_l0"] = m.width_l0
        df.at[i, "height_l0"] = m.height_l0
        df.at[i, "qc_status"] = status
        df.at[i, "exclusion_reason"] = reason
        print(f"{row['slide_id']}: {m.width_l0}x{m.height_l0} mpp={m.mpp_x} "
              f"ds={m.level_downsamples} -> {status}")

    write_table(df, path, "slides")
    n_ok = int((df["qc_status"] == "ok").sum())
    print(f"scanned {len(targets)} slides; {n_ok}/{len(df)} usable")


if __name__ == "__main__":
    main()
