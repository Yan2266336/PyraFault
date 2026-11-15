#!/usr/bin/env python
"""Stage 23 — package the frozen benchmark for release.

What ships is coordinates, not pixels. A context is a box on a slide at a stated mpp;
a query is a box inside it; a pair is (context, query, claimed box, truth). Anyone with
the same slides and the same reader version regenerates byte-identical images from this,
which is why the release is ~220 MB rather than the ~4 TB the slides occupy.

`slides.parquet` carries each slide's GDC file uuid, filename, md5 and size, so the
exact files can be fetched from the GDC API and verified. We do not redistribute the
slides themselves: TCGA diagnostic images are open access but re-hosting them is a
separate question from using them, and the benchmark does not need us to.

Two things are deliberately separable:

* the **core** tables are what you need to use the benchmark;
* the **candidate pools** are what you need to audit how it was sampled -- 520k context
  candidates and 9.0M query candidates, from which 76,508 and 382,230 were selected.
  They are 190 MB of the 410 MB and nobody running the benchmark needs them, so they go
  in an `audit/` subdirectory a user can skip.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq

CORE = ("slides.parquet", "splits.parquet", "contexts.parquet",
        "queries.parquet", "pairs.parquet", "config.yaml")
AUDIT = ("context_candidates.parquet", "query_candidates.parquet")
THRESHOLDS = ("thresholds.json", "thresholds_withholding.json")


def _digest(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data_pan5")
    ap.add_argument("--verifier", default="zoomtrace_loftr",
                    help="whose calibrated thresholds to ship alongside the tables")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-audit", action="store_true",
                    help="omit the candidate pools (190 MB of the 410 MB)")
    a = ap.parse_args()

    src = Path(a.data_root) / "01_benchmark"
    out = Path(a.out)
    (out / "audit").mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict] = {}
    for name in CORE:
        s = src / name
        if not s.exists():
            raise SystemExit(f"{s} is missing; the freeze is incomplete")
        shutil.copy2(s, out / name)
        e = {"bytes": s.stat().st_size, "sha256": _digest(s)}
        if name.endswith(".parquet"):
            m = pq.ParquetFile(s).metadata
            e |= {"rows": m.num_rows, "columns": m.num_columns}
        manifest[name] = e

    if not a.skip_audit:
        for name in AUDIT:
            s = src / name
            if s.exists():
                shutil.copy2(s, out / "audit" / name)
                m = pq.ParquetFile(s).metadata
                manifest[f"audit/{name}"] = {"bytes": s.stat().st_size,
                                             "sha256": _digest(s),
                                             "rows": m.num_rows,
                                             "columns": m.num_columns}

    tdir = Path(a.data_root) / "06_verify" / a.verifier
    for name in THRESHOLDS:
        s = tdir / name
        if s.exists():
            shutil.copy2(s, out / f"{a.verifier}__{name}")
            manifest[f"{a.verifier}__{name}"] = {"bytes": s.stat().st_size,
                                                 "sha256": _digest(s)}

    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    total = sum(v["bytes"] for v in manifest.values())
    print(f"{len(manifest)} files, {total/1048576:.1f} MB -> {out}")
    for k in sorted(manifest):
        v = manifest[k]
        rows = f"{v['rows']:>12,} rows" if "rows" in v else " " * 17
        print(f"  {k:46s} {v['bytes']/1048576:8.1f} MB {rows}")


if __name__ == "__main__":
    main()
