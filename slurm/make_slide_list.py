#!/usr/bin/env python
# slurm/make_slide_list.py — one usable slide_id per line, array index = line number - 1.
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
df = pd.read_parquet(root / "01_benchmark" / "slides.parquet")
ok = sorted(df.loc[df["qc_status"] == "ok", "slide_id"])
if not ok:
    raise SystemExit("no slides with qc_status == 'ok' — run stage 01 first")
out = root / "slides.txt"   # in the data root, not next to this script:
# the slide list belongs to one cohort, so two data roots must not share it.
out.write_text("\n".join(ok) + "\n")
print(f"wrote {out} with {len(ok)} slide ids; use --array=0-{len(ok) - 1}")
