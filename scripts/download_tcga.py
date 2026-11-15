#!/usr/bin/env python
"""Download the slides named by a GDC manifest, and stage the cart exports beside them.

Stage 00 reads three GDC artifacts -- the manifest, the sample sheet and the cart
metadata -- and inner-joins them on the file UUID, auditing every file's size and MD5
against what GDC published. This script fills the directory that stage expects:

    <gdc_dir>/
      gdc_manifest.<stamp>.txt          copied from --metadata-dir
      gdc_sample_sheet.<stamp>.tsv      copied from --metadata-dir
      metadata.cart.<stamp>.json        copied from --metadata-dir
      <file-uuid>/<file-name>.svs       downloaded here, one directory per file

The `<uuid>/<name>` layout is not a choice: it is what `gdc-client` produces, and
`source_relpath` in `slides.parquet` records that path, so a flat download would make
every freeze built here unverifiable against one built with the official client. Either
tool can fill this directory; `gdc-client download -m <manifest> -d <gdc_dir>` is
equivalent and is the better option for large carts, since it resumes and parallelises.

Only open-access files are reachable this way. TCGA diagnostic slide images are open, so
no token is involved anywhere in this script, and none should be: a controlled-access
cart needs the official client and a GDC token file.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

GDC_DATA = "https://api.gdc.cancer.gov/data"
CHUNK = 8 << 20


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def one_metadata_file(metadata_dir: Path, pattern: str) -> Path:
    hits = sorted(metadata_dir.glob(pattern))
    if len(hits) != 1:
        raise SystemExit(f"{metadata_dir}: expected exactly one {pattern}, found {len(hits)}")
    return hits[0]


def download(file_id: str, dest: Path, expect_size: int, expect_md5: str,
             *, verify: bool) -> str:
    """Fetch one file, or report why it was skipped.

    A file already on disk with the published size is left alone, so an interrupted run
    resumes by re-invocation. The size check is what makes that safe to do without
    hashing gigabytes on every run; `--verify-md5` adds the stronger check.
    """
    if dest.exists() and dest.stat().st_size == expect_size:
        if not verify or md5_file(dest) == expect_md5:
            return "present"
        dest.unlink()

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(f"{GDC_DATA}/{file_id}", timeout=120) as resp, \
                open(partial, "wb") as out:
            shutil.copyfileobj(resp, out, CHUNK)
    except urllib.error.URLError as exc:
        partial.unlink(missing_ok=True)
        raise SystemExit(f"{file_id}: download failed: {exc}") from exc

    actual = partial.stat().st_size
    if actual != expect_size:
        partial.unlink()
        raise SystemExit(f"{file_id}: GDC published {expect_size} bytes, received {actual}")
    if verify:
        digest = md5_file(partial)
        if digest != expect_md5:
            partial.unlink()
            raise SystemExit(f"{file_id}: MD5 mismatch -- published {expect_md5}, got {digest}")
    partial.replace(dest)
    return "downloaded"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata-dir", default="examples/gdc_metadata",
                    help="directory holding the manifest, sample sheet and cart metadata")
    ap.add_argument("--gdc-dir", default="data/00_source/gdc_download",
                    help="where to download; must match `source.gdc_dir` in the config")
    ap.add_argument("--verify-md5", action="store_true",
                    help="hash every file after download. Stage 00 does this anyway when "
                         "`source.verify_md5` is true, so it is off here by default")
    ap.add_argument("--limit", type=int, default=None,
                    help="download only the first N files, for a smaller trial")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    metadata_dir, gdc_dir = Path(args.metadata_dir), Path(args.gdc_dir)
    manifest_path = one_metadata_file(metadata_dir, "gdc_manifest.*.txt")
    sheet_path = one_metadata_file(metadata_dir, "gdc_sample_sheet.*.tsv")
    cart_path = one_metadata_file(metadata_dir, "metadata.cart.*.json")

    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str)
    manifest["size"] = manifest["size"].astype("int64")
    if args.limit is not None:
        manifest = manifest.head(args.limit)
    total = int(manifest["size"].sum())
    print(f"{len(manifest)} file(s), {total / 1e9:.2f} GB -> {gdc_dir}")

    if args.dry_run:
        for row in manifest.itertuples():
            print(f"  {row.id}/{row.filename}  {row.size / 1e9:.2f} GB")
        return

    gdc_dir.mkdir(parents=True, exist_ok=True)
    for src in (manifest_path, sheet_path, cart_path):
        shutil.copy2(src, gdc_dir / src.name)
    print(f"staged {manifest_path.name}, {sheet_path.name}, {cart_path.name}")

    done = 0
    for n, row in enumerate(manifest.itertuples(), 1):
        dest = gdc_dir / row.id / row.filename
        state = download(row.id, dest, int(row.size), row.md5, verify=args.verify_md5)
        done += int(row.size)
        print(f"[{n}/{len(manifest)}] {state:10s} {row.filename}  "
              f"({done / 1e9:.2f}/{total / 1e9:.2f} GB)", flush=True)
    print(f"done. Next: python scripts/00_ingest_source.py "
          f"--config configs/example10.yaml --data-root data")


if __name__ == "__main__":
    sys.exit(main())
