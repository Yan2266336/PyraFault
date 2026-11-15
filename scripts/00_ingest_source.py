#!/usr/bin/env python
"""Stage 00 — ingest adapter: emit the slides.parquet skeleton.

`source.mode` selects the adapter (spec §6 "Stage 00 is an adapter, not the
foundation"):

- `gdc` (default) — validate a GDC download: inner-join manifest x sample sheet
  x cart metadata, audit every file's size and MD5 against what GDC published.
- `directory` — walk an arbitrary directory for OpenSlide-readable files,
  derive `slide_id`/`case_id`, record a locally-computed sha256 baseline.

Either way the only thing that matters to every stage downstream is that
`slides.parquet` ends up with `slide_id`, `case_id` and `source_relpath` filled in.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from pyrafault.config import Config, load_config
from pyrafault.schema import SCHEMAS, write_table
from pyrafault.source import (
    join_sources, load_cart_metadata, load_manifest, load_sample_sheet,
    resolve_source_file, verify_files,
)
from pyrafault.source_directory import build_slides_frame


def _gdc_slides_frame(cfg: Config, root: Path) -> tuple[pd.DataFrame, list[Path]]:
    gdc = Path(cfg.source.gdc_dir)

    manifest_path = resolve_source_file(gdc, cfg.source.manifest)
    sheet_path = resolve_source_file(gdc, cfg.source.sample_sheet)
    meta_path = resolve_source_file(gdc, cfg.source.metadata_json)

    manifest = load_manifest(manifest_path)
    sheet = load_sample_sheet(sheet_path)
    meta = load_cart_metadata(meta_path)
    print(f"manifest={len(manifest)} sheet={len(sheet)} metadata={len(meta)}")

    joined = join_sources(manifest, sheet, meta)
    verified = verify_files(joined, gdc, cfg.source.verify_md5)
    print(f"verified {len(verified)} files (md5={cfg.source.verify_md5})")

    df = pd.DataFrame({
        "slide_id": verified["slide_id"],
        "gdc_file_id": verified["gdc_file_id"],
        "case_id": verified["case_id"],
        "gdc_case_uuid": verified["gdc_case_uuid"],
        "sample_id": verified["sample_id"],
        "source_relpath": verified["source_relpath"],
        "file_name": verified["file_name"],
        "md5_gdc": verified["md5"],
        "sha256_local": verified["sha256_local"],
        "file_size": verified["size"].astype("int64"),
        "project_id": verified["project_id"],
        "tissue_type": verified["tissue_type"],
        "tumor_descriptor": verified["tumor_descriptor"],
        "specimen_type": verified["specimen_type"],
        "preservation_method": verified["preservation_method"],
        "experimental_strategy": verified["experimental_strategy"],
    })
    return df, [manifest_path, sheet_path, meta_path]


def _directory_slides_frame(cfg: Config) -> pd.DataFrame:
    df = build_slides_frame(cfg.source.slides_dir, cfg.source.pattern, cfg.source.case_id_from)
    print(f"discovered {len(df)} slides under {cfg.source.slides_dir} "
          f"matching {cfg.source.pattern!r}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/example10.yaml")
    ap.add_argument("--data-root", default="data")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(args.data_root)

    archive_files: list[Path] = []
    if cfg.source.mode == "gdc":
        df, archive_files = _gdc_slides_frame(cfg, root)
    else:
        df = _directory_slides_frame(cfg)

    # Columns filled by stage 01 (or, in gdc mode, absent from directory mode's
    # discovery); declared here so the schema is complete from the start.
    for col in SCHEMAS["slides"].names:
        if col not in df.columns:
            df[col] = None
    df["qc_status"] = "pending_scan"

    write_table(df, root / "01_benchmark" / "slides.parquet", "slides")

    if archive_files:
        archive = root / "00_source" / "source_metadata"
        archive.mkdir(parents=True, exist_ok=True)
        for path in archive_files:
            shutil.copy2(path, archive / path.name)

    (root / "01_benchmark" / "config.yaml").parent.mkdir(parents=True, exist_ok=True)
    (root / "01_benchmark" / "config.yaml").write_text(
        Path(args.config).read_text(encoding="utf-8"), encoding="utf-8"
    )
    print(f"wrote slides.parquet with {len(df)} rows "
          f"(source.mode={cfg.source.mode}); config digest {cfg.digest()[:12]}")


if __name__ == "__main__":
    main()
