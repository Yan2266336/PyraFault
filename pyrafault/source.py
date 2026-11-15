"""GDC source ingestion and integrity audit (guideline §3.4).

Nothing downstream is meaningful if the pixels are not the pixels GDC shipped, so this
module aborts the whole stage on the first inconsistency rather than dropping rows.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


class SourceError(RuntimeError):
    """The downloaded source archive does not match its manifest."""


def resolve_source_file(gdc_dir: str | Path, pattern: str) -> Path:
    """Resolve a configured GDC export name to exactly one file under `gdc_dir`.

    GDC stamps its manifest/sample-sheet/metadata exports with a download
    timestamp, so an exact filename never repeats across cart downloads (spec
    §12). `pattern` may therefore be an exact filename or a glob (e.g.
    `"gdc_manifest.*.txt"`); a literal filename with no glob metacharacters
    matches only itself under `Path.glob`, so both forms share this one lookup.
    Zero matches or more than one is always an error naming what was found —
    never a silent pick of the first.
    """
    gdc_dir = Path(gdc_dir)
    matches = sorted(gdc_dir.glob(pattern))
    if not matches:
        raise SourceError(f"no file in {gdc_dir} matches '{pattern}'")
    if len(matches) > 1:
        names = ", ".join(m.name for m in matches)
        raise SourceError(
            f"pattern '{pattern}' in {gdc_dir} matches {len(matches)} files: {names}"
        )
    return matches[0]


def slide_id_from_filename(name: str) -> str:
    """TCGA slide barcode from a GDC filename, dropping the UUID and extension."""
    return Path(name).name.split(".")[0]


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Load the GDC manifest.

    `filename` is left exactly as GDC published it — that value is provenance, not a
    path, and a stage whose whole job is establishing provenance must not quietly
    discard what the source of truth actually said. `gdc-client` always downloads
    into a `<uuid>/<basename>` directory (the id column is the uuid), so a separate
    `source_relpath` column is added holding that on-disk-relative path; this is the
    name `verify_files` (and `slides.parquet`) use to actually find the file.
    """
    df = pd.read_csv(path, sep="\t", dtype=str)
    df["size"] = df["size"].astype("int64")
    df["source_relpath"] = df["id"] + "/" + df["filename"]
    return df


def load_sample_sheet(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str)


def load_cart_metadata(path: str | Path) -> pd.DataFrame:
    """Parse the GDC cart metadata.cart.*.json export.

    GDC's cart export nests the case UUID under `associated_entities[0].case_id`
    (the format actually produced by the GDC data portal), not a top-level `cases`
    key. Both are checked so an older/alternate export shape is also tolerated,
    but a row missing both is left null and will be caught by `join_sources`,
    which aborts rather than silently keeping an unmatched file.
    """
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    rows = []
    for entry in payload:
        case_uuid = None
        entities = entry.get("associated_entities")
        if entities:
            case_uuid = entities[0].get("case_id")
        if case_uuid is None:
            cases = entry.get("cases")
            if cases:
                case_uuid = cases[0].get("case_id")
        rows.append({
            "file_id": entry.get("file_id"),
            "gdc_case_uuid": case_uuid,
            "experimental_strategy": entry.get("experimental_strategy"),
            "access": entry.get("access"),
        })
    return pd.DataFrame(rows, dtype=object)


def join_sources(
    manifest: pd.DataFrame, sheet: pd.DataFrame, meta: pd.DataFrame
) -> pd.DataFrame:
    """Inner-join the three GDC artifacts on the file UUID, refusing to lose rows."""
    sheet = sheet.rename(columns={
        "File ID": "gdc_file_id", "File Name": "file_name", "Project ID": "project_id",
        "Case ID": "case_id", "Sample ID": "sample_id", "Tissue Type": "tissue_type",
        "Tumor Descriptor": "tumor_descriptor", "Specimen Type": "specimen_type",
        "Preservation Method": "preservation_method",
    })
    m = manifest.rename(columns={"id": "gdc_file_id"})

    a = m.merge(sheet, on="gdc_file_id", how="left", validate="one_to_one")
    lost = a[a["case_id"].isna()]["gdc_file_id"].tolist()
    if lost:
        raise SourceError(f"file UUIDs absent from the sample sheet: {lost}")

    b = a.merge(meta, left_on="gdc_file_id", right_on="file_id", how="left", validate="one_to_one")
    lost = b[b["gdc_case_uuid"].isna()]["gdc_file_id"].tolist()
    if lost:
        raise SourceError(f"file UUIDs absent from the cart metadata: {lost}")

    b["slide_id"] = b["filename"].map(slide_id_from_filename)
    dupes = b["slide_id"][b["slide_id"].duplicated()].tolist()
    if dupes:
        raise SourceError(f"duplicate slide_id after join: {dupes}")
    return b.drop(columns=["file_id"])


def md5_file(path: str | Path, chunk: int = 8 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_file(path: str | Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_files(joined: pd.DataFrame, gdc_dir: str | Path, verify_md5: bool) -> pd.DataFrame:
    """Assert every declared file exists with the declared size and MD5.

    Locates each file via `source_relpath` (the on-disk `<uuid>/<basename>` path),
    not `filename` (GDC's published basename) — one name means one thing end to end,
    matching the `source_relpath` column carried through to `slides.parquet`.
    """
    gdc_dir = Path(gdc_dir)
    out = joined.copy()
    sha = []

    for row in out.itertuples():
        rel = row.source_relpath
        path = gdc_dir / rel
        if not path.is_file():
            raise SourceError(f"{row.slide_id}: missing source file {path}")
        actual = path.stat().st_size
        if actual != int(row.size):
            raise SourceError(
                f"{row.slide_id}: size mismatch — manifest {row.size}, on disk {actual}"
            )
        if verify_md5:
            digest = md5_file(path)
            if digest != row.md5:
                raise SourceError(
                    f"{row.slide_id}: MD5 mismatch — manifest {row.md5}, computed {digest}"
                )
        sha.append(sha256_file(path))

    out["sha256_local"] = sha
    return out
