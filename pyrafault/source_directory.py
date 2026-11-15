"""Directory-mode ingest adapter (spec §6 "Stage 00 is an adapter, not the
foundation"). This is the second `source.mode`, for cohorts that are not a GDC
cart download: it walks a directory for slide files, derives `slide_id` from
each filename, resolves `case_id` by a configured rule, and hands stage 00 a
frame with the same shape `pyrafault.source.verify_files` produces for `mode:
gdc` — everything downstream only ever reads `slide_id`, `case_id` and
`source_relpath`.

**Integrity, honestly.** `mode: gdc` verifies each file's size and MD5 against
what GDC published — an external source of truth. Directory mode has no such
source: there is no manifest to check against, only the files themselves. What
this module records — `sha256_local` and `file_size` — establishes a *baseline*
for detecting later change (re-run stage 00 and compare), not a check against a
publisher's record. Callers must not present the two as equivalent assurance.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from pyrafault.source import SourceError, sha256_file


def discover_slides(slides_dir: str | Path, pattern: str) -> list[Path]:
    """Recursively find files under `slides_dir` matching `pattern`.

    `pattern` is a glob relative to `slides_dir` (e.g. `"*.svs"`, `"**/*.ndpi"`).
    `Path.rglob` already searches every depth, so a bare `"*.svs"` finds slides
    nested in per-case subdirectories without an explicit `"**/"` prefix — though
    one is accepted too. Zero matches is always an error naming the directory and
    pattern searched, never a silent empty benchmark.
    """
    slides_dir = Path(slides_dir)
    matches = sorted(p for p in slides_dir.rglob(pattern) if p.is_file())
    if not matches:
        raise SourceError(f"no files under {slides_dir} match pattern {pattern!r}")
    return matches


def slide_id_from_path(path: str | Path) -> str:
    """`slide_id` is the filename with its final extension stripped.

    Unlike GDC's `source.slide_id_from_filename`, there is no UUID segment to
    also remove — a directory-mode filename is whatever the site that produced
    the cohort named it.
    """
    return Path(path).stem


def _case_ids_from_csv(slide_ids: list[str], csv_path: str | Path) -> dict[str, str]:
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise SourceError(f"case_id_from CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str)
    missing_cols = {"slide_id", "case_id"} - set(df.columns)
    if missing_cols:
        raise SourceError(
            f"case_id_from CSV {csv_path} is missing column(s) {sorted(missing_cols)}; "
            "it must have slide_id,case_id columns"
        )
    mapping = dict(zip(df["slide_id"], df["case_id"]))
    missing = [s for s in slide_ids if s not in mapping]
    if missing:
        raise SourceError(
            f"slide(s) discovered on disk but missing from case_id_from CSV "
            f"{csv_path}: {missing}"
        )
    return {s: mapping[s] for s in slide_ids}


def _case_ids_from_regex(slide_ids: list[str], pattern: str) -> dict[str, str]:
    try:
        rx = re.compile(pattern)
    except re.error as e:
        raise SourceError(f"case_id_from regex {pattern!r} is invalid: {e}") from e
    if rx.groups != 1:
        raise SourceError(
            f"case_id_from regex {pattern!r} must have exactly one capture group, "
            f"got {rx.groups}"
        )
    resolved: dict[str, str] = {}
    unmatched = []
    for slide_id in slide_ids:
        m = rx.search(slide_id)
        if m is None:
            unmatched.append(slide_id)
            continue
        resolved[slide_id] = m.group(1)
    if unmatched:
        raise SourceError(
            f"case_id_from regex {pattern!r} did not match slide_id(s): {unmatched}"
        )
    return resolved


def resolve_case_ids(slide_ids: list[str], case_id_from: str) -> dict[str, str]:
    """Resolve `slide_id -> case_id` per the configured rule (spec §6, three forms):

    - `"slide_id"` — one slide per patient; `case_id` copies `slide_id`.
    - a string ending in `.csv` — a path to a CSV with `slide_id,case_id` columns.
      A discovered slide missing from the mapping is an error naming it.
    - anything else — a regex with exactly one capture group, applied to
      `slide_id`. A slide the pattern does not match is an error naming the
      slide, never a silent fallback to `slide_id`.

    Splits are case-level (spec §8/§12), so a wrong `case_id` here silently lets
    one patient's slides land in both `dev` and `test`; every failure path below
    names the offending slide so that cannot happen unnoticed.
    """
    if case_id_from == "slide_id":
        return {s: s for s in slide_ids}
    if case_id_from.endswith(".csv"):
        return _case_ids_from_csv(slide_ids, case_id_from)
    return _case_ids_from_regex(slide_ids, case_id_from)


def build_slides_frame(slides_dir: str | Path, pattern: str, case_id_from: str) -> pd.DataFrame:
    """Discover slides under `slides_dir` and build the adapter-specific columns
    of `slides.parquet`: `slide_id`, `case_id`, `source_relpath`, `file_name`,
    `sha256_local`, `file_size`. The caller (stage 00) fills in the remaining
    schema columns as null, exactly as `mode: gdc` does for the columns it
    cannot fill until stage 01.
    """
    slides_dir = Path(slides_dir)
    paths = discover_slides(slides_dir, pattern)

    slide_ids = [slide_id_from_path(p) for p in paths]
    dupes = sorted({s for s in slide_ids if slide_ids.count(s) > 1})
    if dupes:
        raise SourceError(
            f"duplicate slide_id derived from filenames under {slides_dir}: {dupes}"
        )

    case_ids = resolve_case_ids(slide_ids, case_id_from)

    rows = []
    for path, slide_id in zip(paths, slide_ids):
        rows.append({
            "slide_id": slide_id,
            "case_id": case_ids[slide_id],
            "source_relpath": path.relative_to(slides_dir).as_posix(),
            "file_name": path.name,
            "sha256_local": sha256_file(path),
            "file_size": path.stat().st_size,
        })
    return pd.DataFrame(rows)
