#!/usr/bin/env python
"""Stage 09 — reconstruction check, overlays, statistics, must-be-zero assertions.

Exit code 1 on any violation. Rows are never deleted; failing rows get qc_status="fail".
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from pyrafault.cache import image_sha256
from pyrafault.config import load_config
from pyrafault.dataset import _degradation, _true_slide, assert_spec_matches_config
from pyrafault.degrade import apply as degrade_apply
from pyrafault.geom import BBoxL0
from pyrafault.io.read import read_at_mpp
from pyrafault.io.slide import open_slide
from pyrafault.qc.overlay import bbox_to_context_px, render_pair
from pyrafault.qc.report import (
    Violation, benchmark_manifest, check_config_provenance, check_invariants,
    check_leakage, check_served_pixels, reconstruction_sample, render_markdown,
    summarize, verify_reconstruction,
)
from pyrafault.schema import read_table, write_table

BOX = ("_x", "_y", "_w", "_h")

# The config sections this stage actually reads, for `check_config_provenance`'s
# per-section scoping. Enumerated by hand from this file's own `cfg.` accesses, which is
# the only place that knows them:
#   seed               -- reconstruction_sample, check_leakage
#   pipeline_version   -- benchmark_manifest
#   extraction_version -- every _check_row_spec, and every read_at_mpp through it
#   source             -- source_root, where the slides are read from
#   context, query     -- the extraction specs every row is checked against
#   faults             -- severity_bands (I3(c))
#   storage, qc        -- require_hashes, the sampling fraction, the overlay count
# `cfg.digest()` is deliberately NOT a dependency: it is recorded verbatim in
# qc_report.md and benchmark_manifest.json as "what ran", never compared here.
#
# This was `sorted(cfg.raw)` -- every top-level section -- which made the scoping a
# no-op at its only call site: declaring everything reproduces the old whole-config
# digest for every section except one ADDED after the archive was frozen. Editing a
# section absent from this list (`features`, `visual_sampling`, `tissue`, ...) must not
# force a rebuild from stage 00 to clear a QC check, which is the entire point of the
# mechanism; and a section this list names but the live config has lost is now a
# violation, which `sorted(cfg.raw)` could not express (a removed section is absent from
# `cfg.raw` too, so it silently left the comparison).
STAGE_SECTIONS = ("context", "extraction_version", "faults", "pipeline_version", "qc",
                  "query", "seed", "source", "storage")


def _box(row, prefix: str) -> BBoxL0:
    return BBoxL0(*(int(row[f"{prefix}{s}"]) for s in BOX))


def _check_row_spec(what, row, cfg_target_mpp, cfg_output_size, cfg_extraction_version):
    """I2: assert the row's own extraction spec matches the live config rather than
    silently reading it from the config as if the row said nothing."""
    target_mpp, output_size, extraction_version = (
        row["target_mpp"], row["output_size"], row["extraction_version"],
    )
    assert_spec_matches_config(what, target_mpp, output_size, extraction_version,
                                cfg_target_mpp, cfg_output_size, cfg_extraction_version)
    return target_mpp, output_size, extraction_version


def _lookup(df, col: str, value):
    """The single row whose `col` equals `value`, or None.

    Never `.iloc[0]` on a possibly-empty frame: a dangling reference is a state
    `check_invariants` has already diagnosed by name, and stage 09 deliberately does
    not exit early on violations -- so raising `IndexError` here would destroy
    qc_report.md, qc_stats.json and qc_violations.parquet for a table whose problem
    was already known, which is the opposite of reporting it.
    """
    hit = df[df[col] == value]
    return None if hit.empty else hit.iloc[0]


def _overlays(cfg, slides, contexts, queries, pairs, source_root: Path, out_dir: Path,
             per_slide: int) -> tuple[int, list[Violation]]:
    """Green ell*, red ell, query beside it — the human check for transposition.

    Returns `(written, violations)`: a row this cannot render (a dangling context,
    query or slide reference) is reported rather than crashing the whole stage, for the
    same reason `_lookup` exists.
    """
    written = 0
    unrendered: list[Violation] = []
    for slide_id, group in pairs.groupby("slide_id", sort=True):
        srow = _lookup(slides, "slide_id", slide_id)
        if srow is None:
            unrendered += [
                Violation("overlay_unrendered", "pairs", row["pair_id"],
                          f"slide_id={slide_id!r} is not in slides.parquet")
                for row in group.to_dict("records")]
            continue
        # Cover every fault_type before covering any of them deeply.
        buckets = defaultdict(list)
        for row in group.sort_values("pair_id").to_dict("records"):
            buckets[row["fault_type"]].append(row)
        chosen, i = [], 0
        while len(chosen) < per_slide and any(len(b) > i for b in buckets.values()):
            for key in sorted(buckets):
                if len(buckets[key]) > i and len(chosen) < per_slide:
                    chosen.append(buckets[key][i])
            i += 1

        dest = out_dir / slide_id
        dest.mkdir(parents=True, exist_ok=True)
        with open_slide(source_root / srow["source_relpath"], slide_id) as slide:
            for row in chosen:
                ctx = _lookup(contexts, "context_id", row["context_id"])
                qrow = _lookup(queries, "query_id", row["query_id"])
                if ctx is None or qrow is None:
                    missing = ("context_id" if ctx is None else "query_id")
                    unrendered.append(Violation(
                        "overlay_unrendered", "pairs", row["pair_id"],
                        f"{missing}={row[missing]!r} has no row to read the overlay's "
                        "extraction spec from"))
                    continue
                ctx_mpp, ctx_size, ctx_ver = _check_row_spec(
                    f"context {ctx['context_id']}", ctx,
                    cfg.context.target_mpp, cfg.context.size_px, cfg.extraction_version)
                q_mpp, q_size, q_ver = _check_row_spec(
                    f"query {qrow['query_id']}", qrow,
                    cfg.query.target_mpp, cfg.query.size_px, cfg.extraction_version)
                ctx_box = _box(ctx, "bbox_l0")
                ctx_img, _ = read_at_mpp(slide, ctx_box, ctx_mpp, ctx_size, ctx_ver)
                # The served patch is not always on this slide: an out-of-slide pair's
                # true_bbox_l0 indexes `true_slide_id`'s frame. Reading it against the
                # open handle here is not a near miss -- it either escapes the canvas
                # (which is how this was found) or, worse, lands inside it and renders a
                # plausible overlay of the wrong tissue. Same reason PyraFaultDataset
                # reads against true_slide_id; this call site was missed when that one
                # was fixed.
                # `_true_slide`, not `row.get(...) or slide_id`: the hand-rolled form
                # treats a NaN `true_slide_id` as truthy, so a cell the loader happily
                # falls back on made this branch look up the slide `"nan"`.
                src_id = _true_slide(row)
                if src_id == slide_id:
                    qry_img, _ = read_at_mpp(slide, _box(row, "true_bbox_l0"),
                                             q_mpp, q_size, q_ver)
                else:
                    src_row = _lookup(slides, "slide_id", src_id)
                    if src_row is None:
                        unrendered.append(Violation(
                            "overlay_unrendered", "pairs", row["pair_id"],
                            f"true_slide_id={src_id!r} is not in slides.parquet"))
                        continue
                    with open_slide(source_root / src_row["source_relpath"], src_id) as src:
                        qry_img, _ = read_at_mpp(src, _box(row, "true_bbox_l0"),
                                                 q_mpp, q_size, q_ver)
                # The row's frozen degradation, replayed exactly as the loader replays
                # it. Without this the montage showed a perfectly registered Q for a
                # 15-degree-rotated row, with nothing on the page saying a rotation was
                # applied -- an overlay is the human check that a freeze is what it
                # claims to be, and this one silently claimed the opposite. The band is
                # in the caption for the same reason: the picture has to be readable as
                # the picture of a degraded pair.
                kind, params = _degradation(row)
                qry_img = degrade_apply(qry_img, kind, params)
                caption = (f"{row['pair_id']}  {row['fault_type']}"
                           f"  sev={row['fault_severity']}"
                           f"  off={row['normalized_offset']}"
                           f"  claim_correct={row['claim_correct']}"
                           f"  degradation={row.get('degradation_band', 'n/a')}"
                           f"/{kind}")
                montage = render_pair(
                    ctx_img, qry_img,
                    bbox_to_context_px(_box(row, "true_bbox_l0"), ctx_box, ctx_size),
                    bbox_to_context_px(_box(row, "claimed_bbox_l0"), ctx_box, ctx_size),
                    caption,
                )
                montage.save(dest / f"{row['pair_id']}__{row['fault_type']}.png")
                written += 1
    return written, unrendered


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/example10.yaml")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--sample-frac", type=float, default=None,
                    help="override qc.reconstruction_sample_frac")
    ap.add_argument("--no-overlays", action="store_true")
    ap.add_argument("--no-reconstruction", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(args.data_root)
    bench = root / "01_benchmark"
    qc_dir = root / "05_qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    source_root = Path(cfg.source.root_dir)

    slides = read_table(bench / "slides.parquet", "slides")
    contexts = read_table(bench / "contexts.parquet", "contexts")
    queries = read_table(bench / "queries.parquet", "queries")
    pairs = read_table(bench / "pairs.parquet", "pairs")
    splits = read_table(bench / "splits.parquet", "splits")

    violations = check_invariants(
        slides, contexts, queries, pairs, splits,
        source_root=source_root,
        require_hashes=cfg.storage.materialize_pilot_png and cfg.qc.verify_materialized_hash,
        severity_bands=cfg.faults.severity_bands,
    )
    print(f"invariants: {len(violations)} violation(s)")

    # The archived config.yaml is supposed to be the frozen record of what built the
    # benchmark. Compare its per-section digests against the ones stage 05 recorded
    # live in feature_manifest.json, and against this run's own live config -- a
    # mismatch on any section this stage cares about means the archive (or the live
    # config) is stale (see task 22 / I7). Scoped to STAGE_SECTIONS rather than the whole
    # config, so a section none of these stages read (a future `degradation:` addition is
    # the motivating example, and this branch is not it -- the bands live in
    # pyrafault.degrade_assign.BAND_SPEC, in code) never forces a rebuild from stage 00
    # just to clear this check -- see pyrafault.qc.report.check_config_provenance's
    # docstring.
    # check_config_provenance runs unconditionally: a missing archived config or a
    # missing manifest is itself a provenance failure (nobody can tell what built
    # this benchmark), not grounds to skip the check and let QC PASS regardless.
    archived_config = bench / "config.yaml"
    feature_manifest = root / "04_features" / "feature_manifest.json"
    config_violations, unchecked_sections = check_config_provenance(
        archived_config, feature_manifest, cfg.section_digests(),
        stage_sections=STAGE_SECTIONS)
    violations += config_violations
    print(f"config provenance: {len(config_violations)} violation(s)")
    if unchecked_sections:
        print(f"config sections not verifiable against this feature manifest: "
              f"{', '.join(unchecked_sections)}")

    # §13.5: pixel-free signals must not beat the base rate at separating clean from
    # not-clean. Fit in-sample and deliberately optimistic -- see check_leakage's
    # docstring before treating this number as anything but an upper bound.
    leakage_violations, leakage_scores = check_leakage(
        contexts, queries, pairs, slides, cfg.query.target_mpp, cfg.query.size_px,
        cfg.seed, cfg.extraction_version)
    violations += leakage_violations
    print("leakage audit (best-threshold balanced accuracy, fit in-sample):")
    for name, score in sorted(leakage_scores.items()):
        print(f"  {name}: {score:.3f}")
    print(f"leakage: {len(leakage_violations)} violation(s)")

    # None (not {"sampled": 0, ...}) when the reconstruction block below never runs at
    # all (--no-reconstruction or qc.verify_materialized_hash off) -- render_markdown
    # only renders a "Served pixels" section when this is not None, so a run that
    # deliberately skipped the whole check is not conflated with one that ran it and
    # sampled/verified nothing.
    served_pixel_summary: dict[str, int] | None = None

    if not args.no_reconstruction and cfg.qc.verify_materialized_hash:
        frac = args.sample_frac if args.sample_frac is not None \
            else cfg.qc.reconstruction_sample_frac
        ctx_sample = reconstruction_sample(contexts, "context_id", frac, cfg.seed)
        qry_sample = reconstruction_sample(queries, "query_id", frac, cfg.seed)
        pair_sample = reconstruction_sample(pairs, "pair_id", frac, cfg.seed)
        print(f"reconstruction: {len(ctx_sample)} contexts + {len(qry_sample)} queries "
              f"+ {len(pair_sample)} pairs (served pixels)")
        for slide_id in sorted(set(ctx_sample["slide_id"]) | set(qry_sample["slide_id"])):
            srow = slides[slides["slide_id"] == slide_id].iloc[0]
            with open_slide(source_root / srow["source_relpath"], slide_id) as slide:
                violations += verify_reconstruction(
                    slide, ctx_sample[ctx_sample["slide_id"] == slide_id].to_dict("records"),
                    "bbox_l0", "context_id", "contexts",
                    cfg.context.target_mpp, cfg.context.size_px, cfg.extraction_version)
                violations += verify_reconstruction(
                    slide, qry_sample[qry_sample["slide_id"] == slide_id].to_dict("records"),
                    "true_bbox_l0", "query_id", "queries",
                    cfg.query.target_mpp, cfg.query.size_px, cfg.extraction_version)
        print(f"after reconstruction: {len(violations)} violation(s)")

        # Task 6: the served-pixel freeze. A pair's served Q is not "this row's own
        # slide re-read at its own spec" the way contexts/queries above are -- it is
        # `true_bbox_l0` read at the *query's* extraction spec, with the row's frozen
        # degradation (a rotation + scale error) replayed on top, exactly
        # `pyrafault.dataset.PyraFaultDataset._image`'s recipe. Sampling this
        # independently of that loader (rather than trusting it) is the point: a
        # collaborator's Pillow/numpy producing a slightly different rotation is
        # precisely what this check exists to catch, at a precision (median IoU 0.99)
        # where it matters.
        qry_by_id = queries.set_index("query_id")

        # Grouped by where the patch actually is, never by pairs.slide_id: an
        # out-of-slide fault type's true_bbox_l0 indexes true_slide_id's frame, and
        # opening the claim's own slide for one of those rows would return a plausible
        # image of the wrong tissue instead of raising (the same hazard _overlays below
        # already guards against).
        pairs_by_source: dict[str, list[dict]] = defaultdict(list)
        for row in pair_sample.to_dict("records"):
            # `_true_slide`, the loader's own resolver, not `row.get(...) or
            # row["slide_id"]`: the latter treats a NaN cell as truthy, so a table the
            # serving path reads happily grouped under the slide `"nan"` here and this
            # loop raised IndexError -- the verification path could not check a table
            # the serving path accepts.
            pairs_by_source[_true_slide(row)].append(row)

        served: list[tuple[dict, Any]] = []
        for src_id in sorted(pairs_by_source):
            srow = _lookup(slides, "slide_id", src_id)
            if srow is None:
                # A dangling slide reference is already a `check_invariants` violation
                # by name; crashing here would destroy qc_report.md, qc_stats.json and
                # qc_violations.parquet for a table whose problem was already diagnosed.
                violations += [
                    Violation("served_pixel_unresolved", "pairs", row["pair_id"],
                              f"slide {src_id!r} holding the served patch is not in "
                              "slides.parquet; served pixels not verified")
                    for row in pairs_by_source[src_id]]
                continue
            with open_slide(source_root / srow["source_relpath"], src_id) as slide:
                for row in pairs_by_source[src_id]:
                    if row["query_id"] not in qry_by_id.index:
                        violations.append(Violation(
                            "served_pixel_unresolved", "pairs", row["pair_id"],
                            f"query_id={row['query_id']!r} is not in queries.parquet; "
                            "served pixels not verified"))
                        continue
                    # q_mpp/q_size/q_ver come from the QUERIES row, not from the pair,
                    # deliberately: that is the lookup `PyraFaultDataset._qry_spec`
                    # performs for the same patch, so this check tracks the spec the
                    # loader would actually serve at rather than a second source for it.
                    qrow = qry_by_id.loc[row["query_id"]]
                    q_mpp, q_size, q_ver = _check_row_spec(
                        f"query {row['query_id']}", qrow,
                        cfg.query.target_mpp, cfg.query.size_px, cfg.extraction_version)
                    # Read against the handle `_true_slide` resolved for this group:
                    # true_bbox_l0 indexes true_slide_id's frame, which is the slide
                    # this loop opened, never necessarily pairs.slide_id.
                    img, _ = read_at_mpp(slide, _box(row, "true_bbox_l0"),
                                         q_mpp, q_size, q_ver)
                    # The pixel recipe below (read_at_mpp -> degrade_apply) is
                    # deliberately stage 09's own; reading the two COLUMNS is not, and
                    # `_degradation` is shared with the loader on purpose -- a NaN
                    # `degradation_kind` that the loader reads as "none" must not become
                    # the kind `"nan"` here and raise mid-loop.
                    kind, params = _degradation(row)
                    served.append((row, degrade_apply(img, kind, params)))

        # A row that already carries a frozen hash is compared now (`check_served_
        # pixels`); a row whose hash has never been computed ("" -- every row, until
        # sampled) gets one written here instead of being flagged, so the *next* run --
        # possibly a collaborator's, on another machine -- has something to compare
        # against. A row sampled twice across two runs is then a real cross-run check.
        # See check_served_pixels's docstring for why an empty hash is normally
        # "unverified" rather than a silent pass: that branch is for a row whose hash
        # SHOULD already be frozen, which a row this run is filling for the first time
        # is not.
        to_check = [(row, img) for row, img in served if row.get("served_pixel_sha256")]
        to_fill = [(row, img) for row, img in served if not row.get("served_pixel_sha256")]
        served_violations = check_served_pixels(to_check)
        violations += served_violations
        if to_fill:
            new_hashes = {row["pair_id"]: image_sha256(img) for row, img in to_fill}
            mapped = pairs["pair_id"].map(new_hashes)
            pairs["served_pixel_sha256"] = mapped.where(mapped.notna(),
                                                        pairs["served_pixel_sha256"])
        # Fix round 1, Finding 1: a `to_fill` row produces no Violation (filling is
        # not a failure) and so, before this, left no trace anywhere the report is
        # persisted -- only in this stdout print, which qc_report.md/qc_stats.json/
        # qc_violations.parquet none of them capture. Right after a freeze is banded
        # every served_pixel_sha256 is "", so `to_check` is empty on that first run and
        # `render_markdown` would otherwise emit "Verdict: PASS" with nothing anywhere
        # in the persisted artefacts saying that zero pairs were actually verified --
        # exactly "a freeze with no hashes reads as fully verified", the failure this
        # whole check exists to prevent. Recorded in `stats` (-> qc_stats.json) and
        # handed to `render_markdown` (-> qc_report.md) so the persisted report, not
        # just this line of stdout, always states the split.
        served_pixel_summary = {
            "sampled": len(served), "compared": len(to_check), "newly_hashed": len(to_fill),
        }
        print(f"served pixels: {len(to_check)} compared, {len(to_fill)} newly hashed, "
              f"{len(served_violations)} violation(s)")

    if not args.no_overlays:
        n, overlay_violations = _overlays(cfg, slides, contexts, queries, pairs,
                                          source_root, qc_dir / "overlays",
                                          cfg.qc.overlays_per_slide)
        violations += overlay_violations
        print(f"overlays: {n} montages under {qc_dir / 'overlays'}, "
              f"{len(overlay_violations)} unrendered")

    stats = summarize(contexts, queries, pairs, splits)
    stats["leakage"] = leakage_scores
    if served_pixel_summary is not None:
        stats["served_pixel_verification"] = served_pixel_summary
    (qc_dir / "qc_report.md").write_text(
        render_markdown(stats, violations, cfg.digest(), unchecked_sections,
                        served_pixel_summary))
    (qc_dir / "qc_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    pd.DataFrame([v.__dict__ for v in violations],
                 columns=["check", "table", "row_id", "detail"]).to_parquet(
        qc_dir / "qc_violations.parquet")

    # Rows are retained, never deleted — a failing row stays auditable.
    failed = {v.row_id for v in violations if v.table == "pairs"}
    pairs["qc_status"] = pairs["pair_id"].map(lambda p: "fail" if p in failed else "pass")
    write_table(pairs, bench / "pairs.parquet", "pairs")

    # I9: checksum the frozen tables now that pairs.parquet carries its final
    # qc_status -- this is the one part of `data/` that gets committed, so a copy of
    # the (gitignored, unversioned) data directory can be verified against the commit
    # that claims to have produced it.
    manifest = benchmark_manifest(bench, cfg.digest(), cfg.pipeline_version)
    # Trailing newline, matching scripts/migrate_pairs_degradation.py and the rest of
    # this repository's JSON writers -- without it the two alternated the file's last
    # byte, so every stage-09 run after a migration showed a one-byte diff.
    (bench / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"benchmark manifest: {len(manifest['tables'])} table(s) -> "
          f"{bench / 'benchmark_manifest.json'}")

    print(f"report: {qc_dir / 'qc_report.md'}")
    if violations:
        print(f"QC FAILED — {len(violations)} violation(s)", file=sys.stderr)
        sys.exit(1)
    print("QC PASSED")


if __name__ == "__main__":
    main()
