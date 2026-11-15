"""Every invariant must be provably detected, not merely documented."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from pyrafault.degrade_assign import BANDS, assign_bands
from pyrafault.geom import centred_inset
from pyrafault.qc.report import (
    Violation, _dupes, _flags, benchmark_manifest, check_config_provenance,
    check_invariants, check_leakage, reconstruction_sample, render_markdown, summarize,
    verify_reconstruction,
)

SLIDE_W = SLIDE_H = 100_000
MPP_X = 0.2520
CTX_FOV, Q_FOV = 16254, 1016   # fov_l0(2048, 2.0, MPP_X), fov_l0(512, 0.5, MPP_X)


def _frames(tmp_path: Path):
    """One slide, one context, two queries, two pairs — all valid."""
    (tmp_path / "S1.svs").write_bytes(b"stub")
    slides = pd.DataFrame([{
        "slide_id": "S1", "case_id": "C1", "source_relpath": "S1.svs",
        "width_l0": SLIDE_W, "height_l0": SLIDE_H, "qc_status": "pass",
        "mpp_x": MPP_X,
    }])
    contexts = pd.DataFrame([{
        "context_id": "S1_ctx000", "slide_id": "S1",
        "bbox_l0_x": 0, "bbox_l0_y": 0, "bbox_l0_w": CTX_FOV, "bbox_l0_h": CTX_FOV,
        "target_mpp": 2.0, "output_size": 2048,
        "split": "dev", "tissue_fraction": 0.9, "quality_flags": "",
        "self_similarity": 0.3, "morphology_summary": "{}",
        "image_sha256": "a" * 64, "materialized_path": "ctx.png",
    }])
    queries = pd.DataFrame([{
        "query_id": f"S1_ctx000_q{i}", "context_id": "S1_ctx000", "slide_id": "S1",
        "true_bbox_l0_x": 2000 + i * Q_FOV, "true_bbox_l0_y": 2000,
        "true_bbox_l0_w": Q_FOV, "true_bbox_l0_h": Q_FOV,
        "target_mpp": 0.5, "output_size": 512,
        "split": "dev", "tissue_fraction": 0.95, "quality_flags": "",
        "morphology_cluster": i, "self_similarity": 0.4,
        "image_sha256": "b" * 64, "materialized_path": f"q{i}.png",
    } for i in range(2)])
    pairs = pd.DataFrame([
        {"pair_id": "S1_p000", "slide_id": "S1", "context_id": "S1_ctx000",
         "query_id": "S1_ctx000_q0",
         "claimed_bbox_l0_x": 2000, "claimed_bbox_l0_y": 2000,
         "claimed_bbox_l0_w": Q_FOV, "claimed_bbox_l0_h": Q_FOV,
         "true_bbox_l0_x": 2000, "true_bbox_l0_y": 2000,
         "true_bbox_l0_w": Q_FOV, "true_bbox_l0_h": Q_FOV,
         "claim_correct": True, "fault_type": "clean", "fault_severity": None,
         "normalized_offset": 0.0, "iou_claim_true": 1.0,
         "true_within_context": True, "local_recovery_possible": True,
         "hardness_score": None, "split": "dev", "qc_status": "pass"},
        {"pair_id": "S1_p001", "slide_id": "S1", "context_id": "S1_ctx000",
         "query_id": "S1_ctx000_q1",
         "claimed_bbox_l0_x": 3016 + 2 * Q_FOV, "claimed_bbox_l0_y": 2000,
         "claimed_bbox_l0_w": Q_FOV, "claimed_bbox_l0_h": Q_FOV,
         "true_bbox_l0_x": 3016, "true_bbox_l0_y": 2000,
         "true_bbox_l0_w": Q_FOV, "true_bbox_l0_h": Q_FOV,
         "claim_correct": False, "fault_type": "adjacent_drift", "fault_severity": "large",
         "normalized_offset": 2.0, "iou_claim_true": 0.0,
         "true_within_context": True, "local_recovery_possible": True,
         "hardness_score": None, "split": "dev", "qc_status": "pass"},
    ])
    splits = pd.DataFrame([{"case_id": "C1", "slide_id": "S1", "split": "dev"}])
    return slides, contexts, queries, pairs, splits


def _checks(v: list[Violation]) -> set[str]:
    return {x.check for x in v}


def test_a_consistent_benchmark_has_zero_violations(tmp_path):
    assert check_invariants(*_frames(tmp_path), source_root=tmp_path) == []


def test_duplicate_pair_id_is_caught(tmp_path):
    s, c, q, p, sp = _frames(tmp_path)
    p = pd.concat([p, p.iloc[[0]]], ignore_index=True)
    assert "duplicate_id" in _checks(check_invariants(s, c, q, p, sp, source_root=tmp_path))


def test_out_of_bounds_context_is_caught(tmp_path):
    s, c, q, p, sp = _frames(tmp_path)
    c.loc[0, "bbox_l0_x"] = SLIDE_W - 10
    assert "out_of_bounds" in _checks(check_invariants(s, c, q, p, sp, source_root=tmp_path))


def test_missing_source_file_is_caught(tmp_path):
    s, c, q, p, sp = _frames(tmp_path)
    (tmp_path / "S1.svs").unlink()
    assert "missing_source" in _checks(check_invariants(s, c, q, p, sp, source_root=tmp_path))


def test_dangling_query_reference_is_caught(tmp_path):
    s, c, q, p, sp = _frames(tmp_path)
    p.loc[0, "query_id"] = "S1_ctx000_q99"
    assert "dangling_reference" in _checks(check_invariants(s, c, q, p, sp, source_root=tmp_path))


def test_clean_pair_with_a_displaced_claim_is_caught(tmp_path):
    s, c, q, p, sp = _frames(tmp_path)
    p.loc[0, "claimed_bbox_l0_x"] = 9000          # still claim_correct=True, but ell != ell*
    assert "claim_correct" in _checks(check_invariants(s, c, q, p, sp, source_root=tmp_path))


def test_fault_pair_labelled_correct_is_caught(tmp_path):
    s, c, q, p, sp = _frames(tmp_path)
    p.loc[1, "claim_correct"] = True
    assert "claim_correct" in _checks(check_invariants(s, c, q, p, sp, source_root=tmp_path))


def test_query_outside_its_context_is_caught(tmp_path):
    s, c, q, p, sp = _frames(tmp_path)
    q.loc[0, "true_bbox_l0_x"] = CTX_FOV + 5000
    p.loc[0, ["claimed_bbox_l0_x", "true_bbox_l0_x"]] = CTX_FOV + 5000
    assert "query_outside_context" in _checks(
        check_invariants(s, c, q, p, sp, source_root=tmp_path))


def test_claimed_location_outside_its_context_is_caught(tmp_path):
    """D2: ell must lie inside C for every fault type in round 1."""
    s, c, q, p, sp = _frames(tmp_path)
    p.loc[1, "claimed_bbox_l0_x"] = CTX_FOV - 100      # straddles the context edge
    assert "claim_outside_context" in _checks(
        check_invariants(s, c, q, p, sp, source_root=tmp_path))


def test_split_leakage_is_caught(tmp_path):
    s, c, q, p, sp = _frames(tmp_path)
    p.loc[0, "split"] = "test"
    assert "split_mismatch" in _checks(check_invariants(s, c, q, p, sp, source_root=tmp_path))


def test_true_within_context_flag_must_match_geometry(tmp_path):
    s, c, q, p, sp = _frames(tmp_path)
    p.loc[0, "true_within_context"] = False
    assert "true_within_context" in _checks(
        check_invariants(s, c, q, p, sp, source_root=tmp_path))


def test_missing_materialized_hash_is_caught(tmp_path):
    s, c, q, p, sp = _frames(tmp_path)
    q.loc[0, "image_sha256"] = None
    assert "missing_hash" in _checks(
        check_invariants(s, c, q, p, sp, source_root=tmp_path, require_hashes=True))


def test_hashes_are_not_required_unless_asked(tmp_path):
    s, c, q, p, sp = _frames(tmp_path)
    q.loc[0, "image_sha256"] = None
    assert check_invariants(s, c, q, p, sp, source_root=tmp_path) == []


# ------------------------------------------------------- the degradation columns


def _banded(frames, band="severe", kind="geometry", params='{"angle_deg": -15.0}'):
    """`_frames` plus consistent degradation columns on every pair -- the shape a
    migrated freeze is in. `_frames` itself deliberately has no degradation columns:
    a freeze built before them must still pass QC unchanged."""
    s, c, q, p, sp = frames
    p["degradation_band"] = band
    p["degradation_kind"] = kind
    p["degradation_params"] = params
    return s, c, q, p, sp


def test_consistent_degradation_columns_are_not_a_violation(tmp_path):
    assert check_invariants(*_banded(_frames(tmp_path)), source_root=tmp_path) == []


def test_a_table_without_the_degradation_columns_is_not_faulted(tmp_path):
    """The check must not turn every pre-Task-2 freeze into a QC failure."""
    s, c, q, p, sp = _frames(tmp_path)
    assert "degradation_band" not in p.columns
    assert check_invariants(s, c, q, p, sp, source_root=tmp_path) == []


def test_a_band_whose_angle_belongs_to_another_band_is_a_violation(tmp_path):
    """The counterpart of `severity_band_mismatch`: the spec's acceptance criterion
    ("the curve is monotone in band") is computed on this label, so a row labelled
    `severe` while carrying mild's 5 degrees moves a point between curves with every
    other artefact still self-consistent."""
    s, c, q, p, sp = _banded(_frames(tmp_path), params='{"angle_deg": 5.0}')
    assert "degradation_band_mismatch" in _checks(
        check_invariants(s, c, q, p, sp, source_root=tmp_path))


def test_a_none_band_carrying_a_rotation_is_a_violation(tmp_path):
    """`none` is the in-freeze control. A control that is silently rotated is worse
    than no control."""
    s, c, q, p, sp = _banded(_frames(tmp_path), band="none", kind="geometry",
                             params='{"angle_deg": 15.0}')
    assert "degradation_band_mismatch" in _checks(
        check_invariants(s, c, q, p, sp, source_root=tmp_path))


def test_a_none_band_with_no_parameters_is_clean(tmp_path):
    s, c, q, p, sp = _banded(_frames(tmp_path), band="none", kind="none", params="{}")
    assert check_invariants(s, c, q, p, sp, source_root=tmp_path) == []


def test_an_unknown_degradation_kind_is_a_violation(tmp_path):
    """`pyrafault.degrade.apply` raises on it at read time -- which means the freeze
    loads and then dies mid-evaluation, instead of failing QC where it was frozen."""
    s, c, q, p, sp = _banded(_frames(tmp_path), kind="rotate_a_lot")
    assert "degradation_kind_unknown" in _checks(
        check_invariants(s, c, q, p, sp, source_root=tmp_path))


def test_unparseable_degradation_params_are_a_violation(tmp_path):
    s, c, q, p, sp = _banded(_frames(tmp_path), params="{not json")
    assert "degradation_params_unparseable" in _checks(
        check_invariants(s, c, q, p, sp, source_root=tmp_path))


def test_an_out_of_vocabulary_band_is_a_violation(tmp_path):
    s, c, q, p, sp = _banded(_frames(tmp_path), band="catastrophic")
    assert "degradation_band_unknown" in _checks(
        check_invariants(s, c, q, p, sp, source_root=tmp_path))


def test_the_band_spec_is_read_from_degrade_assign_not_restated(tmp_path):
    """One home for the angles: a band whose angle changes in `BAND_SPEC` must change
    what this invariant accepts, with no edit here."""
    from pyrafault.degrade_assign import BAND_SPEC
    kind, strength = BAND_SPEC["moderate"]
    s, c, q, p, sp = _banded(_frames(tmp_path), band="moderate", kind=kind,
                             params=json.dumps({"angle_deg": strength}))
    assert check_invariants(s, c, q, p, sp, source_root=tmp_path) == []


def test_summarize_records_the_band_distribution(tmp_path):
    """`qc_stats.json` is where a freeze's balance is recorded; without this the
    four-band split survives only in a migration script's stdout."""
    stats = summarize(*_banded(_frames(tmp_path))[1:])
    assert stats["degradation_band"] == {"severe": 2}


def test_summarize_omits_the_band_block_for_a_table_without_the_columns(tmp_path):
    assert "degradation_band" not in summarize(*_frames(tmp_path)[1:])


def test_reconstruction_sample_is_seeded_and_covers_every_slide(tmp_path):
    df = pd.DataFrame({"query_id": [f"q{i}" for i in range(200)],
                       "slide_id": ["A"] * 100 + ["B"] * 100})
    a = reconstruction_sample(df, "query_id", 0.05, seed=7)
    b = reconstruction_sample(df, "query_id", 0.05, seed=7)
    assert list(a["query_id"]) == list(b["query_id"])
    assert set(a["slide_id"]) == {"A", "B"}          # stratified, never a single-slide sample
    assert len(a) == 10


def test_summarize_reports_every_required_block(tmp_path):
    stats = summarize(*_frames(tmp_path)[1:])
    for block in ("counts", "fault_type", "fault_severity", "tissue_fraction",
                  "quality_flags", "morphology_clusters", "self_similarity",
                  "hardness_score", "normalized_offset", "per_slide", "per_split"):
        assert block in stats, block
    assert stats["counts"]["pairs"] == 2
    assert stats["fault_type"]["clean"] == 1


def test_flags_counts_list_elements_not_a_stringified_repr():
    """quality_flags is schema `list<string>` (the `LS` type in pyrafault.schema),
    delivered from parquet as a list or numpy object array per cell — never a
    pipe-delimited string. Splitting str(value) on '|' invents a flag named "[]" for
    every empty cell and mangles a real one into "['pen_mark']"; this pins the fix."""
    series = pd.Series([[], ["pen_mark"], ["blur", "too_bright"]])
    assert _flags(series) == {"pen_mark": 1, "blur": 1, "too_bright": 1}


def test_flags_treats_null_cells_as_no_flags():
    series = pd.Series([None, float("nan"), []])
    assert _flags(series) == {}


def test_render_markdown_states_the_verdict_in_the_first_lines(tmp_path):
    ok = render_markdown(summarize(*_frames(tmp_path)[1:]), [], "abc123")
    assert "PASS" in ok.splitlines()[2]
    bad = render_markdown(summarize(*_frames(tmp_path)[1:]),
                          [Violation("out_of_bounds", "contexts", "S1_ctx000", "x1 > width")],
                          "abc123")
    assert "FAIL" in bad.splitlines()[2]
    assert "out_of_bounds" in bad


def test_render_markdown_names_unchecked_config_sections_even_when_it_would_otherwise_pass(
        tmp_path):
    """A legacy manifest with zero violations must not read the same as a genuinely
    clean, fully-verified run -- the sections it could not vouch for have to be named
    on the page, not merely absent from the violations table."""
    clean = render_markdown(summarize(*_frames(tmp_path)[1:]), [], "abc123")
    assert "Config sections not verified" not in clean

    reported = render_markdown(summarize(*_frames(tmp_path)[1:]), [], "abc123",
                               unchecked_sections=["degradation"])
    assert "PASS" in reported.splitlines()[2]   # still a pass: zero violations
    assert "Config sections not verified" in reported
    assert "`degradation`" in reported


def test_render_markdown_does_not_conflate_verified_with_newly_hashed(tmp_path):
    """Task 6 fix round 1, Finding 1: right after a freeze is banded every
    served_pixel_sha256 is "" -- stage 09's first run compares zero of them (there is
    nothing yet to compare against) and fills the rest instead. check_served_pixels
    never fires in that state, so a report with no `served_pixel_summary` distinction
    would read as a plain, fully-verified PASS -- exactly "a freeze with no hashes
    reads as fully verified", the failure this whole feature exists to prevent."""
    stats = summarize(*_frames(tmp_path)[1:])

    silent = render_markdown(stats, [], "abc123")
    assert "Served pixels" not in silent  # no summary passed -> no claim either way

    nothing_verified = render_markdown(stats, [], "abc123", served_pixel_summary={
        "sampled": 10, "compared": 0, "newly_hashed": 10})
    assert "PASS" in nothing_verified.splitlines()[2]   # zero violations: still a pass
    assert "Served pixels" in nothing_verified
    assert "0 of 10" in nothing_verified
    assert "NOT confirmed" in nothing_verified

    fully_verified = render_markdown(stats, [], "abc123", served_pixel_summary={
        "sampled": 10, "compared": 10, "newly_hashed": 0})
    assert "Served pixels" in fully_verified
    assert "10 of 10" in fully_verified
    # The two PASS reports must not be textually indistinguishable -- that is the bug.
    assert nothing_verified != fully_verified
    assert "NOT confirmed" not in fully_verified


# --- config-provenance addition (task 22 extension, not in the base brief) -----------
#
# data/01_benchmark/config.yaml is supposed to be the frozen record of exactly what
# built the benchmark. Stage 00 copies it early; if a later stage runs under a resized
# config, the archive silently claims parameters that never actually produced the
# benchmark. feature_manifest.json (written by stage 05, downstream of the resize)
# records the digest that was actually live when features were built, so comparing the
# two is the check that catches the drift.

MINIMAL_CFG = {
    "seed": 1, "pipeline_version": "v1", "extraction_version": "v1",
    "source": {"gdc_dir": "x", "manifest": "m", "sample_sheet": "s",
               "metadata_json": "j", "verify_md5": True},
    "slide": {"require_mpp": True, "max_mpp_l0": 0.5},
    "thumbnail": {"target_mpp": 8.0},
    "tissue": {"method": "hsv_saturation_otsu", "sat_floor": 0.08, "gray_max": 235,
               "close_radius_um": 64, "open_radius_um": 32, "min_component_area_um2": 250000},
    "context": {"target_mpp": 2.0, "size_px": 2048, "stride_frac": 0.25,
                "min_tissue_fraction": 0.5, "num_per_slide": 30, "selection": "spatial_fps"},
    "query": {"target_mpp": 0.5, "size_px": 512, "stride_frac": 1.0,
              "boundary_margin_frac": 1.0, "min_tissue_fraction": 0.8, "num_per_context": 5},
    "quality": {"blur_var_flag": 20.0, "blur_var_exclude": 5.0,
                "mean_intensity_flag": [25, 235], "mean_intensity_exclude": [10, 250],
                "pen_mark_flag_frac": 0.02},
    "visual_sampling": {"enabled": True, "extractor": "uni", "fallback_extractor": "dinov2",
                        "strategy": "cluster_stratified", "n_clusters": 5, "self_similarity_k": 16},
    "features": {"uni": {}, "dinov2": {}, "batch_size": 64, "device": "auto"},
    "faults": {
        "clean": True,
        "adjacent_drift": {"enabled": True, "normalize_by": "query_side",
                           "severity_bands": {"small": [0.25, 0.75], "medium": [0.75, 2.0],
                                              "large": [2.0, 6.0]},
                           "balance": "stratified", "max_theta_retries": 32},
        "random_same_context": {"enabled": True, "max_iou_with_true": 0.0,
                                "min_center_dist_query_widths": 1.5},
        "hard_same_slide": {"enabled": True, "pool": "context", "max_iou_with_true": 0.0,
                            "min_center_dist_query_widths": 1.5, "top_k": 1},
        "out_of_context": {"enabled": False},
    },
    "splits": {"level": "case", "dev": 6, "calibration": 2, "test": 2,
              "min_contexts_per_slide": {"test": 15, "calibration": 15},
              "max_reshuffle_attempts": 1000},
    "storage": {"canonical": "metadata_first", "materialize_pilot_png": True,
               "large_scale_patch_format": "sharded_tar"},
    "qc": {"overlays_per_slide": 25, "verify_source_md5": True,
          "verify_materialized_hash": True, "reconstruction_sample_frac": 0.05},
}


def _write_cfg(path: Path, overrides: dict | None = None) -> dict:
    from pyrafault.config import load_config
    cfg = json.loads(json.dumps(MINIMAL_CFG))  # deep copy
    if overrides:
        cfg["context"]["num_per_slide"] = overrides.get("num_per_slide",
                                                          cfg["context"]["num_per_slide"])
        cfg["query"]["num_per_context"] = overrides.get("num_per_context",
                                                          cfg["query"]["num_per_context"])
        cfg["visual_sampling"]["n_clusters"] = overrides.get(
            "n_clusters", cfg["visual_sampling"]["n_clusters"])
    path.write_text(yaml.safe_dump(cfg))
    loaded = load_config(path)
    return {"digest": loaded.digest(), "sections": loaded.section_digests()}


def test_config_digest_match_yields_no_violation(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    result = _write_cfg(cfg_path)
    manifest_path = tmp_path / "feature_manifest.json"
    manifest_path.write_text(json.dumps({"section_digests": result["sections"]}))
    violations, unchecked = check_config_provenance(
        cfg_path, manifest_path, result["sections"],
        stage_sections=sorted(result["sections"]))
    assert violations == []
    assert unchecked == []


def test_stale_archived_config_is_caught_as_config_drift(tmp_path):
    """The task-20 defect: stage 00's copy of config.yaml is stale relative to the
    config that actually built the benchmark (recorded live by stage 05). Per-section
    scoping surfaces every section that moved -- here three (context, query,
    visual_sampling) -- rather than one opaque whole-config blob."""
    cfg_path = tmp_path / "config.yaml"
    # Archive reflects the OLD (pre-resize) parameters...
    _write_cfg(cfg_path, {"num_per_slide": 25, "num_per_context": 4, "n_clusters": 4})
    # ...but the manifest recorded the digest of the NEW (live) config that actually ran.
    live_path = tmp_path / "live.yaml"
    live = _write_cfg(live_path, {"num_per_slide": 30, "num_per_context": 5, "n_clusters": 5})
    manifest_path = tmp_path / "feature_manifest.json"
    manifest_path.write_text(json.dumps({"section_digests": live["sections"]}))

    violations, _ = check_config_provenance(
        cfg_path, manifest_path, live["sections"], stage_sections=sorted(live["sections"]))
    assert {v.table for v in violations} == {"config"}
    assert all(v.check == "config_drift" for v in violations)
    assert len(violations) == 3   # context, query, visual_sampling all moved


def test_i7_stage09s_own_live_config_disagreeing_is_caught(tmp_path):
    """I7: the archived config and feature_manifest can agree with EACH OTHER (the
    two-way check that predates this fix) while stage 09 itself is running under a
    third, different config edited after stage 05 -- e.g. a severity band changed
    between stage 05 and stage 09. That must also be config_drift.

    Compared per section (using `Config.section_digests()` for the live side, not
    `Config.digest()`): a single whole-config digest cannot say WHICH section a live
    edit touched, only that live disagrees with archived/recorded SOMEWHERE -- which
    would either flag every section (defeating per-section scoping) or none (silently
    dropping this exact check). Per-section digests are the only representation that
    lets "live" participate section-by-section, the same way archived/recorded do.
    """
    cfg_path = tmp_path / "config.yaml"
    result = _write_cfg(cfg_path)
    manifest_path = tmp_path / "feature_manifest.json"
    manifest_path.write_text(json.dumps({"section_digests": result["sections"]}))

    from pyrafault.config import load_config
    edited_path = tmp_path / "edited.yaml"
    cfg = json.loads(json.dumps(MINIMAL_CFG))
    cfg["seed"] = 999999   # a genuine post-stage-05 edit
    edited_path.write_text(yaml.safe_dump(cfg))
    edited = load_config(edited_path)

    violations, _ = check_config_provenance(
        cfg_path, manifest_path, edited.section_digests(),
        stage_sections=sorted(result["sections"]))
    assert len(violations) == 1
    assert violations[0].check == "config_drift"
    assert "seed" in violations[0].detail
    assert "live (stage 09)" in violations[0].detail


def test_missing_archived_config_is_caught_as_config_drift(tmp_path):
    """A missing archive is a provenance failure, not grounds to skip the check --
    mirrors missing_source, which raises rather than skips for an absent slide file."""
    cfg_path = tmp_path / "config.yaml"          # never written
    manifest_path = tmp_path / "feature_manifest.json"
    manifest_path.write_text(json.dumps({"config_digest": "irrelevant"}))

    violations, unchecked = check_config_provenance(cfg_path, manifest_path, {},
                                                     stage_sections=["context"])
    assert len(violations) == 1
    assert violations[0].check == "config_drift"
    assert str(cfg_path) in violations[0].detail
    assert unchecked == []


def test_missing_feature_manifest_is_caught_as_config_drift(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    result = _write_cfg(cfg_path)
    manifest_path = tmp_path / "feature_manifest.json"   # never written

    violations, unchecked = check_config_provenance(cfg_path, manifest_path, result["sections"],
                                                     stage_sections=["context"])
    assert len(violations) == 1
    assert violations[0].check == "config_drift"
    assert str(manifest_path) in violations[0].detail
    assert unchecked == []


def test_both_config_and_manifest_missing_yields_two_violations(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    manifest_path = tmp_path / "feature_manifest.json"

    violations, unchecked = check_config_provenance(cfg_path, manifest_path, {},
                                                     stage_sections=["context"])
    assert len(violations) == 2
    assert all(v.check == "config_drift" for v in violations)
    assert unchecked == []


# --- I3: cheap cross-table QC checks --------------------------------------------------

def test_bbox_size_disagreeing_with_its_own_magnification_claim_is_caught(tmp_path):
    """I3(a): bbox_w == round(output_size * target_mpp / mpp_x) — the one invariant
    tying a stored coordinate to its magnification claim. Reconstruction can't catch
    this because it re-reads from the same (wrong) bbox."""
    s, c, q, p, sp = _frames(tmp_path)
    c.loc[0, "target_mpp"] = 4.0     # bbox_w now claims to be 2.0 um/px but says 4.0
    assert "bbox_mpp_mismatch" in _checks(check_invariants(s, c, q, p, sp, source_root=tmp_path))


def test_pair_true_bbox_disagreeing_with_its_query_is_caught(tmp_path):
    """I3(b): pairs.true_bbox_l0 must equal queries.true_bbox_l0 for the same
    query_id — only reference EXISTENCE was checked before this."""
    s, c, q, p, sp = _frames(tmp_path)
    p.loc[0, "true_bbox_l0_x"] += 500
    assert "true_bbox_mismatch" in _checks(check_invariants(s, c, q, p, sp, source_root=tmp_path))


def test_normalized_offset_disagreeing_with_the_stored_boxes_is_caught(tmp_path):
    """I3(c), part 1: normalized_offset is recomputed from the stored boxes."""
    s, c, q, p, sp = _frames(tmp_path)
    p.loc[1, "normalized_offset"] = 0.5   # geometry says 2.0 (see fixture)
    assert "normalized_offset_mismatch" in _checks(
        check_invariants(s, c, q, p, sp, source_root=tmp_path))


def test_severity_band_disagreeing_with_geometry_is_caught(tmp_path):
    """I3(c), part 2: an adjacent_drift pair's recorded fault_severity band must
    actually contain its (recomputed) normalized_offset."""
    s, c, q, p, sp = _frames(tmp_path)
    p.loc[1, "fault_severity"] = "small"   # band [0.25, 0.75); geometry gives offset=2.0
    assert "severity_band_mismatch" in _checks(
        check_invariants(s, c, q, p, sp, source_root=tmp_path))


# --- I2: verify_reconstruction reads the per-row extraction spec ---------------------

def test_verify_reconstruction_raises_on_row_config_mismatch():
    """I2: a row whose stored target_mpp/output_size/extraction_version disagrees with
    the live config is a provenance failure, caught before any pixel is re-read."""
    row = {"context_id": "S1_ctx000", "bbox_l0_x": 0, "bbox_l0_y": 0,
           "bbox_l0_w": 100, "bbox_l0_h": 100, "target_mpp": 999.0, "output_size": 2048,
           "extraction_version": "v1", "image_sha256": "a" * 64}
    with pytest.raises(ValueError):
        verify_reconstruction(slide=None, rows=[row], bbox_prefix="bbox_l0",
                              id_col="context_id", table="contexts",
                              cfg_target_mpp=2.0, cfg_output_size=2048,
                              cfg_extraction_version="v1")


# --- I9: benchmark_manifest.json -------------------------------------------------------

def test_benchmark_manifest_records_sha256_and_row_count_per_table(tmp_path):
    import hashlib

    s, c, q, p, sp = _frames(tmp_path)
    for name, df in (("slides", s), ("contexts", c), ("queries", q),
                     ("pairs", p), ("splits", sp)):
        df.to_parquet(tmp_path / f"{name}.parquet")

    manifest = benchmark_manifest(tmp_path, cfg_digest="abc123", pipeline_version="v1")
    assert manifest["config_digest"] == "abc123"
    assert manifest["pipeline_version"] == "v1"
    assert set(manifest["tables"]) == {
        "slides.parquet", "contexts.parquet", "queries.parquet",
        "pairs.parquet", "splits.parquet",
    }
    entry = manifest["tables"]["pairs.parquet"]
    assert entry["rows"] == len(p)
    assert entry["sha256"] == hashlib.sha256((tmp_path / "pairs.parquet").read_bytes()).hexdigest()


def test_benchmark_manifest_is_empty_for_a_directory_with_no_parquet(tmp_path):
    manifest = benchmark_manifest(tmp_path, cfg_digest="x", pipeline_version="v1")
    assert manifest["tables"] == {}


# --- §13.5: the leakage audit ----------------------------------------------------
#
# check_leakage measures, per pixel-free signal computable from `inputs` alone
# (claimed_bbox and context_bbox -- never true_bbox), the best in-sample balanced
# accuracy at separating clean from not-clean. n is kept large (200 rows) on purpose:
# with too few rows, best-threshold fitting can overfit pure noise into a spuriously
# high score, which would make the "no violation" test flaky.

LK_SEED, LK_EXTRACTION_VERSION = 1, "v1"
LK_MPP = 0.25
# q_fov = 100; _grid_size_and_remainder(1000, 1000, 100) starts at n=10 (0 remainder)
# but the decrement loop always fires once (0 < 100), landing on n=9, remainder=100 --
# so the centred inset is (50, 50), not (0, 0). Irrelevant to what these fixtures
# test (both tests hold claimed position identical across classes, so on_grid is
# constant either way), but worth getting right in a comment that explains numbers.
LK_Q_SIZE, LK_Q_MPP = 100, 0.25

# Derived from the constants the audit itself iterates, never hand-listed: a field added
# to `inputs` should make this test's expected set grow on its own, so the test keeps
# asserting "every input field is audited" rather than "these four names are".
from pyrafault.dataset import INPUT_SCALAR_KEYS, INPUT_STRING_KEYS

STRING_SCORE_KEYS = {f"string_{n}" for n in INPUT_STRING_KEYS}
SCALAR_SCORE_KEYS = set(INPUT_SCALAR_KEYS)
GEOMETRY_SCORE_KEYS = {"on_grid", "centre_distance", "border_distance",
                       "claimed_width", "claimed_height"}
# The degradation columns (Task 2): present only once a freeze carries them, so they
# are a separately-named set rather than folded into GEOMETRY/SCALAR/STRING above --
# a table without them must keep producing exactly the original three sets.
DEGRADATION_SCORE_KEYS = {"degradation_band", "string_degradation_kind"}


def _leakage_pairs(n_per_class: int, leaking: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One slide, one 1000x1000 context, `n_per_class` clean + `n_per_class`
    not-clean pairs, all claimed at the same position (400, 400), same slide/split.

    `leaking=False`: clean and not-clean pairs are otherwise identical (same claimed
    width/height) -- every signal has the same distribution in both classes, so no
    single threshold (or, for string fields, substring) can do better than chance.
    `leaking=True`: clean pairs claim width=height=120, not-clean pairs claim
    width=height=100 -- a deliberate, perfectly-separating leak on `claimed_width`/
    `claimed_height` (spec §13.5's "ℓ's width and height ... a non-constant value
    would itself be a defect"). `pair_id` itself contains the fault-type name
    (`S1_pclean_0`, `S1_padjacent_drift_0`, ...) in both cases -- on purpose, to prove
    `item_id = sha256(seed|pair_id)` does NOT leak it through either (hex cannot spell
    these words; see `make_item_id`).
    """
    slides = pd.DataFrame([{"slide_id": "S1", "mpp_x": LK_MPP}])
    contexts = pd.DataFrame([{
        "context_id": "S1_ctx000",
        "bbox_l0_x": 0, "bbox_l0_y": 0, "bbox_l0_w": 1000, "bbox_l0_h": 1000,
    }])
    rows = []
    for i in range(n_per_class):
        for fault_type, side in (("clean", 120 if leaking else 100),
                                 ("adjacent_drift", 100)):
            rows.append({
                "pair_id": f"S1_p{fault_type}_{i}", "slide_id": "S1", "split": "dev",
                "context_id": "S1_ctx000", "fault_type": fault_type,
                "claimed_bbox_l0_x": 400, "claimed_bbox_l0_y": 400,
                "claimed_bbox_l0_w": side, "claimed_bbox_l0_h": side,
            })
    pairs = pd.DataFrame(rows)
    return contexts, pairs, slides


def _check(contexts, pairs, slides, queries=None):
    """`queries` supplies the per-record `mq` the scalar audit reconstructs; the
    fixtures below predate `inputs` carrying mc/mq, so build a minimal one when the
    caller has not."""
    if queries is None:
        queries = pd.DataFrame({
            "query_id": sorted(set(pairs.get("query_id", pairs["pair_id"]))),
        })
        queries["target_mpp"] = LK_Q_MPP
    if "query_id" not in pairs.columns:
        pairs = pairs.assign(query_id=pairs["pair_id"])
        queries = pd.DataFrame({"query_id": pairs["pair_id"], "target_mpp": LK_Q_MPP})
    if "target_mpp" not in contexts.columns:
        contexts = contexts.assign(target_mpp=LK_Q_MPP * 4)
    return check_leakage(contexts, queries, pairs, slides, LK_Q_MPP, LK_Q_SIZE,
                         LK_SEED, LK_EXTRACTION_VERSION)


def test_leakage_audit_fires_on_a_deliberately_leaking_signal():
    contexts, pairs, slides = _leakage_pairs(n_per_class=100, leaking=True)
    violations, scores = _check(contexts, pairs, slides)
    assert violations, "a perfectly label-separating claimed_width/height must be caught"
    assert all(v.check == "leakage" for v in violations)
    assert {"claimed_width", "claimed_height"} & {v.row_id for v in violations}
    assert scores["claimed_width"] > 0.55
    assert scores["claimed_height"] > 0.55
    # the deliberately fault-type-laden pair_id must NOT leak through item_id
    assert scores["string_item_id"] <= 0.55


def test_leakage_audit_is_silent_on_a_clean_benchmark():
    contexts, pairs, slides = _leakage_pairs(n_per_class=100, leaking=False)
    violations, scores = _check(contexts, pairs, slides)
    assert violations == []
    assert all(score <= 0.55 for score in scores.values()), scores
    assert set(scores) == (GEOMETRY_SCORE_KEYS | SCALAR_SCORE_KEYS | STRING_SCORE_KEYS)


def test_leakage_audit_fires_on_a_band_assigned_only_to_faulty_items():
    """The failure this guards: bands correlated with `claim_correct` let a verifier
    predict the label with no pixels at all, and no must-be-zero invariant would notice --
    the same shape as the lattice defect §15.1 responds to."""
    contexts, pairs, slides = _leakage_pairs(n_per_class=100, leaking=False)
    clean = pairs["fault_type"] == "clean"
    pairs["degradation_band"] = np.where(clean, "none", "severe")
    pairs["degradation_kind"] = np.where(clean, "none", "geometry")
    pairs["degradation_params"] = "{}"

    violations, scores = _check(contexts, pairs, slides)

    assert scores["degradation_band"] > 0.9
    assert "degradation_band" in {v.row_id for v in violations}


def test_leakage_audit_is_silent_on_balanced_bands():
    contexts, pairs, slides = _leakage_pairs(n_per_class=100, leaking=False)
    pairs = assign_bands(pairs, seed=LK_SEED, bands=BANDS)

    violations, scores = _check(contexts, pairs, slides)

    assert violations == []
    assert abs(scores["degradation_band"] - 0.5) < 0.05
    assert set(scores) == (GEOMETRY_SCORE_KEYS | SCALAR_SCORE_KEYS
                           | STRING_SCORE_KEYS | DEGRADATION_SCORE_KEYS)


def test_an_out_of_vocabulary_band_is_not_scored_as_a_fourth_band():
    """A corrupt band cell must not become a band the threshold search can separate on.

    Mapped to a sentinel ordinal (the previous `band_index.get(str(b), -1.0)`), the
    values below separate clean from not-clean perfectly and the audit reports a
    leakage violation naming `degradation_band` -- pointing the reader at
    `assign_bands`, which assigned nothing of the sort. The corrupt rows are excluded
    from this signal instead; they are reported where they can actually be acted on, as
    `check_invariants`' `degradation_band_unknown` on the row itself.
    """
    contexts, pairs, slides = _leakage_pairs(n_per_class=100, leaking=False)
    clean = pairs["fault_type"] == "clean"
    pairs["degradation_band"] = np.where(clean, "none", "catastrophic")
    pairs["degradation_kind"] = "none"
    pairs["degradation_params"] = "{}"

    violations, scores = _check(contexts, pairs, slides)

    assert scores["degradation_band"] == 0.5
    assert "degradation_band" not in {v.row_id for v in violations}


def test_the_band_signal_survives_an_empty_pairs_table():
    """The in-vocabulary mask is a numpy bool array; built from an empty list it would
    default to float64, which cannot index, so a zero-row table would raise here instead
    of scoring 0.500 like every other signal."""
    contexts, pairs, slides = _leakage_pairs(n_per_class=1, leaking=False)
    pairs = pairs.iloc[:0].copy()
    pairs["degradation_band"] = pd.Series(dtype=object)
    pairs["degradation_kind"] = pd.Series(dtype=object)

    _violations, scores = _check(contexts, pairs, slides)

    assert scores["degradation_band"] == 0.5


def test_leakage_audit_fires_on_a_deliberately_leaking_string_field():
    """spec §8/§13.5: the actual shipped defect was a string field (`inputs["pair_id"]`),
    not geometry -- this pins that check_leakage's string audit, not just its geometry
    audit, is wired up and catches a field that spells the label out in plain text."""
    contexts, pairs, slides = _leakage_pairs(n_per_class=100, leaking=False)
    # slide_id itself now encodes the label -- exactly the class of bug that shipped.
    pairs.loc[pairs["fault_type"] == "clean", "slide_id"] = "S1_clean_slide"
    pairs.loc[pairs["fault_type"] != "clean", "slide_id"] = "S1_other_slide"
    slides = pd.concat([slides, pd.DataFrame([
        {"slide_id": "S1_clean_slide", "mpp_x": LK_MPP},
        {"slide_id": "S1_other_slide", "mpp_x": LK_MPP},
    ])], ignore_index=True)

    violations, scores = _check(contexts, pairs, slides)
    assert violations, "a slide_id that spells out the fault type must be caught"
    assert all(v.check == "leakage" for v in violations)
    assert "string_slide_id" in {v.row_id for v in violations}
    assert scores["string_slide_id"] > 0.55


def _on_grid_leak_pairs(n_per_class: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reproduces the pre-jitter §15.1 on-grid defect in miniature: `clean`,
    `random_same_context` and `hard_mismatch` claim a position exactly on the
    *centred* lattice (`centred_inset`); `adjacent_drift` claims a position offset
    from that lattice by an amount that is not a multiple of `q_fov` on either axis.

    Every other `_leakage_pairs`-based fixture holds `claimed_bbox` position IDENTICAL
    across classes (see its docstring), so `on_grid` is constant there and the signal
    is never actually exercised as a genuinely-discriminating one -- only
    `claimed_width`/`claimed_height` are. This fixture is what a broken
    `centred_inset` call or broken modulo arithmetic in `check_leakage`'s on-grid
    computation would need to be caught by.
    """
    slides = pd.DataFrame([{"slide_id": "S1", "mpp_x": LK_MPP}])
    contexts = pd.DataFrame([{
        "context_id": "S1_ctx000",
        "bbox_l0_x": 0, "bbox_l0_y": 0, "bbox_l0_w": 1000, "bbox_l0_h": 1000,
    }])
    inset_x, inset_y = centred_inset(1000, 1000, LK_Q_SIZE)   # (50, 50)
    rows = []
    for i in range(n_per_class):
        k = i % 5   # a handful of distinct lattice cells, all still on-grid
        on_x, on_y = inset_x + k * LK_Q_SIZE, inset_y
        off_x, off_y = on_x + 37, on_y + 11   # not a multiple of q_fov away
        for fault_type, (x, y) in (
            ("clean", (on_x, on_y)),
            ("random_same_context", (on_x, on_y)),
            ("hard_mismatch", (on_x, on_y)),
            ("adjacent_drift", (off_x, off_y)),
        ):
            rows.append({
                "pair_id": f"S1_p{fault_type}_{i}", "slide_id": "S1", "split": "dev",
                "context_id": "S1_ctx000", "fault_type": fault_type,
                "claimed_bbox_l0_x": x, "claimed_bbox_l0_y": y,
                "claimed_bbox_l0_w": LK_Q_SIZE, "claimed_bbox_l0_h": LK_Q_SIZE,
            })
    pairs = pd.DataFrame(rows)
    return contexts, pairs, slides


def test_leakage_audit_fires_on_the_pre_jitter_on_grid_defect_reproduced_in_miniature():
    """spec §13.5: the on-grid signal itself (not just claimed_width/height) must be
    a live, exercised code path -- verified by mutation while writing this test:
    changing the on-grid modulo comparison in `check_leakage` from `== 0` to `== 1`
    (so it matches neither the on-grid nor the deliberately-off-grid rows below) made
    this test fail with `on_grid` missing from the violations, then passed again once
    reverted. See the leakage-hardening report for the exact diff checked."""
    contexts, pairs, slides = _on_grid_leak_pairs(n_per_class=100)
    violations, scores = _check(contexts, pairs, slides)
    assert violations
    assert "on_grid" in {v.row_id for v in violations}
    assert scores["on_grid"] > 0.55


def test_dupes_catches_rows_sharing_a_null_id():
    """value_counts() defaults to dropna=True, which would make several rows with a
    null id invisible to the duplicate check -- the same "check can't see its own
    worst case" pattern hit elsewhere on this plan. dropna=False fixes it."""
    df = pd.DataFrame({"pair_id": [None, None, "S1_p000"]})
    violations = _dupes(df, "pair_id", "pairs")
    assert len(violations) == 1
    assert violations[0].check == "duplicate_id"
    assert violations[0].table == "pairs"


def test_every_input_field_is_covered_by_some_audit():
    """spec §8/§13.5. `inputs` gained mc/mq; this pins the property that made adding
    them safe -- every name in INPUT_KEYS falls into exactly one audited category, so a
    field cannot be shipped to verifiers without some check looking at it. The shipped
    `inputs["pair_id"]` defect is what happens when one slips through."""
    from pyrafault.dataset import (
        INPUT_GEOMETRY_KEYS, INPUT_IMAGE_KEYS, INPUT_KEYS, INPUT_SCALAR_KEYS,
        INPUT_STRING_KEYS,
    )
    categories = [INPUT_IMAGE_KEYS, INPUT_GEOMETRY_KEYS, INPUT_SCALAR_KEYS,
                  INPUT_STRING_KEYS]
    union = set().union(*categories)
    assert union == set(INPUT_KEYS), f"uncategorised input field(s): {set(INPUT_KEYS) ^ union}"
    for i, a in enumerate(categories):
        for b in categories[i + 1:]:
            assert not (a & b), f"field in two categories at once: {a & b}"


def test_unknown_scalar_field_raises_rather_than_going_unaudited():
    from pyrafault.qc.report import _reconstruct_input_scalar_field
    with pytest.raises(NotImplementedError, match="INPUT_SCALAR_KEYS"):
        _reconstruct_input_scalar_field("objective_power", pd.DataFrame(),
                                        pd.DataFrame(), pd.DataFrame())


def test_scalar_audit_fires_when_magnification_predicts_the_label():
    """mc/mq are constant in one freeze, so they score 0.500 -- but the audit is not
    decorative. Pool two magnification combinations whose fault composition differs and
    magnification becomes a pixel-free predictor; this is that scenario in miniature."""
    contexts, pairs, slides = _leakage_pairs(n_per_class=100, leaking=False)
    pairs = pairs.assign(query_id=[f"q{i}" for i in range(len(pairs))])
    # combo A holds every clean pair, combo B every fault: mq alone answers the task.
    is_clean = pairs["fault_type"] == "clean"
    queries = pd.DataFrame({
        "query_id": pairs["query_id"],
        "target_mpp": np.where(is_clean, 0.25, 0.5),
    })
    contexts = contexts.assign(target_mpp=1.0)
    _, scores = check_leakage(contexts, queries, pairs, slides, LK_Q_MPP, LK_Q_SIZE,
                              LK_SEED, LK_EXTRACTION_VERSION)
    assert scores["query_mpp"] > 0.55, scores
    assert scores["context_mpp"] <= 0.55, "constant mc must stay at chance"
