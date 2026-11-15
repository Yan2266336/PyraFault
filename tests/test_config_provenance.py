"""A data-side config change must not invalidate stage 05.

One digest over the whole config means adding a `degradation:` section forces a rebuild
from stage 00, re-embedding 2.7M query candidates to change three columns -- which is
what happened when the last fault types landed.

`live_sections` (the third positional argument below) is `Config.section_digests()`,
not `Config.digest()`. A single whole-config digest cannot answer a per-section
question by construction: a hash hides structure on purpose, so two whole digests
disagreeing says nothing about *which* section moved. Telling "the config changed"
from "the config that mattered to this stage changed" -- the entire point of this
check -- requires the live side to carry the same per-section shape as the archived
and recorded sides.
"""
from __future__ import annotations

import json

from pyrafault.config import load_config
from pyrafault.qc.report import check_config_provenance


def test_adding_a_section_a_stage_never_read_does_not_fault_it(tmp_path, write_config):
    archived = write_config(tmp_path / "archived.yaml", extra=None)
    live = write_config(tmp_path / "live.yaml", extra={"degradation": {"bands": 4}})
    manifest = tmp_path / "feature_manifest.json"
    manifest.write_text(json.dumps({
        "section_digests": load_config(archived).section_digests()}))

    violations, unchecked = check_config_provenance(
        archived, manifest, load_config(live).section_digests(),
        stage_sections=["context", "query", "features", "visual_sampling"])

    assert violations == []
    assert unchecked == []


def test_a_changed_section_a_stage_did_read_is_a_violation(tmp_path, write_config):
    archived = write_config(tmp_path / "archived.yaml", extra=None)
    live = write_config(tmp_path / "live.yaml", context_size_px=1024)
    manifest = tmp_path / "feature_manifest.json"
    manifest.write_text(json.dumps({
        "section_digests": load_config(archived).section_digests()}))

    violations, _ = check_config_provenance(
        archived, manifest, load_config(live).section_digests(),
        stage_sections=["context"])

    assert [v.check for v in violations] == ["config_drift"]


def test_a_partially_upgraded_manifest_reports_the_section_it_never_recorded(
        tmp_path, write_config):
    """A "modern" manifest (it HAS `section_digests`) can still be missing an entry
    for one particular section -- e.g. it was written just before that section
    landed. Without this check, a section present and AGREEING in both `archived`
    and `live` but absent from `recorded` would be compared archived-vs-live only,
    which passes, silently treating a manifest that never vouched for the section as
    if it had -- the exact "silence indistinguishable from a pass" failure this
    whole mechanism exists to prevent."""
    archived = write_config(tmp_path / "archived.yaml", extra=None)
    live = write_config(tmp_path / "live.yaml", extra=None)   # unchanged from archived
    sections = load_config(archived).section_digests()
    del sections["context"]   # this manifest never recorded "context"
    manifest = tmp_path / "feature_manifest.json"
    manifest.write_text(json.dumps({"section_digests": sections}))

    violations, unchecked = check_config_provenance(
        archived, manifest, load_config(live).section_digests(),
        stage_sections=["context"])

    assert violations == []          # archived and live agree on "context"...
    assert unchecked == ["context"]  # ...but the manifest never vouched for it


def test_an_unrecorded_section_can_be_both_flagged_and_unchecked(tmp_path, write_config):
    """Documents the co-occurrence the docstring calls out: when `recorded` never
    captured a section, but `archived` and `live` disagree on it, that disagreement
    is real and reported on its own terms -- being unchecked (the manifest cannot
    vouch for it) does not suppress being flagged (two sources we DO have disagree).
    "Flagged" and "unverifiable" are independent facts about the same section, not
    alternatives."""
    archived = write_config(tmp_path / "archived.yaml", extra=None)
    live = write_config(tmp_path / "live.yaml", context_size_px=1024)   # disagrees
    sections = load_config(archived).section_digests()
    del sections["context"]   # this manifest never recorded "context"
    manifest = tmp_path / "feature_manifest.json"
    manifest.write_text(json.dumps({"section_digests": sections}))

    violations, unchecked = check_config_provenance(
        archived, manifest, load_config(live).section_digests(),
        stage_sections=["context"])

    assert [v.check for v in violations] == ["config_drift"]
    assert unchecked == ["context"]


def test_a_legacy_manifest_reports_what_it_cannot_check(tmp_path, write_config):
    """Written before section digests existed. Re-running stage 05 to obtain them is not
    an acceptable cost for a data-side change, so the sections it cannot vouch for are
    named in the report rather than passed over."""
    archived = write_config(tmp_path / "archived.yaml", extra=None)
    live = write_config(tmp_path / "live.yaml", extra={"degradation": {"bands": 4}})
    manifest = tmp_path / "feature_manifest.json"
    manifest.write_text(json.dumps({"config_digest": load_config(archived).digest()}))

    violations, unchecked = check_config_provenance(
        archived, manifest, load_config(live).section_digests(),
        stage_sections=["context", "degradation"])

    assert violations == []
    assert unchecked == ["degradation"]


def test_a_legacy_manifest_still_compares_the_live_config(tmp_path, write_config):
    """The legacy branch's whole-config leg is not a substitute for the live leg.

    Every `feature_manifest.json` on disk is legacy, so this is the state the check
    actually runs in: stage 09 running under a config edited after stage 00 archived
    its own copy must still fault, section by section, even though the manifest can
    only vouch for the config as a whole. Returning after the whole-config comparison
    (which passes -- the manifest and the archive still agree; it is *live* that
    moved) would mean no manifest in this repository has its live leg checked at all.
    """
    archived = write_config(tmp_path / "archived.yaml", extra=None)
    live = write_config(tmp_path / "live.yaml", context_size_px=1024)
    manifest = tmp_path / "feature_manifest.json"
    manifest.write_text(json.dumps({"config_digest": load_config(archived).digest()}))

    violations, unchecked = check_config_provenance(
        archived, manifest, load_config(live).section_digests(),
        stage_sections=["context"])

    assert [v.check for v in violations] == ["config_drift"]
    assert "context" in violations[0].detail
    assert unchecked == []   # the archive HAS `context`; only per-section vouching is missing


def test_a_section_removed_from_the_live_config_is_a_violation(tmp_path, write_config):
    """The half of "config drift" that scoping lost. Under the old whole-config digest,
    deleting a section stage 09 reads faulted; with the sections declared by the live
    config itself (`sorted(cfg.raw)`), a deleted section simply stopped being in
    `stage_sections` and was never compared or reported -- the archive and the manifest
    both remember it, which is two sources saying the live config is missing something.
    """
    archived = write_config(tmp_path / "archived.yaml", extra=None)
    live_sections = load_config(archived).section_digests()
    del live_sections["context"]          # the live config no longer has `context:`
    manifest = tmp_path / "feature_manifest.json"
    manifest.write_text(json.dumps({
        "section_digests": load_config(archived).section_digests()}))

    violations, unchecked = check_config_provenance(
        archived, manifest, live_sections, stage_sections=["context"])

    assert [v.check for v in violations] == ["config_drift"]
    assert "absent from the live config" in violations[0].detail
    assert unchecked == []


def test_a_section_absent_everywhere_is_reported_as_unchecked_not_removed(
        tmp_path, write_config):
    """A section that never existed is not a removal: nothing was lost, the stage just
    declares a dependency no config on any of the three legs carries (a section added
    to the declaration before the config). That is `unchecked`, not `config_drift`."""
    archived = write_config(tmp_path / "archived.yaml", extra=None)
    manifest = tmp_path / "feature_manifest.json"
    manifest.write_text(json.dumps({
        "section_digests": load_config(archived).section_digests()}))

    violations, unchecked = check_config_provenance(
        archived, manifest, load_config(archived).section_digests(),
        stage_sections=["degradation"])

    assert violations == []
    assert unchecked == ["degradation"]


def test_stage09_declares_the_sections_it_reads_not_every_section():
    """`stage_sections=sorted(cfg.raw)` made the scoping a no-op at its only call site:
    declaring every top-level section reproduces the old whole-config digest for
    everything except a newly ADDED section. This pins the declaration to a real,
    narrower list -- and to sections that exist, so a typo cannot quietly move a
    section into "unchecked" forever."""
    import importlib.util
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("stage09", root / "scripts" / "09_run_qc.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    raw = yaml.safe_load((root / "configs" / "example10.yaml").read_text())
    declared = set(mod.STAGE_SECTIONS)

    assert declared <= set(raw), f"declared sections that pilot.yaml lacks: {declared - set(raw)}"
    assert declared < set(raw), "declaring every section is the no-op this replaced"
    # The expensive ones: stage 09 reads no embedding, tissue or sampling parameter, so
    # editing them must not force a rebuild from stage 00 to clear this check.
    assert not declared & {"features", "visual_sampling", "tissue", "thumbnail", "splits"}
    for section in ("context", "query", "qc", "source", "faults", "storage",
                    "seed", "extraction_version", "pipeline_version"):
        assert section in declared, section
