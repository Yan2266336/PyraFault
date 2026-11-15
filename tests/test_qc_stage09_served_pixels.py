"""Task 6 fix round 1, Finding 2: an executable check of stage 09's cross-slide
served-pixel wiring (scripts/09_run_qc.py, the block grouping a sample by
`true_slide_id` rather than `pairs.slide_id`).

tests/test_out_of_slide.py::test_every_pairs_pixel_read_resolves_the_source_slide is
a textual grep for `true_slide_id` appearing near a `read_at_mpp` call -- it would
keep passing even if that block's `row.get("true_slide_id") or row["slide_id"]`
fallback were ever silently inverted, because the string `true_slide_id` would still
be sitting right there in the source while the pixels came from the wrong slide. This
runs the real `main()` against two slides whose pixels are built to differ
unmistakably, and checks the hash that lands in `pairs.parquet`, not the source text.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

from pyrafault.cache import image_sha256
from pyrafault.dataset import PyraFaultDataset
from pyrafault.degrade import apply as degrade_apply
from pyrafault.geom import BBoxL0
from pyrafault.io.read import read_at_mpp
from pyrafault.io.slide import SlideMeta
from pyrafault.schema import (
    CONTEXTS, PAIRS, QUERIES, SLIDES, SPLITS, read_table, write_table,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeSlide:
    """A synthetic pyramid whose pixel value is salted by its own slide id, so
    reading the same box from the wrong slide provably returns different bytes
    rather than accidentally matching."""

    def __init__(self, slide_id, width=8000, height=8000, mpp=0.25):
        self.meta = SlideMeta(
            slide_id=slide_id, path="/dev/null", width_l0=width, height_l0=height,
            mpp_x=mpp, mpp_y=mpp, objective_power=40.0, vendor="fake",
            level_count=2, level_dimensions=((width, height), (width // 4, height // 4)),
            level_downsamples=(1.0, 4.0),
        )
        self._id = slide_id

    def read_region(self, location, level, size):
        w, h = size
        ds = self.meta.level_downsamples[level]
        salt = sum(self._id.encode())
        xs = (np.arange(w) * ds + location[0] + salt) % 251
        ys = (np.arange(h) * ds + location[1]) % 241
        r = np.broadcast_to(xs.astype(np.uint8), (h, w))
        g = np.broadcast_to(ys.astype(np.uint8)[:, None], (h, w))
        b = np.full((h, w), level * 40, np.uint8)
        a = np.full((h, w), 255, np.uint8)
        return Image.fromarray(np.dstack([r, g, b, a]), mode="RGBA")

    def close(self) -> None:
        """A real handle has one, and `PyraFaultDataset.close()` calls it."""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _defaults(schema, overrides):
    out = {}
    for f in schema:
        if f.name in overrides:
            out[f.name] = overrides[f.name]
        elif str(f.type) == "string":
            out[f.name] = ""
        elif str(f.type) == "int64":
            out[f.name] = 0
        elif str(f.type) == "double":
            out[f.name] = 0.0
        elif str(f.type) == "bool":
            out[f.name] = False
        else:
            out[f.name] = None
    return out


@pytest.fixture
def stage09_cross_slide_fixture(tmp_path, monkeypatch):
    """One out_of_slide pair whose claim/context sit on S1 but whose served patch
    (true_bbox_l0) is frozen against S2 (true_slide_id="S2"). Returns
    (mod, bench_dir, run, degradation, true_box, q_spec, fakes) so the test can both
    drive stage 09's real main() and independently recompute the "correct" and
    "wrong-slide" hashes to compare against what it wrote.
    """
    mod = _load_script("09_run_qc.py")
    fakes = {"S1": FakeSlide("S1"), "S2": FakeSlide("S2")}
    monkeypatch.setattr(mod, "open_slide", lambda path, slide_id: fakes[slide_id])
    # Also on the loader, so `PyraFaultDataset` can be driven over the same two
    # synthetic slides (see the loader-vs-frozen-hash test at the end of this file).
    monkeypatch.setattr("pyrafault.dataset.open_slide",
                        lambda path, slide_id: fakes[slide_id])

    root = tmp_path / "data"
    bench = root / "01_benchmark"
    bench.mkdir(parents=True)
    (root / "04_features").mkdir(parents=True)
    source_root = tmp_path / "source"
    source_root.mkdir()
    for sid in ("S1", "S2"):
        (source_root / f"{sid}.svs").write_bytes(b"x")

    slides = pd.DataFrame([
        _defaults(SLIDES, {"slide_id": sid, "source_relpath": f"{sid}.svs",
                           "width_l0": 8000, "height_l0": 8000, "mpp_x": 0.25,
                           "mpp_y": 0.25})
        for sid in ("S1", "S2")
    ])
    write_table(slides, bench / "slides.parquet", "slides")

    contexts = pd.DataFrame([_defaults(CONTEXTS, {
        "context_id": "c0", "slide_id": "S1",
        "bbox_l0_x": 0, "bbox_l0_y": 0, "bbox_l0_w": 512, "bbox_l0_h": 512,
        "target_mpp": 2.0, "output_size": 64, "extraction_version": "v1",
        "split": "test",
    })])
    write_table(contexts, bench / "contexts.parquet", "contexts")

    true_box_kwargs = dict(true_bbox_l0_x=100, true_bbox_l0_y=100,
                           true_bbox_l0_w=64, true_bbox_l0_h=64)
    none_box_kwargs = dict(true_bbox_l0_x=200, true_bbox_l0_y=200,
                           true_bbox_l0_w=64, true_bbox_l0_h=64)
    queries = pd.DataFrame([
        _defaults(QUERIES, {
            "query_id": "q_cross", "context_id": "c0", "slide_id": "S1",
            "target_mpp": 0.25, "output_size": 64, "extraction_version": "v1",
            "split": "test", **true_box_kwargs,
        }),
        _defaults(QUERIES, {
            "query_id": "q_none", "context_id": "c0", "slide_id": "S1",
            "target_mpp": 0.25, "output_size": 64, "extraction_version": "v1",
            "split": "test", **none_box_kwargs,
        }),
    ])
    write_table(queries, bench / "queries.parquet", "queries")

    params = {"angle_deg": 15.0, "scale": 1.0}
    pairs = pd.DataFrame([_defaults(PAIRS, {
        "pair_id": "p_cross", "slide_id": "S1", "context_id": "c0", "query_id": "q_cross",
        "claimed_bbox_l0_x": 0, "claimed_bbox_l0_y": 0,
        "claimed_bbox_l0_w": 64, "claimed_bbox_l0_h": 64,
        "true_bbox_l0_x": 100, "true_bbox_l0_y": 100,
        "true_bbox_l0_w": 64, "true_bbox_l0_h": 64,
        "claim_correct": False, "fault_type": "out_of_slide",
        "true_slide_id": "S2", "same_slide": False,
        "normalized_offset": None, "offset_frac_context": None,
        "iou_claim_true": 0.0, "true_within_context": False,
        "local_recovery_possible": False,
        "context_mpp": 2.0, "query_mpp": 0.25, "scale_ratio": 8.0,
        "degradation_kind": "geometry", "degradation_params": json.dumps(params),
        "degradation_band": "severe", "served_pixel_sha256": "",
        "split": "test", "qc_status": "pending",
    }), _defaults(PAIRS, {
        # The in-freeze control, in the same fixture: band "none", the identity
        # transform. A loader that stopped replaying degradations would still
        # reproduce THIS row's frozen hash, which is exactly why the severe row above
        # is checked beside it rather than on its own.
        "pair_id": "p_none", "slide_id": "S1", "context_id": "c0", "query_id": "q_none",
        "claimed_bbox_l0_x": 200, "claimed_bbox_l0_y": 200,
        "claimed_bbox_l0_w": 64, "claimed_bbox_l0_h": 64,
        "true_bbox_l0_x": 200, "true_bbox_l0_y": 200,
        "true_bbox_l0_w": 64, "true_bbox_l0_h": 64,
        "claim_correct": True, "fault_type": "clean",
        "true_slide_id": "S1", "same_slide": True,
        "normalized_offset": 0.0, "offset_frac_context": 0.0,
        "iou_claim_true": 1.0, "true_within_context": True,
        "local_recovery_possible": True,
        "context_mpp": 2.0, "query_mpp": 0.25, "scale_ratio": 8.0,
        "degradation_kind": "none", "degradation_params": "{}",
        "degradation_band": "none", "served_pixel_sha256": "",
        "split": "test", "qc_status": "pending",
    })])
    write_table(pairs, bench / "pairs.parquet", "pairs")

    splits = pd.DataFrame([
        _defaults(SPLITS, {"case_id": "case1", "slide_id": "S1", "split": "test"}),
        _defaults(SPLITS, {"case_id": "case2", "slide_id": "S2", "split": "test"}),
    ])
    write_table(splits, bench / "splits.parquet", "splits")

    raw = yaml.safe_load(open(ROOT / "configs" / "example10.yaml"))
    raw["context"]["target_mpp"] = 2.0
    raw["context"]["size_px"] = 64
    raw["query"]["target_mpp"] = 0.25
    raw["query"]["size_px"] = 64
    raw["slide"]["max_mpp_l0"] = 0.1
    raw["qc"]["reconstruction_sample_frac"] = 1.0
    raw["qc"]["overlays_per_slide"] = 0
    raw["storage"]["materialize_pilot_png"] = False
    raw["source"]["gdc_dir"] = str(source_root)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    (bench / "config.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))

    from pyrafault.config import load_config
    cfg = load_config(cfg_path)
    (root / "04_features" / "feature_manifest.json").write_text(json.dumps({
        "section_digests": cfg.section_digests(),
    }))

    def run():
        import sys
        argv = sys.argv
        sys.argv = ["09_run_qc.py", "--config", str(cfg_path), "--data-root", str(root),
                    "--no-overlays"]
        try:
            try:
                mod.main()
            except SystemExit:
                pass  # unrelated violations (e.g. leakage, on a 1-row table) are expected
        finally:
            sys.argv = argv

    true_box = BBoxL0(100, 100, 64, 64)
    return run, bench, fakes, true_box, params


def test_out_of_slide_hash_comes_from_true_slide_id_not_slide_id(stage09_cross_slide_fixture):
    run, bench, fakes, true_box, params = stage09_cross_slide_fixture
    run()

    written = pd.read_parquet(bench / "pairs.parquet").set_index("pair_id") \
        .loc["p_cross", "served_pixel_sha256"]
    assert written, "stage 09 must have filled the empty hash on its first sample"

    correct_img, _ = read_at_mpp(fakes["S2"], true_box, 0.25, 64, "v1")
    correct = image_sha256(degrade_apply(correct_img, "geometry", params))

    wrong_img, _ = read_at_mpp(fakes["S1"], true_box, 0.25, 64, "v1")
    wrong = image_sha256(degrade_apply(wrong_img, "geometry", params))

    assert written == correct, (
        "the hash stage 09 wrote for an out_of_slide pair must be the patch read from "
        "true_slide_id (S2), not slide_id (S1)"
    )
    # This is the assertion an inverted true_slide_id/slide_id fallback would fail:
    # reading the identical box from the wrong slide must produce different bytes.
    assert written != wrong
    assert correct != wrong


def test_a_second_run_compares_the_cross_slide_hash_and_still_passes(
        stage09_cross_slide_fixture):
    """The written hash is not just self-consistent with one run's own read -- a
    second, independent pass through the same (now frozen) cross-slide row must
    re-derive the identical bytes and be reported as compared, not re-filled."""
    run, bench, fakes, true_box, params = stage09_cross_slide_fixture
    run()
    first = pd.read_parquet(bench / "pairs.parquet").set_index("pair_id") \
        .loc["p_cross", "served_pixel_sha256"]

    run()
    second = pd.read_parquet(bench / "pairs.parquet").set_index("pair_id") \
        .loc["p_cross", "served_pixel_sha256"]
    status = pd.read_parquet(bench / "pairs.parquet").set_index("pair_id") \
        .loc["p_cross", "qc_status"]

    assert second == first
    assert status == "pass"

    violations = pd.read_parquet(bench.parent / "05_qc" / "qc_violations.parquet")
    assert "served_pixel_mismatch" not in set(violations["check"])


def test_qc_stats_json_records_the_verify_fill_split(stage09_cross_slide_fixture):
    """The one untested link in that wiring: `served_pixel_verification` has to reach
    the persisted `qc_stats.json`, not just stdout and the markdown report."""
    run, bench, _fakes, _true_box, _params = stage09_cross_slide_fixture
    run()

    stats = json.loads((bench.parent / "05_qc" / "qc_stats.json").read_text())

    assert stats["served_pixel_verification"] == {
        "sampled": 2, "compared": 0, "newly_hashed": 2}


def test_the_loader_reproduces_the_frozen_served_pixel_hash(stage09_cross_slide_fixture):
    """`pairs.served_pixel_sha256` is frozen by stage 09's own copy of the serving
    recipe (`read_at_mpp` -> `degrade_apply` -> cache), deliberately independent of
    `PyraFaultDataset._image`. Nothing asserted that the loader -- the thing a
    collaborator actually runs -- reproduces it, so a change to `_image` would have
    made every frozen hash wrong while stage 09 kept "verifying" against its own
    unchanged copy: the failure the hash exists to prevent, inverted.

    Over the synthetic fixture rather than the built pilot benchmark, on purpose: the
    closest existing coverage (tests/test_dataset_degradation.py) depends on the
    `benchmark_root`/`gdc_root` fixtures and therefore skips on a fresh clone, in CI,
    and on the collaborator's machine -- precisely the machine the hash protects.

    Both bands are checked: `none` alone would pass for a loader that stopped applying
    degradations altogether, and `severe` alone would not show that the identity path
    is byte-identical.
    """
    run, bench, fakes, _true_box, params = stage09_cross_slide_fixture
    run()

    pairs = pd.read_parquet(bench / "pairs.parquet").set_index("pair_id")
    frozen = pairs["served_pixel_sha256"]
    assert frozen["p_none"] and frozen["p_cross"], "stage 09 froze no hash to compare to"
    assert pairs.loc["p_none", "degradation_band"] == "none"
    assert pairs.loc["p_cross", "degradation_band"] == "severe"

    # `open_slide` is patched on the loader too (see the fixture), so this path is only
    # joined with `source_relpath`, never opened.
    source_root = bench.parent.parent / "source"
    with PyraFaultDataset(bench, source_root, split="test", materialize=False) as ds:
        for pair_id in ("p_none", "p_cross"):
            served = ds[pair_id]["inputs"]["query"]
            assert image_sha256(served) == frozen[pair_id], (
                f"{pair_id}: the loader's served pixels do not reproduce the hash "
                "stage 09 froze -- the two copies of the serving recipe have drifted"
            )

    # And the severe row really is degraded, so the assertion above is not vacuous for
    # a loader that dropped `degrade_apply`.
    undegraded, _ = read_at_mpp(fakes["S2"], BBoxL0(100, 100, 64, 64), 0.25, 64, "v1")
    assert image_sha256(undegraded) != frozen["p_cross"]
    assert image_sha256(degrade_apply(undegraded, "geometry", params)) == frozen["p_cross"]


def test_the_overlay_renders_the_degraded_query_and_names_the_band(
        stage09_cross_slide_fixture, monkeypatch):
    """An overlay is the human check that a freeze is what it claims to be. Before
    this, the montage for a 15-degree-rotated row showed a perfectly registered Q with
    nothing on the page saying a rotation had been applied -- the artefact positively
    suggested the degradation had not taken effect."""
    run, bench, fakes, true_box, params = stage09_cross_slide_fixture
    mod = _load_script("09_run_qc.py")
    monkeypatch.setattr(mod, "open_slide", lambda path, slide_id: fakes[slide_id])

    calls = {}

    def fake_render_pair(ctx_img, qry_img, true_px, claim_px, caption):
        calls[caption.split()[0]] = (qry_img, caption)
        return Image.new("RGB", (4, 4))

    monkeypatch.setattr(mod, "render_pair", fake_render_pair)

    from pyrafault.config import load_config
    cfg = load_config(bench / "config.yaml")
    tables = {name: read_table(bench / f"{name}.parquet", name)
              for name in ("slides", "contexts", "queries", "pairs")}
    written, unrendered = mod._overlays(
        cfg, tables["slides"], tables["contexts"], tables["queries"], tables["pairs"],
        Path(cfg.source.root_dir), bench.parent / "05_qc" / "overlays", per_slide=2)

    assert written == 2 and unrendered == []
    degraded, caption = calls["p_cross"]
    undegraded, _ = read_at_mpp(fakes["S2"], true_box, 0.25, 64, "v1")
    assert image_sha256(degraded) == image_sha256(
        degrade_apply(undegraded, "geometry", params))
    assert image_sha256(degraded) != image_sha256(undegraded)
    assert "degradation=severe/geometry" in caption
    assert "degradation=none/none" in calls["p_none"][1]


def test_a_dangling_query_reference_is_reported_not_raised(stage09_cross_slide_fixture):
    """`check_invariants` has already recorded this row's dangling reference by name,
    and stage 09 deliberately does not exit early on violations -- so raising here
    destroyed qc_report.md, qc_stats.json and qc_violations.parquet for a table whose
    problem was already diagnosed."""
    run, bench, *_ = stage09_cross_slide_fixture
    pairs = read_table(bench / "pairs.parquet", "pairs")
    pairs.loc[pairs["pair_id"] == "p_none", "query_id"] = "q_gone"
    write_table(pairs, bench / "pairs.parquet", "pairs")

    run()   # must not raise KeyError

    qc = bench.parent / "05_qc"
    checks = set(pd.read_parquet(qc / "qc_violations.parquet")["check"])
    assert "served_pixel_unresolved" in checks
    assert "dangling_reference" in checks    # the diagnosis itself still gets made
    assert json.loads((qc / "qc_stats.json").read_text())["counts"]["pairs"] == 2


def test_a_dangling_slide_reference_does_not_destroy_the_report(
        stage09_cross_slide_fixture):
    """The other lookup in the same loop: `slides[slides.slide_id == src].iloc[0]`
    raised IndexError on a pair whose served patch names a slide the table never had."""
    run, bench, *_ = stage09_cross_slide_fixture
    pairs = read_table(bench / "pairs.parquet", "pairs")
    pairs.loc[pairs["pair_id"] == "p_cross", "true_slide_id"] = "S_gone"
    write_table(pairs, bench / "pairs.parquet", "pairs")

    run()   # must not raise IndexError

    qc = bench.parent / "05_qc"
    detail = pd.read_parquet(qc / "qc_violations.parquet")
    assert "served_pixel_unresolved" in set(detail["check"])
    assert (qc / "qc_report.md").exists()


def test_the_benchmark_manifest_is_written_with_a_trailing_newline(
        stage09_cross_slide_fixture):
    """scripts/migrate_pairs_degradation.py writes one and this did not, so the two
    alternated the file's last byte: every stage-09 run after a migration produced a
    one-byte diff in a committed file for no reason."""
    run, bench, *_ = stage09_cross_slide_fixture
    run()

    assert (bench / "benchmark_manifest.json").read_text().endswith("}\n")
