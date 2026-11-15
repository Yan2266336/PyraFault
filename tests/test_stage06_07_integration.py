"""Stage 06/07 script-level regression tests.

These pin two fixes that had no regression test on the plan's own ledger
(the random_same_context hardness backfill and the hard_pool guard), plus I4
(negative pools must exclude stage-05 hard-failed candidates), plus Task 3's fix round 1
(stage 07 must assign bands to the rows it appends, not just leave stage 06's alone).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import yaml

from pyrafault.degrade_assign import BANDS
from pyrafault.faults import SUBSTITUTED_TYPES
from pyrafault.schema import CONTEXTS, PAIRS, QUERIES, QUERY_CANDIDATES, write_table

ROOT = Path(__file__).resolve().parents[1]
CTX_FOV = 16254
Q_FOV = 1016
MPP_X = 0.2520


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _defaults(schema: pa.Schema) -> dict:
    out = {}
    for f in schema:
        t = f.type
        if pa.types.is_integer(t):
            out[f.name] = 0
        elif pa.types.is_floating(t):
            out[f.name] = 0.0
        elif pa.types.is_boolean(t):
            out[f.name] = False
        elif pa.types.is_string(t):
            out[f.name] = ""
        elif pa.types.is_list(t):
            out[f.name] = []
        else:
            out[f.name] = None
    return out


BASE_CFG = {
    "seed": 20260812, "pipeline_version": "v1", "extraction_version": "v1",
    "source": {"gdc_dir": "unused", "manifest": "m", "sample_sheet": "s",
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


def _write_cfg(path: Path, hard_pool: str = "context") -> None:
    cfg = json.loads(json.dumps(BASE_CFG))
    cfg["faults"]["hard_same_slide"]["pool"] = hard_pool
    path.write_text(yaml.safe_dump(cfg))


def _build_benchmark(tmp_path: Path, n_candidates: int = 5):
    """One slide, one context, `n_candidates` non-overlapping query-candidate boxes
    spaced 2000px apart (> 1.5 * Q_FOV eligibility distance), one frozen query on
    candidate 0."""
    root = tmp_path / "data"
    bench = root / "01_benchmark"
    bench.mkdir(parents=True)

    contexts = pd.DataFrame([{**_defaults(CONTEXTS),
        "context_id": "S1__c0000", "slide_id": "S1",
        "bbox_l0_x": 0, "bbox_l0_y": 0, "bbox_l0_w": CTX_FOV, "bbox_l0_h": CTX_FOV,
        "target_mpp": 2.0, "output_size": 2048, "extraction_version": "v1",
        "split": "dev",
    }])
    write_table(contexts, bench / "contexts.parquet", "contexts")

    cand_rows = []
    for i in range(n_candidates):
        cand_rows.append({**_defaults(QUERY_CANDIDATES),
            "candidate_id": f"S1__c0000__cc{i:04d}", "context_id": "S1__c0000",
            "slide_id": "S1", "bbox_l0_x": 1000 + i * 2000, "bbox_l0_y": 1000,
            "bbox_l0_w": Q_FOV, "bbox_l0_h": Q_FOV, "tissue_fraction": 1.0,
            "selected": i == 0,
        })
    cands = pd.DataFrame(cand_rows)
    write_table(cands, bench / "query_candidates.parquet", "query_candidates")

    queries = pd.DataFrame([{**_defaults(QUERIES),
        "query_id": "S1__c0000__q0", "context_id": "S1__c0000", "slide_id": "S1",
        "candidate_id": "S1__c0000__cc0000",
        "true_bbox_l0_x": 1000, "true_bbox_l0_y": 1000,
        "true_bbox_l0_w": Q_FOV, "true_bbox_l0_h": Q_FOV,
        "target_mpp": 0.5, "output_size": 512, "extraction_version": "v1",
        "split": "dev",
    }])
    write_table(queries, bench / "queries.parquet", "queries")

    return root, cand_rows


def _write_features(root: Path, cand_rows: list[dict], dim: int = 4,
                    invalid_ids: set[str] | None = None):
    invalid_ids = invalid_ids or set()
    (root / "04_features").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    ids = [c["candidate_id"] for c in cand_rows]
    emb = rng.normal(size=(len(ids), dim)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    valid = np.array([cid not in invalid_ids for cid in ids], dtype=bool)
    emb[~valid] = 0.0
    np.savez_compressed(root / "04_features" / "S1.npz",
                        candidate_id=np.array(ids), embedding=emb, valid=valid)
    (root / "04_features" / "feature_manifest.json").write_text(
        json.dumps({"name": "handcrafted", "revision": "v1", "dim": dim,
                    "config_digest": "irrelevant"}))
    return dict(zip(ids, emb))


def test_row_true_within_context_is_computed_not_hardcoded():
    """Promoted deferred item: stage 06's _row wrote true_within_context and
    local_recovery_possible as literal True. In round 1 assert_pair_invariants (called
    before _row) guarantees true is always inside context, so the only way to prove
    this is computed rather than hardcoded is to call _row directly with a true bbox
    manufactured to sit outside the context."""
    import types

    from pyrafault.faults import FaultResult
    from pyrafault.geom import BBoxL0

    stage06 = _load_script("06_build_pairs.py")

    class Cfg:
        seed = 1
        pipeline_version = "v1"

    ctx = BBoxL0(0, 0, 16254, 16254)
    outside_true = BBoxL0(20000, 20000, Q_FOV, Q_FOV)   # deliberately NOT inside ctx
    q = types.SimpleNamespace(
        query_id="q0", slide_id="S1", context_id="S1__c0000",
        true_bbox_l0_x=outside_true.x, true_bbox_l0_y=outside_true.y,
        true_bbox_l0_w=outside_true.w, true_bbox_l0_h=outside_true.h,
        split="dev",
    )
    res = FaultResult(claimed=outside_true, fault_type="clean", severity=None,
                      normalized_offset=0.0, offset_frac_context=0.0, iou_claim_true=1.0)

    row = stage06._row(Cfg, ctx, q, res)
    assert row["true_within_context"] is False
    assert row["local_recovery_possible"] is False

    inside_true = BBoxL0(1000, 1000, Q_FOV, Q_FOV)
    q2 = types.SimpleNamespace(
        query_id="q1", slide_id="S1", context_id="S1__c0000",
        true_bbox_l0_x=inside_true.x, true_bbox_l0_y=inside_true.y,
        true_bbox_l0_w=inside_true.w, true_bbox_l0_h=inside_true.h,
        split="dev",
    )
    res2 = FaultResult(claimed=inside_true, fault_type="clean", severity=None,
                       normalized_offset=0.0, offset_frac_context=0.0, iou_claim_true=1.0)
    row2 = stage06._row(Cfg, ctx, q2, res2)
    assert row2["true_within_context"] is True
    assert row2["local_recovery_possible"] is True


def test_hard_pool_slide_is_refused_before_touching_data(tmp_path, monkeypatch):
    """The hard_pool guard: mine_hard only ever pools the parent context's own
    candidates, so a config asking for slide-wide pooling must fail loudly rather
    than silently label context-pooled negatives as slide-pooled."""
    cfg_path = tmp_path / "config.yaml"
    _write_cfg(cfg_path, hard_pool="slide")
    data_root = tmp_path / "data"  # deliberately never populated

    stage07 = _load_script("07_mine_hard_negatives.py")
    monkeypatch.setattr(sys, "argv",
                        ["07_mine_hard_negatives.py", "--config", str(cfg_path),
                         "--data-root", str(data_root)])
    with pytest.raises(NotImplementedError):
        stage07.main()


def test_random_same_context_hardness_score_is_backfilled(tmp_path, monkeypatch):
    """Ruling X: hardness_score must be populated for random_same_context, not left
    null. Stage 06 has no embedding access; stage 07 backfills it via a coordinate
    round-trip to the matching query_candidates row."""
    cfg_path = tmp_path / "config.yaml"
    _write_cfg(cfg_path)
    root, cand_rows = _build_benchmark(tmp_path)
    embeddings = _write_features(root, cand_rows)

    stage06 = _load_script("06_build_pairs.py")
    monkeypatch.setattr(sys, "argv",
                        ["06_build_pairs.py", "--config", str(cfg_path),
                         "--data-root", str(root)])
    stage06.main()

    stage07 = _load_script("07_mine_hard_negatives.py")
    monkeypatch.setattr(sys, "argv",
                        ["07_mine_hard_negatives.py", "--config", str(cfg_path),
                         "--data-root", str(root)])
    stage07.main()

    pairs = pd.read_parquet(root / "01_benchmark" / "pairs.parquet")
    random_rows = pairs[pairs["fault_type"] == "random_same_context"]
    assert len(random_rows) == 1
    r = random_rows.iloc[0]
    assert r["hardness_score"] is not None and not pd.isna(r["hardness_score"])

    claimed_id = None
    for c in cand_rows:
        if (c["bbox_l0_x"], c["bbox_l0_y"]) == (r["claimed_bbox_l0_x"], r["claimed_bbox_l0_y"]):
            claimed_id = c["candidate_id"]
    assert claimed_id is not None
    expected = float(embeddings["S1__c0000__cc0000"] @ embeddings[claimed_id])
    assert r["hardness_score"] == pytest.approx(expected, abs=1e-5)


def test_hard_failed_candidate_is_excluded_from_both_negative_pools(tmp_path, monkeypatch):
    """I4: a candidate stage 05 hard-failed (valid=False, all-zero embedding) must
    never be selected as a random_same_context claim or a hard_mismatch claim, even
    when its (fake) embedding would otherwise be the top cosine match."""
    cfg_path = tmp_path / "config.yaml"
    _write_cfg(cfg_path)
    root, cand_rows = _build_benchmark(tmp_path, n_candidates=5)
    invalid_id = "S1__c0000__cc0001"  # would otherwise be the closest match to true
    _write_features(root, cand_rows, invalid_ids={invalid_id})

    stage06 = _load_script("06_build_pairs.py")
    monkeypatch.setattr(sys, "argv",
                        ["06_build_pairs.py", "--config", str(cfg_path),
                         "--data-root", str(root)])
    stage06.main()

    stage07 = _load_script("07_mine_hard_negatives.py")
    monkeypatch.setattr(sys, "argv",
                        ["07_mine_hard_negatives.py", "--config", str(cfg_path),
                         "--data-root", str(root)])
    stage07.main()

    pairs = pd.read_parquet(root / "01_benchmark" / "pairs.parquet")
    invalid_box = next(c for c in cand_rows if c["candidate_id"] == invalid_id)
    for fault_type in ("random_same_context", "hard_mismatch"):
        row = pairs[pairs["fault_type"] == fault_type].iloc[0]
        claimed = (row["claimed_bbox_l0_x"], row["claimed_bbox_l0_y"])
        assert claimed != (invalid_box["bbox_l0_x"], invalid_box["bbox_l0_y"]), (
            f"{fault_type} must never claim the hard-failed candidate {invalid_id}"
        )


def test_stage07_assigns_bands_to_the_rows_it_appends(tmp_path, monkeypatch):
    """Task 3 fix round 1: stage 07 appends hard_mismatch (and, when enabled, the
    out_of_context/out_of_slide families) after stage 06 has already assigned bands to
    its own rows. Without this, every row stage 07 owns keeps a null
    degradation_kind -- which the loader's `_degradation` reads as "none" -- so a fresh
    stage00->09 build would silently ship those fault types undegraded while stage 06's
    fault types were not."""
    cfg_path = tmp_path / "config.yaml"
    _write_cfg(cfg_path)
    root, cand_rows = _build_benchmark(tmp_path, n_candidates=5)
    _write_features(root, cand_rows)

    stage06 = _load_script("06_build_pairs.py")
    monkeypatch.setattr(sys, "argv",
                        ["06_build_pairs.py", "--config", str(cfg_path),
                         "--data-root", str(root)])
    stage06.main()

    stage07 = _load_script("07_mine_hard_negatives.py")
    monkeypatch.setattr(sys, "argv",
                        ["07_mine_hard_negatives.py", "--config", str(cfg_path),
                         "--data-root", str(root)])
    stage07.main()

    pairs = pd.read_parquet(root / "01_benchmark" / "pairs.parquet")
    assert "hard_mismatch" in set(pairs["fault_type"])  # sanity: stage 07 did append rows
    null_kind = pairs[pairs["degradation_kind"].isna()]
    assert null_kind.empty, (
        f"rows with no degradation_kind assigned: "
        f"{null_kind['fault_type'].value_counts().to_dict()}"
    )
    hard_rows = pairs[pairs["fault_type"] == "hard_mismatch"]
    assert hard_rows["degradation_band"].isin(BANDS).all()
    assert (hard_rows["served_pixel_sha256"] == "").all()


def test_stage07_does_not_reassign_stage06_rows_bands(tmp_path, monkeypatch):
    """The other half of the fix: assigning bands to the rows stage 07 appends must not
    redraw the bands stage 06 already froze for its own rows -- a rerun of stage 07
    (or a stage 07 run at all) must leave clean/adjacent_drift/random_same_context's
    degradation columns exactly as stage 06 wrote them."""
    cfg_path = tmp_path / "config.yaml"
    _write_cfg(cfg_path)
    root, cand_rows = _build_benchmark(tmp_path, n_candidates=5)
    _write_features(root, cand_rows)

    stage06 = _load_script("06_build_pairs.py")
    monkeypatch.setattr(sys, "argv",
                        ["06_build_pairs.py", "--config", str(cfg_path),
                         "--data-root", str(root)])
    stage06.main()

    before = pd.read_parquet(root / "01_benchmark" / "pairs.parquet")
    before_band = before.set_index("pair_id")["degradation_band"].to_dict()
    before_params = before.set_index("pair_id")["degradation_params"].to_dict()
    stage06_pair_ids = set(before["pair_id"])

    stage07 = _load_script("07_mine_hard_negatives.py")
    monkeypatch.setattr(sys, "argv",
                        ["07_mine_hard_negatives.py", "--config", str(cfg_path),
                         "--data-root", str(root)])
    stage07.main()

    after = pd.read_parquet(root / "01_benchmark" / "pairs.parquet")
    owned = ("hard_mismatch",) + SUBSTITUTED_TYPES
    stage06_rows = after[~after["fault_type"].isin(owned)]
    assert set(stage06_rows["pair_id"]) == stage06_pair_ids, (
        "stage 07 must not add, drop or rename a row it does not own"
    )
    for r in stage06_rows.itertuples():
        assert r.degradation_band == before_band[r.pair_id], r.pair_id
        assert r.degradation_params == before_params[r.pair_id], r.pair_id
