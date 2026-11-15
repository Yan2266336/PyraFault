"""Stage 05 --gather regression tests (I6).

feature_manifest.json used to be written unconditionally by every array task to one
shared path -- ten concurrent non-atomic writers racing on the same file. Now each
array task writes its own per-slide shard manifest, and only the --gather phase
produces the canonical file, after requiring the shards to unanimously agree.

Also pins the promoted deferred item: the context-descriptor gather must refuse to
shrink the populated self_similarity/morphology_summary count under a partial shard set.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest
import yaml

from pyrafault.schema import CONTEXTS, QUERIES, QUERY_CANDIDATES, write_table

ROOT = Path(__file__).resolve().parents[1]


def _load_stage05():
    path = ROOT / "scripts" / "05_sample_visual_diversity.py"
    spec = importlib.util.spec_from_file_location("stage05_gather_test", path)
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
    "seed": 1, "pipeline_version": "v1", "extraction_version": "v1",
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


def _write_cfg(path: Path) -> None:
    path.write_text(yaml.safe_dump(json.loads(json.dumps(BASE_CFG))))


def _run_gather(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    _write_cfg(cfg_path)
    stage05 = _load_stage05()
    monkeypatch.setattr(sys, "argv",
                        ["05_sample_visual_diversity.py", "--config", str(cfg_path),
                         "--data-root", str(tmp_path / "data"), "--gather"])
    stage05.main()


def test_gather_raises_on_manifest_disagreement(tmp_path, monkeypatch):
    root = tmp_path / "data"
    shard = root / "05_qc" / "stage05"
    shard.mkdir(parents=True)
    (shard / "S1__manifest.json").write_text(json.dumps({"name": "uni", "revision": "a" * 40}))
    (shard / "S2__manifest.json").write_text(json.dumps({"name": "uni", "revision": "b" * 40}))

    with pytest.raises(RuntimeError, match="disagreement"):
        _run_gather(tmp_path, monkeypatch)


def test_gather_with_no_shard_manifests_raises(tmp_path, monkeypatch):
    (tmp_path / "data" / "05_qc" / "stage05").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="no per-slide feature manifests"):
        _run_gather(tmp_path, monkeypatch)


def test_gather_writes_the_agreed_manifest(tmp_path, monkeypatch):
    root = tmp_path / "data"
    shard = root / "05_qc" / "stage05"
    shard.mkdir(parents=True)
    desc = {"name": "uni", "revision": "a" * 40, "dim": 1024}
    (shard / "S1__manifest.json").write_text(json.dumps(desc))
    (shard / "S2__manifest.json").write_text(json.dumps(desc))
    # minimal valid empty-ish query/candidate/context shards so the rest of gather runs
    (root / "01_benchmark").mkdir(parents=True)
    ctx = pd.DataFrame([{**_defaults(CONTEXTS), "context_id": "S1__c0000", "slide_id": "S1",
                        "bbox_l0_w": 16254, "bbox_l0_h": 16254, "split": "dev"}])
    write_table(ctx, root / "01_benchmark" / "contexts.parquet", "contexts")
    q = pd.DataFrame([{**_defaults(QUERIES), "query_id": "q0", "context_id": "S1__c0000",
                      "slide_id": "S1", "true_bbox_l0_w": 1016, "true_bbox_l0_h": 1016,
                      "split": "dev"}])
    q.to_parquet(shard / "S1__q.parquet")
    qc = pd.DataFrame([{**_defaults(QUERY_CANDIDATES), "candidate_id": "c0",
                       "context_id": "S1__c0000", "slide_id": "S1",
                       "bbox_l0_w": 1016, "bbox_l0_h": 1016}])
    qc.to_parquet(shard / "S1__qc.parquet")
    ctx_shard = pd.DataFrame([{"context_id": "S1__c0000", "self_similarity": 0.5,
                              "morphology_summary": "{}"}])
    ctx_shard.to_parquet(shard / "S1__ctx.parquet")

    _run_gather(tmp_path, monkeypatch)

    written = json.loads((root / "04_features" / "feature_manifest.json").read_text())
    assert written == desc


def test_gather_refuses_to_shrink_populated_context_descriptors(tmp_path, monkeypatch):
    """I6 promoted item: a partial shard set (missing S2's ctx shard) must never
    silently overwrite S2's already-populated self_similarity with NaN."""
    root = tmp_path / "data"
    shard = root / "05_qc" / "stage05"
    shard.mkdir(parents=True)
    desc = {"name": "uni", "revision": "a" * 40, "dim": 1024}
    (shard / "S1__manifest.json").write_text(json.dumps(desc))

    (root / "01_benchmark").mkdir(parents=True)
    # contexts.parquet already has S2's descriptor populated from an earlier full run.
    ctx = pd.DataFrame([
        {**_defaults(CONTEXTS), "context_id": "S1__c0000", "slide_id": "S1",
         "bbox_l0_w": 16254, "bbox_l0_h": 16254, "split": "dev",
         "self_similarity": 0.6, "morphology_summary": "{}"},
        {**_defaults(CONTEXTS), "context_id": "S2__c0000", "slide_id": "S2",
         "bbox_l0_w": 16254, "bbox_l0_h": 16254, "split": "dev",
         "self_similarity": 0.7, "morphology_summary": "{}"},
    ])
    write_table(ctx, root / "01_benchmark" / "contexts.parquet", "contexts")

    q = pd.DataFrame([{**_defaults(QUERIES), "query_id": "q0", "context_id": "S1__c0000",
                      "slide_id": "S1", "true_bbox_l0_w": 1016, "true_bbox_l0_h": 1016,
                      "split": "dev"}])
    q.to_parquet(shard / "S1__q.parquet")
    qc = pd.DataFrame([{**_defaults(QUERY_CANDIDATES), "candidate_id": "c0",
                       "context_id": "S1__c0000", "slide_id": "S1",
                       "bbox_l0_w": 1016, "bbox_l0_h": 1016}])
    qc.to_parquet(shard / "S1__qc.parquet")
    # Only S1's context descriptor shard is present -- S2's is missing (partial set).
    ctx_shard = pd.DataFrame([{"context_id": "S1__c0000", "self_similarity": 0.5,
                              "morphology_summary": "{}"}])
    ctx_shard.to_parquet(shard / "S1__ctx.parquet")

    with pytest.raises(RuntimeError, match="shrink"):
        _run_gather(tmp_path, monkeypatch)
