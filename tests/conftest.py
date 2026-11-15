import hashlib
import json
import os

import pandas as pd
import pytest
import yaml

PILOT = (
    "data/00_source/gdc_download/0c5b9279-e9d5-4ac5-bd71-5c21d73ef417/"
    "TCGA-55-8094-01Z-00-DX1.8dc29615-e124-4f17-81a1-c0b20c38d12c.svs"
)

BENCH = "data/01_benchmark"
GDC = "data/00_source/gdc_download"

# A minimal but load_config-valid pipeline config, for freezes fabricated in tests
# rather than built by the real stages. Shape copied from
# tests/test_stage06_07_integration.py's BASE_CFG, which is exercised against the real
# stage 06/07 scripts.
_MIGRATION_CFG: dict = {
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


@pytest.fixture
def lung_like_freeze():
    """A factory for a pairs.parquet built before Task 2's degradation columns
    existed -- the shape every freeze on disk is in until it is migrated -- plus its
    config.yaml and a benchmark_manifest.json pinning the pre-migration table.

    Named for the freeze the migration script exists to unblock: rebuilding it from
    stage 00 would re-embed millions of query candidates with UNI to add three columns
    that move no coordinate (see tests/test_migrate_pairs_degradation.py).
    """
    def _make(tmp_path, seed: int = 20260929, n_per_cell: int = 12):
        root = tmp_path / "data_lung_like"
        bench = root / "01_benchmark"
        bench.mkdir(parents=True)

        rows = []
        for split in ("calibration", "test"):
            for fault in ("clean", "adjacent_drift", "out_of_slide"):
                for i in range(n_per_cell):
                    pair_id = f"{split}-{fault}-{i}"
                    rows.append({
                        "pair_id": pair_id, "slide_id": "S1", "context_id": "S1__c0",
                        "query_id": f"{pair_id}__q",
                        "claimed_bbox_l0_x": 0, "claimed_bbox_l0_y": 0,
                        "claimed_bbox_l0_w": 512, "claimed_bbox_l0_h": 512,
                        "true_bbox_l0_x": 0, "true_bbox_l0_y": 0,
                        "true_bbox_l0_w": 512, "true_bbox_l0_h": 512,
                        "claim_correct": fault == "clean",
                        "fault_type": fault, "fault_severity": "small",
                        "normalized_offset": 0.0, "offset_frac_context": 0.0,
                        "iou_claim_true": 1.0,
                        "true_within_context": True, "local_recovery_possible": True,
                        "hardness_score": None, "hard_negative_miner": None,
                        "miner_model": None, "miner_revision": None,
                        "candidate_rank": None, "hard_pool": None,
                        "true_slide_id": "S1" if fault != "out_of_slide" else "S2",
                        "same_slide": fault != "out_of_slide",
                        "context_mpp": 2.0, "query_mpp": 0.5, "scale_ratio": 4.0,
                        "construction_seed": seed, "pipeline_version": "v1",
                        "split": split, "qc_status": "pending",
                    })
        pairs = pd.DataFrame(rows)
        pairs.to_parquet(bench / "pairs.parquet", compression="snappy", index=False)

        cfg = json.loads(json.dumps(_MIGRATION_CFG))
        cfg["seed"] = seed
        (bench / "config.yaml").write_text(yaml.safe_dump(cfg))

        digest = hashlib.sha256((bench / "pairs.parquet").read_bytes()).hexdigest()
        manifest = {
            "pipeline_version": "v1",
            "config_digest": "irrelevant-for-this-fixture",
            "tables": {"pairs.parquet": {"sha256": digest, "rows": len(pairs)}},
        }
        (bench / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2))

        return root
    return _make


@pytest.fixture
def write_config():
    """Write a `load_config`-valid config to `path`, built from `configs/example10.yaml`
    with the given mutations applied.

    Lifted from the ad hoc `yaml.safe_load(open(CFG)); mutate; write_text` pattern
    that `tests/test_config.py` used to repeat per test (see
    `test_digest_is_stable_and_sensitive`, `test_rejects_missing_section`) -- one
    builder that knows the pilot schema, not two that can silently drift apart.

    `extra`, if given, is merged in as additional top-level sections a config might
    one day carry that no stage has been taught to read yet (e.g. a future
    `degradation:` section) -- see tests/test_config_provenance.py. Any other keyword
    is `<section>_<field>` (`context_size_px` -> `raw["context"]["size_px"]`) or a
    bare top-level scalar key (`seed`); the section name is resolved against
    `raw`'s own top-level keys so `visual_sampling_n_clusters` still splits correctly
    despite the section name itself containing an underscore.
    """
    def _write(path, extra: dict | None = None, **overrides):
        raw = yaml.safe_load(open("configs/example10.yaml"))
        if extra:
            raw.update(extra)
        for dotted, value in overrides.items():
            if dotted in raw and not isinstance(raw[dotted], dict):
                raw[dotted] = value
                continue
            section = next(s for s in raw if isinstance(raw[s], dict)
                           and dotted.startswith(s + "_"))
            raw[section][dotted[len(section) + 1:]] = value
        path.write_text(yaml.safe_dump(raw, sort_keys=False))
        return path
    return _write


@pytest.fixture
def pilot_slide_path():
    if not os.path.exists(PILOT):
        pytest.skip("pilot slide not available")
    return PILOT


@pytest.fixture
def benchmark_root():
    if not os.path.exists(os.path.join(BENCH, "pairs.parquet")):
        pytest.skip("benchmark not built yet")
    return BENCH


@pytest.fixture
def gdc_root():
    if not os.path.exists(GDC):
        pytest.skip("GDC source not available")
    return GDC
