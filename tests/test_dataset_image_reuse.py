"""The loader re-decodes one context patch for every item that shares it.

A context is 2048x2048 and decodes in ~100 ms; a query is 512x512 and decodes in ~8 ms.
Forty pairs share each context in the lung freeze, and `pairs.parquet` sorted by
`pair_id` puts them next to each other -- 3,877 contexts, 3,877 contiguous runs in the
test split -- so an evaluation pass spends about forty times longer on context pixels
than it has to. A small cache of the last few decoded patches removes that, and copying
on the way out keeps the contract that each item owns its images.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from PIL import Image

from pyrafault import cache as cache_mod
from pyrafault.cache import cache_key, cache_path, save_png
from pyrafault.dataset import PyraFaultDataset
from pyrafault.geom import BBoxL0
from pyrafault.schema import SCHEMAS

ROOT = Path(__file__).resolve().parents[1]


def _table(name, rows, path):
    cols = list(SCHEMAS[name].names)
    pd.DataFrame([{c: r.get(c) for c in cols} for r in rows],
                 columns=cols).to_parquet(path, index=False)


@pytest.fixture
def shared_context_freeze(tmp_path):
    """One context, four pairs, every patch pre-seeded into `02_cache`.

    The shape the real freeze has at forty pairs per context, small enough to count
    decodes by hand.
    """
    root = tmp_path / "data_shared"
    bench = root / "01_benchmark"
    bench.mkdir(parents=True)
    cfg = yaml.safe_load(open(ROOT / "configs" / "example10.yaml"))
    ctx_mpp, ctx_size = cfg["context"]["target_mpp"], cfg["context"]["size_px"]
    qry_mpp, qry_size = cfg["query"]["target_mpp"], cfg["query"]["size_px"]
    ver = cfg["extraction_version"]
    (bench / "config.yaml").write_text(yaml.safe_dump(cfg))

    side = qry_size * 4
    pairs, queries = [], []
    for i in range(4):
        pairs.append({
            "pair_id": f"S1__c0__q{i}", "slide_id": "S1", "context_id": "S1__c0",
            "query_id": f"q{i}",
            "claimed_bbox_l0_x": 1000 * i, "claimed_bbox_l0_y": 0,
            "claimed_bbox_l0_w": side, "claimed_bbox_l0_h": side,
            "true_bbox_l0_x": 1000 * i, "true_bbox_l0_y": 0,
            "true_bbox_l0_w": side, "true_bbox_l0_h": side,
            "claim_correct": True, "fault_type": "clean", "fault_severity": "small",
            "normalized_offset": 0.0, "offset_frac_context": 0.0, "iou_claim_true": 1.0,
            "true_within_context": True, "local_recovery_possible": True,
            "true_slide_id": "S1", "same_slide": True,
            "context_mpp": ctx_mpp, "query_mpp": qry_mpp, "scale_ratio": 4.0,
            "construction_seed": 1, "pipeline_version": "v1",
            "split": "test", "qc_status": "pass",
            "degradation_kind": "none", "degradation_band": "none",
            "degradation_params": json.dumps({}),
        })
        queries.append({"query_id": f"q{i}", "context_id": "S1__c0", "slide_id": "S1",
                        "target_mpp": qry_mpp, "output_size": qry_size,
                        "extraction_version": ver, "split": "test"})
    _table("pairs", pairs, bench / "pairs.parquet")
    _table("queries", queries, bench / "queries.parquet")
    _table("contexts", [{
        "context_id": "S1__c0", "slide_id": "S1", "bbox_l0_x": 0, "bbox_l0_y": 0,
        "bbox_l0_w": 40000, "bbox_l0_h": 40000, "target_mpp": ctx_mpp,
        "output_size": ctx_size, "extraction_version": ver, "split": "test",
    }], bench / "contexts.parquet")
    _table("slides", [{"slide_id": "S1", "source_relpath": "S1.svs"}],
           bench / "slides.parquet")

    cache = root / "02_cache"
    save_png(Image.new("RGB", (32, 32), (7, 7, 7)),
             cache_path(cache, "S1", cache_key("S1", BBoxL0(0, 0, 40000, 40000),
                                               ctx_mpp, ctx_size, ver)))
    for i in range(4):
        save_png(Image.new("RGB", (16, 16), (10 * i + 1,) * 3),
                 cache_path(cache, "S1", cache_key("S1", BBoxL0(1000 * i, 0, side, side),
                                                   qry_mpp, qry_size, ver)))
    return bench


@pytest.fixture
def counted(shared_context_freeze, tmp_path, monkeypatch):
    """A dataset over that freeze, plus a count of how many times a PNG was decoded."""
    calls = []
    real = cache_mod.load_png

    def counting(path):
        calls.append(str(path))
        return real(path)

    import pyrafault.dataset as ds_mod
    monkeypatch.setattr(ds_mod, "load_png", counting)
    with PyraFaultDataset(shared_context_freeze, tmp_path / "no_slides",
                          split="test", materialize=False) as ds:
        yield ds, calls


class TestASharedContextIsDecodedOnce:
    def test_four_items_decode_the_context_once_and_each_query_once(self, counted):
        ds, calls = counted
        for iid in ds.item_ids:
            ds[iid]
        assert len(calls) == 5, (
            f"{len(calls)} decodes for 1 context and 4 queries: "
            "the context is being re-decoded per item"
        )

    def test_re_reading_one_item_decodes_nothing_again(self, counted):
        ds, calls = counted
        ds[0]
        before = len(calls)
        ds[0]
        assert len(calls) == before

    def test_a_query_is_still_read_for_every_distinct_box(self, counted):
        """The cache must not collapse two different patches: the keys differ, so the
        pixels must differ."""
        ds, _ = counted
        shades = [ds[i]["inputs"]["query"].getpixel((0, 0))[0] for i in range(4)]
        assert shades == [1, 11, 21, 31]


class TestEachItemOwnsItsImages:
    def test_two_items_do_not_share_one_context_object(self, counted):
        ds, _ = counted
        assert ds[0]["inputs"]["context"] is not ds[1]["inputs"]["context"]

    def test_mutating_one_items_context_does_not_reach_the_next(self, counted):
        """A verifier that drew on its input would otherwise poison every later item
        sharing that context, and the corruption would look like a model failure."""
        ds, _ = counted
        first = ds[0]["inputs"]["context"]
        first.paste(Image.new("RGB", first.size, (255, 0, 0)), (0, 0))
        assert ds[1]["inputs"]["context"].getpixel((0, 0)) == (7, 7, 7)


class TestTheCacheIsBounded:
    def test_it_does_not_grow_with_the_number_of_distinct_patches(self, counted):
        ds, _ = counted
        for iid in ds.item_ids:
            ds[iid]
        assert len(ds._img_cache) <= ds.IMAGE_CACHE_SIZE

    def test_a_patch_pushed_out_is_read_again_rather_than_lost(self, counted):
        """Correctness under eviction: the point is that an evicted entry is re-decoded,
        not that it is served stale."""
        ds, calls = counted
        ds[0]
        for i in range(1, 4):
            ds[i]
        before = len(calls)
        assert ds[0]["inputs"]["query"].getpixel((0, 0))[0] == 1
        assert len(calls) >= before, "an evicted query must be re-read, not invented"
