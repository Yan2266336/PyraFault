"""`PyraFaultDataset.meta` -- the item without its pixels.

A full-scale evaluation localises in parallel shards and then decides in one place. The
deciding pass needs the claim and the answer key and nothing else, so routing it through
`__getitem__` would re-read two cached patches for every row: 600,000 PNG decodes on the
lung freeze to recover integers that are already in `pairs.parquet`. `meta` is that half
of the item, and these pin that it really is the same half -- identical values, identical
gates, no pixels.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from pyrafault.dataset import INPUT_KEYS, TARGET_KEYS, PyraFaultDataset
from pyrafault.schema import SCHEMAS
from pyrafault.verify.runner import cache_key


def _table(name: str, rows: list[dict], path: Path) -> None:
    """Write `rows` as `name`, filling every column the schema demands.

    Spelling out all forty of `pairs`' columns here would make this fixture something
    to maintain against the schema; it only cares about a handful of them, so the rest
    are nulls taken from the schema itself.
    """
    cols = list(SCHEMAS[name].names)
    pd.DataFrame([{c: r.get(c) for c in cols} for r in rows], columns=cols).to_parquet(
        path, index=False
    )


@pytest.fixture
def pixel_less_freeze(tmp_path):
    """A complete, readable freeze whose slide files do not exist.

    That is the point: anything that touches pixels raises, so a test cannot pass by
    quietly reading them. The tables are the minimum `PyraFaultDataset` reads.
    """
    bench = tmp_path / "01_benchmark"
    bench.mkdir(parents=True)
    cfg = yaml.safe_load(open("configs/example10.yaml"))
    ctx_mpp = cfg["context"]["target_mpp"]
    ctx_size = cfg["context"]["size_px"]
    qry_mpp = cfg["query"]["target_mpp"]
    qry_size = cfg["query"]["size_px"]
    ver = cfg["extraction_version"]
    (bench / "config.yaml").write_text(yaml.safe_dump(cfg))

    rows = []
    for i, fault in enumerate(("clean", "adjacent_drift", "out_of_slide")):
        rows.append({
            "pair_id": f"p{i}", "slide_id": "S1", "context_id": "S1__c0",
            "query_id": f"p{i}__q",
            "claimed_bbox_l0_x": 100 * i, "claimed_bbox_l0_y": 0,
            "claimed_bbox_l0_w": 512, "claimed_bbox_l0_h": 512,
            "true_bbox_l0_x": 0, "true_bbox_l0_y": 0,
            "true_bbox_l0_w": 512, "true_bbox_l0_h": 512,
            "claim_correct": fault == "clean", "fault_type": fault,
            "fault_severity": "small", "normalized_offset": 0.0,
            "offset_frac_context": 0.0, "iou_claim_true": 1.0,
            "true_within_context": True, "local_recovery_possible": True,
            "hardness_score": None, "hard_negative_miner": None, "miner_model": None,
            "miner_revision": None, "candidate_rank": None, "hard_pool": None,
            "true_slide_id": "S1" if fault != "out_of_slide" else "S2",
            "same_slide": fault != "out_of_slide",
            "context_mpp": ctx_mpp, "query_mpp": qry_mpp, "scale_ratio": 4.0,
            "construction_seed": 1, "pipeline_version": "v1",
            "split": "test", "qc_status": "pass",
            "degradation_kind": "none", "degradation_band": "none",
            "degradation_params": json.dumps({}),
        })
    _table("pairs", rows, bench / "pairs.parquet")
    _table("contexts", [{
        "context_id": "S1__c0", "slide_id": "S1",
        "bbox_l0_x": 0, "bbox_l0_y": 0, "bbox_l0_w": 4096, "bbox_l0_h": 4096,
        "target_mpp": ctx_mpp, "output_size": ctx_size, "extraction_version": ver,
        "split": "test",
    }], bench / "contexts.parquet")
    _table("queries", [{
        "query_id": f"p{i}__q", "context_id": "S1__c0", "slide_id": "S1",
        "target_mpp": qry_mpp, "output_size": qry_size, "extraction_version": ver,
        "split": "test",
    } for i in range(len(rows))], bench / "queries.parquet")
    _table("slides", [{"slide_id": s, "source_relpath": f"{s}.svs"}
                      for s in ("S1", "S2")], bench / "slides.parquet")
    return bench


@pytest.fixture
def ds(pixel_less_freeze, tmp_path):
    with PyraFaultDataset(pixel_less_freeze, tmp_path / "no_slides_here",
                          split="test", materialize=False) as d:
        yield d


class TestMetaCostsNoPixels:
    def test_a_full_read_of_this_freeze_does_fail(self, ds):
        """The premise of the next test: the slides are genuinely unreadable, so a
        `meta` that silently fell back to reading them could not pass."""
        with pytest.raises(Exception):
            ds[0]

    def test_meta_reads_every_item_anyway(self, ds):
        for iid in ds.item_ids:
            assert ds.meta(iid)["inputs"]["item_id"] == iid

    def test_meta_never_calls_the_image_reader(self, ds, monkeypatch):
        def boom(*a, **kw):
            raise AssertionError("meta touched pixels")

        monkeypatch.setattr(ds, "_image", boom)
        assert ds.meta(0)["target"]["pair_id"] == "p0"


class TestMetaIsTheSameItem:
    def test_inputs_is_the_declared_key_set_minus_the_two_images(self, ds):
        assert set(ds.meta(0)["inputs"]) == set(INPUT_KEYS) - {"context", "query"}

    def test_target_is_whole(self, ds):
        assert set(ds.meta(0)["target"]) == set(TARGET_KEYS)

    def test_every_shared_value_matches_a_full_read(self, ds, monkeypatch):
        """Compared against a `__getitem__` whose only difference is a stubbed reader,
        so this is an equality against the real item-building code rather than against
        a restatement of it here."""
        monkeypatch.setattr(ds, "_image", lambda *a, **kw: "PIXELS")
        for iid in ds.item_ids:
            full, meta = ds[iid], ds.meta(iid)
            assert {k: v for k, v in full["inputs"].items()
                    if k not in ("context", "query")} == meta["inputs"]
            assert full["target"] == meta["target"]

    def test_the_localise_cache_key_is_computable_from_meta_alone(self, ds, monkeypatch):
        monkeypatch.setattr(ds, "_image", lambda *a, **kw: "PIXELS")
        for iid in ds.item_ids:
            assert cache_key(ds.meta(iid)) == cache_key(ds[iid])

    def test_the_three_indexing_forms_all_work(self, ds):
        by_pos = ds.meta(0)
        assert ds.meta(by_pos["inputs"]["item_id"]) == by_pos
        assert ds.meta(by_pos["target"]["pair_id"]) == by_pos

    def test_an_unknown_key_raises_keyerror(self, ds):
        with pytest.raises(KeyError):
            ds.meta("not-an-id")


class TestMetaKeepsTheFrozenSpecGate:
    """Skipping the pixels must not skip the I2 check. A freeze whose own rows disagree
    with the live config has to fail on the cheap path too, or a sharded run would
    decide happily on records it was never allowed to serve."""

    def test_a_context_whose_frozen_mpp_disagrees_raises(self, pixel_less_freeze,
                                                          tmp_path):
        p = pixel_less_freeze / "contexts.parquet"
        ctx = pd.read_parquet(p)
        ctx.loc[0, "target_mpp"] = 999.0
        ctx.to_parquet(p, index=False)
        with PyraFaultDataset(pixel_less_freeze, tmp_path / "nope", split="test",
                              materialize=False) as d:
            with pytest.raises(ValueError, match="context"):
                d.meta(0)

    def test_a_query_whose_frozen_size_disagrees_raises(self, pixel_less_freeze,
                                                         tmp_path):
        p = pixel_less_freeze / "queries.parquet"
        q = pd.read_parquet(p)
        q.loc[:, "output_size"] = 7
        q.to_parquet(p, index=False)
        with PyraFaultDataset(pixel_less_freeze, tmp_path / "nope", split="test",
                              materialize=False) as d:
            with pytest.raises(ValueError, match="query"):
                d.meta(0)
