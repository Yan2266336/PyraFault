"""Localising in shards and deciding once, which is what makes a full-scale run finish.

ZoomTrace localises at ~240 ms and the NCC control at ~360 ms, so the lung freeze's
298,080 calibration+test rows are 20 to 30 hours in one process -- past the partition's
20-hour wall, before counting the second method or the fit pass. Nothing about the task
is sequential, though: `localise` sees one (C, Q) and the decision is a pure function of
(localisation, claim, tau, delta). So the expensive half shards and the cheap half runs
once over the shards' output.

What these pin is that the split path is not a second implementation. The sharded route
must produce the identical predictions table -- same rows, same estimates, same
decisions -- or the parallel number in the paper is not the number the protocol defines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from pyrafault.geom import BBoxL0
from pyrafault.verify.base import Localisation, Verifier
from pyrafault.verify.runner import (
    LOCALISATION_COLUMNS, PREDICTION_COLUMNS, fit_thresholds, load_localisations,
    localise_items, run_verifier, shard_of,
)
from pyrafault.verify.thresholds import Thresholds


class _Deterministic(Verifier):
    """Estimate and reliability are a function of the query's shade, so a shard that
    localised the wrong items produces visibly wrong rows rather than plausible ones."""
    name, revision = "det", "7"

    def __init__(self):
        self.calls = 0

    def localise(self, inputs):
        self.calls += 1
        shade = inputs["query"].getpixel((0, 0))[0]
        return Localisation(BBoxL0(1000 + shade, 2000, 100, 100), shade / 255.0)


class _NoEstimate(Verifier):
    name, revision = "blind", "1"

    def localise(self, inputs):
        return Localisation(None, 0.1)


class _Freeze:
    """Twelve items over two splits: four distinct (C, Q) each, with `clean` and
    `hard_mismatch` sharing one patch under different claims -- the collision the
    runner's cache exists for, so sharding has something to get wrong."""

    def __init__(self, split="test"):
        self._items = {}
        for q in range(4):
            for fault in ("clean", "hard_mismatch", "adjacent_drift"):
                iid = f"{split}-{fault}-{q}"
                claim = (BBoxL0(1000 + 20 * q, 2000, 100, 100) if fault == "clean"
                         else BBoxL0(60000, 2000, 100, 100))
                self._items[iid] = {
                    "inputs": {
                        "context": Image.new("RGB", (64, 64)),
                        "query": Image.new("RGB", (16, 16), (20 * q,) * 3),
                        "context_bbox": BBoxL0(0, 0, 100000, 100000),
                        "claimed_bbox": claim,
                        "context_mpp": 2.0, "query_mpp": 0.5, "slide_id": "S1",
                        "item_id": iid, "split": split, "extraction_version": "v1",
                    },
                    "target": {
                        "claim_correct": fault == "clean",
                        "true_bbox": BBoxL0(1000 + 20 * q, 2000, 100, 100),
                        "true_slide_id": "S1", "pair_id": iid,
                    },
                }

    @property
    def item_ids(self):
        return list(self._items)

    def __getitem__(self, k):
        return self._items[k]


def _thresholds():
    return Thresholds(tau=0.3, delta=0.5, fitted_on="calibration", n_items=12,
                      residuals={"n": 12, "p50": 0.1, "p75": 0.2, "p90": 0.3,
                                 "p95": 0.4, "p99": 0.5})


class TestShardOf:
    """The partition itself, independent of any verifier."""

    def test_the_shards_cover_the_split_exactly_once(self):
        ids = [f"i{i}" for i in range(97)]
        seen = [i for s in range(7) for i in shard_of(ids, s, 7)]
        assert sorted(seen) == sorted(ids)
        assert len(seen) == len(ids)

    def test_the_shards_are_within_one_of_each_other(self):
        ids = [f"i{i}" for i in range(97)]
        sizes = [len(shard_of(ids, s, 7)) for s in range(7)]
        assert max(sizes) - min(sizes) <= 1

    def test_one_shard_is_the_whole_split(self):
        ids = [f"i{i}" for i in range(10)]
        assert shard_of(ids, 0, 1) == ids

    def test_more_shards_than_items_leaves_some_empty_rather_than_raising(self):
        ids = ["a", "b"]
        assert [shard_of(ids, s, 5) for s in range(5)].count([]) == 3

    @pytest.mark.parametrize("shard,n", [(-1, 4), (4, 4), (0, 0)])
    def test_an_out_of_range_shard_raises(self, shard, n):
        with pytest.raises(ValueError):
            shard_of(["a"], shard, n)


class TestLocaliseItems:
    def test_it_emits_the_declared_threshold_free_columns(self):
        df = localise_items(_Deterministic(), _Freeze())
        assert list(df.columns) == list(LOCALISATION_COLUMNS)

    def test_no_threshold_or_decision_leaks_into_it(self):
        """The whole point of the split: this half is computed before tau and delta
        exist, so a decision column here would be a decision made under thresholds
        nobody fitted yet."""
        df = localise_items(_Deterministic(), _Freeze())
        assert "decision" not in df.columns
        assert "thresholds_digest" not in df.columns

    def test_one_row_per_item_in_the_given_order(self):
        ds = _Freeze()
        df = localise_items(_Deterministic(), ds)
        assert list(df["item_id"]) == ds.item_ids

    def test_it_localises_a_shared_patch_once(self):
        v = _Deterministic()
        localise_items(v, _Freeze())
        assert v.calls == 4, "four distinct (C, Q) across twelve items"

    def test_a_shard_only_localises_its_own_items(self):
        ds = _Freeze()
        ids = shard_of(ds.item_ids, 1, 3)
        df = localise_items(_Deterministic(), ds, item_ids=ids)
        assert list(df["item_id"]) == ids

    def test_a_none_estimate_round_trips_as_null(self):
        df = localise_items(_NoEstimate(), _Freeze())
        assert df["estimated_bbox_l0_x"].isna().all()

    def test_the_concatenated_shards_equal_the_whole_split(self):
        ds = _Freeze()
        whole = localise_items(_Deterministic(), ds)
        parts = pd.concat([localise_items(_Deterministic(), ds,
                                          item_ids=shard_of(ds.item_ids, s, 5))
                           for s in range(5)], ignore_index=True)
        cols = [c for c in LOCALISATION_COLUMNS if c not in ("item_id", "runtime_ms")]
        pd.testing.assert_frame_equal(
            parts.set_index("item_id").loc[whole["item_id"], cols],
            whole.set_index("item_id")[cols],
        )


class TestDecidingFromPrecomputedLocalisations:
    def test_predictions_match_the_single_process_route(self):
        ds = _Freeze()
        direct = run_verifier(_Deterministic(), ds, _thresholds())
        locs = load_localisations(localise_items(_Deterministic(), ds))
        staged = run_verifier(_Deterministic(), ds, _thresholds(), precomputed=locs)
        cols = [c for c in PREDICTION_COLUMNS if c != "runtime_ms"]
        pd.testing.assert_frame_equal(staged[cols], direct[cols])

    def test_predictions_match_when_the_localisations_came_in_shards(self):
        ds = _Freeze()
        direct = run_verifier(_Deterministic(), ds, _thresholds())
        locs = load_localisations(pd.concat(
            [localise_items(_Deterministic(), ds,
                            item_ids=shard_of(ds.item_ids, s, 5))
             for s in range(5)], ignore_index=True))
        staged = run_verifier(_Deterministic(), ds, _thresholds(), precomputed=locs)
        cols = [c for c in PREDICTION_COLUMNS if c != "runtime_ms"]
        pd.testing.assert_frame_equal(staged[cols], direct[cols])

    def test_the_verifier_is_never_called_again(self):
        ds = _Freeze()
        locs = load_localisations(localise_items(_Deterministic(), ds))
        v = _Deterministic()
        run_verifier(v, ds, _thresholds(), precomputed=locs)
        assert v.calls == 0, "the expensive half already happened in the shards"

    def test_runtime_ms_is_carried_through_from_the_shard(self):
        ds = _Freeze()
        loc_df = localise_items(_Deterministic(), ds)
        staged = run_verifier(_Deterministic(), ds, _thresholds(),
                              precomputed=load_localisations(loc_df))
        assert (staged.set_index("item_id")["runtime_ms"]
                == loc_df.set_index("item_id")["runtime_ms"]).all()

    def test_a_missing_item_names_itself_rather_than_raising_a_bare_keyerror(self):
        ds = _Freeze()
        locs = load_localisations(localise_items(_Deterministic(), ds))
        dropped = ds.item_ids[3]
        del locs[dropped]
        with pytest.raises(SystemExit, match=dropped):
            run_verifier(_Deterministic(), ds, _thresholds(), precomputed=locs)

    def test_it_reads_meta_rather_than_pixels_when_the_dataset_offers_it(self):
        """The reason the deciding pass is cheap. A dataset with a `meta` must have it
        used -- going through `__getitem__` would re-decode two cached patches per row,
        600,000 of them on the lung freeze, for integers already in `pairs.parquet`."""
        ds = _Freeze()
        locs = load_localisations(localise_items(_Deterministic(), ds))

        class MetaOnly:
            item_ids = ds.item_ids

            def meta(self, k):
                item = ds[k]
                return {"inputs": {x: y for x, y in item["inputs"].items()
                                   if x not in ("context", "query")},
                        "target": item["target"]}

            def __getitem__(self, k):
                raise AssertionError("the deciding pass read pixels")

        run_verifier(_Deterministic(), MetaOnly(), _thresholds(), precomputed=locs)


class TestFittingFromPrecomputedLocalisations:
    def test_the_fitted_thresholds_match_the_single_process_route(self):
        ds = _Freeze("calibration")
        direct = fit_thresholds(_Deterministic(), ds, "calibration")
        locs = load_localisations(localise_items(_Deterministic(), ds))
        staged = fit_thresholds(_Deterministic(), ds, "calibration", precomputed=locs)
        assert (staged.tau, staged.delta) == (direct.tau, direct.delta)
        assert staged.n_items == direct.n_items
        assert staged.residuals == direct.residuals

    def test_the_digest_matches_so_predictions_record_the_same_calibration(self):
        ds = _Freeze("calibration")
        direct = fit_thresholds(_Deterministic(), ds, "calibration")
        locs = load_localisations(localise_items(_Deterministic(), ds))
        staged = fit_thresholds(_Deterministic(), ds, "calibration", precomputed=locs)
        assert staged.digest() == direct.digest()

    def test_the_verifier_is_never_called_again(self):
        ds = _Freeze("calibration")
        locs = load_localisations(localise_items(_Deterministic(), ds))
        v = _Deterministic()
        fit_thresholds(v, ds, "calibration", precomputed=locs)
        assert v.calls == 0


class TestLoadLocalisations:
    def test_it_survives_a_parquet_round_trip(self, tmp_path):
        """Where the shards actually come from. `None` estimates become pandas nulls on
        the way through, and an estimate must not come back as a box at (nan, nan)."""
        ds = _Freeze()
        p = tmp_path / "loc.parquet"
        localise_items(_NoEstimate(), ds).to_parquet(p, index=False)
        locs = load_localisations(pd.read_parquet(p))
        assert all(loc.estimated_bbox is None for loc, _ in locs.values())

    def test_a_box_round_trips_as_integers(self, tmp_path):
        ds = _Freeze()
        p = tmp_path / "loc.parquet"
        localise_items(_Deterministic(), ds).to_parquet(p, index=False)
        locs = load_localisations(pd.read_parquet(p))
        box = locs[ds.item_ids[0]][0].estimated_bbox
        assert (box.x, box.y, box.w, box.h) == (1000, 2000, 100, 100)
        assert all(isinstance(v, (int, np.integer)) for v in (box.x, box.y, box.w, box.h))

    def test_a_duplicated_item_id_across_shards_is_refused(self):
        """Two shards that overlap would mean one item decided twice -- silently, with
        whichever row happened to land last. The partition makes it impossible; this is
        the check that says so if a resubmission ever does it by hand."""
        ds = _Freeze()
        df = localise_items(_Deterministic(), ds)
        with pytest.raises(SystemExit, match="duplicate"):
            load_localisations(pd.concat([df, df.head(1)], ignore_index=True))


class TestShardingByGroup:
    """Striding groups rather than items, which is how a real run shards.

    Items sharing one (C, Q) share a context, so grouping by context keeps the runner's
    dedupe intact across shards -- and, for ZoomTrace, keeps each shard's window-bank
    read down to its own contexts. The lung freeze's banks are 12.8 GB; twenty jobs each
    reading all of them would spend longer on that than on localising.
    """

    def test_the_shards_still_cover_the_split_exactly_once(self):
        ids = [f"i{i}" for i in range(60)]
        groups = [f"c{i % 7}" for i in range(60)]
        seen = [i for s in range(3) for i in shard_of(ids, s, 3, groups=groups)]
        assert sorted(seen) == sorted(ids)

    def test_every_item_of_a_group_lands_in_one_shard(self):
        ids = [f"i{i}" for i in range(60)]
        groups = [f"c{i % 7}" for i in range(60)]
        where = {}
        for s in range(3):
            for i in shard_of(ids, s, 3, groups=groups):
                where.setdefault(groups[ids.index(i)], set()).add(s)
        assert all(len(v) == 1 for v in where.values())

    def test_the_partition_does_not_depend_on_row_order(self):
        ids = [f"i{i}" for i in range(20)]
        groups = [f"c{i % 5}" for i in range(20)]
        forward = set(shard_of(ids, 1, 3, groups=groups))
        backward = set(shard_of(ids[::-1], 1, 3, groups=groups[::-1]))
        assert forward == backward

    def test_a_grouped_shard_preserves_the_dedupe(self):
        """The point of grouping. Twelve items over four patches, three shards: each
        patch is localised exactly once in total, not once per shard that saw it."""
        ds = _Freeze()
        groups = [iid.rsplit("-", 1)[1] for iid in ds.item_ids]   # the query index
        calls = 0
        for s in range(3):
            v = _Deterministic()
            localise_items(v, ds, item_ids=shard_of(ds.item_ids, s, 3, groups=groups))
            calls += v.calls
        assert calls == 4

    def test_mismatched_group_length_raises(self):
        with pytest.raises(ValueError, match="groups"):
            shard_of(["a", "b"], 0, 2, groups=["g"])
