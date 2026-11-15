"""The loader replays a frozen degradation and nothing else.

A verifier is never told that its patch was degraded -- `inputs` gains no field. The
point is to make the pixels harder, not to hand the method a flag saying so.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from pyrafault.dataset import PyraFaultDataset, _degradation
from pyrafault.degrade import apply, item_rng, sample_params


def test_a_row_without_the_columns_reads_as_undegraded():
    """An old freeze must still load, rather than needing a migration before it can be
    read at all."""
    assert _degradation({}) == ("none", {})
    assert _degradation({"degradation_kind": None}) == ("none", {})
    assert _degradation({"degradation_kind": float("nan")}) == ("none", {})


def test_a_row_with_the_columns_parses_its_frozen_parameters():
    row = {"degradation_kind": "geometry",
           "degradation_params": json.dumps({"angle_deg": 15.0, "scale": 1.0})}

    assert _degradation(row) == ("geometry", {"angle_deg": 15.0, "scale": 1.0})


def test_a_nan_params_value_is_treated_as_no_parameters():
    """`degradation_params` can be NaN independently of `degradation_kind` -- a real
    `kind` with an unreadable `params` cell (a partially-migrated row, say) must not
    reach `json.loads` with a float, which raises `TypeError` rather than the
    `JSONDecodeError` the sibling test below pins for a malformed string."""
    row = {"degradation_kind": "geometry", "degradation_params": float("nan")}

    assert _degradation(row) == ("geometry", {})


def test_an_unparseable_params_string_raises():
    """Serving undegraded pixels for a row whose parameters cannot be read would make a
    whole band silently absent from the results."""
    with pytest.raises(json.JSONDecodeError):
        _degradation({"degradation_kind": "geometry", "degradation_params": "{not json"})


def _serve(bench, gdc, pairs_edit):
    """Serve item 0's query after applying `pairs_edit` to the loader's own frame."""
    with PyraFaultDataset(bench, gdc, split="test", materialize=False) as ds:
        pairs_edit(ds.pairs)
        return ds[0]["inputs"]["query"], ds.pairs.iloc[0]


def test_a_none_row_is_served_the_pixels_of_an_undegraded_build(benchmark_root, gdc_root):
    """The in-freeze control must be a control."""
    plain, _ = _serve(benchmark_root, gdc_root, lambda df: None)
    served, _ = _serve(benchmark_root, gdc_root, lambda df: df.__setitem__(
        "degradation_kind", "none"))

    assert np.array_equal(np.asarray(plain), np.asarray(served))


def test_a_severe_row_is_served_exactly_the_recorded_rendering(benchmark_root, gdc_root):
    """Not a rendering the loader re-drew: re-drawing would serve different pixels on
    every read, which no single run can reveal."""
    plain, _ = _serve(benchmark_root, gdc_root, lambda df: None)
    params = {"angle_deg": 15.0, "scale": 1.0}

    def edit(df):
        df["degradation_kind"] = "geometry"
        df["degradation_params"] = json.dumps(params)
        df["degradation_band"] = "severe"

    served, _ = _serve(benchmark_root, gdc_root, edit)

    assert np.array_equal(np.asarray(served),
                          np.asarray(apply(plain, "geometry", params)))


def test_an_unknown_degradation_kind_raises(benchmark_root, gdc_root):
    def edit(df):
        df["degradation_kind"] = "rotate_a_lot"
        df["degradation_params"] = "{}"

    with pytest.raises(ValueError, match="unknown degradation kind"):
        _serve(benchmark_root, gdc_root, edit)
