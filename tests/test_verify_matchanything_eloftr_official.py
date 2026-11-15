"""Opt-in CPU checks using the author's public test photo and synthetic noise."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pyrafault.geom import BBoxL0, iou
from pyrafault.verify.matchanything_eloftr import (
    MatchAnythingELoFTRBaseline,
    OfficialMatchAnythingELoFTR,
)


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def matcher():
    source = os.environ.get("MATCHANYTHING_ELOFTR_SOURCE")
    checkpoint = os.environ.get("MATCHANYTHING_ELOFTR_CHECKPOINT")
    if not source or not checkpoint:
        pytest.skip("set MATCHANYTHING_ELOFTR_SOURCE and MATCHANYTHING_ELOFTR_CHECKPOINT")
    import pyarrow  # noqa: F401  (before Kornia)
    import torch

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(4, previous_threads))
    aliases = {key: sys.modules.get(key) for key in ("src", "src.config", "src.config.default")}
    try:
        model = OfficialMatchAnythingELoFTR(source, checkpoint, device="cpu")
        for key, previous in aliases.items():
            assert sys.modules.get(key) is previous
        assert not model._model.training
        assert all(not parameter.requires_grad for parameter in model._model.parameters())
        yield model
    finally:
        torch.set_num_threads(previous_threads)


def _images():
    # The pinned Space checkout already contains this public natural-scene image.
    # No user's WSI pixels or images are accessed by these opt-in tests.
    source = Path(os.environ["MATCHANYTHING_ELOFTR_SOURCE"])
    with Image.open(source.parents[2] / "tests" / "data" / "02928139_3448003521.jpg") as image:
        context = image.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
    query = context.crop((64, 64, 192, 192))
    return context, query


def test_native_checkpoint_localises_an_unequal_size_crop(matcher):
    context, query = _images()
    loc = MatchAnythingELoFTRBaseline(matcher).localise({
        "context": context, "query": query,
        "context_bbox": BBoxL0(1000, 2000, 2560, 1280),
        "claimed_bbox": BBoxL0(-100, -100, 1, 1),
        "context_mpp": 1.0, "query_mpp": 1.0,
    })
    assert loc.estimated_bbox is not None
    assert loc.estimated_bbox.w == 1280 and loc.estimated_bbox.h == 640
    assert iou(loc.estimated_bbox, BBoxL0(1640, 2320, 1280, 640)) > 0.9
    assert loc.reliability >= 0.8
    assert loc.diagnostics["inlier_matches"] >= 4
    print("native crop:", loc.estimated_bbox, "rho:", loc.reliability,
          "inliers:", loc.diagnostics["inlier_matches"])


def test_native_checkpoint_mpp_alignment_and_claim_independence(matcher):
    context, query = _images()
    # A high-resolution raster representing the same physical crop: the adapter
    # must downsample to 128, not independently enlarge to the context's size.
    query = query.resize((512, 512), Image.Resampling.NEAREST)
    inputs = {"context": context, "query": query,
              "context_bbox": BBoxL0(1000, 2000, 2560, 2560),
              "context_mpp": 2.0, "query_mpp": 0.5,
              "claimed_bbox": BBoxL0(0, 0, 1280, 1280)}
    verifier = MatchAnythingELoFTRBaseline(matcher)
    first = verifier.localise(inputs)
    inputs["claimed_bbox"] = BBoxL0(90000, 90000, 1280, 1280)
    second = verifier.localise(inputs)
    assert first.estimated_bbox is not None
    assert first.estimated_bbox == second.estimated_bbox
    assert first.reliability == second.reliability
    assert first.diagnostics == second.diagnostics
    assert iou(first.estimated_bbox, BBoxL0(1640, 2640, 1280, 1280)) > 0.9
    assert first.diagnostics["aligned_query_size_px"] == [128, 128]
    print("native MPP:", first.estimated_bbox, "rho:", first.reliability,
          "inliers:", first.diagnostics["inlier_matches"])


def test_native_checkpoint_small_query_full_context_runs_without_assuming_localisation(matcher):
    # Pilot-sized rasters test shape/cost feasibility only, not WSI accuracy.
    context, _ = _images()
    context = context.convert("L").resize((2048, 2048), Image.Resampling.LANCZOS)
    loc = MatchAnythingELoFTRBaseline(matcher).localise({
        "context": context, "query": context.crop((1024, 768, 1152, 896)),
        "context_bbox": BBoxL0(0, 0, 2048, 2048),
        "context_mpp": 1.0, "query_mpp": 1.0,
    })
    assert 0.0 <= loc.reliability <= 1.0
    assert loc.diagnostics["valid_matches"] >= 0
    assert loc.diagnostics.get("reason") not in {
        "input_exceeds_model_limit", "query_footprint_too_large",
    }
    if loc.diagnostics["valid_matches"] < 4:
        assert loc.estimated_bbox is None and loc.reliability == 0.0
        assert loc.diagnostics["reason"] == "too_few_valid_matches"
    print("native pilot-sized localisation:", loc.estimated_bbox,
          "rho:", loc.reliability, "diagnostics:", dict(loc.diagnostics))


def test_native_checkpoint_noise_crop_is_not_assumed_matchable(matcher):
    # A matching network need not recover arbitrary white noise, unlike NCC.
    # Keep this observed limitation rather than tuning thresholds to pass it.
    rng = np.random.default_rng(4)
    context = Image.fromarray(rng.integers(0, 256, (256, 256), dtype=np.uint8)).convert("RGB")
    loc = MatchAnythingELoFTRBaseline(matcher).localise({
        "context": context, "query": context.crop((80, 64, 208, 192)),
        "context_bbox": BBoxL0(0, 0, 256, 256), "context_mpp": 1, "query_mpp": 1,
    })
    assert loc.estimated_bbox is None and loc.reliability == 0.0
    assert loc.diagnostics["reason"] == "too_few_valid_matches"
    print("native white-noise crop:", dict(loc.diagnostics))
