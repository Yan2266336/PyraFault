"""Opt-in native SuperPoint + LightGlue checks on public images, never WSI."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from pyrafault.geom import BBoxL0, iou
from pyrafault.verify.lightglue import LightGlueBaseline, OfficialSuperPointLightGlue


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def matcher():
    names = ("LIGHTGLUE_SOURCE", "SUPERPOINT_CHECKPOINT", "LIGHTGLUE_CHECKPOINT")
    paths = [os.environ.get(name) for name in names]
    if not all(paths):
        pytest.skip("set LIGHTGLUE_SOURCE, SUPERPOINT_CHECKPOINT and LIGHTGLUE_CHECKPOINT")
    import pyarrow  # noqa: F401 (before Kornia)
    import torch

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(4, previous_threads))
    previous = (torch.backends.cuda.flash_sdp_enabled(), torch.backends.cudnn.deterministic,
                torch.backends.cudnn.allow_tf32)
    try:
        backend = OfficialSuperPointLightGlue(*paths, device=os.environ.get("LIGHTGLUE_TEST_DEVICE", "cpu"))
        assert previous == (torch.backends.cuda.flash_sdp_enabled(), torch.backends.cudnn.deterministic,
                            torch.backends.cudnn.allow_tf32)
        for model in (backend._extractor, backend._model):
            assert not model.training and all(not p.requires_grad for p in model.parameters())
        yield backend
    finally:
        torch.set_num_threads(previous_threads)


def _image(size):
    with Image.open(Path(os.environ["LIGHTGLUE_SOURCE"]) / "assets" / "sacre_coeur1.jpg") as image:
        return image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)


def _inputs(context, query, *, query_mpp=1.0):
    return {"context": context, "query": query,
            "context_bbox": BBoxL0(1000, 2000, context.width * 10, context.height * 10),
            "context_mpp": 1.0, "query_mpp": query_mpp,
            "claimed_bbox": BBoxL0(0, 0, 1, 1)}


def _report(label, loc, truth):
    assert 0 <= loc.reliability <= 1
    if loc.estimated_bbox is None:
        assert loc.reliability == 0 and "reason" in loc.diagnostics
        overlap = 0.0
    else:
        overlap = iou(loc.estimated_bbox, truth)
    print(label, "bbox:", loc.estimated_bbox, "rho:", loc.reliability,
          "IoU:", overlap, "diagnostics:", dict(loc.diagnostics))
    # Report accuracy honestly, without assuming every positive must succeed.


def test_native_public_crop_mpp_alignment_and_independence(matcher):
    context = _image(256)
    query = context.crop((64, 64, 192, 192)).resize((512, 512), Image.Resampling.NEAREST)
    inputs = _inputs(context, query, query_mpp=0.25)
    verifier = LightGlueBaseline(matcher)
    first = verifier.localise(inputs)
    inputs["claimed_bbox"] = BBoxL0(99999, 99999, 1, 1)
    second = verifier.localise(inputs)
    assert first.estimated_bbox == second.estimated_bbox
    assert first.reliability == pytest.approx(second.reliability, rel=1e-9, abs=1e-9)
    assert first.diagnostics == second.diagnostics
    assert first.diagnostics["aligned_query_size_px"] == [128, 128]
    _report("native public crop", first, BBoxL0(1640, 2640, 1280, 1280))


def test_native_pilot_size_cost_and_shape_feasibility(matcher):
    context = _image(2048)
    query = context.crop((1024, 768, 1152, 896))
    loc = LightGlueBaseline(matcher).localise(_inputs(context, query))
    assert loc.diagnostics["context_size_px"] == [2048, 2048]
    assert loc.diagnostics["aligned_query_size_px"] == [128, 128]
    _report("native pilot-sized crop", loc, BBoxL0(11240, 9680, 1280, 1280))


def test_native_blank_images_do_not_invent_matches(matcher):
    context = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    query = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    loc = LightGlueBaseline(matcher).localise(_inputs(context, query))
    assert loc.estimated_bbox is None and loc.reliability == 0
    assert loc.diagnostics["reason"] == "too_few_valid_matches"
    print("native blank:", dict(loc.diagnostics))
