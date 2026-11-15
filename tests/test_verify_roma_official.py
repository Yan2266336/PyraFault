"""Opt-in native RoMa checkpoint smoke; never downloads weights or reads WSI."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image

from pyrafault.geom import BBoxL0, iou
from pyrafault.verify.roma import OfficialRoMa, RoMaBaseline


pytestmark = pytest.mark.slow


def test_official_roma_public_crop_and_claim_independence():
    values = [os.environ.get(name) for name in
              ("ROMA_SOURCE", "ROMA_CHECKPOINT", "ROMA_DINOV2_WEIGHTS")]
    if not all(values):
        pytest.skip("set ROMA_SOURCE, ROMA_CHECKPOINT and ROMA_DINOV2_WEIGHTS")
    import pyarrow  # noqa: F401  (before Kornia)
    import torch

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(4, previous_threads))
    try:
        matcher = OfficialRoMa(*values, device=os.environ.get("ROMA_TEST_DEVICE", "cpu"))
        assert not matcher._model.training
        assert all(not p.requires_grad for p in matcher._model.parameters())
        dino = matcher._model.encoder.dinov2_vitl14[0]
        assert not dino.training and all(not p.requires_grad for p in dino.parameters())
        with Image.open(Path(values[0]) / "assets" / "toronto_A.jpg") as image:
            context = image.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
        inputs = {"context": context, "query": context.crop((64, 64, 192, 192)),
                  "context_bbox": BBoxL0(1000, 2000, 2560, 2560),
                  "context_mpp": 1.0, "query_mpp": 1.0,
                  "claimed_bbox": BBoxL0(0, 0, 1, 1)}
        verifier = RoMaBaseline(matcher)
        first = verifier.localise(inputs)
        inputs["claimed_bbox"] = BBoxL0(99999, 99999, 1, 1)
        second = verifier.localise(inputs)
        assert first.estimated_bbox == second.estimated_bbox
        assert first.reliability == pytest.approx(second.reliability, rel=1e-9, abs=1e-9)
        assert 0 <= first.reliability <= 1
        # This smoke checks the executable/deterministic contract, not accuracy.
        # Report a failed positive honestly rather than asserting all crops work.
        if first.estimated_bbox is None:
            assert first.reliability == 0 and "reason" in first.diagnostics
            overlap = 0.0
        else:
            overlap = iou(first.estimated_bbox, BBoxL0(1640, 2640, 1280, 1280))
        print("RoMa public crop:", first.estimated_bbox, "rho:", first.reliability,
              "IoU:", overlap, "diagnostics:", dict(first.diagnostics))
    finally:
        torch.set_num_threads(previous_threads)
