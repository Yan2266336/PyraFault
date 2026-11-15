"""Opt-in smoke test for the pinned official source and checkpoint."""
from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

from pyrafault.geom import BBoxL0
from pyrafault.verify.efficient_loftr import (
    EfficientLoFTRBaseline,
    OfficialEfficientLoFTR,
)


pytestmark = pytest.mark.slow


def test_official_checkpoint_recovers_a_crop_on_cpu():
    source = os.environ.get("EFFICIENT_LOFTR_SOURCE")
    checkpoint = os.environ.get("EFFICIENT_LOFTR_CHECKPOINT")
    if not source or not checkpoint:
        pytest.skip(
            "set EFFICIENT_LOFTR_SOURCE and EFFICIENT_LOFTR_CHECKPOINT"
        )

    matcher = OfficialEfficientLoFTR(source, checkpoint, device="cpu")
    assert not matcher._model.training
    assert all(not parameter.requires_grad for parameter in matcher._model.parameters())
    rng = np.random.default_rng(4)
    gray = rng.integers(0, 256, (128, 128), dtype=np.uint8)
    context = Image.fromarray(np.repeat(gray[..., None], 3, axis=2), "RGB")
    crop = gray[32:96, 40:104]
    query = Image.fromarray(np.repeat(crop[..., None], 3, axis=2), "RGB")
    loc = EfficientLoFTRBaseline(matcher).localise({
        "context": context,
        "query": query,
        "context_bbox": BBoxL0(1000, 2000, 1280, 640),
        "claimed_bbox": BBoxL0(999, 999, 1, 1),
        "context_mpp": 1.0,
        "query_mpp": 1.0,
    })

    assert loc.estimated_bbox is not None
    assert loc.estimated_bbox.w == 640 and loc.estimated_bbox.h == 320
    assert loc.estimated_bbox.x == pytest.approx(1400, abs=20)
    assert loc.estimated_bbox.y == pytest.approx(2160, abs=10)
    assert loc.reliability >= 0.9
    assert loc.diagnostics["inlier_matches"] >= 4
