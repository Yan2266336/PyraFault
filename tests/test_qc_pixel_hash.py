"""The hand-off guarantee: another machine either reproduces these pixels or fails.

Without this, a Pillow or numpy difference on the collaborator's side produces a
slightly different Q, and both sides report numbers computed on different images at a
precision where that matters (median IoU 0.99).
"""
from __future__ import annotations

from pyrafault.cache import image_sha256
from pyrafault.qc.report import check_served_pixels


import numpy as np
from PIL import Image

from pyrafault.degrade import apply


def _patch(seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    small = rng.integers(60, 240, size=(16, 16, 3), dtype=np.uint8)
    return Image.fromarray(small).resize((64, 64), Image.BILINEAR)


def test_a_matching_hash_passes():
    served = apply(_patch(), "geometry", {"angle_deg": 15.0, "scale": 1.0})
    row = {"pair_id": "p0", "degradation_band": "severe",
           "served_pixel_sha256": image_sha256(served)}

    assert check_served_pixels([(row, served)]) == []


def test_a_different_rendering_is_a_violation():
    plain = _patch()
    served = apply(plain, "geometry", {"angle_deg": 15.0, "scale": 1.0})
    row = {"pair_id": "p0", "degradation_band": "severe",
           "served_pixel_sha256": image_sha256(plain)}      # the undegraded hash

    violations = check_served_pixels([(row, served)])

    assert [v.check for v in violations] == ["served_pixel_mismatch"]


def test_an_empty_hash_is_not_silently_a_pass():
    """A row whose hash was never computed must be reported as unverified, not counted as
    verified -- otherwise a freeze with no hashes at all reads as fully checked."""
    served = _patch()
    row = {"pair_id": "p0", "degradation_band": "severe", "served_pixel_sha256": ""}

    violations = check_served_pixels([(row, served)])

    assert [v.check for v in violations] == ["served_pixel_unverified"]
