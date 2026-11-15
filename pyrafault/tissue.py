"""Tissue vs background only (guideline §6.2).

This module deliberately knows nothing about tumour, stroma or necrosis. It answers
one question — does this region contain enough tissue to be a valid CMPV sample — and
answers it fast enough to be asked tens of thousands of times per slide.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import closing, disk, opening, remove_small_objects

from pyrafault.geom import BBoxL0
from pyrafault.io.read import read_region_at_mpp


def _saturation(rgb: np.ndarray) -> np.ndarray:
    a = rgb.astype(np.float32)
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    return np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)


def build_mask(thumb_rgb: np.ndarray, mpp_thumb: float, cfg) -> np.ndarray:
    """HSV-saturation Otsu mask, cleaned by morphology and a minimum component area."""
    sat = _saturation(thumb_rgb)
    gray = thumb_rgb.astype(np.float32).mean(axis=2)

    try:
        otsu = float(threshold_otsu(sat))
    except ValueError:              # constant image
        otsu = 1.0
    # The floor stops Otsu from splitting pure background into "tissue" and "more background"
    # on a blank slide, where the between-class variance is meaningless.
    thr = max(otsu, float(cfg.sat_floor))

    mask = (sat >= thr) & (gray < float(cfg.gray_max))

    close_px = max(1, int(round(cfg.close_radius_um / mpp_thumb)))
    open_px = max(1, int(round(cfg.open_radius_um / mpp_thumb)))
    # I10: skimage >= 0.26 deprecates binary_closing/binary_opening in favour of the
    # unified closing/opening (which dispatch to the binary implementation for a bool
    # array). Verified empirically (tests/test_tissue.py) that this substitution is
    # exact, including for features touching the array border, where the two
    # functions' differing `mode` defaults could plausibly have differed but do not.
    mask = closing(mask, disk(close_px))
    mask = opening(mask, disk(open_px))

    min_px = max(1, int(round(cfg.min_component_area_um2 / (mpp_thumb ** 2))))
    # I10: skimage >= 0.26 deprecates `min_size` in favour of `max_size`, and the two
    # are NOT equivalent at the boundary -- `min_size` removed objects strictly
    # smaller than the threshold (kept size >= min_size); `max_size` removes objects
    # smaller than OR EQUAL to the threshold (kept size > max_size). `max_size =
    # min_px - 1` reproduces the old "keep size >= min_px" rule exactly (verified in
    # tests/test_tissue.py against the deprecated min_size call, suppressing only the
    # warning, not the behaviour).
    mask = remove_small_objects(mask, max_size=min_px - 1)
    return np.ascontiguousarray(mask, dtype=bool)


class TissueMask:
    """A thumbnail-resolution tissue mask with O(1) level-0 fraction queries."""

    def __init__(self, mask: np.ndarray, width_l0: int, height_l0: int) -> None:
        self.mask = np.ascontiguousarray(mask, dtype=bool)
        self.width_l0 = int(width_l0)
        self.height_l0 = int(height_l0)
        self.mh, self.mw = self.mask.shape
        self.scale_x = self.mw / self.width_l0
        self.scale_y = self.mh / self.height_l0
        self._sat = np.pad(self.mask.astype(np.float64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)

    def fraction(self, bbox: BBoxL0) -> float:
        x0 = int(bbox.x * self.scale_x)
        y0 = int(bbox.y * self.scale_y)
        x1 = min(self.mw, int(round(bbox.x1 * self.scale_x)))
        y1 = min(self.mh, int(round(bbox.y1 * self.scale_y)))
        if x1 <= x0 or y1 <= y0:
            return 0.0
        total = (
            self._sat[y1, x1] - self._sat[y0, x1] - self._sat[y1, x0] + self._sat[y0, x0]
        )
        return float(total / ((x1 - x0) * (y1 - y0)))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, mask=self.mask, width_l0=self.width_l0, height_l0=self.height_l0
        )

    @classmethod
    def load(cls, path: str | Path) -> "TissueMask":
        z = np.load(path)
        return cls(z["mask"], int(z["width_l0"]), int(z["height_l0"]))


def thumbnail(slide, target_mpp: float, extraction_version: str) -> tuple[np.ndarray, float]:
    """Whole-slide thumbnail at approximately `target_mpp`, read through read_region_at_mpp.

    A whole-slide thumbnail is essentially never square, so it is read once at its
    true (out_w, out_h) aspect ratio via `read_region_at_mpp` rather than squashed
    into a square and stretched back.
    """
    m = slide.meta
    box = BBoxL0(0, 0, m.width_l0, m.height_l0)
    out_w = max(1, int(round(m.width_l0 * m.mpp_x / target_mpp)))
    out_h = max(1, int(round(m.height_l0 * m.mpp_x / target_mpp)))
    img, _ = read_region_at_mpp(slide, box, target_mpp, out_w, out_h, extraction_version)
    arr = np.asarray(img)
    return np.ascontiguousarray(arr), target_mpp
