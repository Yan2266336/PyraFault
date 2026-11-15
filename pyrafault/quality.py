"""Image validity metrics (guideline §6.3).

The goal is "is this image valid", not "is this image pretty". Nothing here modifies
pixels; the strongest action available is to drop a candidate from the pool.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def blur_var(img: np.ndarray) -> float:
    return float(cv2.Laplacian(_gray(img), cv2.CV_64F).var())


def mean_intensity(img: np.ndarray) -> float:
    return float(_gray(img).mean())


def pen_mark_fraction(img: np.ndarray) -> float:
    """Fraction of pixels that look like marker ink: highly saturated and not pink.

    H&E stains occupy the magenta/purple band; surgical markers are blue, green or black.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0].astype(np.int16), hsv[..., 1], hsv[..., 2]
    # OpenCV hue is 0..179. H&E purple/pink sits roughly in 120..175.
    ink_hue = ((h >= 85) & (h < 120)) | ((h >= 35) & (h < 85))
    ink = (s > 120) & (v > 40) & ink_hue
    black = (v < 45) & (s < 90)
    return float((ink | black).mean())


@dataclass(frozen=True)
class QualityResult:
    flags: list[str] = field(default_factory=list)
    blur_var: float = 0.0
    mean_intensity: float = 0.0
    pen_mark_frac: float = 0.0
    hard_fail: bool = False


def evaluate(img: np.ndarray, cfg) -> QualityResult:
    bv = blur_var(img)
    mi = mean_intensity(img)
    pm = pen_mark_fraction(img)

    flags: list[str] = []
    hard = False

    if bv < cfg.blur_var_flag:
        flags.append("blur")
        if bv < cfg.blur_var_exclude:
            hard = True

    lo_flag, hi_flag = cfg.mean_intensity_flag
    lo_hard, hi_hard = cfg.mean_intensity_exclude
    if mi < lo_flag:
        flags.append("too_dark")
        if mi < lo_hard:
            hard = True
    elif mi > hi_flag:
        flags.append("too_bright")
        if mi > hi_hard:
            hard = True

    if pm > cfg.pen_mark_flag_frac:
        flags.append("pen_mark")      # recorded only; never a hard failure

    return QualityResult(
        flags=flags, blur_var=bv, mean_intensity=mi, pen_mark_frac=pm, hard_fail=hard
    )
