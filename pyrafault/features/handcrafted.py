"""Network-free fallback descriptor.

Not competitive with a pathology foundation model, but it keeps stages 05 and 07
runnable on a node with no internet and gives the tests something deterministic to
assert against.
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
from PIL import Image

# Ruifrok & Johnston H&E stain vectors, used only to summarise colour composition.
_HE = np.linalg.inv(np.array([
    [0.65, 0.70, 0.29],
    [0.07, 0.99, 0.11],
    [0.27, 0.57, 0.78],
]))


class HandcraftedExtractor:
    name = "handcrafted"
    revision = "v1"
    input_size = 224
    preprocessing = "resize-224-bilinear; stain-deconv + LBP(8,1) + grad-orientation"
    hf_repo = None
    timm_model = None

    def __init__(self, batch_size: int = 64) -> None:
        self.batch_size = batch_size
        self.dim = 3 * 8 + 59 + 16      # stain stats + LBP hist + gradient hist

    def _one(self, img: Image.Image) -> np.ndarray:
        a = np.asarray(img.convert("RGB").resize((self.input_size, self.input_size),
                                                 Image.BILINEAR), dtype=np.float32)
        od = -np.log10(np.clip(a, 1.0, 255.0) / 255.0)
        stains = od.reshape(-1, 3) @ _HE.T
        stain_stats = np.concatenate([
            np.concatenate([
                np.percentile(stains[:, c], [10, 50, 90]),
                [stains[:, c].mean(), stains[:, c].std(),
                 stains[:, c].min(), stains[:, c].max(), np.abs(stains[:, c]).mean()],
            ]) for c in range(3)
        ])

        gray = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        from skimage.feature import local_binary_pattern
        lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
        lbp_hist, _ = np.histogram(lbp, bins=59, range=(0, 59), density=True)

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        ang = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)
        mag = np.hypot(gx, gy)
        if mag.sum() > 0:
            # density=True normalises by weights.sum(); a flat patch (smooth background
            # or stroma, common in production) has zero gradient everywhere, which would
            # divide by zero. Skip normalisation and emit an all-zero histogram instead.
            grad_hist, _ = np.histogram(ang, bins=16, range=(0, 1), weights=mag, density=True)
        else:
            grad_hist = np.zeros(16, dtype=np.float64)

        return np.concatenate([stain_stats, lbp_hist, grad_hist]).astype(np.float32)

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        out = np.stack([self._one(im) for im in images]).astype(np.float32)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return (out / np.maximum(norms, 1e-8)).astype(np.float32)
