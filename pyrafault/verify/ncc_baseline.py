"""A minimal verifier: normalised cross-correlation of a downsampled Q over C."""
from __future__ import annotations

import numpy as np
from PIL import Image

from pyrafault.geom import BBoxL0
from pyrafault.verify.base import Localisation, Verifier


def _gray(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float64)


def _ncc_map(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """NCC of `template` at every valid offset of `image`, stride 1.

    Stride 1 is not an optimisation detail. A true source sits at an arbitrary pixel
    offset, and on tissue the correlation falls to about zero a few pixels away from
    it -- a coarser stride usually never samples the peak at all.
    """
    th, tw = template.shape
    t = template - template.mean()
    # Sum of (window * zero-mean template) at every offset, by FFT convolution with
    # the flipped template.
    k = np.zeros_like(image)
    k[:th, :tw] = t[::-1, ::-1]
    full = np.real(np.fft.ifft2(np.fft.fft2(image) * np.fft.fft2(k)))
    cross = full[th - 1:, tw - 1:]
    # Per-window variance from integral images.
    def window_sum(a):
        s = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        return s[th:, tw:] - s[:-th, tw:] - s[th:, :-tw] + s[:-th, :-tw]
    n = th * tw
    s1 = window_sum(image)
    var = np.maximum(window_sum(image ** 2) - s1 ** 2 / n, 0.0)
    denom = np.sqrt(var * (t ** 2).sum())
    return np.divide(cross, denom, out=np.zeros_like(cross), where=denom > 1e-6)


class NccBaseline(Verifier):
    name = "ncc_baseline"
    revision = "2"

    def localise(self, inputs) -> Localisation:
        context, query = inputs["context"], inputs["query"]
        scale = inputs["context_mpp"] / inputs["query_mpp"]  # mc / mq
        side = max(1, round(query.size[0] / scale))
        c_arr = _gray(context)
        t_arr = _gray(query.resize((side, side), Image.LANCZOS))

        if c_arr.shape[0] < side or c_arr.shape[1] < side:
            return Localisation(estimated_bbox=None, reliability=0.0)

        scores = _ncc_map(c_arr, t_arr)
        y, x = np.unravel_index(int(np.argmax(scores)), scores.shape)
        best = float(scores[y, x])

        ctx_bbox = inputs["context_bbox"]
        px_scale = ctx_bbox.w / context.width
        est = BBoxL0(
            int(round(ctx_bbox.x + x * px_scale)),
            int(round(ctx_bbox.y + y * px_scale)),
            int(round(side * px_scale)),
            int(round(side * px_scale)),
        )
        # NCC is in [-1, 1]; rescale to [0, 1] so `reliability` behaves monotonically
        # with the same sign convention the decision rule assumes (higher is better).
        return Localisation(estimated_bbox=est, reliability=(best + 1.0) / 2.0)
