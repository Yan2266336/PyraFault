"""Imaging degradation of the served query patch.

Why this exists: the eight fault types change *which* pixels are served, never how they
were imaged. `Q` is a resampled crop of the same scan as `C` -- same scanner, stain batch
and focus -- so recovering its source is pixel alignment, and a grayscale
cross-correlation baseline matches ZoomTrace on it (spec
docs/superpowers/specs/2026-09-29-imaging-degradation-conditions.md, §1).

Every operation here is deterministic array arithmetic over parameters that are drawn
once, frozen into the tables, and replayed at read time -- never drawn while reading. No
library filter, resampling call or encoder is involved: a second machine regenerating
these pixels must get the same bytes, and a filter whose implementation changed between
versions would break that silently. That is also why JPEG recompression is not here.

One property worth knowing before choosing strengths: a verifier's first step is to
downsample `Q` to `C`'s resolution (a factor of `mc/mq`, 4 for the shipped freezes), and
that averaging attenuates anything living in the high spatial frequencies. A blur or
resolution loss finer than `C`'s pixel grid is invisible to it. Degradations that change
intensity rather than spatial frequency survive the averaging.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np
from PIL import Image

KINDS = ("none", "stain_jitter", "blur", "resolution_loss", "sensor_noise", "geometry")

# Ruifrok & Johnston H&E optical-density vectors, columns normalised to unit length.
_HE = np.array([[0.65, 0.07], [0.70, 0.99], [0.29, 0.11]], dtype=np.float64)
_HE = _HE / np.linalg.norm(_HE, axis=0, keepdims=True)


def degradation_digest(kind: str, params: Mapping[str, Any]) -> str:
    """Identity of a degradation, for the cache key.

    The empty string for `none`, so an undegraded entry keeps the key it has today and
    every cache entry written before degradation existed stays valid.
    """
    if kind == "none":
        return ""
    payload = json.dumps({"kind": kind, "params": dict(sorted(params.items()))},
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def item_rng(seed: int, pair_id: str) -> np.random.Generator:
    """Per-item generator. Keyed by `pair_id` so a row's parameters do not depend on the
    order rows were assigned in, or on how many rows came before it.

    One limit worth stating: NumPy's compatibility policy covers `RandomState`'s method
    streams, not `Generator`'s, so a future NumPy could in principle draw different
    `uniform`/`choice` values from the same seed. This cannot affect a SHIPPED freeze --
    parameters are drawn once by `assign_bands`, frozen into `degradation_params`, and
    read back from the row at serve time, never redrawn (see `apply`). It does mean
    "rebuild the freeze from stage 00 and get the same `angle_deg`/`scale` draws" is not
    guaranteed across NumPy versions; reproducing the PIXELS of an existing freeze is,
    because that path never touches this function.
    """
    key = hashlib.sha256(f"{seed}|{pair_id}|degradation".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(key[:8], "big"))


def sample_params(kind: str, strength: float, rng: np.random.Generator) -> dict:
    """Draw the frozen parameters for one item. `strength` scales the band."""
    if kind == "none":
        return {}
    if kind == "stain_jitter":
        # Rotate each stain vector in its own plane and rescale its concentration.
        # Concentration scaling is what reaches luminance, and therefore what a
        # grayscale matcher can feel; rotation alone mostly moves hue.
        return {
            "angle_h": float(rng.uniform(-1, 1) * 0.15 * strength),
            "angle_e": float(rng.uniform(-1, 1) * 0.15 * strength),
            "scale_h": float(1.0 + rng.uniform(-1, 1) * 0.35 * strength),
            "scale_e": float(1.0 + rng.uniform(-1, 1) * 0.35 * strength),
        }
    if kind == "blur":
        return {"sigma": float(strength)}
    if kind == "resolution_loss":
        return {"factor": int(strength)}
    if kind == "sensor_noise":
        return {"sigma": float(strength)}
    if kind == "geometry":
        # `strength` is the rotation in degrees; the scale error rides along with it,
        # because a tool that returns a misregistered tile usually gets both wrong.
        return {
            "angle_deg": float(rng.choice([-1.0, 1.0]) * strength),
            "scale": float(1.0 + rng.uniform(-1, 1) * 0.02 * (strength / 5.0)),
        }
    raise ValueError(f"unknown degradation kind: {kind!r}")


def _gaussian_kernel(sigma: float) -> np.ndarray:
    radius = int(math.ceil(3.0 * sigma))
    xs = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(xs ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def _convolve_separable(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Separable convolution with edge padding, written out rather than delegated, so the
    result cannot change with a library version."""
    radius = (len(k) - 1) // 2
    out = np.zeros_like(a)
    padded = np.pad(a, ((radius, radius), (0, 0), (0, 0)), mode="edge")
    for i, w in enumerate(k):
        out += w * padded[i:i + a.shape[0]]
    padded = np.pad(out, ((0, 0), (radius, radius), (0, 0)), mode="edge")
    out = np.zeros_like(a)
    for i, w in enumerate(k):
        out += w * padded[:, i:i + a.shape[1]]
    return out


def _box_downsample(a: np.ndarray, r: int) -> np.ndarray:
    h, w = a.shape[:2]
    if h % r or w % r:
        raise ValueError(f"resolution_loss factor {r} must divide the patch side {h}x{w}")
    return a.reshape(h // r, r, w // r, r, a.shape[2]).mean(axis=(1, 3))


def _bilinear_upsample(a: np.ndarray, r: int) -> np.ndarray:
    h, w = a.shape[:2]
    coords = (np.arange(h * r, dtype=np.float64) + 0.5) / r - 0.5
    coords = np.clip(coords, 0, h - 1)
    lo = np.floor(coords).astype(int)
    hi = np.minimum(lo + 1, h - 1)
    frac = (coords - lo)[:, None, None]
    rows = a[lo] * (1 - frac) + a[hi] * frac
    frac_w = (coords - lo)[None, :, None]
    return rows[:, lo] * (1 - frac_w) + rows[:, hi] * frac_w


def _affine_sample(a: np.ndarray, angle_deg: float, scale: float) -> np.ndarray:
    """Rotate about the centre and rescale, sampling bilinearly from the source.

    Written out rather than delegated to a library's affine transform for the same
    reason as the blur: the pixels have to be reproducible on another machine. Sampling
    is clamped at the border, so no padding colour enters and the patch stays a patch of
    tissue -- a fill value would hand a verifier a free cue about the transform.
    """
    h, w = a.shape[:2]
    theta = math.radians(angle_deg)
    cos, sin = math.cos(theta) / scale, math.sin(theta) / scale
    ys, xs = np.meshgrid(np.arange(h, dtype=np.float64) - (h - 1) / 2.0,
                         np.arange(w, dtype=np.float64) - (w - 1) / 2.0, indexing="ij")
    src_x = cos * xs + sin * ys + (w - 1) / 2.0
    src_y = -sin * xs + cos * ys + (h - 1) / 2.0
    src_x = np.clip(src_x, 0, w - 1)
    src_y = np.clip(src_y, 0, h - 1)
    x0 = np.floor(src_x).astype(int); x1 = np.minimum(x0 + 1, w - 1)
    y0 = np.floor(src_y).astype(int); y1 = np.minimum(y0 + 1, h - 1)
    fx = (src_x - x0)[..., None]; fy = (src_y - y0)[..., None]
    top = a[y0, x0] * (1 - fx) + a[y0, x1] * fx
    bot = a[y1, x0] * (1 - fx) + a[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def apply(img: Image.Image, kind: str, params: Mapping[str, Any]) -> Image.Image:
    """Replay a frozen degradation. Pure function of (pixels, kind, params)."""
    if kind == "none":
        return img
    a = np.asarray(img.convert("RGB"), dtype=np.float64)

    if kind == "stain_jitter":
        od = -np.log10((a + 1.0) / 256.0)
        stains = _HE.copy()
        for j, key in enumerate(("angle_h", "angle_e")):
            theta = float(params[key])
            c, s = math.cos(theta), math.sin(theta)
            v = stains[:, j]
            other = stains[:, 1 - j]
            perp = other - v * float(v @ other)
            n = np.linalg.norm(perp)
            if n > 1e-8:
                stains[:, j] = v * c + (perp / n) * s
        conc = np.linalg.pinv(_HE) @ od.reshape(-1, 3).T
        conc[0] *= float(params["scale_h"])
        conc[1] *= float(params["scale_e"])
        od = (stains @ conc).T.reshape(a.shape)
        a = 256.0 * np.power(10.0, -od) - 1.0
    elif kind == "blur":
        a = _convolve_separable(a, _gaussian_kernel(float(params["sigma"])))
    elif kind == "resolution_loss":
        r = int(params["factor"])
        a = _bilinear_upsample(_box_downsample(a, r), r)
    elif kind == "geometry":
        a = _affine_sample(a, float(params["angle_deg"]), float(params["scale"]))
    elif kind == "sensor_noise":
        # Seeded from the frozen parameters, not from a live generator, so the noise
        # field itself is part of the record rather than of the run.
        seed = int(hashlib.sha256(
            json.dumps(dict(sorted(params.items())), sort_keys=True).encode()
        ).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        a = a + rng.normal(0.0, float(params["sigma"]), size=a.shape)
    else:
        raise ValueError(f"unknown degradation kind: {kind!r}")

    return Image.fromarray(np.clip(np.rint(a), 0, 255).astype(np.uint8), mode="RGB")
