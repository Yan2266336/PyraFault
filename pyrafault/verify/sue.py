"""Frozen patch retrieval + Spatial Uncertainty Estimation (SUE).

The top-1 window supplies the position; SUE supplies uncertainty, not a transform.
This WSI adaptation follows the paper's weighted population covariance, without
the example script's metre-specific clipping or cross-query min/max scaling.
"""
from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from numbers import Integral
from typing import Mapping

import numpy as np
from PIL import Image

from pyrafault.geom import BBoxL0
from pyrafault.verify.base import Localisation, Verifier


def spatial_uncertainty(positions, distances, distance_decay: float = 350.0):
    """Return covariance trace and normalized exp(-lambda * L2) weights.

    Coordinates may be physical or dimensionless, but must share a fixed unit.
    Subtracting the smallest distance preserves weights and avoids underflow of
    every weight. This is a population variance: there is no K-1 correction.
    """
    p = np.asarray(positions, dtype=np.float64)
    d = np.asarray(distances, dtype=np.float64)
    decay = float(distance_decay)
    if (p.ndim != 2 or p.shape[1] != 2 or d.ndim != 1
            or len(p) != len(d) or len(d) < 2):
        raise ValueError("SUE requires at least two (N, 2) positions and (N,) distances")
    if not np.isfinite(p).all() or not np.isfinite(d).all() or (d < 0).any():
        raise ValueError("positions/distances must be finite; distances must be nonnegative")
    if not math.isfinite(decay) or decay < 0:
        raise ValueError("distance_decay must be finite and nonnegative")
    with np.errstate(over="ignore"):
        weights = np.exp(-decay * (d - d.min()))
    weights /= weights.sum()
    mean = weights @ p
    uncertainty = float(np.sum(weights * np.sum((p - mean) ** 2, axis=1)))
    return uncertainty, weights


class SUEBaseline(Verifier):
    """Claim-independent top-1 patch retrieval with top-K spatial uncertainty.

    No geometric refinement, tissue filtering, model training, or persistent bank
    is involved. All windows are embedded afresh, making first-version costs and
    behavior explicit. Model construction belongs to the caller/registry's CLI.
    """

    name = "sue_baseline"

    def __init__(self, extractor, *, top_k: int = 10,
                 distance_decay: float = 350.0, stride_fraction: float = 0.5):
        if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k < 2:
            raise ValueError("top_k must be an integer >= 2")
        try:
            distance_decay, stride_fraction = float(distance_decay), float(stride_fraction)
        except (TypeError, ValueError) as exc:
            raise ValueError("distance_decay and stride_fraction must be numeric") from exc
        if not math.isfinite(distance_decay) or distance_decay < 0:
            raise ValueError("distance_decay must be finite and nonnegative")
        if not math.isfinite(stride_fraction) or not 0 < stride_fraction <= 1:
            raise ValueError("stride_fraction must be finite and in (0, 1]")
        if not getattr(extractor, "name", None) or not getattr(extractor, "revision", None):
            raise ValueError("extractor must declare its name and frozen revision")
        self._extractor = extractor
        self._top_k = int(top_k)
        self._distance_decay = float(distance_decay)
        self._stride_fraction = float(stride_fraction)
        self.configuration = {
            "extractor": {key: str(getattr(extractor, key, "unspecified")) for key in
                          ("name", "revision", "preprocessing", "dim", "input_size",
                           "device", "batch_size", "hf_repo", "timm_model")},
            "top_k": self._top_k, "distance_decay": self._distance_decay,
            "stride_fraction": self._stride_fraction,
            "distance": "L2-unit-descriptors", "variance": "population-trace",
            "coordinates": "query-footprint-normalized-xy",
            "reliability": "1/(1+variance)", "edge_windows": True,
            "numpy": np.__version__, "pillow": Image.__version__,
        }
        # Record inference libraries without importing/loading any neural model.
        libraries = {"dinov2": ("torch", "transformers", "torchvision"),
                     "uni": ("torch", "timm", "torchvision")}.get(extractor.name, ())
        for library in libraries:
            try:
                self.configuration[library] = version(library)
            except PackageNotFoundError:
                self.configuration[library] = "not-installed"
        digest = hashlib.sha256(json.dumps(self.configuration, sort_keys=True).encode()).hexdigest()
        self.revision = f"1-{extractor.name}-{extractor.revision}-cfg{digest[:16]}"

    @staticmethod
    def _failure(reason, **diagnostics):
        return Localisation(None, 0.0, {"reason": reason, **diagnostics})

    def localise(self, inputs: Mapping) -> Localisation:
        # These five fields are the complete algorithm input, never claimed_bbox.
        context, query = inputs["context"], inputs["query"]
        ctx = inputs["context_bbox"]
        try:
            mc, mq = float(inputs["context_mpp"]), float(inputs["query_mpp"])
        except (TypeError, ValueError) as exc:
            raise ValueError("context_mpp and query_mpp must be finite and positive") from exc
        if not all(math.isfinite(v) and v > 0 for v in (mc, mq)):
            raise ValueError("context_mpp and query_mpp must be finite and positive")
        if mq > mc:
            raise ValueError("would upsample query; context must be the coarser image")
        if min(*context.size, *query.size, ctx.w, ctx.h) <= 0:
            raise ValueError("images and context_bbox must have positive dimensions")
        footprint = np.array([query.width * mq / mc, query.height * mq / mc])
        if footprint[0] > context.width or footprint[1] > context.height:
            return self._failure("query_footprint_too_large")
        width, height = (max(1, round(float(v))) for v in footprint)
        aligned = query.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)

        def origins(size, raster_side, physical_side):
            last = min(size - raster_side, math.floor(size - physical_side))
            stride = max(1, round(raster_side * self._stride_fraction))
            coords = list(range(0, last + 1, stride))
            if coords[-1] != last:
                coords.append(last)
            return coords

        xs = origins(context.width, width, footprint[0])
        ys = origins(context.height, height, footprint[1])
        grid = np.asarray([(x, y) for y in ys for x in xs], dtype=np.float64)
        if len(grid) < 2:
            return self._failure("too_few_candidates", n_candidates=len(grid))
        crops = [context.crop((int(x), int(y), int(x) + width, int(y) + height))
                 for x, y in grid]
        embeddings = np.asarray(self._extractor.embed([aligned, *crops]), dtype=np.float64)
        if (embeddings.ndim != 2 or len(embeddings) != len(grid) + 1
                or embeddings.shape[1] == 0 or not np.isfinite(embeddings).all()):
            raise ValueError("extractor must return a finite (1 + windows, D) embedding matrix")
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.isfinite(norms).all():
            raise ValueError("descriptor norms must be finite")
        if norms[0] <= 1e-12:
            return self._failure("zero_query_descriptor")
        valid = norms[1:] > 1e-12
        if not valid.any():
            return self._failure("no_valid_window_descriptors")
        grid = grid[valid]
        if len(grid) < 2:
            return self._failure("too_few_candidates", n_candidates=len(grid))
        q = embeddings[0] / norms[0]
        windows = embeddings[1:][valid] / norms[1:][valid, None]
        distances = np.linalg.norm(windows - q, axis=1)  # L2, not squared L2.
        chosen = np.argsort(distances, kind="stable")[:self._top_k]
        candidates, d = grid[chosen], distances[chosen]
        uncertainty, weights = spatial_uncertainty(
            candidates / footprint, d, self._distance_decay
        )
        x, y = candidates[0]  # Not the weighted mean between separate locations.
        sx, sy = ctx.w / context.width, ctx.h / context.height
        estimate = BBoxL0(round(ctx.x + x * sx), round(ctx.y + y * sy),
                         max(1, round(footprint[0] * sx)),
                         max(1, round(footprint[1] * sy)))
        return Localisation(estimate, 1.0 / (1.0 + uncertainty), {
            "n_candidates": len(grid), "top_k_used": len(chosen),
            "aligned_query_size_px": (width, height),
            "query_footprint_context_px": footprint.tolist(),
            "best_origin_context": candidates[0].tolist(),
            "candidate_origins_context": candidates.tolist(),
            "candidate_l2_distances": d.tolist(), "candidate_weights": weights.tolist(),
            "normalised_spatial_uncertainty": uncertainty,
            "effective_candidates": float(1.0 / np.sum(weights ** 2)),
        })
