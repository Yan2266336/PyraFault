"""Frozen window retrieval + Statistical Uncertainty (SU), paper-formula adapter.

SU evaluates competition among similarity scores; it does not localise images.
The top-1 retrieval supplies the position. No author code, score translation,
cross-query normalization, geometric refinement, or additional training is used.
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


# Our declared numerical/domain guard, not a parameter supplied by the paper.
MIN_BEST_SIMILARITY = 1e-8


def statistical_uncertainty(similarities):
    """Return RS, SD and equal-weight SU for descending nonnegative scores.

    Scores must belong to ONE query. They are not shifted, clipped or pooled with
    other queries. Unsupported signed/near-zero scores raise instead of quietly
    modifying the ratio's definition. Dividing first avoids large-score overflow.
    """
    scores = np.asarray(similarities, dtype=np.float64)
    if scores.ndim != 1 or len(scores) < 2:
        raise ValueError("SU requires a one-dimensional vector of at least two scores")
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("SU scores must be finite and nonnegative")
    if (np.diff(scores) > 0).any():
        raise ValueError("SU scores must be sorted in descending order")
    if scores[0] <= MIN_BEST_SIMILARITY:
        raise ValueError("best similarity is too small for a meaningful ratio")
    ratios = scores / scores[0]
    rs = float(np.mean(ratios[1:]))
    sd = float(np.median(ratios))
    return {"ratio_spread": rs, "similarity_distribution": sd,
            "statistical_uncertainty": 0.5 * rs + 0.5 * sd}


class StatisticalUncertaintyBaseline(Verifier):
    """Claim-independent top-1 window retrieval with SU confidence.

    The retrieval steps deliberately match SUE's grid and unit-L2 ranking so the
    two methods compare confidence rather than different localisation pipelines.
    This remains an independent Verifier, not a subclass or ensemble of SUE.
    """

    name = "statistical_uncertainty_baseline"

    def __init__(self, extractor, *, top_k: int = 10, stride_fraction: float = 0.5):
        if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k < 2:
            raise ValueError("top_k must be an integer >= 2")
        try:
            stride_fraction = float(stride_fraction)
        except (TypeError, ValueError) as exc:
            raise ValueError("stride_fraction must be numeric") from exc
        if not math.isfinite(stride_fraction) or not 0 < stride_fraction <= 1:
            raise ValueError("stride_fraction must be finite and in (0, 1]")
        if not getattr(extractor, "name", None) or not getattr(extractor, "revision", None):
            raise ValueError("extractor must declare its name and frozen revision")
        self._extractor = extractor
        self._top_k = int(top_k)
        self._stride_fraction = stride_fraction
        self.configuration = {
            "extractor": {key: str(getattr(extractor, key, "unspecified")) for key in
                          ("name", "revision", "preprocessing", "dim", "input_size",
                           "device", "batch_size", "hf_repo", "timm_model")},
            "top_k": self._top_k, "stride_fraction": self._stride_fraction,
            "ranking": "stable-ascending-unit-L2", "score": "raw-unit-cosine-dot-product",
            "score_order": "descending-selected-cosines",
            "alpha": 0.5, "formula": "0.5*mean(s[1:]/s[0])+0.5*median(s/s[0])",
            "minimum_best_similarity": MIN_BEST_SIMILARITY,
            "score_domain_policy": "nonnegative-topk-otherwise-rho0-preserve-bbox",
            "reliability": "1-SU", "edge_windows": True,
            "numpy": np.__version__, "pillow": Image.__version__,
        }
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
        # These five fields are the only algorithm input: never claim, id or target.
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
        distances = np.linalg.norm(windows - q, axis=1)
        chosen = np.argsort(distances, kind="stable")[:self._top_k]
        candidates, d = grid[chosen], distances[chosen]
        # Dot products avoid the cancellation that can turn an orthogonal score
        # into a tiny negative via 1-L2^2/2. Do not shift or clip raw cosine.
        scores = windows[chosen] @ q
        # L2/cosine rankings are mathematically equivalent, but roundoff ties may
        # differ. Sort these same selected scores for SU, not for the top-1 bbox.
        su_scores = np.sort(scores)[::-1]
        x, y = candidates[0]
        sx, sy = ctx.w / context.width, ctx.h / context.height
        estimate = BBoxL0(round(ctx.x + x * sx), round(ctx.y + y * sy),
                         max(1, round(footprint[0] * sx)),
                         max(1, round(footprint[1] * sy)))
        diagnostics = {
            "n_candidates": len(grid), "top_k_used": len(chosen),
            "aligned_query_size_px": (width, height),
            "query_footprint_context_px": footprint.tolist(),
            "best_origin_context": candidates[0].tolist(),
            "candidate_origins_context": candidates.tolist(),
            "candidate_l2_distances": d.tolist(),
            "candidate_cosine_similarities": scores.tolist(),
            "su_cosine_similarities": su_scores.tolist(),
            "ratio_spread": None, "similarity_distribution": None,
            "statistical_uncertainty": None, "score_domain_supported": False,
        }
        # The paper does not specify signed/near-zero score handling. Keep the
        # retrieval result but conservatively report rho=0 and the exact reason.
        if su_scores[0] <= MIN_BEST_SIMILARITY:
            diagnostics["reason"] = "best_similarity_too_small"
            return Localisation(estimate, 0.0, diagnostics)
        if (scores < 0).any():
            diagnostics["reason"] = "negative_top_k_similarity"
            return Localisation(estimate, 0.0, diagnostics)
        metrics = statistical_uncertainty(su_scores)
        diagnostics.update(metrics, score_domain_supported=True)
        return Localisation(estimate, 1.0 - metrics["statistical_uncertainty"], diagnostics)
