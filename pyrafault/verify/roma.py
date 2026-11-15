"""RoMa v1 dense correspondences adapted to claim-independent WSI localisation.

One fixed query grid, no random sampling, sliding-window retrieval or training.
RoMa sees the original RGB rasters; MPP conversion happens on the returned points.
"""
from __future__ import annotations

import hashlib
import importlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys
from typing import Mapping

import numpy as np
from PIL import Image

from pyrafault.geom import BBoxL0
from pyrafault.verify.base import Localisation, Verifier
from pyrafault.verify.efficient_loftr import Matches, _dominant_translation, _git_revision, _sha256


OFFICIAL_SOURCE_REVISION = "77f8d68803526dcddfd9b7a46bc76125bdc25f15"
MODEL_CONFIG = {
    "coarse_res": 560, "upsample_res": 864, "symmetric": False,
    "use_custom_corr": False, "upsample_preds": True,
    "with_padding": False, "do_compile": False,
}
GRID_SIDE = 32


def _load_official_symbols(source_root: Path):
    """Load the external native package, never substitute an unrelated install."""
    existing = sys.modules.get("romatch")
    if existing is not None:
        locations = [Path(path).resolve() for path in getattr(existing, "__path__", [])]
        if source_root / "romatch" not in locations:
            raise RuntimeError("another romatch checkout is already loaded; start a fresh process")
    source = str(source_root)
    sys.path.insert(0, source)
    try:
        module = importlib.import_module("romatch")
    finally:
        sys.path.remove(source)
    return module.roma_outdoor


def _attention_configuration():
    """Record the native DINO backend selected during import, without changing it."""
    attention = importlib.import_module("romatch.models.transformer.layers.attention")
    uses_xformers = attention.XFORMERS_AVAILABLE
    return {"backend": "xformers" if uses_xformers else "torch-sdpa",
            "xformers_version": version("xformers") if uses_xformers else None}


class OfficialRoMa:
    """Pinned, frozen outdoor RoMa with two explicitly supplied local weight files.

Weights are safely loaded and their actual hashes recorded, not downloaded here.
The native factory strictly loads both RoMa and DINOv2 state dictionaries.
"""

    def __init__(self, source_root, checkpoint, dinov2_weights, *, device="auto"):
        source_root = Path(source_root).expanduser().resolve()
        checkpoint = Path(checkpoint).expanduser().resolve()
        dinov2_weights = Path(dinov2_weights).expanduser().resolve()
        if not (source_root / "romatch" / "models" / "model_zoo" / "__init__.py").is_file():
            raise ValueError("roma_source must point to the official RoMa git checkout")
        for label, path in (("RoMa checkpoint", checkpoint), ("DINOv2 weights", dinov2_weights)):
            if not path.is_file():
                raise ValueError(f"{label} does not exist: {path}")
        try:
            source_revision = _git_revision(source_root)
        except ValueError as exc:
            raise ValueError("RoMa source must be a clean git checkout at the documented revision") from exc
        if source_revision != OFFICIAL_SOURCE_REVISION:
            raise ValueError(f"RoMa source revision mismatch: expected {OFFICIAL_SOURCE_REVISION}, "
                             f"got {source_revision}")
        checkpoint_digest, dino_digest = _sha256(checkpoint), _sha256(dinov2_weights)
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("RoMa requires PyTorch and the project's inference dependencies") from exc
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device not in {"cpu", "cuda"}:
            raise ValueError("RoMa v1 supports auto, cpu, or cuda in this adapter")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but torch.cuda.is_available() is false")
        if torch.get_float32_matmul_precision() != "highest":
            raise ValueError("RoMa requires torch float32 matmul precision 'highest'; "
                             "set it explicitly before loading the model")
        try:
            factory = _load_official_symbols(source_root)
            attention_config = _attention_configuration()
        except ImportError as exc:
            raise RuntimeError("RoMa inference dependencies are missing; use the project environment "
                               "and requirements-efficient-loftr.txt, not its full training stack") from exc
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
        dino_weights = torch.load(dinov2_weights, map_location="cpu", weights_only=True)
        if not isinstance(weights, Mapping) or not isinstance(dino_weights, Mapping):
            raise ValueError("RoMa and DINOv2 files must contain native state dictionaries")
        model = factory(device=device, weights=weights, dinov2_weights=dino_weights, **MODEL_CONFIG)
        self._model = model.eval().requires_grad_(False).to(device)
        # Native DINO is stored in a plain list, outside nn.Module recursion.
        self._model.encoder.dinov2_vitl14[0].eval().requires_grad_(False).to(device)
        self._torch, self._device = torch, device
        versions = {name: version(name) for name in ("Pillow", "torchvision", "einops", "loguru")}
        versions.update(numpy=np.__version__, torch=torch.__version__)
        self.configuration = {
            "source_revision": source_revision, "checkpoint_sha256": checkpoint_digest,
            "dinov2_sha256": dino_digest, "model_config": dict(MODEL_CONFIG),
            "sampling": {"method": "query-fixed-grid", "grid_side": GRID_SIDE},
            "device": device, "dependencies": versions,
            "attention": attention_config,
            "precision": "native-cpu-fp32" if device == "cpu" else "native-cuda-mixed-fp16",
            "matmul_precision": "highest",
        }
        digest = hashlib.sha256(json.dumps(self.configuration, sort_keys=True).encode()).hexdigest()
        self.revision = (f"official-{source_revision[:12]}-ckpt-{checkpoint_digest[:12]}-"
                         f"dino-{dino_digest[:12]}-cfg{digest[:12]}-{device}-grid{GRID_SIDE}")

    def match(self, query: np.ndarray, context: np.ndarray) -> Matches:
        for image in (query, context):
            if (image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8
                    or min(image.shape[:2]) <= 0):
                raise ValueError("RoMa inputs must be nonempty RGB uint8 arrays")
        with self._torch.inference_mode():
            warp, certainty = self._model.match(
                Image.fromarray(np.array(query, copy=True, order="C")),
                Image.fromarray(np.array(context, copy=True, order="C")),
                batched=True, device=self._device,
            )
        if (warp.ndim != 4 or warp.shape[0] != 1 or warp.shape[-1] != 4
                or certainty.shape != warp.shape[:3] or min(warp.shape[1:3]) <= 0):
            raise ValueError("RoMa must return warp (1,H,W,4) and certainty (1,H,W)")
        height, width = warp.shape[1:3]
        rows = ((np.arange(min(GRID_SIDE, height)) + 0.5) * height / min(GRID_SIDE, height)).astype(int)
        cols = ((np.arange(min(GRID_SIDE, width)) + 0.5) * width / min(GRID_SIDE, width)).astype(int)
        torch = self._torch
        row_ids = torch.as_tensor(rows, device=warp.device)
        col_ids = torch.as_tensor(cols, device=warp.device)
        points = warp[0, row_ids[:, None], col_ids[None, :]].detach().float().cpu().numpy().reshape(-1, 4)
        scores = certainty[0, row_ids[:, None], col_ids[None, :]].detach().float().cpu().numpy().reshape(-1)
        valid = (np.isfinite(points).all(axis=1) & np.isfinite(scores)
                 & (points >= -1).all(axis=1) & (points < 1).all(axis=1)
                 & (scores > 0) & (scores <= 1))
        points, scores = points[valid], scores[valid]
        # Author to_pixel_coordinates convention: raster-edge coordinates of pixel
        # centres. Do not subtract .5 or confuse normalized warp with pixel offsets.
        query_points = (points[:, :2] + 1) * np.array([query.shape[1], query.shape[0]]) / 2
        context_points = (points[:, 2:] + 1) * np.array([context.shape[1], context.shape[0]]) / 2
        return Matches(query_points, context_points, scores)


class RoMaBaseline(Verifier):
    """RoMa point localisation; native certainty times geometric support is rho."""

    name = "roma_baseline"

    def __init__(self, matcher, *, min_matches=4, translation_tolerance_px=3.0):
        if isinstance(min_matches, bool) or int(min_matches) != min_matches or min_matches < 4:
            raise ValueError("min_matches must be an integer at least 4")
        if not math.isfinite(translation_tolerance_px) or translation_tolerance_px <= 0:
            raise ValueError("translation_tolerance_px must be finite and positive")
        self._matcher, self._min_matches = matcher, int(min_matches)
        self._translation_tolerance = float(translation_tolerance_px)
        self.revision = (f"1-{getattr(matcher, 'revision', 'unversioned-matcher')}-rgb-native-"
                         f"translation{self._translation_tolerance:g}-min{self._min_matches}-rho-certxratio")

    @staticmethod
    def _failure(reason, **diagnostics):
        return Localisation(None, 0.0, {"reason": reason, **diagnostics})

    def localise(self, inputs: Mapping) -> Localisation:
        # The only five fields read. No claim, target, split, labels or identifiers.
        context, query, ctx = inputs["context"], inputs["query"], inputs["context_bbox"]
        try:
            context_mpp, query_mpp = float(inputs["context_mpp"]), float(inputs["query_mpp"])
        except (TypeError, ValueError) as exc:
            raise ValueError("context_mpp and query_mpp must be finite and positive") from exc
        if not all(math.isfinite(value) and value > 0 for value in (context_mpp, query_mpp)):
            raise ValueError("context_mpp and query_mpp must be finite and positive")
        if min(*context.size, *query.size, ctx.w, ctx.h) <= 0:
            raise ValueError("images and context_bbox must have positive dimensions")
        ratio = query_mpp / context_mpp
        footprint = np.array(query.size, dtype=float) * ratio
        if not np.isfinite(footprint).all() or (footprint > np.array(context.size)).any():
            return self._failure("query_footprint_too_large")
        raw = self._matcher.match(np.asarray(query.convert("RGB"), dtype=np.uint8),
                                  np.asarray(context.convert("RGB"), dtype=np.uint8))
        qp, cp = np.asarray(raw.query_points, dtype=float), np.asarray(raw.context_points, dtype=float)
        confidence = np.asarray(raw.confidence, dtype=float).reshape(-1)
        if (qp.ndim != 2 or cp.ndim != 2 or qp.shape[1:] != (2,) or cp.shape[1:] != (2,)
                or len(qp) != len(cp) or len(qp) != len(confidence)):
            raise ValueError("matcher must return equally sized (N,2), (N,2), and (N,) arrays")
        total = len(confidence)
        keep = (np.isfinite(qp).all(axis=1) & np.isfinite(cp).all(axis=1)
                & np.isfinite(confidence) & (confidence > 0) & (confidence <= 1)
                & (qp >= 0).all(axis=1) & (qp < np.array(query.size)).all(axis=1)
                & (cp >= 0).all(axis=1) & (cp < np.array(context.size)).all(axis=1))
        qp, cp, confidence = qp[keep], cp[keep], confidence[keep]
        if len(confidence):
            pairs = np.round(np.concatenate([qp, cp], axis=1), 4)
            _, indices = np.unique(pairs, axis=0, return_index=True)
            indices.sort()
            qp, cp, confidence = qp[indices], cp[indices], confidence[indices]
        unique_q = len(np.unique(np.round(qp, 3), axis=0))
        unique_c = len(np.unique(np.round(cp, 3), axis=0))
        diagnostics = {"total_matches": total, "valid_matches": len(confidence),
                       "unique_query_points": unique_q, "unique_context_points": unique_c}
        if min(len(confidence), unique_q, unique_c) < self._min_matches:
            return self._failure("too_few_valid_matches", **diagnostics, min_matches=self._min_matches)
        translation, inliers = _dominant_translation(cp - qp * ratio, confidence,
                                                    self._translation_tolerance)
        n_inliers = int(inliers.sum())
        unique_iq = len(np.unique(np.round(qp[inliers], 3), axis=0))
        unique_ic = len(np.unique(np.round(cp[inliers], 3), axis=0))
        diagnostics.update(inlier_matches=n_inliers, unique_inlier_query_points=unique_iq,
                           unique_inlier_context_points=unique_ic)
        if min(n_inliers, unique_iq, unique_ic) < self._min_matches:
            return self._failure("too_few_geometric_inliers", **diagnostics, min_matches=self._min_matches)
        x, y = map(float, translation)
        max_x, max_y = np.array(context.size) - footprint
        tolerance = self._translation_tolerance
        if x < -tolerance or y < -tolerance or x > max_x + tolerance or y > max_y + tolerance:
            return self._failure("translation_outside_context", **diagnostics,
                                 translation_xy_context=[x, y])
        x, y = min(max(x, 0.0), max_x), min(max(y, 0.0), max_y)
        scale_x, scale_y = ctx.w / context.width, ctx.h / context.height
        estimate = BBoxL0(round(ctx.x + x * scale_x), round(ctx.y + y * scale_y),
                          max(1, round(footprint[0] * scale_x)), max(1, round(footprint[1] * scale_y)))
        inlier_fraction = n_inliers / len(confidence)
        mean_certainty = float(confidence[inliers].mean())
        diagnostics.update(inlier_fraction=inlier_fraction, mean_inlier_certainty=mean_certainty,
                           translation_xy_context=[x, y], query_to_context_point_scale=[ratio, ratio],
                           query_size_px=list(query.size), context_size_px=list(context.size))
        return Localisation(estimate, inlier_fraction * mean_certainty, diagnostics)
