"""EfficientLoFTR baseline adapted to PyraFault's localisation protocol.

The learned matcher supplies point correspondences.  This adapter does the
benchmark-specific work: align Q to C's physical pixel size, recover the dominant
translation, and return the query footprint in slide level-0 coordinates.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import math
import subprocess
import sys
import types
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image

from pyrafault.geom import BBoxL0
from pyrafault.verify.base import Localisation, Verifier


OFFICIAL_SOURCE_REVISION = "07e9c1401e71e9c556b1fda9d86d060af36ba176"
OFFICIAL_CHECKPOINT_SHA256 = (
    "0af6291141c736e75e94b7f8aae4399b77c7731b3f08758212b2cfe370188878"
)
NPE_LONG_SIDE = 2048


@dataclass(frozen=True)
class Matches:
    """Point correspondences in the two input rasters.

    ``query_points[i]`` corresponds to ``context_points[i]``.  Both arrays use
    ``(x, y)`` pixel coordinates; confidence is EfficientLoFTR's coarse score.
    """

    query_points: np.ndarray
    context_points: np.ndarray
    confidence: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        )
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain",
             "--untracked-files=no"],
            check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            "EfficientLoFTR source must be a git checkout at the documented revision"
        ) from exc
    if status.stdout.strip():
        raise ValueError(
            "EfficientLoFTR source has modified tracked files; use a clean checkout"
        )
    return result.stdout.strip()


def _install_official_compatibility_shims() -> type:
    """Bridge two inference-only imports in the official research checkout.

    The pinned source imports the old ``kornia.utils.grid`` path and imports one
    PyTorch-Lightning symbol only while defining logging/checkpoint helpers.  Neither
    requires the old training stack at inference time.  These narrow shims let the
    code run with the project's modern Kornia and safe ``torch.load``.
    """
    import kornia.utils

    try:
        importlib.import_module("kornia.utils.grid")
    except ModuleNotFoundError:
        grid = types.ModuleType("kornia.utils.grid")
        grid.create_meshgrid = kornia.utils.create_meshgrid
        sys.modules[grid.__name__] = grid

    if "pytorch_lightning" in sys.modules:
        from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

        return ModelCheckpoint
    if importlib.util.find_spec("pytorch_lightning") is not None:
        from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

        return ModelCheckpoint

    lightning = types.ModuleType("pytorch_lightning")
    utilities = types.ModuleType("pytorch_lightning.utilities")
    callbacks = types.ModuleType("pytorch_lightning.callbacks")
    checkpoint_module = types.ModuleType(
        "pytorch_lightning.callbacks.model_checkpoint"
    )

    def rank_zero_only(function):
        return function

    rank_zero_only.rank = 0

    class ModelCheckpoint:
        pass

    ModelCheckpoint.__module__ = checkpoint_module.__name__
    ModelCheckpoint.__qualname__ = "ModelCheckpoint"
    utilities.rank_zero_only = rank_zero_only
    checkpoint_module.ModelCheckpoint = ModelCheckpoint
    callbacks.model_checkpoint = checkpoint_module
    lightning.utilities = utilities
    lightning.callbacks = callbacks
    sys.modules[lightning.__name__] = lightning
    sys.modules[utilities.__name__] = utilities
    sys.modules[callbacks.__name__] = callbacks
    sys.modules[checkpoint_module.__name__] = checkpoint_module
    return ModelCheckpoint


def _load_official_symbols(source_root: Path):
    package = source_root / "src" / "loftr" / "__init__.py"
    if not package.is_file():
        raise ValueError(
            f"{source_root} is not an EfficientLoFTR checkout: missing {package}"
        )

    existing = sys.modules.get("src")
    if existing is not None:
        locations = [Path(p).resolve() for p in getattr(existing, "__path__", [])]
        expected = (source_root / "src").resolve()
        if expected not in locations:
            raise RuntimeError(
                "another top-level 'src' package is already loaded; start a fresh "
                "process before constructing the official EfficientLoFTR backend"
            )

    source = str(source_root)
    sys.path.insert(0, source)
    try:
        module = importlib.import_module("src.loftr")
    finally:
        sys.path.remove(source)
    return module.LoFTR, module.full_default_cfg, module.reparameter


class OfficialEfficientLoFTR:
    """Thin inference wrapper around the pinned official implementation.

    The source checkout and checkpoint remain external; no third-party model code or
    193 MB weight file is copied into this repository.
    """

    max_input_side = 2048

    def __init__(self, source_root, checkpoint, *, device: str = "auto"):
        source_root = Path(source_root).expanduser().resolve()
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise ValueError(f"EfficientLoFTR checkpoint does not exist: {checkpoint}")

        source_revision = _git_revision(source_root)
        if source_revision != OFFICIAL_SOURCE_REVISION:
            raise ValueError(
                "EfficientLoFTR source revision mismatch: expected "
                f"{OFFICIAL_SOURCE_REVISION}, got {source_revision}"
            )
        checkpoint_digest = _sha256(checkpoint)
        if checkpoint_digest != OFFICIAL_CHECKPOINT_SHA256:
            raise ValueError(
                "EfficientLoFTR checkpoint sha256 mismatch: expected "
                f"{OFFICIAL_CHECKPOINT_SHA256}, got {checkpoint_digest}"
            )

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("EfficientLoFTR requires PyTorch") from exc
        try:
            import kornia
        except ImportError as exc:
            raise RuntimeError(
                "EfficientLoFTR optional dependencies are missing; install "
                "requirements-efficient-loftr.txt"
            ) from exc

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but torch.cuda.is_available() is false")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but torch.backends.mps.is_available() is false")
        if device not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be one of auto, cpu, cuda, or mps")

        try:
            checkpoint_class = _install_official_compatibility_shims()
            LoFTR, default_config, reparameter = _load_official_symbols(source_root)
        except ImportError as exc:
            raise RuntimeError(
                "EfficientLoFTR optional dependencies are missing; install "
                "requirements-efficient-loftr.txt"
            ) from exc

        config = deepcopy(default_config)
        # The official config explicitly recommends setting NPE to the test long side
        # when it exceeds the 832 px training size.  PyraFault's pilot context is 2048.
        config["coarse"]["npe"] = [832, 832, NPE_LONG_SIDE, NPE_LONG_SIDE]
        model = LoFTR(config=config)
        with torch.serialization.safe_globals([checkpoint_class]):
            saved = torch.load(
                checkpoint, map_location="cpu", weights_only=True
            )
        model.load_state_dict(saved["state_dict"], strict=True)
        self._model = reparameter(model).eval().to(device)
        self._model.requires_grad_(False)
        self._torch = torch
        self._device = device
        self.revision = (
            f"official-{source_revision[:12]}-ckpt-{checkpoint_digest[:12]}-"
            f"full-fp32-npe{NPE_LONG_SIDE}-{device}-"
            f"torch{torch.__version__}-kornia{kornia.__version__}"
        )

    def match(self, query: np.ndarray, context: np.ndarray) -> Matches:
        """Match normalized grayscale ``image0=query`` to ``image1=context``."""
        torch = self._torch

        def tensor(image):
            if image.ndim != 2 or image.dtype != np.uint8:
                raise ValueError("matcher inputs must be 2-D uint8 grayscale arrays")
            return (
                torch.from_numpy(np.array(image, dtype=np.uint8, copy=True, order="C"))
                [None, None]
                .to(device=self._device, dtype=torch.float32)
                .div_(255.0)
            )

        batch = {"image0": tensor(query), "image1": tensor(context)}
        with torch.inference_mode():
            self._model(batch)
        return Matches(
            batch["mkpts0_f"].detach().cpu().numpy(),
            batch["mkpts1_f"].detach().cpu().numpy(),
            batch["mconf"].detach().cpu().numpy(),
        )


def _pad_to_multiple(image: np.ndarray, multiple: int = 32) -> np.ndarray:
    height, width = image.shape
    padded_h = math.ceil(height / multiple) * multiple
    padded_w = math.ceil(width / multiple) * multiple
    if (padded_h, padded_w) == (height, width):
        return np.ascontiguousarray(image)
    return np.pad(
        image, ((0, padded_h - height), (0, padded_w - width)), mode="edge"
    )


def _dominant_translation(displacements, confidence, tolerance):
    """Return a deterministic translation and its inlier mask.

    Four shifted grids avoid the common histogram-boundary failure, then each grid's
    mode is refined by a median and evaluated with the same Euclidean tolerance.
    Runtime is linear in the number of matches rather than quadratic.
    """
    candidates = []
    for offset_x in (0.0, tolerance / 2):
        for offset_y in (0.0, tolerance / 2):
            cells = np.floor(
                (displacements + np.array([offset_x, offset_y])) / tolerance
            ).astype(np.int64)
            unique, inverse, counts = np.unique(
                cells, axis=0, return_inverse=True, return_counts=True
            )
            best_count = int(counts.max())
            tied = np.flatnonzero(counts == best_count)
            # Confidence only breaks equal-support grid ties; reliability remains the
            # transparent geometric inlier ratio.
            sums = [float(confidence[inverse == index].sum()) for index in tied]
            best = int(tied[int(np.argmax(sums))])
            centre = np.median(displacements[inverse == best], axis=0)
            inliers = np.linalg.norm(displacements - centre, axis=1) <= tolerance
            if inliers.any():
                centre = np.median(displacements[inliers], axis=0)
                inliers = np.linalg.norm(displacements - centre, axis=1) <= tolerance
            candidates.append((centre, inliers))

    # The global median is a useful fifth candidate when a compact cluster straddles
    # every tested grid boundary.
    centre = np.median(displacements, axis=0)
    inliers = np.linalg.norm(displacements - centre, axis=1) <= tolerance
    if inliers.any():
        centre = np.median(displacements[inliers], axis=0)
        inliers = np.linalg.norm(displacements - centre, axis=1) <= tolerance
    candidates.append((centre, inliers))

    def rank(candidate):
        centre, inliers = candidate
        return (
            int(inliers.sum()),
            float(confidence[inliers].sum()),
            -float(centre[0]),
            -float(centre[1]),
        )

    return max(candidates, key=rank)


class EfficientLoFTRBaseline(Verifier):
    """Localise an MPP-aligned query from EfficientLoFTR correspondences.

    Reliability is the fraction of valid, unique matches consistent with the
    recovered translation.  It is a method-specific ranking score, not a probability;
    ``tau`` must be fitted on PyraFault calibration data.
    """

    name = "efficient_loftr_baseline"

    def __init__(self, matcher, *, min_matches: int = 4,
                 translation_tolerance_px: float = 3.0):
        if min_matches < 2:
            raise ValueError("min_matches must be at least 2")
        if not math.isfinite(translation_tolerance_px) or translation_tolerance_px <= 0:
            raise ValueError("translation_tolerance_px must be finite and positive")
        self._matcher = matcher
        self._min_matches = int(min_matches)
        self._translation_tolerance = float(translation_tolerance_px)
        matcher_revision = getattr(matcher, "revision", "unversioned-matcher")
        self.revision = (
            f"1-{matcher_revision}-pad32-translation"
            f"{self._translation_tolerance:g}-min{self._min_matches}"
        )

    @staticmethod
    def _failure(reason: str, **diagnostics) -> Localisation:
        return Localisation(None, 0.0, {"reason": reason, **diagnostics})

    def localise(self, inputs: Mapping) -> Localisation:
        # These five fields are the entire algorithm input.  claimed_bbox, ids, split,
        # and target metadata must not affect the estimate.
        context, query = inputs["context"], inputs["query"]
        ctx = inputs["context_bbox"]
        try:
            context_mpp = float(inputs["context_mpp"])
            query_mpp = float(inputs["query_mpp"])
        except (TypeError, ValueError) as exc:
            raise ValueError("context_mpp and query_mpp must be finite and positive") from exc
        if not all(math.isfinite(value) and value > 0
                   for value in (context_mpp, query_mpp)):
            raise ValueError("context_mpp and query_mpp must be finite and positive")
        if query_mpp > context_mpp:
            raise ValueError("would upsample query; context must be the coarser image")
        if min(*context.size, *query.size, ctx.w, ctx.h) <= 0:
            raise ValueError("images and context_bbox must have positive dimensions")

        footprint_w = query.width * query_mpp / context_mpp
        footprint_h = query.height * query_mpp / context_mpp
        aligned_w = max(1, round(footprint_w))
        aligned_h = max(1, round(footprint_h))
        if footprint_w > context.width or footprint_h > context.height:
            return self._failure("query_footprint_too_large")

        aligned = query.convert("L").resize(
            (aligned_w, aligned_h), Image.Resampling.LANCZOS
        )
        query_gray = np.asarray(aligned, dtype=np.uint8)
        context_gray = np.asarray(context.convert("L"), dtype=np.uint8)
        padded_query = _pad_to_multiple(query_gray)
        padded_context = _pad_to_multiple(context_gray)
        max_side = getattr(self._matcher, "max_input_side", None)
        if max_side is not None and max(
            *padded_query.shape, *padded_context.shape
        ) > max_side:
            return self._failure(
                "input_exceeds_model_limit", max_input_side=int(max_side)
            )

        raw = self._matcher.match(padded_query, padded_context)
        query_points = np.asarray(raw.query_points, dtype=np.float64)
        context_points = np.asarray(raw.context_points, dtype=np.float64)
        confidence = np.asarray(raw.confidence, dtype=np.float64).reshape(-1)
        if (query_points.ndim != 2 or context_points.ndim != 2
                or query_points.shape[1:] != (2,)
                or context_points.shape[1:] != (2,)
                or len(query_points) != len(context_points)
                or len(query_points) != len(confidence)):
            raise ValueError(
                "matcher must return equally sized (N, 2), (N, 2), and (N,) arrays"
            )

        total_matches = len(confidence)
        finite = (
            np.isfinite(query_points).all(axis=1)
            & np.isfinite(context_points).all(axis=1)
            & np.isfinite(confidence)
        )
        inside = (
            (query_points[:, 0] >= 0) & (query_points[:, 0] < aligned_w)
            & (query_points[:, 1] >= 0) & (query_points[:, 1] < aligned_h)
            & (context_points[:, 0] >= 0) & (context_points[:, 0] < context.width)
            & (context_points[:, 1] >= 0) & (context_points[:, 1] < context.height)
        )
        keep = finite & inside
        query_points = query_points[keep]
        context_points = context_points[keep]
        confidence = confidence[keep]

        if len(confidence):
            pairs = np.round(
                np.concatenate([query_points, context_points], axis=1), decimals=4
            )
            _, indices = np.unique(pairs, axis=0, return_index=True)
            indices.sort()
            query_points = query_points[indices]
            context_points = context_points[indices]
            confidence = confidence[indices]
        valid_matches = len(confidence)
        unique_query = len(np.unique(np.round(query_points, 3), axis=0))
        unique_context = len(np.unique(np.round(context_points, 3), axis=0))
        if min(valid_matches, unique_query, unique_context) < self._min_matches:
            return self._failure(
                "too_few_valid_matches", total_matches=total_matches,
                valid_matches=valid_matches, unique_query_points=unique_query,
                unique_context_points=unique_context,
                min_matches=self._min_matches,
            )

        # The network sees an integer raster, while its physical footprint can be
        # fractional in C pixels.  Map Q keypoints back to C's pixel units before
        # estimating translation; otherwise non-integer MPP ratios bias the origin.
        query_scale = np.array(
            [footprint_w / aligned_w, footprint_h / aligned_h], dtype=np.float64
        )
        query_points_context = query_points * query_scale
        displacement = context_points - query_points_context
        translation, inliers = _dominant_translation(
            displacement, confidence, self._translation_tolerance
        )
        inlier_matches = int(inliers.sum())
        inlier_unique_query = len(
            np.unique(np.round(query_points[inliers], 3), axis=0)
        )
        inlier_unique_context = len(
            np.unique(np.round(context_points[inliers], 3), axis=0)
        )
        if min(inlier_matches, inlier_unique_query,
               inlier_unique_context) < self._min_matches:
            return self._failure(
                "too_few_geometric_inliers", total_matches=total_matches,
                valid_matches=valid_matches, inlier_matches=inlier_matches,
                unique_inlier_query_points=inlier_unique_query,
                unique_inlier_context_points=inlier_unique_context,
                min_matches=self._min_matches,
            )

        x, y = map(float, translation)
        max_x = context.width - footprint_w
        max_y = context.height - footprint_h
        tolerance = self._translation_tolerance
        if x < -tolerance or y < -tolerance or x > max_x + tolerance or y > max_y + tolerance:
            return self._failure(
                "translation_outside_context", translation_xy_context=[x, y],
                valid_matches=valid_matches, inlier_matches=inlier_matches,
            )
        x = min(max(x, 0.0), max_x)
        y = min(max(y, 0.0), max_y)

        scale_x, scale_y = ctx.w / context.width, ctx.h / context.height
        estimate = BBoxL0(
            round(ctx.x + x * scale_x),
            round(ctx.y + y * scale_y),
            max(1, round(footprint_w * scale_x)),
            max(1, round(footprint_h * scale_y)),
        )
        reliability = inlier_matches / valid_matches
        return Localisation(
            estimate,
            reliability,
            {
                "total_matches": total_matches,
                "valid_matches": valid_matches,
                "inlier_matches": inlier_matches,
                "unique_inlier_query_points": inlier_unique_query,
                "unique_inlier_context_points": inlier_unique_context,
                "median_inlier_confidence": float(np.median(confidence[inliers])),
                "translation_xy_context": [x, y],
                "query_to_context_point_scale": query_scale.tolist(),
                "aligned_query_size_px": [aligned_w, aligned_h],
                "padded_query_size_px": [padded_query.shape[1], padded_query.shape[0]],
                "padded_context_size_px": [padded_context.shape[1], padded_context.shape[0]],
            },
        )
