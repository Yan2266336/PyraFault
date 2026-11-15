"""Frozen SuperPoint + LightGlue adapted to claim-independent WSI localisation.

MPP-align Q before feature extraction; no padding, automatic image resizing,
sliding-window retrieval, training, or claim-dependent search.
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
from unittest.mock import patch

import numpy as np
from PIL import Image

from pyrafault.geom import BBoxL0
from pyrafault.verify.base import Localisation, Verifier
from pyrafault.verify.efficient_loftr import Matches, _dominant_translation, _git_revision, _sha256


OFFICIAL_SOURCE_REVISION = "eb42fee2d71449efb0aa5c10549752b5d75384d8"
SUPERPOINT_URL = "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth"
SUPERPOINT_SHA256 = "52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e"
LIGHTGLUE_SHA256 = "6ff7040d0a497fc6639337946d7538dae07428c18f77a067a0b5a960e7cc551a"
SUPERPOINT_CONFIG = {"descriptor_dim": 256, "nms_radius": 4,
                     "max_num_keypoints": 2048, "detection_threshold": 0.0005,
                     "remove_borders": 4}
LIGHTGLUE_CONFIG = {"input_dim": 256, "descriptor_dim": 256, "add_scale_ori": False,
                   "n_layers": 9, "num_heads": 4, "flash": False, "mp": False,
                   "depth_confidence": 0.95, "width_confidence": 0.99,
                   "filter_threshold": 0.1, "weights": None}


def _load_official_symbols(source_root: Path):
    existing = sys.modules.get("lightglue")
    if existing is not None:
        locations = [Path(path).resolve() for path in getattr(existing, "__path__", [])]
        if source_root / "lightglue" not in locations:
            raise RuntimeError("another lightglue checkout is loaded; start a fresh process")
    source = str(source_root)
    import torch

    previous_deterministic = torch.backends.cudnn.deterministic
    sys.path.insert(0, source)
    try:
        module = importlib.import_module("lightglue")
    finally:
        sys.path.remove(source)
        torch.backends.cudnn.deterministic = previous_deterministic
    return module.SuperPoint, module.LightGlue


class OfficialSuperPointLightGlue:
    """Pinned native models and verified local weights, never auto-downloaded.

    SuperPoint has no download-disable switch. During its constructor only, its
    one documented URL is redirected to the safely loaded local state dictionary.
    The original loader is restored even if construction fails. Initialise serially.
    """

    def _expected_lightglue_sha256(self):
        """The fixed published checkpoint for this matcher, not a caller option."""
        return LIGHTGLUE_SHA256

    def __init__(self, source_root, superpoint_checkpoint, lightglue_checkpoint, *, device="auto"):
        source_root = Path(source_root).expanduser().resolve()
        sp_path = Path(superpoint_checkpoint).expanduser().resolve()
        lg_path = Path(lightglue_checkpoint).expanduser().resolve()
        if not all((source_root / "lightglue" / name).is_file()
                   for name in ("superpoint.py", "lightglue.py")):
            raise ValueError("lightglue_source must point to the official LightGlue git checkout")
        for label, path in (("SuperPoint", sp_path), ("LightGlue", lg_path)):
            if not path.is_file():
                raise ValueError(f"{label} checkpoint does not exist: {path}")
        try:
            source_revision = _git_revision(source_root)
        except ValueError as exc:
            raise ValueError("LightGlue source must be a clean git checkout at the documented revision") from exc
        if source_revision != OFFICIAL_SOURCE_REVISION:
            raise ValueError(f"LightGlue source revision mismatch: expected {OFFICIAL_SOURCE_REVISION}, "
                             f"got {source_revision}")
        sp_digest, lg_digest = _sha256(sp_path), _sha256(lg_path)
        if sp_digest != SUPERPOINT_SHA256 or lg_digest != self._expected_lightglue_sha256():
            raise ValueError("SuperPoint/LightGlue checkpoint SHA-256 mismatch; use the documented official files")
        try:
            import torch
            superpoint, lightglue = _load_official_symbols(source_root)
        except ImportError as exc:
            raise RuntimeError("LightGlue needs PyTorch and Kornia from the project inference environment") from exc
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device not in {"cpu", "cuda"}:
            raise ValueError("LightGlue supports auto, cpu, or cuda in this adapter")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but torch.cuda.is_available() is false")
        if torch.get_float32_matmul_precision() != "highest":
            raise ValueError("LightGlue fp32 baseline requires torch matmul precision 'highest'")
        sp_weights = torch.load(sp_path, map_location="cpu", weights_only=True)
        lg_weights = torch.load(lg_path, map_location="cpu", weights_only=True)
        if not isinstance(sp_weights, Mapping) or not isinstance(lg_weights, Mapping):
            raise ValueError("both files must contain native state dictionaries")

        def local_superpoint(url, *args, **kwargs):
            if url != SUPERPOINT_URL:
                raise RuntimeError(f"unexpected model download requested: {url}")
            return sp_weights

        with patch.object(torch.hub, "load_state_dict_from_url", local_superpoint):
            extractor = superpoint(**SUPERPOINT_CONFIG)
        # features=None avoids LightGlue's own automatic download. Set every
        # SuperPoint-specific architecture choice explicitly, then load strictly.
        previous_flash = torch.backends.cuda.flash_sdp_enabled()
        try:
            model = lightglue(features=None, **LIGHTGLUE_CONFIG)
        finally:
            torch.backends.cuda.enable_flash_sdp(previous_flash)
        converted = dict(lg_weights)
        for layer in range(LIGHTGLUE_CONFIG["n_layers"]):
            for kind in ("self_attn", "cross_attn"):
                converted = {key.replace(f"{kind}.{layer}", f"transformers.{layer}.{kind}"): value
                             for key, value in converted.items()}
            # The official constructor's strict=False discards these obsolete
            # scalar head entries from the verified legacy release. Restrict the
            # exception to this exact list, then check all current keys strictly.
            converted.pop(f"log_assignment.{layer}.r", None)
        # This sole deterministic schedule buffer is absent from the released
        # legacy checkpoint; never allow missing learned parameters silently.
        if "confidence_thresholds" not in converted:
            converted["confidence_thresholds"] = model.confidence_thresholds.detach().clone()
        model.load_state_dict(converted, strict=True)
        self._extractor = extractor.eval().requires_grad_(False).to(device)
        self._model = model.eval().requires_grad_(False).to(device)
        self._torch, self._device = torch, device
        self.configuration = {
            "source_revision": source_revision, "superpoint_sha256": sp_digest,
            "lightglue_sha256": lg_digest, "superpoint": dict(SUPERPOINT_CONFIG),
            "lightglue": dict(LIGHTGLUE_CONFIG), "extract_resize": None,
            "device": device, "precision": "fp32-flash-off-matmul-highest",
            "autocast": False,
            "attention_backend": "torch-sdpa-math",
            "dependencies": {"torch": torch.__version__, "numpy": np.__version__,
                             "kornia": version("kornia"), "Pillow": version("Pillow")},
            "cudnn_allow_tf32": False, "cudnn_deterministic": True,
            "pruning_keypoint_thresholds": dict(model.pruning_keypoint_thresholds),
        }
        digest = hashlib.sha256(json.dumps(self.configuration, sort_keys=True).encode()).hexdigest()
        self.revision = (f"official-{source_revision[:12]}-sp{sp_digest[:12]}-lg{lg_digest[:12]}-"
                         f"cfg{digest[:12]}-{device}")

    def match(self, query: np.ndarray, context: np.ndarray) -> Matches:
        torch = self._torch
        if torch.get_float32_matmul_precision() != "highest":
            raise ValueError("LightGlue fp32 baseline requires torch matmul precision 'highest'")

        def tensor(image):
            if (image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8
                    or min(image.shape[:2]) < 8):
                raise ValueError("SuperPoint requires RGB uint8 arrays with sides at least 8 pixels")
            return torch.from_numpy(np.array(image, copy=True, order="C")).permute(2, 0, 1).to(
                device=self._device, dtype=torch.float32).div_(255)

        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH), \
                torch.backends.cudnn.flags(benchmark=False, deterministic=True, allow_tf32=False), \
                torch.autocast(device_type=self._device, enabled=False), \
                torch.inference_mode():
            q_features = self._extractor.extract(tensor(query), resize=None)
            c_features = self._extractor.extract(tensor(context), resize=None)
            predictions = self._model({"image0": q_features, "image1": c_features})
        pairs = predictions["matches"][0].detach().cpu().numpy()
        scores = predictions["scores"][0].detach().float().cpu().numpy()
        qp = q_features["keypoints"][0].detach().float().cpu().numpy()
        cp = c_features["keypoints"][0].detach().float().cpu().numpy()
        if (pairs.ndim != 2 or pairs.shape[1:] != (2,) or pairs.dtype.kind not in "iu"
                or scores.ndim != 1 or len(pairs) != len(scores)
                or (pairs < 0).any() or (pairs[:, 0] >= len(qp)).any()
                or (pairs[:, 1] >= len(cp)).any()):
            raise ValueError("LightGlue returned malformed match indices or scores")
        return Matches(qp[pairs[:, 0]], cp[pairs[:, 1]], scores)


class LightGlueBaseline(Verifier):
    """MPP-aligned sparse matching; rho combines geometry with pair scores."""

    name = "lightglue_baseline"

    def __init__(self, matcher, *, min_matches=4, translation_tolerance_px=3.0):
        try:
            minimum = int(min_matches)
            tolerance = float(translation_tolerance_px)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError("invalid minimum support or translation tolerance") from exc
        if isinstance(min_matches, bool) or minimum != min_matches or minimum < 4:
            raise ValueError("min_matches must be an integer at least 4")
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("translation_tolerance_px must be finite and positive")
        self._matcher, self._min_matches, self._translation_tolerance = matcher, minimum, tolerance
        self.revision = (f"1-{getattr(matcher, 'revision', 'unversioned-matcher')}-mpp-rgb-noresize-"
                         f"pixelcentres-translation{tolerance:g}-min{minimum}-rho-scorexratio")

    @staticmethod
    def _failure(reason, **diagnostics):
        return Localisation(None, 0.0, {"reason": reason, **diagnostics})

    def localise(self, inputs: Mapping) -> Localisation:
        # Exactly five fields; never inspect a claim, label, ID, target, or split.
        context, query, ctx = inputs["context"], inputs["query"], inputs["context_bbox"]
        try:
            cm, qm = float(inputs["context_mpp"]), float(inputs["query_mpp"])
        except (TypeError, ValueError) as exc:
            raise ValueError("context_mpp and query_mpp must be finite and positive") from exc
        if not all(math.isfinite(value) and value > 0 for value in (cm, qm)):
            raise ValueError("context_mpp and query_mpp must be finite and positive")
        if qm > cm:
            raise ValueError("would upsample query; context must be the coarser image")
        if min(*context.size, *query.size, ctx.w, ctx.h) <= 0:
            raise ValueError("images and context_bbox must have positive dimensions")
        footprint = np.array(query.size, dtype=float) * (qm / cm)
        if not np.isfinite(footprint).all() or (footprint > np.array(context.size)).any():
            return self._failure("query_footprint_too_large")
        aligned_size = tuple(max(1, round(value)) for value in footprint)
        sizes = {"aligned_query_size_px": list(aligned_size), "context_size_px": list(context.size)}
        if min(*aligned_size, *context.size) < 8:
            return self._failure("input_too_small_for_superpoint", **sizes)
        aligned = query.convert("RGB").resize(aligned_size, Image.Resampling.LANCZOS)
        raw = self._matcher.match(np.asarray(aligned, dtype=np.uint8),
                                  np.asarray(context.convert("RGB"), dtype=np.uint8))
        qp, cp = np.asarray(raw.query_points, dtype=float), np.asarray(raw.context_points, dtype=float)
        confidence = np.asarray(raw.confidence, dtype=float).reshape(-1)
        if (qp.ndim != 2 or cp.ndim != 2 or qp.shape[1:] != (2,) or cp.shape[1:] != (2,)
                or len(qp) != len(cp) or len(qp) != len(confidence)):
            raise ValueError("matcher must return equally sized (N,2), (N,2), and (N,) arrays")
        total = len(confidence)
        keep = (np.isfinite(qp).all(axis=1) & np.isfinite(cp).all(axis=1)
                & np.isfinite(confidence) & (confidence > 0) & (confidence <= 1)
                & (qp >= 0).all(axis=1) & (qp < np.array(aligned_size)).all(axis=1)
                & (cp >= 0).all(axis=1) & (cp < np.array(context.size)).all(axis=1))
        qp, cp, confidence = qp[keep], cp[keep], confidence[keep]
        if len(confidence):
            _, indices = np.unique(np.round(np.concatenate([qp, cp], axis=1), 4),
                                   axis=0, return_index=True)
            indices.sort()
            qp, cp, confidence = qp[indices], cp[indices], confidence[indices]
        uq, uc = len(np.unique(np.round(qp, 3), axis=0)), len(np.unique(np.round(cp, 3), axis=0))
        diagnostics = {"total_matches": total, "valid_matches": len(confidence),
                       "unique_query_points": uq, "unique_context_points": uc, **sizes}
        if min(len(confidence), uq, uc) < self._min_matches:
            return self._failure("too_few_valid_matches", **diagnostics, min_matches=self._min_matches)
        point_scale = footprint / np.array(aligned_size)
        # Native SP coordinates use integer pixel centres. The +.5 conversion
        # preserves raster-edge origins under fractional MPP/raster rounding.
        displacement = (cp + 0.5) - (qp + 0.5) * point_scale
        translation, inliers = _dominant_translation(displacement, confidence,
                                                    self._translation_tolerance)
        count = int(inliers.sum())
        uiq = len(np.unique(np.round(qp[inliers], 3), axis=0))
        uic = len(np.unique(np.round(cp[inliers], 3), axis=0))
        diagnostics.update(inlier_matches=count, unique_inlier_query_points=uiq,
                           unique_inlier_context_points=uic)
        if min(count, uiq, uic) < self._min_matches:
            return self._failure("too_few_geometric_inliers", **diagnostics, min_matches=self._min_matches)
        x, y = map(float, translation)
        max_x, max_y = np.array(context.size) - footprint
        tolerance = self._translation_tolerance
        if x < -tolerance or y < -tolerance or x > max_x + tolerance or y > max_y + tolerance:
            return self._failure("translation_outside_context", **diagnostics,
                                 translation_xy_context=[x, y])
        x, y = min(max(x, 0.0), max_x), min(max(y, 0.0), max_y)
        sx, sy = ctx.w / context.width, ctx.h / context.height
        estimate = BBoxL0(round(ctx.x + x * sx), round(ctx.y + y * sy),
                          max(1, round(footprint[0] * sx)), max(1, round(footprint[1] * sy)))
        fraction, mean_score = count / len(confidence), float(confidence[inliers].mean())
        diagnostics.update(inlier_fraction=fraction, mean_inlier_match_score=mean_score,
                           translation_xy_context=[x, y], query_to_context_point_scale=point_scale.tolist())
        return Localisation(estimate, fraction * mean_score, diagnostics)
