"""Zero-argument baseline plugins for the unmodified stage-13 harness.

The seven algorithm classes still accept injected matchers/extractors for testing.
These small entry points only prepare their frozen backends from environment paths.
Construction does not load weights or access the network, so deciding from recorded
localisations remains cheap. First localisation (or reading revision) loads once.
"""
from __future__ import annotations

import os
from pathlib import Path

from pyrafault.verify.base import Verifier

DINO_REPO = "facebook/dinov2-base"
DINO_REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"


class _LazyPlugin(Verifier):
    path_variables: tuple[str, ...] = ()

    def __init__(self):
        self._paths = {key: os.environ.get(key) for key in self.path_variables}
        self.device = os.environ.get("PYRAFAULT_BASELINE_DEVICE", "auto")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("PYRAFAULT_BASELINE_DEVICE must be auto, cpu, or cuda")
        self._backend_instance = None

    def _path(self, variable, *, directory=False):
        value = self._paths.get(variable)
        if not value:
            raise ValueError(f"{self.name}: set {variable}; see BASELINE_HANDOVER.zh-CN.md")
        path = Path(value).expanduser().resolve()
        valid = path.is_dir() if directory else path.is_file()
        if not valid:
            raise ValueError(f"{variable}: {'directory' if directory else 'file'} does not exist: {path}")
        return path

    def _build_backend(self):
        raise NotImplementedError

    def _backend(self):
        if self._backend_instance is None:
            backend = self._build_backend()
            if backend.name != self.name:
                raise ValueError(f"plugin/backend identity mismatch: {self.name} != {backend.name}")
            self._backend_instance = backend
        return self._backend_instance

    @property
    def revision(self):
        return self._backend().revision

    def localise(self, inputs):
        return self._backend().localise(inputs)


class EfficientLoFTRPlugin(_LazyPlugin):
    name = "efficient_loftr_baseline"
    path_variables = ("EFFICIENT_LOFTR_SOURCE", "EFFICIENT_LOFTR_CHECKPOINT")

    def _build_backend(self):
        from pyrafault.verify.efficient_loftr import EfficientLoFTRBaseline, OfficialEfficientLoFTR
        return EfficientLoFTRBaseline(OfficialEfficientLoFTR(
            self._path("EFFICIENT_LOFTR_SOURCE", directory=True),
            self._path("EFFICIENT_LOFTR_CHECKPOINT"), device=self.device))


class MatchAnythingELoFTRPlugin(_LazyPlugin):
    name = "matchanything_eloftr_baseline"
    path_variables = ("MATCHANYTHING_ELOFTR_SOURCE", "MATCHANYTHING_ELOFTR_CHECKPOINT")

    def _build_backend(self):
        from pyrafault.verify.matchanything_eloftr import MatchAnythingELoFTRBaseline, OfficialMatchAnythingELoFTR
        return MatchAnythingELoFTRBaseline(OfficialMatchAnythingELoFTR(
            self._path("MATCHANYTHING_ELOFTR_SOURCE", directory=True),
            self._path("MATCHANYTHING_ELOFTR_CHECKPOINT"), device=self.device))


class RoMaPlugin(_LazyPlugin):
    name = "roma_baseline"
    path_variables = ("ROMA_SOURCE", "ROMA_CHECKPOINT", "ROMA_DINOV2_WEIGHTS")

    def _build_backend(self):
        from pyrafault.verify.roma import OfficialRoMa, RoMaBaseline
        return RoMaBaseline(OfficialRoMa(
            self._path("ROMA_SOURCE", directory=True), self._path("ROMA_CHECKPOINT"),
            self._path("ROMA_DINOV2_WEIGHTS"), device=self.device))


class LightGluePlugin(_LazyPlugin):
    name = "lightglue_baseline"
    path_variables = ("LIGHTGLUE_SOURCE", "SUPERPOINT_CHECKPOINT", "LIGHTGLUE_CHECKPOINT")

    def _build_backend(self):
        from pyrafault.verify.lightglue import LightGlueBaseline, OfficialSuperPointLightGlue
        return LightGlueBaseline(OfficialSuperPointLightGlue(
            self._path("LIGHTGLUE_SOURCE", directory=True), self._path("SUPERPOINT_CHECKPOINT"),
            self._path("LIGHTGLUE_CHECKPOINT"), device=self.device))


class MinimaLightGluePlugin(_LazyPlugin):
    name = "minima_lightglue_baseline"
    path_variables = ("LIGHTGLUE_SOURCE", "SUPERPOINT_CHECKPOINT", "MINIMA_LIGHTGLUE_CHECKPOINT")

    def _build_backend(self):
        from pyrafault.verify.minima_lightglue import MinimaLightGlueBaseline, OfficialMinimaLightGlue
        return MinimaLightGlueBaseline(OfficialMinimaLightGlue(
            self._path("LIGHTGLUE_SOURCE", directory=True), self._path("SUPERPOINT_CHECKPOINT"),
            self._path("MINIMA_LIGHTGLUE_CHECKPOINT"), device=self.device))


class _PinnedDinoV2Extractor:
    """Baseline-only, frozen DINOv2 from an explicitly prepared offline snapshot.

    Keep the upstream construction/mining extractor untouched. Unlike its default
    Hub-main lookup, this path is pinned and never downloads during evaluation.
    """
    name = "dinov2"
    revision = DINO_REVISION
    hf_repo = DINO_REPO
    timm_model = None
    input_size = 224
    batch_size = 64

    def __init__(self, snapshot, device):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        from PIL import Image

        if snapshot.name != DINO_REVISION:
            raise ValueError(f"PYRAFAULT_DINOV2_SNAPSHOT must be the pinned snapshots/{DINO_REVISION} directory")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but torch.cuda.is_available() is false")
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(str(snapshot), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(snapshot), local_files_only=True).eval().to(device)
        self.model.requires_grad_(False)
        self.dim = int(self.model.config.hidden_size)
        if self.model.config.model_type != "dinov2" or self.dim != 768:
            raise ValueError("snapshot must contain facebook/dinov2-base (768-dimensional DINOv2)")
        probe = self.processor(images=[Image.new("RGB", (672, 448))], return_tensors="pt")
        if tuple(probe["pixel_values"].shape[-2:]) != (224, 224):
            raise ValueError("pinned DINOv2 processor must produce 224x224 inputs")
        self.preprocessing = str(self.processor)

    def embed(self, images):
        import numpy as np
        import torch
        out = []
        with torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                batch = [im.convert("RGB") for im in images[start:start+self.batch_size]]
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                out.append(self.model(**inputs).last_hidden_state[:, 0].float().cpu().numpy())
        arr = np.concatenate(out).astype(np.float32)
        return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-8)


class SUEPlugin(_LazyPlugin):
    name = "sue_baseline"
    path_variables = ("PYRAFAULT_DINOV2_SNAPSHOT",)

    def _build_backend(self):
        from pyrafault.verify.sue import SUEBaseline
        extractor = _PinnedDinoV2Extractor(self._path("PYRAFAULT_DINOV2_SNAPSHOT", directory=True), self.device)
        return SUEBaseline(extractor)


class StatisticalUncertaintyPlugin(_LazyPlugin):
    name = "statistical_uncertainty_baseline"
    path_variables = ("PYRAFAULT_DINOV2_SNAPSHOT",)

    def _build_backend(self):
        from pyrafault.verify.statistical_uncertainty import StatisticalUncertaintyBaseline
        extractor = _PinnedDinoV2Extractor(self._path("PYRAFAULT_DINOV2_SNAPSHOT", directory=True), self.device)
        return StatisticalUncertaintyBaseline(extractor)


class ClaimConditionedPlugin(_LazyPlugin):
    """The claim-conditioned baseline on kornia's LoFTR -- the same matcher, built the
    same way, as the `--matcher loftr` ZoomTrace runs the paper reports.

    No `path_variables`: the weights come from the torch hub cache the ZoomTrace runs
    already populated, so there is no handover path to set. `reads_claim` is mirrored
    from the backend rather than restated, because the harness reads it off whatever
    `load_plugin` returns -- which is this object, not the class it wraps, and a
    mismatch between the two would put the claim in the key for one and not the other.
    """

    name = "claim_conditioned"
    reads_claim = True

    def _build_backend(self):
        from pyrafault.verify.claim_conditioned import ClaimConditionedBaseline
        from pyrafault.verify.matchers import LoftrMatcher
        matcher = LoftrMatcher(device=None if self.device == "auto" else self.device)
        backend = ClaimConditionedBaseline(matcher)
        if backend.reads_claim is not self.reads_claim:
            raise ValueError("plugin/backend disagree on reads_claim")
        return backend
