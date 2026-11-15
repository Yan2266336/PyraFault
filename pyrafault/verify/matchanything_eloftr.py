"""MatchAnything's ELoFTR backend with the existing WSI localisation adapter.

Only the learned point matcher changes.  MPP alignment, translation estimation,
level-0 coordinates and geometric reliability are inherited, not reimplemented.
The author source/checkpoint stay external and are never fetched at inference time.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
from importlib.metadata import version
import json
import sys
import types
from pathlib import Path

import numpy as np

from pyrafault.verify.efficient_loftr import (
    EfficientLoFTRBaseline,
    Matches,
    _git_revision,
    _sha256,
)


OFFICIAL_SOURCE_REVISION = "6a7bcb589ec8da3a9e861e799122beaa5eba2193"
# Measured from the native checkpoint in the author's linked weights.zip.
OFFICIAL_CHECKPOINT_SHA256 = (
    "b394fd368488e5e3136906daac825fadd3dd5376755dcdf5cb6ede0de20f6e7e"
)
NPE_LONG_SIDE = 2048
COARSE_MATCH_THRESHOLD = 0.1


def _load_official_symbols(source_root: Path):
    """Import only native inference modules without occupying the name 'src'.

    The model's imports are relative; its Python config alone imports
    ``src.config.default`` absolutely.  Temporary aliases during config loading
    preserve an already-loaded original EfficientLoFTR backend in the same process.
    No training wrapper, Lightning module or RoMa model is imported.
    """
    namespace = "_pyrafault_matchanything_" + hashlib.sha256(
        str(source_root).encode()
    ).hexdigest()[:12]
    if namespace not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            namespace, source_root / "src" / "__init__.py",
            submodule_search_locations=[str(source_root / "src")],
        )
        package = importlib.util.module_from_spec(spec)
        sys.modules[namespace] = package
        spec.loader.exec_module(package)

    defaults = importlib.import_module(f"{namespace}.config.default")
    config_package = importlib.import_module(f"{namespace}.config")
    config = defaults.get_cfg_defaults()
    aliases = {"src": sys.modules[namespace], "src.config": config_package,
               "src.config.default": defaults}
    saved = {name: sys.modules.get(name) for name in aliases}
    try:
        sys.modules.update(aliases)
        config.merge_from_file(str(source_root / "configs" / "models" / "eloftr_model.py"))
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    # Unlike the demo's .001 override, retain the author model/evaluator default.
    config.LOFTR.MATCH_COARSE.THR = COARSE_MATCH_THRESHOLD
    config.LOFTR.COARSE.NPE = [832, 832, NPE_LONG_SIDE, NPE_LONG_SIDE]

    def lower(value):
        # Equivalent to the author's lower_config; avoid its Lightning import.
        if isinstance(value, dict):
            return {key.lower(): lower(item) for key, item in value.items()}
        return value

    model_config = lower(config.LOFTR)
    # Kornia renamed this import; use its modern geometry API without changing
    # any third-party source or mocking training dependencies.
    try:
        importlib.import_module("kornia.utils.grid")
    except ModuleNotFoundError:
        from kornia.geometry import create_meshgrid

        grid = types.ModuleType("kornia.utils.grid")
        grid.create_meshgrid = create_meshgrid
        sys.modules[grid.__name__] = grid
    module = importlib.import_module(f"{namespace}.loftr")
    return module.LoFTR, model_config


class OfficialMatchAnythingELoFTR:
    """Frozen native author model, distinct from the Transformers conversion."""

    max_input_side = NPE_LONG_SIDE

    def __init__(self, source_root, checkpoint, *, device: str = "auto"):
        source_root = Path(source_root).expanduser().resolve()
        checkpoint = Path(checkpoint).expanduser().resolve()
        required = [source_root / "src" / "loftr" / "__init__.py",
                    source_root / "src" / "config" / "default.py",
                    source_root / "configs" / "models" / "eloftr_model.py"]
        if not all(path.is_file() for path in required):
            raise ValueError("matchanything_eloftr_source must point to the author's "
                             "imcui/third_party/MatchAnything directory")
        if not checkpoint.is_file():
            raise ValueError(f"MatchAnything-ELoFTR checkpoint does not exist: {checkpoint}")
        source_revision = _git_revision(source_root)
        if source_revision != OFFICIAL_SOURCE_REVISION:
            raise ValueError("MatchAnything-ELoFTR source revision mismatch: expected "
                             f"{OFFICIAL_SOURCE_REVISION}, got {source_revision}")
        checkpoint_digest = _sha256(checkpoint)
        if checkpoint_digest != OFFICIAL_CHECKPOINT_SHA256:
            raise ValueError("MatchAnything-ELoFTR checkpoint sha256 mismatch: expected "
                             f"{OFFICIAL_CHECKPOINT_SHA256}, got {checkpoint_digest}")
        try:
            import torch
            import kornia
            LoFTR, config = _load_official_symbols(source_root)
        except ImportError as exc:
            raise RuntimeError("MatchAnything-ELoFTR inference dependencies are missing; "
                               "install requirements-efficient-loftr.txt") from exc
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be one of auto, cpu, cuda, or mps")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but torch.cuda.is_available() is false")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but torch.backends.mps.is_available() is false")
        model = LoFTR(config=config)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["state_dict"], strict=True)
        # Match the native author eval path, which does not fuse RepVGG here.
        self._model = model.eval().requires_grad_(False).to(device)
        self._torch = torch
        self._device = device
        config_digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        versions = {name: version(name) for name in ("Pillow", "einops", "loguru", "yacs")}
        versions.update(numpy=np.__version__, torch=torch.__version__, kornia=kornia.__version__)
        dependencies_digest = hashlib.sha256(json.dumps(versions, sort_keys=True).encode()).hexdigest()
        self.configuration = {
            "source_revision": source_revision, "checkpoint_sha256": checkpoint_digest,
            "native_model_config": config, "device": device, "dependencies": versions,
            "precision": "fp32-native-rope-buffers", "reparameterized": False,
        }
        self.revision = (
            f"native-{source_revision[:12]}-ckpt-{checkpoint_digest[:12]}-"
            f"cfg{config_digest[:12]}-fp32-npe{NPE_LONG_SIDE}-thr{COARSE_MATCH_THRESHOLD:g}-"
            f"unfused-{device}-torch{torch.__version__}-kornia{kornia.__version__}-"
            f"deps{dependencies_digest[:12]}"
        )

    def match(self, query: np.ndarray, context: np.ndarray) -> Matches:
        """Two independently sized, already-padded grayscale uint8 rasters."""
        torch = self._torch

        def tensor(image):
            if (image.ndim != 2 or image.dtype != np.uint8 or min(image.shape) < 32
                    or any(side % 32 for side in image.shape)
                    or max(image.shape) > self.max_input_side):
                raise ValueError("matcher inputs must be 2-D uint8 grayscale arrays "
                                 "with sides divisible by 32, in [32, 2048]")
            return (torch.from_numpy(np.array(image, copy=True, order="C"))[None, None]
                    .to(device=self._device, dtype=torch.float32).div_(255.0))

        batch = {"image0": tensor(query), "image1": tensor(context)}
        # Padding was minimal edge extension, not a square canvas; invalid padded
        # matches are filtered by the inherited localise. No same-size mask stacking.
        with torch.inference_mode():
            self._model(batch)
        return Matches(batch["mkpts0_f"].detach().cpu().numpy(),
                       batch["mkpts1_f"].detach().cpu().numpy(),
                       batch["mconf"].detach().cpu().numpy())


class MatchAnythingELoFTRBaseline(EfficientLoFTRBaseline):
    """The existing claim-independent localisation adapter with a new backend."""

    name = "matchanything_eloftr_baseline"
