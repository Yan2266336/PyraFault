# pyrafault/features/dinov2.py
"""facebook/dinov2-base — ungated general-purpose fallback (768-d CLS token)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from PIL import Image

from pyrafault.features.registry import ExtractorError, resolve_local_revision


class DinoV2Extractor:
    name = "dinov2"

    def __init__(self, cfg: dict, device: str, batch_size: int) -> None:
        from transformers import AutoImageProcessor, AutoModel

        repo = cfg["hf_repo"]
        self.input_size = int(cfg.get("input_size", 224))
        self.batch_size = batch_size
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(repo)
        self.model = AutoModel.from_pretrained(repo).eval().to(device)
        self.dim = int(self.model.config.hidden_size)
        # `input_size` is recorded in the manifest as a claim about what the processor
        # actually does — the processor resizes/crops using its own defaults (shortest
        # edge + centre-crop, independent of this config value), so verify the claim
        # against real output rather than trusting it unchecked.
        probe = self.processor(
            images=[Image.new("RGB", (self.input_size * 3, self.input_size * 2))],
            return_tensors="pt",
        )
        actual = tuple(probe["pixel_values"].shape[-2:])
        if actual != (self.input_size, self.input_size):
            raise ExtractorError(
                f"configured features.dinov2.input_size={self.input_size} does not match "
                f"what the loaded processor actually produces ({actual[0]}x{actual[1]}); "
                "fix the config to describe reality rather than record a stale claim"
            )
        self.hf_repo = repo
        self.timm_model = None   # dinov2 is loaded via transformers, not timm
        self.revision = resolve_local_revision(repo)
        # The real processor object, not a hand-built string from the config-declared
        # size: it carries resize/crop strategy and normalisation mean/std, so a
        # library-upgrade change to those defaults shows up in the manifest.
        self.preprocessing = str(self.processor)

    @torch.inference_mode()
    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        out = []
        for i in range(0, len(images), self.batch_size):
            batch = [im.convert("RGB") for im in images[i:i + self.batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            feats = self.model(**inputs).last_hidden_state[:, 0].float().cpu().numpy()
            out.append(feats)
        arr = np.concatenate(out).astype(np.float32)
        return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-8)
