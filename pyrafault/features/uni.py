# pyrafault/features/uni.py
"""MahmoodLab/UNI — a pathology foundation model (ViT-L/16, 1024-d)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from PIL import Image

from pyrafault.features.registry import ExtractorError, resolve_local_revision


def assert_timm_model_matches_hf_repo(timm_model: str, hf_repo: str) -> None:
    """I5: `timm_model` is what actually loads the weights; `hf_repo` is what the
    revision gets resolved from and recorded under. They must name the same model or
    the manifest's provenance describes a different model than the one that produced
    the embeddings -- e.g. `timm_model: vit_large_patch16_224` (an ImageNet checkpoint)
    paired with `hf_repo: MahmoodLab/UNI` would silently record UNI's revision against
    ImageNet weights. Non-`hf-hub:` timm_model values (a bare timm architecture name)
    carry no repo reference to check, so they pass through unchecked."""
    if timm_model.startswith("hf-hub:"):
        referenced = timm_model[len("hf-hub:"):]
        if referenced != hf_repo:
            raise ExtractorError(
                f"features.uni.timm_model={timm_model!r} references {referenced!r}, "
                f"which disagrees with features.uni.hf_repo={hf_repo!r}; refusing to "
                "record a revision for a model that is not the one actually loaded"
            )


class UNIExtractor:
    name = "uni"

    def __init__(self, cfg: dict, device: str, batch_size: int) -> None:
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        self.input_size = int(cfg.get("input_size", 224))
        self.batch_size = batch_size
        self.device = device
        self.model = timm.create_model(
            cfg["timm_model"], pretrained=True, init_values=1e-5, dynamic_img_size=True
        ).eval().to(device)
        self.transform = create_transform(
            **resolve_data_config(self.model.pretrained_cfg, model=self.model)
        )
        # `input_size` is recorded in the manifest as a claim about what the model
        # actually does to its input — so verify it against the transform's real
        # output rather than trusting the config unchecked. A probe image with
        # unequal, non-input_size dimensions catches both "config disagrees with
        # reality" and "transform doesn't resize/crop at all".
        probe = self.transform(Image.new("RGB", (self.input_size * 3, self.input_size * 2)))
        actual = tuple(probe.shape[-2:])
        if actual != (self.input_size, self.input_size):
            raise ExtractorError(
                f"configured features.uni.input_size={self.input_size} does not match "
                f"what the loaded transform actually produces ({actual[0]}x{actual[1]}); "
                "fix the config to describe reality rather than record a stale claim"
            )
        self.dim = int(self.model.num_features)

        self.hf_repo = cfg["hf_repo"]
        self.timm_model = cfg["timm_model"]
        assert_timm_model_matches_hf_repo(self.timm_model, self.hf_repo)
        self.revision = resolve_local_revision(self.hf_repo)
        self.preprocessing = str(self.transform)

    @torch.inference_mode()
    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        out = []
        for i in range(0, len(images), self.batch_size):
            batch = torch.stack([
                self.transform(im.convert("RGB")) for im in images[i:i + self.batch_size]
            ]).to(self.device)
            feats = self.model(batch).float().cpu().numpy()
            out.append(feats)
        arr = np.concatenate(out).astype(np.float32)
        return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-8)
