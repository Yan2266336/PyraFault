"""Frozen feature-extractor registry (guideline §8.2, §9.4).

The extractor drives diversity sampling and hard-negative mining, so which model produced
an embedding — and at which revision — is part of the benchmark's provenance, not an
implementation detail.
"""
from __future__ import annotations

import re
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np
from PIL import Image

KNOWN = ("uni", "dinov2", "handcrafted")

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class ExtractorError(RuntimeError):
    """The requested extractor could not be constructed."""


def resolve_local_revision(repo: str) -> str:
    """Resolve the commit sha of the local snapshot that was actually loaded.

    Deliberately NOT a fresh `model_info()` Hub call: that queries the Hub's current
    HEAD *after* the weights are already in memory, which is a TOCTOU gap (the recorded
    sha is not guaranteed to match the snapshot that was loaded) and can silently
    degrade to a placeholder like "unknown" on a transient network blip or offline node.

    `local_files_only=True` is load-bearing here, not optional: without it,
    `hf_hub_download`'s default `revision="main"` is a ref name, not a commit hash, so
    the local-cache shortcut is skipped and it makes a live HTTP call to resolve "main"
    to whatever commit is current *right now* — i.e. exactly the second, unpinned,
    post-load Hub query this function exists to eliminate, just routed through a
    different client method. With `local_files_only=True` the lookup can only succeed by
    reading the snapshot directory already on disk from the weight load above, and that
    directory is named after the exact commit sha of the files inside it — closing the
    gap between "what was loaded" and "what gets recorded" for real.

    Raises `ExtractorError` — never returns a placeholder — if the sha cannot be
    determined locally or does not look like a resolved commit hash.
    """
    import os
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(
            repo, "config.json", token=os.environ.get("HF_TOKEN"), local_files_only=True
        )
        sha = Path(path).parent.name
    except Exception as exc:
        raise ExtractorError(
            f"could not resolve a pinned revision for {repo} from the local cache: {exc}"
        ) from exc
    if not _SHA_RE.match(sha):
        raise ExtractorError(
            f"resolved revision for {repo} is not a 40-character commit sha: {sha!r} "
            "— refusing to record an unpinned or placeholder revision"
        )
    return sha


@runtime_checkable
class Extractor(Protocol):
    name: str
    revision: str
    dim: int
    input_size: int
    preprocessing: str

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray: ...


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_extractor(name: str, cfg, device: str | None = None) -> Extractor:
    if name not in KNOWN:
        raise ExtractorError(f"unknown extractor {name!r}; known: {KNOWN}")

    batch = int(getattr(cfg, "batch_size", 64))
    if name == "handcrafted":
        from pyrafault.features.handcrafted import HandcraftedExtractor
        return HandcraftedExtractor(batch_size=batch)

    dev = _resolve_device(device or getattr(cfg, "device", "auto"))

    if name == "dinov2":
        from pyrafault.features.dinov2 import DinoV2Extractor
        try:
            return DinoV2Extractor(cfg.dinov2, dev, batch)
        except Exception as exc:
            raise ExtractorError(f"could not load facebook/dinov2-base: {exc}") from exc

    from pyrafault.features.uni import UNIExtractor
    try:
        return UNIExtractor(cfg.uni, dev, batch)
    except ExtractorError:
        # Already actionable (e.g. a revision-resolution or input_size-mismatch
        # failure raised from inside UNIExtractor) — reporting it as "gated" would
        # send someone to re-accept a licence they already hold, chasing the wrong
        # problem. Let it surface unchanged.
        raise
    except Exception as exc:
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

        is_gated = isinstance(exc, GatedRepoError) or (
            isinstance(exc, HfHubHTTPError)
            and getattr(getattr(exc, "response", None), "status_code", None) == 403
        )
        if not is_gated:
            raise ExtractorError(f"could not load MahmoodLab/UNI: {exc}") from exc
        raise ExtractorError(
            "could not load MahmoodLab/UNI. It is a gated repository with `gated: auto`, "
            "so access is granted immediately once the licence is accepted at "
            "https://huggingface.co/MahmoodLab/UNI while signed in to the account whose "
            f"token is in $HF_TOKEN. Until then set visual_sampling.extractor to "
            f"'{getattr(cfg, 'fallback_extractor', 'dinov2')}'. Underlying error: {exc}"
        ) from exc


def describe(extractor: Extractor) -> dict[str, Any]:
    # I5: record both hf_repo (what the revision was resolved from) and timm_model
    # (what actually loaded, for timm-based extractors) so a future disagreement
    # between the two is visible in the frozen manifest rather than only in code.
    return {
        "name": extractor.name,
        "revision": extractor.revision,
        "dim": int(extractor.dim),
        "input_size": int(extractor.input_size),
        "preprocessing": extractor.preprocessing,
        "hf_repo": getattr(extractor, "hf_repo", None),
        "timm_model": getattr(extractor, "timm_model", None),
    }
