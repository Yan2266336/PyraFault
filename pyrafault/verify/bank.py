"""Per-context window bank: the retrieval grid and its cached descriptors.

Windows are cropped from the already materialised context image, so building a bank
needs no WSI access. One bank serves every query in its context -- there are 674
contexts across the three freezes against 3,370 queries -- and it serves both the
top-K retrieval and the decoy null, which must share a scoring function for the tail
probability to mean anything.
"""
from __future__ import annotations

import re

import json

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np


def window_side(cfg) -> int:
    """The pixels a query occupies in the context image.

    Derived, not configured: it is what makes a window directly comparable to `Q`
    after `Q` is downsampled to the context's physical resolution.
    """
    return round(cfg.query.size_px * cfg.query.target_mpp / cfg.context.target_mpp)


def window_origins(context_px: int, side: int, stride: int | None = None) -> list[tuple[int, int]]:
    """Top-left corners of a tiling of `context_px`² by `side`² windows.

    `stride` defaults to half the window. That is not a tuning choice: a true position
    a full window from the nearest grid point produces as many matches as a random
    window does, so the grid must guarantee the truth falls within half a window.
    """
    if stride is None:
        stride = side // 2
    if stride <= 0 or side <= 0 or side > context_px:
        raise ValueError(f"bad geometry: context_px={context_px} side={side} stride={stride}")
    coords = list(range(0, context_px - side + 1, stride))
    return [(x, y) for y in coords for x in coords]


def bank_key(slide_id: str, context_bbox) -> tuple:
    """The runtime index for a bank.

    Not `context_id`: a verifier never receives one -- it is not in `INPUT_KEYS` -- while
    `slide_id` and `context_bbox` are. The slide is part of the key because a bbox alone
    is not unique across slides: the pilot freeze's 187 contexts occupy only 136 distinct
    boxes, so a bbox-only index would hand 27% of items another context's bank.
    """
    return (slide_id, context_bbox.x, context_bbox.y, context_bbox.w, context_bbox.h)


#: The file a bank directory carries to say which extractor built it. Leading
#: underscore so `*.npz` globs never see it.
BANK_MANIFEST = "_manifest.json"


def bank_dir(data_root: str | Path, extractor: str) -> Path:
    """Where the window banks built with `extractor` live.

    Keyed by extractor because a bank file records `context_id`, `side`, `origins` and
    `embeddings` -- and nothing that says which model produced the embeddings. One
    shared directory would mean an extractor ablation either overwrites the previous
    run's banks or retrieves against them, and the second one is silent: every candidate
    score becomes a dot product between this extractor's query and another's windows,
    which is a number rather than an error.

    The name is validated rather than interpolated blindly, so `--extractor ../..`
    cannot write outside the freeze.
    """
    if not extractor or not re.fullmatch(r"[A-Za-z0-9_.-]+", extractor):
        raise ValueError(
            f"extractor name {extractor!r} is not a safe directory component; "
            "expected letters, digits, underscore, dot or hyphen"
        )
    return Path(data_root) / "06_verify" / f"window_bank__{extractor}"


def write_bank_manifest(directory: str | Path, extractor: str, n_banks: int) -> None:
    """Record what built this directory. Rewritten on every build, so `n_banks` is the
    count as of the last one rather than a high-water mark."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / BANK_MANIFEST).write_text(
        json.dumps({"extractor": extractor, "n_banks": int(n_banks)},
                   sort_keys=True, indent=2) + "\n"
    )


def read_bank_manifest(directory: str | Path) -> dict | None:
    """The manifest, or None when the directory has never been written by a build.

    A corrupt manifest raises instead of reading as absent: "never built" and "built and
    then damaged" need different responses, and collapsing them would let a damaged
    freeze pass as one that simply predates the manifest.
    """
    path = Path(directory) / BANK_MANIFEST
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not readable JSON: {exc}") from exc


@dataclass(frozen=True)
class WindowBank:
    context_id: str
    side: int
    origins: list[tuple[int, int]]
    embeddings: np.ndarray            # (N, dim) float16

    def __post_init__(self) -> None:
        if len(self.origins) != len(self.embeddings):
            raise ValueError(
                f"{len(self.embeddings)} embeddings for {len(self.origins)} origins"
            )

    @cached_property
    def normalized_embeddings(self) -> np.ndarray:
        """`embeddings` as unit-norm float32 rows, computed once and cached.

        Retrieval (`ZoomTrace.localise`) used to rebuild this -- an `astype(float32)`
        plus a per-row `np.linalg.norm` over a (900-3844, 1024) matrix -- on every
        single query. Section 3.1 justifies the bank's existence as amortised cost;
        redoing this per query defeats it. `cached_property` works on a frozen
        dataclass because it writes `self.__dict__` directly rather than going through
        the generated `__setattr__` that raises on assignment.
        """
        emb = self.embeddings.astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / (norms + 1e-12)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, context_id=self.context_id, side=self.side,
            origins=np.asarray(self.origins, dtype=np.int32),
            embeddings=self.embeddings,
        )

    @classmethod
    def load(cls, path: str | Path) -> "WindowBank":
        z = np.load(path, allow_pickle=False)
        return cls(
            context_id=str(z["context_id"]), side=int(z["side"]),
            origins=[tuple(int(v) for v in row) for row in z["origins"]],
            embeddings=z["embeddings"],
        )
