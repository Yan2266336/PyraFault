"""Level-0 pixel geometry. Every canonical bbox in PyraFault is a BBoxL0."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BBoxL0:
    """An axis-aligned box in WSI level-0 pixel coordinates."""

    x: int
    y: int
    w: int
    h: int

    @property
    def x1(self) -> int:
        return self.x + self.w

    @property
    def y1(self) -> int:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    def contains(self, other: "BBoxL0") -> bool:
        return (
            other.x >= self.x
            and other.y >= self.y
            and other.x1 <= self.x1
            and other.y1 <= self.y1
        )

    def within_bounds(self, width: int, height: int) -> bool:
        return self.x >= 0 and self.y >= 0 and self.x1 <= width and self.y1 <= height

    def as_dict(self, prefix: str) -> dict[str, int]:
        return {
            f"{prefix}_x": self.x,
            f"{prefix}_y": self.y,
            f"{prefix}_w": self.w,
            f"{prefix}_h": self.h,
        }


def fov_l0(output_size: int, target_mpp: float, mpp_l0: float) -> int:
    """Level-0 side length that yields `output_size` px at `target_mpp` µm/px."""
    return int(round(output_size * target_mpp / mpp_l0))


def grid_positions(width: int, height: int, fov: int, stride: int) -> list[BBoxL0]:
    """Fully in-bounds square positions on a regular grid over the level-0 canvas."""
    if fov <= 0 or stride <= 0 or fov > width or fov > height:
        return []
    return [
        BBoxL0(x, y, fov, fov)
        for y in range(0, height - fov + 1, stride)
        for x in range(0, width - fov + 1, stride)
    ]


def context_jitter(
    seed: int, context_id: str, remainder_x: int, remainder_y: int
) -> tuple[int, int]:
    """Deterministic, per-axis-independent grid-origin offset for `subgrid`.

    `(jx, jy)` is derived from `(seed, context_id)` via SHA-256 -- never Python's
    `hash()`, which is randomised per process and would make the benchmark
    non-reproducible across runs. The two axes are hashed with different domain tags
    ("x" vs "y") and therefore drawn independently: sharing one offset across both axes
    would leave `(claimed.x - ctx.x) mod fov == (claimed.y - ctx.y) mod fov` true for
    every on-grid point and false for drifted ones, which is the same shortcut in a
    different shape (spec 15.1).
    """
    if remainder_x < 0 or remainder_y < 0:
        raise ValueError(f"remainder must be >= 0, got ({remainder_x}, {remainder_y})")

    def _axis(tag: str, remainder: int) -> int:
        digest = hashlib.sha256(f"{seed}|{context_id}|{tag}".encode()).digest()
        return int.from_bytes(digest[:8], "big") % (remainder + 1)

    return _axis("x", remainder_x), _axis("y", remainder_y)


def _grid_size_and_remainder(width: int, height: int, fov: int) -> tuple[int, int, int]:
    """The number of grid cells per side and the per-axis leftover space.

    `n` is `width // fov` (or `height // fov` if smaller), decremented while the
    leftover space on either axis is smaller than `fov` itself. This guarantees the
    leftover always spans a full period of `fov`, so a grid origin can land on any
    residue class mod `fov` on either axis -- without the decrement, a small leftover
    (e.g. 1 px) would make most residues unreachable no matter the offset.
    """
    n = min(width // fov, height // fov)
    while n > 0 and min(width, height) - n * fov < fov:
        n -= 1
    return n, width - n * fov, height - n * fov


def centred_inset(width: int, height: int, fov: int) -> tuple[int, int]:
    """The grid-origin offset that centres a `subgrid` lattice inside a `width` x
    `height` parent -- `subgrid`'s origin when no jitter is supplied (`remainder // 2`
    on each axis).

    Exposed on its own because the §13.5 leakage audit needs exactly this "no jitter"
    assumption: it is what a verifier without knowledge of the per-context jitter
    would guess the grid origin to be, from `ℓ` and `C` alone (spec §15.1).
    """
    _, rem_x, rem_y = _grid_size_and_remainder(width, height, fov)
    return rem_x // 2, rem_y // 2


def subgrid(
    parent: BBoxL0,
    fov: int,
    margin_rings: int = 0,
    seed: int | None = None,
    context_id: str | None = None,
) -> list[BBoxL0]:
    """Non-overlapping square grid inside `parent`, minus `margin_rings` outer rings.

    The number of cells per side is `parent.w // fov` (or `parent.h // fov` if smaller),
    but decremented while the leftover space on either axis is smaller than `fov` itself.
    This guarantees the leftover always spans a full period of `fov`, so the grid origin
    can land on any residue class mod `fov` on either axis -- without the decrement, a
    small leftover (e.g. 1 px) would make most residues unreachable no matter the offset.

    With `seed` and `context_id` given, the origin is `context_jitter(seed, context_id,
    ...)`: a deterministic offset that differs per context and is independent per axis, so
    the grid cannot be reconstructed from the parent bbox alone (spec 15.1). Without them
    the grid is centred (`centred_inset`) -- which is what every caller that has no
    context identity to key a jitter off of (including these tests) should keep using.
    """
    if fov <= 0 or margin_rings < 0:
        return []
    n, rem_x, rem_y = _grid_size_and_remainder(parent.w, parent.h, fov)
    lo, hi = margin_rings, n - margin_rings
    if hi <= lo:
        return []
    if seed is None:
        off_x, off_y = rem_x // 2, rem_y // 2
    else:
        if context_id is None:
            raise ValueError("context_id is required when seed is given")
        off_x, off_y = context_jitter(seed, context_id, rem_x, rem_y)
    return [
        BBoxL0(parent.x + off_x + i * fov, parent.y + off_y + j * fov, fov, fov)
        for j in range(lo, hi)
        for i in range(lo, hi)
    ]


def iou(a: BBoxL0, b: BBoxL0) -> float:
    ix = min(a.x1, b.x1) - max(a.x, b.x)
    iy = min(a.y1, b.y1) - max(a.y, b.y)
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    union = a.w * a.h + b.w * b.h - inter
    return inter / union


def center_distance(a: BBoxL0, b: BBoxL0) -> float:
    return math.hypot(a.cx - b.cx, a.cy - b.cy)
