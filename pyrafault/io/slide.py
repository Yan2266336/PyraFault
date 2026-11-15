"""OpenSlide wrapper. Reading pixels through this class is allowed; interpreting
them at a target magnification is not — that belongs to `pyrafault.io.read`."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import openslide
from PIL import Image


class SlideError(RuntimeError):
    """Base class for slide-level failures."""


class UpsamplingError(SlideError):
    """The requested resolution is finer than the slide can provide."""


class MissingMPPError(SlideError):
    """The slide does not declare a usable microns-per-pixel value."""


@dataclass(frozen=True)
class SlideMeta:
    slide_id: str
    path: str
    width_l0: int
    height_l0: int
    mpp_x: float | None
    mpp_y: float | None
    objective_power: float | None
    vendor: str | None
    level_count: int
    level_dimensions: tuple[tuple[int, int], ...]
    level_downsamples: tuple[float, ...]


def select_level(level_downsamples: Sequence[float], need_ds: float, tol: float = 1e-3) -> int:
    """Index of the coarsest native level that does not require upsampling.

    `need_ds` is `target_mpp / mpp_level0`. A level qualifies when its downsample is at
    most `need_ds` (within a relative tolerance, so an exactly-matching level is not
    rejected by float noise). Among qualifying levels the coarsest wins, because reading
    less data and downscaling once is both faster and — with a fixed resample filter —
    deterministic.
    """
    limit = need_ds * (1.0 + tol)
    eligible = [i for i, ds in enumerate(level_downsamples) if ds <= limit]
    if not eligible:
        raise UpsamplingError(
            f"no native level at or below downsample {need_ds:.4f}; "
            f"available: {tuple(level_downsamples)}"
        )
    return max(eligible, key=lambda i: level_downsamples[i])


def _as_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class SlideHandle:
    """Read-only handle on one WSI."""

    def __init__(self, path: str | Path, slide_id: str) -> None:
        self.path = str(path)
        self._os = openslide.OpenSlide(self.path)
        p = self._os.properties
        w, h = self._os.dimensions
        self.meta = SlideMeta(
            slide_id=slide_id,
            path=self.path,
            width_l0=int(w),
            height_l0=int(h),
            mpp_x=_as_float(p.get(openslide.PROPERTY_NAME_MPP_X)),
            mpp_y=_as_float(p.get(openslide.PROPERTY_NAME_MPP_Y)),
            objective_power=_as_float(p.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)),
            vendor=p.get(openslide.PROPERTY_NAME_VENDOR),
            level_count=int(self._os.level_count),
            level_dimensions=tuple(tuple(int(v) for v in d) for d in self._os.level_dimensions),
            level_downsamples=tuple(float(d) for d in self._os.level_downsamples),
        )

    def read_region(self, location: tuple[int, int], level: int, size: tuple[int, int]) -> Image.Image:
        """Raw OpenSlide read. `location` is in level-0 coordinates, `size` in level pixels."""
        return self._os.read_region(location, level, size)

    def close(self) -> None:
        self._os.close()

    def __enter__(self) -> "SlideHandle":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def open_slide(path: str | Path, slide_id: str) -> SlideHandle:
    return SlideHandle(path, slide_id)


def scan_status(
    meta: SlideMeta, require_mpp: bool, max_mpp_l0: float
) -> tuple[str, str | None]:
    """Decide whether a slide can enter the benchmark.

    Guideline §6.1: a slide whose MPP cannot be recovered is excluded rather than
    assumed to be 40x, because a wrong MPP silently corrupts every bbox derived from it.
    """
    if meta.mpp_x is None:
        return ("excluded", "missing_mpp") if require_mpp else ("ok", None)
    if meta.mpp_x > max_mpp_l0:
        return "excluded", "insufficient_resolution"
    return "ok", None
