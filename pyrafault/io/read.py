"""The single place where WSI pixels enter PyraFault.

Every context and query image in the benchmark is produced here. Because the
benchmark stores coordinates rather than pixels, this function's behaviour *is*
the benchmark's ground truth, and any change to it requires bumping
`extraction_version`.
"""
from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from pyrafault.geom import BBoxL0
from pyrafault.io.slide import MissingMPPError, SlideError, UpsamplingError, select_level

RESAMPLE = Image.LANCZOS


class OutOfBoundsError(SlideError):
    """The requested bbox is not fully inside the level-0 canvas."""


class CorruptReadError(SlideError):
    """OpenSlide returned pixels too transparent to be defined data."""


#: The lowest alpha a read may contain and still be trusted.
#:
#: Not 255. Requiring full opacity threw away 696 of 2,920 slides in the five-cohort
#: freeze -- 61% of UCEC -- and the failure rate tracked `mpp_x` rather than the tissue:
#: 212 of 216 slides at mpp 0.248 failed, 0 of 382 at mpp 0.2525. A rate that follows the
#: scanner is a property of the encoder, not of the slide, and the surviving slides were
#: therefore biased by site.
#:
#: Measured directly, on the 38 boxes that failed this check during an agent run over
#: slides that had already passed QC:
#:
#: | alpha | boxes | shape | under those pixels |
#: |---|---|---|---|
#: | 254 | 35 | scattered, ~0.0015% of pixels | RGB equal to the whole-image mean |
#: | 253 | 1 | scattered | same |
#: | 240 | 1 | one contiguous edge row | near-white, 243/243/243 |
#: | 118 | 1 | one contiguous edge row | near-white |
#:
#: So two unlike things were being refused by one rule. A handful of alpha-254 pixels
#: whose colour matches the surrounding tissue is JPEG/colour-conversion rounding; a
#: solid row at alpha 118 is the edge of the scanned area, where OpenSlide's premultiplied
#: RGB really is wrong. 250 separates them exactly, which is why no area rule is needed
#: on top of it.
#:
#: `extraction_version` is deliberately NOT bumped. The version exists so that two images
#: produced under one version cannot differ, and this change cannot make them differ: a
#: read that passed before still contains no pixel under 255 and returns identical bytes.
#: Only the set of readable boxes grows. Bumping would invalidate every cached image in
#: `02_cache` to record a widening that changed none of them.
MIN_ALPHA = 250


def _to_rgb(img: Image.Image, meta, bbox_l0: BBoxL0) -> Image.Image:
    """Drop the alpha channel, refusing a read that is too transparent to trust.

    One function rather than the same six lines at each call site: `faults.FAULT_TYPES`
    already carries a comment about three hand-copied lists coming apart silently, and a
    threshold that drifted between the context path and the query path would be worse
    than either value, because the two would disagree about which slides exist.
    """
    if img.mode == "RGBA":
        lo, _ = img.getchannel("A").getextrema()
        if lo < MIN_ALPHA:
            raise CorruptReadError(
                f"{meta.slide_id}: bbox {bbox_l0} returned pixels at alpha {lo}, below "
                f"the {MIN_ALPHA} this reader trusts; treat as out-of-bounds or corrupt"
            )
        return img.convert("RGB")
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


@dataclass(frozen=True)
class ExtractionRecord:
    native_level: int
    native_downsample: float
    effective_mpp: float
    target_mpp: float
    output_size: int
    resize_required: bool
    extraction_version: str


def read_at_mpp(
    slide,
    bbox_l0: BBoxL0,
    target_mpp: float,
    output_size: int,
    extraction_version: str,
) -> tuple[Image.Image, ExtractionRecord]:
    """Read `bbox_l0` and return it as `output_size x output_size` RGB at `target_mpp`.

    Deterministic by construction: the level choice, the resample filter and the bounds
    policy are all fixed, so the same arguments always yield the same bytes.
    """
    meta = slide.meta

    if meta.mpp_x is None:
        raise MissingMPPError(f"{meta.slide_id}: no level-0 MPP; refusing to assume a magnification")

    if not bbox_l0.within_bounds(meta.width_l0, meta.height_l0):
        raise OutOfBoundsError(
            f"{meta.slide_id}: bbox {bbox_l0} escapes the level-0 canvas "
            f"{meta.width_l0}x{meta.height_l0}"
        )

    need_ds = target_mpp / meta.mpp_x
    level = select_level(meta.level_downsamples, need_ds)
    ds = meta.level_downsamples[level]

    read_w = int(round(bbox_l0.w / ds))
    read_h = int(round(bbox_l0.h / ds))
    if read_w < output_size or read_h < output_size:
        raise UpsamplingError(
            f"{meta.slide_id}: bbox {bbox_l0} yields {read_w}x{read_h} px at level {level}, "
            f"which is smaller than the requested output {output_size}x{output_size}"
        )

    img = slide.read_region((bbox_l0.x, bbox_l0.y), level, (read_w, read_h))

    img = _to_rgb(img, meta, bbox_l0)

    resize_required = (read_w, read_h) != (output_size, output_size)
    if resize_required:
        img = img.resize((output_size, output_size), RESAMPLE)

    return img, ExtractionRecord(
        native_level=level,
        native_downsample=float(ds),
        effective_mpp=meta.mpp_x * bbox_l0.w / output_size,
        target_mpp=target_mpp,
        output_size=output_size,
        resize_required=resize_required,
        extraction_version=extraction_version,
    )


def read_region_at_mpp(
    slide,
    bbox_l0: BBoxL0,
    target_mpp: float,
    output_w: int,
    output_h: int,
    extraction_version: str,
) -> tuple[Image.Image, ExtractionRecord]:
    """Non-square sibling of read_at_mpp, for DERIVED artifacts only.

    Same level selection and the same refusal to upsample, but an explicit (w, h)
    instead of one side, so a whole-slide thumbnail is resampled once at its true
    aspect ratio rather than squashed square and stretched back.

    Never use this for benchmark pixels. Every context and query is square and must
    go through read_at_mpp, so that provenance has exactly one code path.
    """
    meta = slide.meta

    if meta.mpp_x is None:
        raise MissingMPPError(f"{meta.slide_id}: no level-0 MPP; refusing to assume a magnification")

    if not bbox_l0.within_bounds(meta.width_l0, meta.height_l0):
        raise OutOfBoundsError(
            f"{meta.slide_id}: bbox {bbox_l0} escapes the level-0 canvas "
            f"{meta.width_l0}x{meta.height_l0}"
        )

    need_ds = target_mpp / meta.mpp_x
    level = select_level(meta.level_downsamples, need_ds)
    ds = meta.level_downsamples[level]

    read_w = int(round(bbox_l0.w / ds))
    read_h = int(round(bbox_l0.h / ds))
    if read_w < output_w or read_h < output_h:
        raise UpsamplingError(
            f"{meta.slide_id}: bbox {bbox_l0} yields {read_w}x{read_h} px at level {level}, "
            f"which is smaller than the requested output {output_w}x{output_h}"
        )

    img = slide.read_region((bbox_l0.x, bbox_l0.y), level, (read_w, read_h))

    img = _to_rgb(img, meta, bbox_l0)

    resize_required = (read_w, read_h) != (output_w, output_h)
    if resize_required:
        img = img.resize((output_w, output_h), RESAMPLE)

    return img, ExtractionRecord(
        native_level=level,
        native_downsample=float(ds),
        effective_mpp=meta.mpp_x * bbox_l0.w / output_w,
        target_mpp=target_mpp,
        output_size=max(output_w, output_h),
        resize_required=resize_required,
        extraction_version=extraction_version,
    )
