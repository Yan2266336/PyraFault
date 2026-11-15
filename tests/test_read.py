import hashlib

import numpy as np
import pytest
from PIL import Image

from pyrafault.geom import BBoxL0
from pyrafault.io.read import (
    read_at_mpp, read_region_at_mpp, ExtractionRecord, MIN_ALPHA, OutOfBoundsError,
    CorruptReadError,
)
from pyrafault.io.slide import SlideMeta, UpsamplingError, MissingMPPError

EV = "v1"


class FakeSlide:
    """A synthetic pyramid whose pixel value encodes its level-0 position."""

    def __init__(self, width=40000, height=30000, mpp=0.2520,
                 downsamples=(1.0, 4.0, 16.0, 32.0), alpha=255):
        self.alpha = alpha
        self.meta = SlideMeta(
            slide_id="FAKE", path="/dev/null", width_l0=width, height_l0=height,
            mpp_x=mpp, mpp_y=mpp, objective_power=40.0, vendor="fake",
            level_count=len(downsamples),
            level_dimensions=tuple((int(width / d), int(height / d)) for d in downsamples),
            level_downsamples=tuple(downsamples),
        )
        self.calls = []

    def read_region(self, location, level, size):
        self.calls.append((location, level, size))
        w, h = size
        ds = self.meta.level_downsamples[level]
        xs = (np.arange(w) * ds + location[0]) % 251
        ys = (np.arange(h) * ds + location[1]) % 241
        r = np.broadcast_to(xs.astype(np.uint8), (h, w))
        g = np.broadcast_to(ys.astype(np.uint8)[:, None], (h, w))
        b = np.full((h, w), level * 40, np.uint8)
        a = np.full((h, w), self.alpha, np.uint8)
        return Image.fromarray(np.dstack([r, g, b, a]), mode="RGBA")


def test_context_read_uses_level1_and_resizes():
    s = FakeSlide()
    img, rec = read_at_mpp(s, BBoxL0(0, 0, 16254, 16254), 2.0, 2048, EV)
    assert img.size == (2048, 2048) and img.mode == "RGB"
    assert rec == ExtractionRecord(
        native_level=1, native_downsample=4.0,
        effective_mpp=pytest.approx(16254 * 0.2520 / 2048),
        target_mpp=2.0, output_size=2048, resize_required=True, extraction_version="v1",
    )
    (loc, level, size) = s.calls[0]
    assert loc == (0, 0) and level == 1 and size == (4064, 4064)


def test_query_read_uses_level0_and_resizes():
    s = FakeSlide()
    img, rec = read_at_mpp(s, BBoxL0(5000, 6000, 1016, 1016), 0.5, 512, EV)
    assert img.size == (512, 512)
    assert rec.native_level == 0 and rec.resize_required is True
    (loc, level, size) = s.calls[0]
    assert loc == (5000, 6000) and level == 0 and size == (1016, 1016)


def test_location_is_always_level0_never_level_scaled():
    s = FakeSlide()
    read_at_mpp(s, BBoxL0(20000, 10000, 16254, 16254), 2.0, 2048, EV)
    loc, level, _ = s.calls[0]
    assert loc == (20000, 10000)      # NOT (5000, 2500)
    assert level == 1


def test_no_resize_when_read_size_already_matches():
    s = FakeSlide(downsamples=(1.0, 2.0))
    img, rec = read_at_mpp(s, BBoxL0(0, 0, 1024, 1024), 0.504, 512, EV)
    assert rec.resize_required is False and img.size == (512, 512)


def test_out_of_bounds_raises_and_does_not_read():
    s = FakeSlide(width=1000, height=1000)
    for bad in [BBoxL0(-1, 0, 10, 10), BBoxL0(0, -1, 10, 10),
                BBoxL0(995, 0, 10, 10), BBoxL0(0, 995, 10, 10)]:
        with pytest.raises(OutOfBoundsError):
            read_at_mpp(s, bad, 0.252, 10, EV)
    assert s.calls == []


@pytest.mark.parametrize("alpha", [249, 240, 118, 0])
def test_a_read_below_the_alpha_floor_is_a_corrupt_read(alpha):
    s = FakeSlide(alpha=alpha)
    with pytest.raises(CorruptReadError, match=f"alpha {alpha}"):
        read_at_mpp(s, BBoxL0(0, 0, 1016, 1016), 0.5, 512, EV)


@pytest.mark.parametrize("alpha", [255, 254, 253, 250])
def test_near_opaque_is_read_not_refused(alpha):
    """Requiring alpha == 255 exactly cost 696 of 2,920 slides, 61% of UCEC, and the
    failure rate followed `mpp_x` -- the encoder -- not the tissue. Measured on the 38
    boxes that failed it: 36 were a few scattered alpha-254 pixels whose colour equalled
    the whole-image mean, i.e. rounding; the 2 genuine ones sat at alpha 240 and 118, a
    solid row at the edge of the scanned area. `MIN_ALPHA` is set between them.
    """
    s = FakeSlide(alpha=alpha)
    img, _ = read_at_mpp(s, BBoxL0(0, 0, 1016, 1016), 0.5, 512, EV)
    assert img.mode == "RGB" and img.size == (512, 512)


def test_the_floor_sits_between_what_was_measured():
    """Pinned as a number, not left implicit in two call sites: the context path and the
    query path read the same slides, and a threshold that drifted between them would make
    them disagree about which slides exist."""
    assert MIN_ALPHA == 250


def test_relaxing_the_floor_changed_no_image_that_already_read():
    """Why `extraction_version` is not bumped. The version exists so two images made
    under one version cannot differ; a fully opaque read is unaffected by the floor, so
    every cached image stays valid and only the set of readable boxes grows.
    """
    a, rec_a = read_at_mpp(FakeSlide(alpha=255), BBoxL0(0, 0, 1016, 1016), 0.5, 512, EV)
    assert rec_a.extraction_version == EV
    assert a.tobytes() == read_at_mpp(
        FakeSlide(alpha=255), BBoxL0(0, 0, 1016, 1016), 0.5, 512, EV)[0].tobytes()


def test_missing_mpp_raises():
    s = FakeSlide(mpp=None)
    with pytest.raises(MissingMPPError):
        read_at_mpp(s, BBoxL0(0, 0, 1016, 1016), 0.5, 512, EV)


def test_upsampling_is_refused():
    s = FakeSlide(mpp=0.5)
    with pytest.raises(UpsamplingError):
        read_at_mpp(s, BBoxL0(0, 0, 1016, 1016), 0.25, 512, EV)


def test_output_smaller_than_read_is_required():
    """A bbox that would have to be enlarged to fill output_size is rejected."""
    s = FakeSlide()
    with pytest.raises(UpsamplingError):
        read_at_mpp(s, BBoxL0(0, 0, 1016, 1016), 0.5, 4096, EV)


def test_repeated_reads_are_byte_identical():
    s = FakeSlide()
    box = BBoxL0(3000, 4000, 16254, 16254)
    h = [hashlib.sha256(read_at_mpp(s, box, 2.0, 2048, EV)[0].tobytes()).hexdigest()
         for _ in range(3)]
    assert len(set(h)) == 1


def test_read_region_at_mpp_reads_non_square_region_at_right_size():
    """A non-square bbox resamples to an explicit (w, h), not a squash-then-stretch square."""
    s = FakeSlide()
    img, rec = read_region_at_mpp(s, BBoxL0(0, 0, 16254, 8127), 2.0, 2032, 1016, EV)
    assert img.size == (2032, 1016) and img.mode == "RGB"
    assert rec.native_level == 1 and rec.resize_required is True
    (loc, level, size) = s.calls[0]
    assert loc == (0, 0) and level == 1 and size == (4064, 2032)


def test_read_region_at_mpp_refuses_upsampling_when_output_w_too_large():
    """output_w exceeding the native read width is refused, even though output_h fits."""
    s = FakeSlide()
    with pytest.raises(UpsamplingError):
        read_region_at_mpp(s, BBoxL0(0, 0, 16254, 8127), 2.0, 5000, 1000, EV)
    assert s.calls == []


def test_read_region_at_mpp_refuses_upsampling_when_output_h_too_large():
    """output_h exceeding the native read height is refused, even though output_w fits.

    Checked independently of the width case: output_w here is well within bounds, so a
    guard that only inspects one axis would let this slip through.
    """
    s = FakeSlide()
    with pytest.raises(UpsamplingError):
        read_region_at_mpp(s, BBoxL0(0, 0, 16254, 8127), 2.0, 1000, 3000, EV)
    assert s.calls == []


@pytest.mark.slow
def test_real_slide_reconstruction_is_deterministic(pilot_slide_path):
    import subprocess, sys, json
    from pyrafault.io.slide import open_slide
    box = BBoxL0(8000, 8000, 16254, 16254)
    with open_slide(pilot_slide_path, "s") as s:
        img, rec = read_at_mpp(s, box, 2.0, 2048, EV)
    first = hashlib.sha256(img.tobytes()).hexdigest()
    assert rec.native_level == 1 and rec.resize_required is True

    code = (
        "import hashlib,sys;"
        "from pyrafault.geom import BBoxL0;"
        "from pyrafault.io.slide import open_slide;"
        "from pyrafault.io.read import read_at_mpp;"
        f"s=open_slide({pilot_slide_path!r},'s');"
        f"i,_=read_at_mpp(s,BBoxL0(8000,8000,16254,16254),2.0,2048,'v1');"
        "print(hashlib.sha256(i.tobytes()).hexdigest())"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == first
