import pytest

from pyrafault.io.slide import SlideMeta, scan_status


def meta(mpp, ds=(1.0, 4.0, 16.0, 32.0)):
    return SlideMeta(
        slide_id="S", path="p", width_l0=1000, height_l0=1000,
        mpp_x=mpp, mpp_y=mpp, objective_power=40.0, vendor="aperio",
        level_count=len(ds),
        level_dimensions=tuple((1000, 1000) for _ in ds), level_downsamples=ds,
    )


def test_pilot_slides_are_ok():
    assert scan_status(meta(0.2520), True, 0.5) == ("ok", None)
    assert scan_status(meta(0.2527), True, 0.5) == ("ok", None)


def test_missing_mpp_is_excluded_never_assumed_40x():
    status, reason = scan_status(meta(None), True, 0.5)
    assert status == "excluded" and reason == "missing_mpp"


def test_missing_mpp_may_be_allowed_only_when_explicitly_configured():
    assert scan_status(meta(None), False, 0.5) == ("ok", None)


def test_coarse_slide_cannot_deliver_20x():
    status, reason = scan_status(meta(0.6), True, 0.5)
    assert status == "excluded" and reason == "insufficient_resolution"


def test_boundary_mpp_equal_to_limit_is_accepted():
    assert scan_status(meta(0.5), True, 0.5) == ("ok", None)
