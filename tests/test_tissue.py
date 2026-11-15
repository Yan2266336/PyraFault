import warnings

import numpy as np
import pytest
from skimage.morphology import binary_closing, binary_opening, disk, remove_small_objects

from pyrafault.geom import BBoxL0
from pyrafault.tissue import TissueMask, build_mask


class TissueCfg:
    method = "hsv_saturation_otsu"
    sat_floor = 0.08
    gray_max = 235
    close_radius_um = 64
    open_radius_um = 32
    min_component_area_um2 = 250000


def _synthetic_slide(h=400, w=600):
    """White background with one saturated magenta tissue blob."""
    img = np.full((h, w, 3), 245, np.uint8)
    img[100:300, 150:450] = (180, 60, 150)
    return img


def test_mask_finds_the_blob_and_rejects_background():
    m = build_mask(_synthetic_slide(), mpp_thumb=8.0, cfg=TissueCfg)
    assert m.dtype == bool and m.shape == (400, 600)
    assert m[200, 300]                      # inside the blob
    assert not m[10, 10]                    # background corner
    assert 0.15 < m.mean() < 0.35


def test_small_specks_are_removed():
    img = _synthetic_slide()
    img[5:8, 5:8] = (180, 60, 150)          # 3x3 speck, far below min area
    m = build_mask(img, mpp_thumb=8.0, cfg=TissueCfg)
    assert not m[6, 6]


def test_all_white_slide_yields_empty_mask():
    m = build_mask(np.full((200, 200, 3), 250, np.uint8), mpp_thumb=8.0, cfg=TissueCfg)
    assert m.sum() == 0, "sat_floor must stop Otsu from inventing tissue on a blank slide"


def test_fraction_matches_brute_force_on_random_windows():
    rng = np.random.default_rng(7)
    mask = rng.random((300, 400)) < 0.4
    tm = TissueMask(mask, width_l0=40000, height_l0=30000)
    sx, sy = 400 / 40000, 300 / 30000
    for _ in range(50):
        w = int(rng.integers(1000, 12000))
        x = int(rng.integers(0, 40000 - w))
        y = int(rng.integers(0, 30000 - w))
        x0, y0 = int(x * sx), int(y * sy)
        x1, y1 = min(400, int(round((x + w) * sx))), min(300, int(round((y + w) * sy)))
        expected = mask[y0:y1, x0:x1].mean()
        assert tm.fraction(BBoxL0(x, y, w, w)) == pytest.approx(expected, abs=1e-9)


def test_fraction_of_a_fully_covered_box_is_one():
    tm = TissueMask(np.ones((100, 100), bool), 10000, 10000)
    assert tm.fraction(BBoxL0(0, 0, 5000, 5000)) == pytest.approx(1.0)


def test_fraction_of_an_empty_region_is_zero():
    mask = np.zeros((100, 100), bool)
    mask[80:, 80:] = True
    tm = TissueMask(mask, 10000, 10000)
    assert tm.fraction(BBoxL0(0, 0, 4000, 4000)) == pytest.approx(0.0)


def test_build_mask_emits_no_deprecation_warning():
    """I10: closing/opening + remove_small_objects(max_size=...) must not warn."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        build_mask(_synthetic_slide(), mpp_thumb=8.0, cfg=TissueCfg)


@pytest.mark.parametrize("seed", range(5))
def test_morphology_migration_matches_the_deprecated_api_exactly(seed):
    """I10: closing/opening must reproduce binary_closing/binary_opening exactly, and
    remove_small_objects(max_size=min_px - 1) must reproduce remove_small_objects(
    min_size=min_px) exactly -- the two size parameters disagree at the boundary
    (max_size keeps size > threshold; min_size kept size >= threshold), so the
    replacement threshold must be off by one, not the same value."""
    rng = np.random.default_rng(seed)
    mask = rng.random((150, 150)) < 0.15
    # deliberately include features touching the border, where the differing `mode`
    # defaults of the two APIs could plausibly (but empirically do not) disagree.
    mask[0:8, 0:8] = True
    mask[-6:, -6:] = True
    min_px = 40

    close_px, open_px = 5, 3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old = binary_closing(mask, disk(close_px))
        old = binary_opening(old, disk(open_px))
        old = remove_small_objects(old, min_size=min_px)

    from pyrafault.tissue import closing, opening
    new = closing(mask, disk(close_px))
    new = opening(new, disk(open_px))
    new = remove_small_objects(new, max_size=min_px - 1)

    assert np.array_equal(old, new)


def test_save_load_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    mask = rng.random((50, 60)) < 0.5
    tm = TissueMask(mask, 6000, 5000)
    p = tmp_path / "m.npz"
    tm.save(p)
    back = TissueMask.load(p)
    assert np.array_equal(back.mask, mask)
    assert (back.width_l0, back.height_l0) == (6000, 5000)
    assert back.fraction(BBoxL0(0, 0, 3000, 2500)) == pytest.approx(
        tm.fraction(BBoxL0(0, 0, 3000, 2500))
    )
