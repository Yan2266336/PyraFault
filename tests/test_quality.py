import numpy as np
import pytest

from pyrafault.config import QualityCfg
from pyrafault.quality import blur_var, evaluate, mean_intensity, pen_mark_fraction

CFG = QualityCfg(
    blur_var_flag=20.0, blur_var_exclude=5.0,
    mean_intensity_flag=(25.0, 235.0), mean_intensity_exclude=(10.0, 250.0),
    pen_mark_flag_frac=0.02,
)


def _texture(seed=0, size=128):
    """An H&E-like patch: magenta tint, channels correlated, real texture.

    Independent per-channel noise is NOT tissue. Its hue is uniformly distributed, so
    ~47% of its pixels land in pen_mark_fraction's marker-ink hue band by construction
    and its mean saturation clears the s > 120 gate — a random-RGB patch is measured at
    up to 0.28 ink fraction. A stain moves all three channels together with density, so
    real tissue holds a fixed magenta hue (~155 in OpenCV units) well outside that band.
    """
    rng = np.random.default_rng(seed)
    base = rng.integers(-30, 30, (size, size, 1), dtype=np.int16)   # shared luminance
    tint = np.array([200, 140, 190], dtype=np.int16)                # H&E magenta
    return np.clip(base + tint, 0, 255).astype(np.uint8)


def _flat(value, size=128):
    return np.full((size, size, 3), value, np.uint8)


def test_blur_var_higher_for_texture_than_flat():
    assert blur_var(_texture()) > 100.0
    assert blur_var(_flat(128)) == pytest.approx(0.0, abs=1e-6)


def test_mean_intensity():
    assert mean_intensity(_flat(200)) == pytest.approx(200.0)


def test_textured_patch_is_clean():
    r = evaluate(_texture(), CFG)
    assert r.flags == [] and r.hard_fail is False


def test_flat_grey_patch_is_flagged_and_excluded_for_blur():
    r = evaluate(_flat(128), CFG)
    assert "blur" in r.flags and r.hard_fail is True


def test_near_white_patch_is_too_bright_and_excluded():
    r = evaluate(_flat(252), CFG)
    assert "too_bright" in r.flags and r.hard_fail is True


def test_mildly_bright_patch_is_flagged_but_kept():
    img = _texture()
    img = np.clip(img.astype(int) + 90, 0, 255).astype(np.uint8)    # mean ~240
    r = evaluate(img, CFG)
    assert "too_bright" in r.flags and r.hard_fail is False


def test_black_patch_is_too_dark_and_excluded():
    r = evaluate(_flat(5), CFG)
    assert "too_dark" in r.flags and r.hard_fail is True


def test_pen_mark_is_flagged_but_never_excluded():
    img = _texture()
    img[:40, :40] = (20, 30, 200)          # saturated blue ink, 9.8% of the patch
    assert pen_mark_fraction(img) > CFG.pen_mark_flag_frac
    r = evaluate(img, CFG)
    assert "pen_mark" in r.flags
    assert r.hard_fail is False, "artifacts are recorded, never repaired or dropped"
