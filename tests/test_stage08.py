"""Materialized assets must be reproducible from coordinates alone."""
import numpy as np
import pytest
from PIL import Image

from pyrafault.cache import cache_key, cache_path, image_sha256, load_png, save_png
from pyrafault.geom import BBoxL0


def test_materialized_filename_is_its_cache_key(tmp_path):
    key = cache_key("S1", BBoxL0(0, 0, 1016, 1016), 0.5, 512, "v1")
    p = cache_path(tmp_path, "S1", key)
    save_png(Image.fromarray(np.zeros((512, 512, 3), np.uint8)), p)
    assert p.stem == key
    assert p.parent.name == key[:2]


def test_hash_survives_a_save_load_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
    p = tmp_path / "x.png"
    save_png(img, p)
    assert image_sha256(load_png(p)) == image_sha256(img)


def test_a_different_extraction_version_writes_to_a_different_path(tmp_path):
    box = BBoxL0(0, 0, 1016, 1016)
    k1 = cache_key("S1", box, 0.5, 512, "v1")
    k2 = cache_key("S1", box, 0.5, 512, "v2")
    assert cache_path(tmp_path, "S1", k1) != cache_path(tmp_path, "S1", k2)
