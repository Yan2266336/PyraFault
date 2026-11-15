import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from pyrafault.cache import cache_key, cache_path, image_sha256, load_png, save_png
from pyrafault.geom import BBoxL0

BOX = BBoxL0(1000, 2000, 16254, 16254)


def test_key_is_hex64_and_deterministic():
    a = cache_key("S1", BOX, 2.0, 2048, "v1")
    b = cache_key("S1", BOX, 2.0, 2048, "v1")
    assert a == b and len(a) == 64 and all(c in "0123456789abcdef" for c in a)


def test_key_changes_with_every_component():
    base = cache_key("S1", BOX, 2.0, 2048, "v1")
    assert cache_key("S2", BOX, 2.0, 2048, "v1") != base
    assert cache_key("S1", BBoxL0(1001, 2000, 16254, 16254), 2.0, 2048, "v1") != base
    assert cache_key("S1", BOX, 2.0, 1024, "v1") != base
    assert cache_key("S1", BOX, 1.0, 2048, "v1") != base
    assert cache_key("S1", BOX, 2.0, 2048, "v2") != base, "extraction_version must invalidate"


def test_key_is_stable_across_processes():
    code = (
        "from pyrafault.cache import cache_key;"
        "from pyrafault.geom import BBoxL0;"
        "print(cache_key('S1', BBoxL0(1000,2000,16254,16254), 2.0, 2048, 'v1'))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == cache_key("S1", BOX, 2.0, 2048, "v1")


def test_cache_path_is_sharded():
    key = cache_key("S1", BOX, 2.0, 2048, "v1")
    p = cache_path("/tmp/cache", "S1", key)
    assert p.as_posix() == f"/tmp/cache/S1/{key[:2]}/{key}.png"


def test_image_sha256_tracks_pixels_not_encoding():
    a = Image.fromarray(np.zeros((8, 8, 3), np.uint8))
    b = Image.fromarray(np.zeros((8, 8, 3), np.uint8))
    c = Image.fromarray(np.ones((8, 8, 3), np.uint8))
    assert image_sha256(a) == image_sha256(b)
    assert image_sha256(a) != image_sha256(c)
    assert len(image_sha256(a)) == 64


def test_png_round_trip_preserves_pixels(tmp_path):
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    p = tmp_path / "a" / "b" / "x.png"
    save_png(img, p)
    assert p.exists()
    assert image_sha256(load_png(p)) == image_sha256(img)


def test_save_png_leaves_no_temp_file(tmp_path):
    img = Image.fromarray(np.zeros((4, 4, 3), np.uint8))
    p = tmp_path / "x.png"
    save_png(img, p)
    assert [f.name for f in tmp_path.iterdir()] == ["x.png"]


BOX_DEGRADE = BBoxL0(1000, 2000, 512, 512)


def test_two_bands_of_one_box_do_not_share_a_cache_entry():
    """The bug: the mild and severe renderings of one true box differ only in their
    degradation, so under the old key the second item is served the first's pixels."""
    mild = cache_key("S1", BOX_DEGRADE, 0.5, 512, "v2", "aaaaaaaaaaaaaaaa")
    severe = cache_key("S1", BOX_DEGRADE, 0.5, 512, "v2", "bbbbbbbbbbbbbbbb")

    assert mild != severe


def test_an_undegraded_key_is_unchanged_by_the_new_argument():
    """Every entry already on disk was written without it; they must stay reachable."""
    assert cache_key("S1", BOX_DEGRADE, 0.5, 512, "v1") == cache_key("S1", BOX_DEGRADE, 0.5, 512, "v1", "")


def test_concurrent_writers_of_one_key_do_not_corrupt_it(tmp_path):
    """Two processes materialising the same patch must not interleave into one temp file.

    The cache is content-addressed, so a sharded materialisation legitimately has two
    shards write the same key at the same time -- a cross-slide pair's patch belongs to
    both shards' work. A fixed temp name makes that an interleaved write followed by two
    renames, and the survivor can be a mixture of both. The failure is silent: the file
    exists, its size looks plausible, and it decodes to garbage or raises much later.
    """
    import multiprocessing as mp

    target = tmp_path / "patch.png"

    def write(seed):
        rng = np.random.default_rng(seed)
        img = Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8))
        for _ in range(12):
            save_png(img, target)

    procs = [mp.Process(target=write, args=(s,)) for s in (1, 2, 3, 4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)

    assert all(p.exitcode == 0 for p in procs)
    # The survivor must be one writer's image in full, not a splice of several.
    img = load_png(target)
    assert img.size == (256, 256)
    assert not list(tmp_path.glob("*.tmp")), "a temp file was left behind"
