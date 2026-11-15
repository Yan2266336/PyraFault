"""A slide's ICC profile must not make its own cache entry unreadable.

Pillow copies `info["icc_profile"]` into every PNG it writes, and then refuses to read
a PNG whose decompressed iCCP chunk exceeds `MAX_TEXT_CHUNK` (1 MB). Several TCGA lung
slides carry a profile well past that, so the cache was writable but not readable for
them -- and the ValueError surfaced hours into a verifier run, on whichever stage first
read one of those patches back.
"""
from __future__ import annotations

from PIL import Image

from pyrafault.cache import image_sha256, load_png, save_png

OVERSIZED = b"\x00" * (3 << 20)      # 3 MB uncompressed, a few KB on disk


def _image() -> Image.Image:
    img = Image.new("RGB", (16, 16))
    img.putdata([(i % 251, (i * 7) % 253, (i * 13) % 257) for i in range(16 * 16)])
    return img


def test_save_png_does_not_embed_the_source_profile(tmp_path):
    img = _image()
    img.info["icc_profile"] = OVERSIZED
    path = tmp_path / "patch.png"

    save_png(img, path)

    assert b"iCCP" not in path.read_bytes()


def test_a_cached_patch_round_trips_pixels_despite_the_profile(tmp_path):
    img = _image()
    img.info["icc_profile"] = OVERSIZED
    path = tmp_path / "patch.png"

    save_png(img, path)

    assert image_sha256(load_png(path)) == image_sha256(img)


def test_load_png_reads_an_entry_written_before_the_profile_was_stripped(tmp_path):
    """Entries already on disk carry the profile; re-reading them from the WSIs to get
    rid of it would cost hours, so they have to stay readable."""
    img = _image()
    img.info["icc_profile"] = OVERSIZED
    path = tmp_path / "legacy.png"
    img.save(path, format="PNG")          # what save_png used to do
    assert b"iCCP" in path.read_bytes()

    assert image_sha256(load_png(path)) == image_sha256(img)
