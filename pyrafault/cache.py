"""Content-addressed cache for materialized C/Q images.

The key covers everything that can change the pixels, `extraction_version` included,
so bumping the version makes stale entries unreachable rather than silently reused.
"""
from __future__ import annotations

import errno
import hashlib
import os
import time
from pathlib import Path

from PIL import Image, PngImagePlugin

from pyrafault.geom import BBoxL0

# Some vendor slides carry an ICC profile of several megabytes. Pillow copies it into
# every PNG it writes, then refuses to read that PNG back: its zip-bomb guard caps a
# decompressed iCCP chunk at MAX_TEXT_CHUNK (1 MB) and raises ValueError above it. The
# cache is therefore writable but not readable for those slides -- and the failure
# surfaces hours later, in whatever stage first reads a cached patch from one.
#
# `save_png` stops writing the profile at all (below). This raises the read guard so
# that entries written before that fix still load: the alternative is re-reading them
# from the WSIs. The guard exists against hostile PNGs; these are files this pipeline
# wrote itself, into a directory it owns, and the bound stays finite.
PngImagePlugin.MAX_TEXT_CHUNK = 64 << 20


def cache_key(
    slide_id: str,
    bbox: BBoxL0,
    target_mpp: float,
    output_size: int,
    extraction_version: str,
    degradation_digest: str = "",
) -> str:
    """Everything that can change the pixels.

    `degradation_digest` is empty for an undegraded patch, and the payload then has the
    exact bytes it had before degradation existed -- so every entry already in a cache
    keeps its key. A non-empty value is what keeps two severities of the same true box
    from sharing an entry, where the second item would otherwise be served the first's
    pixels.
    """
    payload = (
        f"{slide_id}|{bbox.x}|{bbox.y}|{bbox.w}|{bbox.h}"
        f"|{target_mpp:.6f}|{output_size}|{extraction_version}"
    )
    if degradation_digest:
        payload += f"|{degradation_digest}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_path(root: str | Path, slide_id: str, key: str) -> Path:
    return Path(root) / slide_id / key[:2] / f"{key}.png"


def image_sha256(img: Image.Image) -> str:
    """Hash the raw RGB pixel buffer, not the encoded file.

    PNG encoder output is not stable across Pillow/zlib versions, so hashing the file
    would make a QC failure indistinguishable from a library upgrade.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    return hashlib.sha256(img.tobytes()).hexdigest()


def save_png(img: Image.Image, path: str | Path) -> None:
    """Write the pixels, and only the pixels.

    `icc_profile=None` is load-bearing: without it Pillow copies the source slide's
    colour profile into the file, which for some vendors is large enough that Pillow
    itself will not read the result back (see MAX_TEXT_CHUNK above). Dropping it cannot
    change what this cache is for -- `image_sha256` hashes the RGB buffer, so the
    canonical identity of an entry never included the profile.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The temp name carries this process's pid, so two shards materialising the same key
    # at once cannot share it. A fixed name made that an interleaved write followed by two
    # renames: whichever process replaced second raised FileNotFoundError on a temp file
    # the first had already moved, and a differently-timed interleaving would have left a
    # splice of both writers' bytes under a name that looks fine. The cache is
    # content-addressed, so concurrent writers of one key are normal in a sharded
    # materialisation rather than a misuse to be prevented.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        img.save(tmp, format="PNG", optimize=False, compress_level=6, icc_profile=None)
        # `os.link`, not `os.replace`. Replace re-points the name at a new inode and
        # unlinks the old one; on NFS every process that already had the entry open then
        # gets ESTALE on its next read, which is how a sharded warm-up of this cache dies
        # hours in with the writers all looking healthy. Link is the same atomic publish
        # when the name is free and simply fails with EEXIST when it is not -- and EEXIST
        # is not a problem to resolve here: the key covers everything that determines the
        # pixels, so the entry already there is the entry this call would have written.
        try:
            os.link(tmp, path)
        except FileExistsError:
            pass
    finally:
        # A crash between save and link would otherwise leave this behind forever, and
        # nothing ever looks for stale temp files.
        if tmp.exists():
            tmp.unlink()


#: How many times `load_png` re-resolves a path whose handle came back stale, and how
#: long it waits between tries. `save_png` no longer creates that window, but entries
#: written by jobs queued before it stopped doing so still can, and an NFS client can
#: produce a stale handle by itself when a rename lands between lookup and read. Three
#: tries over ~0.3s covers that; beyond it the mount is broken and the caller should see
#: the error rather than a process that never finishes.
STALE_RETRIES = 3
STALE_BACKOFF_S = 0.1


def load_png(path: str | Path) -> Image.Image:
    """Read a cache entry, retrying a stale NFS handle.

    Only ESTALE is retried. ENOENT means the key is not in the cache, which is a bug in
    the caller or a cache that was cleared under it -- retrying would make a clear
    failure a slow one.
    """
    for attempt in range(STALE_RETRIES):
        try:
            with Image.open(path) as img:
                # Inside the retry on purpose: `Image.open` is lazy, so a stale handle
                # surfaces from `convert`, where the decode actually reads bytes.
                return img.convert("RGB")
        except OSError as exc:
            if exc.errno != errno.ESTALE or attempt == STALE_RETRIES - 1:
                raise
            time.sleep(STALE_BACKOFF_S)
    raise AssertionError("unreachable")
