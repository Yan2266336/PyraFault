"""Bank construction core, importable so it can be tested without a CLI."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from pyrafault.verify.bank import WindowBank, window_origins


def _already_built(path: Path, side: int) -> bool:
    """True when `path` holds a usable bank for this `side`.

    The side is checked rather than assumed: it comes from the config, and silently
    reusing a bank built under a different one would hand the verifier windows of the
    wrong size with nothing to indicate it. A file that cannot be read at all counts as
    not built, so a shard killed mid-write resumes over its own truncated output.
    """
    if not path.exists():
        return False
    try:
        return WindowBank.load(path).side == side
    except Exception:
        return False


def build(contexts, load_image, extractor, side: int, out_dir: Path,
          batch: int = 256, resume: bool = True) -> int:
    """One bank per context. `load_image(context_id) -> PIL.Image`. Returns banks written.

    `resume` skips contexts that already have a bank at this side, which is what makes a
    shard restartable: a full-scale build is 7,469 contexts at roughly eight seconds each,
    dominated by reading the slide, so a shard that hits its wall clock must not begin
    again from nothing. Pass `resume=False` to force a rebuild.
    """
    out_dir = Path(out_dir)
    n = 0
    for row in contexts.itertuples():
        path = out_dir / f"{row.context_id}.npz"
        if resume and _already_built(path, side):
            continue
        img = load_image(row.context_id)
        origins = window_origins(img.width, side)
        crops = [img.crop((x, y, x + side, y + side)) for x, y in origins]
        vecs = []
        for i in range(0, len(crops), batch):
            vecs.append(np.asarray(extractor.embed(crops[i:i + batch]),
                                   dtype=np.float16))
        WindowBank(row.context_id, side, origins,
                   np.concatenate(vecs)).save(path)
        n += 1
    return n
