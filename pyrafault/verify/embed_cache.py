"""Memoise a frozen extractor's per-image embeddings across items.

Two of the collaborator's baselines, `SUEBaseline` and
`StatisticalUncertaintyBaseline`, embed the context's whole window grid on every
item. Their docstring says so deliberately -- "All windows are embedded afresh,
making first-version costs and behavior explicit" -- and at pilot scale that is
the right call, because a bank is a second thing that can be wrong.

At full scale it is 37x of avoidable work, measured rather than guessed:

* a 2048 px context with a 128 px window at stride 0.5 gives ~961 windows, so
  each item embeds 1 query + 961 crops = 962 images;
* each context is shared by **38.4** localisations on the lung freeze
  (924,217 unique `(context, served patch, degradation)` triples over 24,054
  contexts), and every one of them re-embeds the same 961 crops;
* 286,368 localisations x 962 = 275M embeddings, of which 7.45M are distinct.

That is 720 of the 815 GPU-hours a full seven-baseline run would cost.

This module does not change either method. It wraps the *extractor*, memoising a
deterministic function of an image, so the matrix each baseline receives is the
same matrix it would have computed. Their algorithm, their window grid, their
reliability -- untouched. The one thing a caller must check is that it really is
the same matrix, which `tests/test_embed_cache.py` asserts against the unwrapped
extractor on both synthetic and real items, including the `Localisation` the
baseline ends up returning.

Why wrap the extractor rather than add a bank: a bank is persistent state with a
manifest, an extractor identity and an invalidation rule -- all of which
`pyrafault.verify.bank` already gets right for ZoomTrace and none of which these
two baselines claim. An in-process cache keyed by image content has no such
state: it cannot be stale, because its key *is* the input.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict

import numpy as np
from PIL import Image

#: Cached rows, not images. One DINOv2-base row is 768 float32 = 3 KB, so the
#: default holds about 12 MB -- four contexts' worth of windows at ~961 each.
#: Sized for how the run actually arrives: `runner.shard_of` groups a shard by
#: context, so consecutive items share one context's crops and an LRU this size
#: never evicts a window it is about to need again.
DEFAULT_MAX_ENTRIES = 4000


def image_key(img: Image.Image) -> bytes:
    """A content key for one image.

    The raw bytes would be a sound key and a ruinous one: 961 windows of
    128x128x3 is 47 MB of keys per context. blake2b over the same bytes plus the
    size and mode is 16 bytes, and the size and mode are included because
    `tobytes()` alone does not distinguish a 128x256 from a 256x128, nor RGB
    from L.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(f"{img.mode}|{img.width}x{img.height}|".encode())
    h.update(img.tobytes())
    return h.digest()


class CachingExtractor:
    """An extractor that embeds each distinct image once.

    Presents the wrapped extractor's own identity -- `name`, `revision`, `dim`,
    and anything else a baseline reads off it -- so a baseline cannot tell it is
    wrapped, and so the `verifier_revision` recorded in a predictions file is
    still the model's. The cache is this object's, not a module global: two
    extractors in one process do not share rows, and nothing survives the run.
    """

    def __init__(self, inner, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self._inner = inner
        self._max = int(max_entries)
        self._rows: OrderedDict[bytes, np.ndarray] = OrderedDict()
        self.n_hits = 0
        self.n_misses = 0

    def __getattr__(self, item):
        # Only reached for attributes this wrapper does not define, which is
        # every one the baselines read (`name`, `revision`, `dim`, ...).
        return getattr(self._inner, item)

    def embed(self, images) -> np.ndarray:
        """Return the same matrix `self._inner.embed(images)` would return.

        The misses go to the inner extractor in **one** call, in their original
        relative order, so they are batched the way the unwrapped path would
        batch them. Rows then land back in the caller's order.
        """
        images = list(images)
        if not images:
            return self._inner.embed(images)

        keys = [image_key(im) for im in images]
        missing_at, missing_keys, seen = [], set(), set()
        for i, k in enumerate(keys):
            if k in self._rows or k in seen:
                continue
            seen.add(k)
            missing_at.append(i)
            missing_keys.add(k)

        if missing_at:
            fresh = np.asarray(self._inner.embed([images[i] for i in missing_at]))
            if fresh.ndim != 2 or len(fresh) != len(missing_at):
                raise ValueError(
                    f"wrapped extractor returned {fresh.shape} for "
                    f"{len(missing_at)} images; expected a 2-D matrix with one "
                    "row per image"
                )
            for i, row in zip(missing_at, fresh):
                self._rows[keys[i]] = row
        self.n_misses += len(missing_at)
        self.n_hits += len(images) - len(missing_at)

        out = np.empty((len(images), self._rows[keys[0]].shape[0]),
                       dtype=self._rows[keys[0]].dtype)
        for i, k in enumerate(keys):
            out[i] = self._rows[k]
            self._rows.move_to_end(k)
        while len(self._rows) > self._max:
            self._rows.popitem(last=False)
        return out

    @property
    def hit_rate(self) -> float:
        n = self.n_hits + self.n_misses
        return float(self.n_hits / n) if n else 0.0


# --------------------------------------------------------------------------------------
# The two plugins that benefit, with the cache in place of nothing else changed.
# --------------------------------------------------------------------------------------
#
# These subclass the collaborator's plugins and override only how the extractor is
# built. `name` is inherited, so each still identifies as `sue_baseline` /
# `statistical_uncertainty_baseline`, their `_backend()` identity check still passes,
# and stage 13 writes to the same directory a cacheless run would. That is deliberate:
# this is the same method at the same revision, computed once instead of 38 times.

def _cached_dino(plugin):
    """The plugin's own pinned DINOv2, wrapped. Imported late so that constructing a
    plugin still loads no weights and touches no network."""
    from pyrafault.verify.plugins import _PinnedDinoV2Extractor
    inner = _PinnedDinoV2Extractor(
        plugin._path("PYRAFAULT_DINOV2_SNAPSHOT", directory=True), plugin.device)
    plugin.cache = CachingExtractor(inner)
    return plugin.cache


def _sue_plugin_base():
    from pyrafault.verify.plugins import SUEPlugin
    return SUEPlugin


def _su_plugin_base():
    from pyrafault.verify.plugins import StatisticalUncertaintyPlugin
    return StatisticalUncertaintyPlugin


class CachedSUEPlugin(_sue_plugin_base()):
    """`sue_baseline`, with per-image embeddings memoised across items."""

    def _build_backend(self):
        from pyrafault.verify.sue import SUEBaseline
        return SUEBaseline(_cached_dino(self))


class CachedStatisticalUncertaintyPlugin(_su_plugin_base()):
    """`statistical_uncertainty_baseline`, with the same memoisation."""

    def _build_backend(self):
        from pyrafault.verify.statistical_uncertainty import StatisticalUncertaintyBaseline
        return StatisticalUncertaintyBaseline(_cached_dino(self))
