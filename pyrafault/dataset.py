"""The one interface every downstream verifier uses (guideline §11).

Reading a sample is a lookup, not a decision: membership, coordinates and labels were all
frozen during construction. This module therefore contains no sampling, no filtering and
decides no pixel perturbation of its own -- a deliberate absence, enforced by a test that
greps this file. The one pixel-changing call it makes (`_image`, via `degrade_apply`) only
replays a degradation whose kind and parameters were frozen at construction time and read
back off the row; it never draws, chooses, or varies one at read time.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from pyrafault.cache import cache_key, cache_path, load_png, save_png
from pyrafault.config import load_config
from pyrafault.degrade import apply as degrade_apply, degradation_digest
from pyrafault.geom import BBoxL0
from pyrafault.io.read import read_at_mpp
from pyrafault.io.slide import open_slide
from pyrafault.schema import read_table

_BBOX = ("x", "y", "w", "h")

# spec §8: the exact key set `__getitem__` returns under "inputs"/"target". This is
# the single source of truth for anyone that needs to know what `inputs` exposes --
# the loader-level leakage test (tests/test_dataset.py) checks `set(item["inputs"])`
# against it, and the stage-09 audit (pyrafault.qc.report.check_leakage) imports it
# rather than restating its own separate list. A field added to the `inputs` dict
# literal below without updating INPUT_KEYS is caught immediately by that key-set
# test; a field added to INPUT_KEYS without a way to reconstruct it from the frozen
# tables is caught immediately by check_leakage raising rather than skipping it (see
# that module). Restating the same four names in three unrelated places -- the dict
# literal, a test, and the audit -- is exactly how `pair_id` shipped inside `inputs`
# unaudited the first time this loader was split: every guard agreed with a
# hand-copied list, and none of them looked at what the others said.
INPUT_KEYS = frozenset({
    "context", "query", "claimed_bbox", "context_bbox",
    "item_id", "slide_id", "split", "extraction_version",
    "context_mpp", "query_mpp",
})
# Images: pixels, out of scope for a "pixel-free signal" audit (spec §13.5) by
# definition.
INPUT_IMAGE_KEYS = frozenset({"context", "query"})
# Geometry: audited via the numeric signals derived FROM these boxes (on_grid,
# centre/border distance, width/height), not via the string/substring audit.
INPUT_GEOMETRY_KEYS = frozenset({"claimed_bbox", "context_bbox"})
# Numeric scalars: the physical resolutions `mc` and `mq` the task formulation lists as
# CMPV inputs. A verifier needs them -- scale alignment (downsampling `C` to `Q`'s
# physical resolution) is arithmetic on `mc/mq`, so without them the first step of any
# cross-magnification method is impossible. They are constant across a single freeze and
# therefore carry no label information, but they are audited numerically rather than
# assumed harmless: pooling several magnification combinations into one table would make
# them vary, and if the combos then differed in fault composition, magnification would
# leak. The audit notices that; an exemption would not.
INPUT_SCALAR_KEYS = frozenset({"context_mpp", "query_mpp"})
# Whatever's left of INPUT_KEYS is string-valued and audited by check_leakage's
# substring signal.
INPUT_STRING_KEYS = (
    INPUT_KEYS - INPUT_IMAGE_KEYS - INPUT_GEOMETRY_KEYS - INPUT_SCALAR_KEYS
)

TARGET_KEYS = frozenset({
    "claim_correct", "true_bbox", "true_slide_id", "pair_id", "fault_type",
    "fault_severity", "normalized_offset", "iou_claim_true", "hardness_score",
    # Not an answer, and not something a verifier may see either: it is what a run
    # harness needs to key its localise cache so two bands of one true box are not
    # collapsed into one localisation. See `runner.cache_key`.
    "degradation_digest",
})


def _true_slide(row) -> str:
    """The slide the served patch came from. Falls back to `slide_id` for a table built
    before `true_slide_id` existed, where no out-of-slide pair can occur."""
    value = row.get("true_slide_id")
    if value is None or (isinstance(value, float) and value != value):
        return row["slide_id"]
    return str(value)


def _degradation(row) -> tuple[str, dict]:
    """The frozen degradation of one row, or `none` for a table built before the
    columns existed -- so an old freeze still loads rather than needing a migration
    before it can be read at all."""
    kind = row.get("degradation_kind")
    if kind is None or (isinstance(kind, float) and kind != kind):
        return ("none", {})
    raw = row.get("degradation_params")
    if raw is None or (isinstance(raw, float) and raw != raw):
        raw = "{}"
    return (str(kind), json.loads(raw or "{}"))


def make_item_id(seed: int, pair_id: str) -> str:
    """The opaque identifier `inputs` carries instead of `pair_id`.

    `pair_id` is minted as `f"{query_id}__{fault_type}"` (`06_build_pairs.py`,
    `07_mine_hard_negatives.py`), so `"__clean" in pair_id` recovers `claim_correct`
    with 100% precision on every pair -- a worse leak than §15.1's, and the shape it
    shipped in the first time this loader was split, because the guarding test
    compared *key names* rather than *values*. `sha256(seed|pair_id)` is a stable,
    deterministic digest, unlike Python's per-process `hash()`; its hex alphabet
    (0-9a-f) cannot spell any of `pyrafault.faults`'s fault-type names -- every one of
    them contains a letter outside a-f -- so the identifier is structurally incapable
    of leaking the label the way the raw `pair_id` did.
    """
    return hashlib.sha256(f"{seed}|{pair_id}".encode()).hexdigest()


def assert_spec_matches_config(
    what: str,
    target_mpp: float, output_size: int, extraction_version: str,
    cfg_target_mpp: float, cfg_output_size: int, cfg_extraction_version: str,
) -> None:
    """I2: the extraction spec stored on a `contexts`/`queries` row is the canonical
    record — a row that disagrees with the live config is a provenance failure, not
    something to silently paper over by preferring one side."""
    if (target_mpp, output_size, extraction_version) != (
        cfg_target_mpp, cfg_output_size, cfg_extraction_version
    ):
        raise ValueError(
            f"{what} extraction spec (target_mpp={target_mpp}, output_size={output_size}, "
            f"extraction_version={extraction_version!r}) does not match the loaded config "
            f"(target_mpp={cfg_target_mpp}, output_size={cfg_output_size}, "
            f"extraction_version={cfg_extraction_version!r})"
        )


class PyraFaultDataset:
    """Indexed by integer position, by the opaque `item_id`, or by `pair_id`."""

    #: How many decoded patches to keep. Forty pairs share each context in the lung
    #: freeze, and `pairs.parquet` sorted by `pair_id` puts them next to each other, so
    #: holding the last few decodes turns forty reads of one 2048x2048 context (~100 ms
    #: each) into one. Two entries would already do it -- an item reads its context then
    #: its query -- and four leaves room for a pass that interleaves differently. It is a
    #: bound, not a target: an evicted entry is re-decoded, never served stale.
    IMAGE_CACHE_SIZE = 4

    def __init__(
        self,
        benchmark_dir: str | Path,
        gdc_dir: str | Path,
        split: str | None = None,
        cache_dir: str | Path | None = None,
        materialize: bool = True,
    ) -> None:
        self.benchmark_dir = Path(benchmark_dir)
        self.gdc_dir = Path(gdc_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.benchmark_dir.parent / "02_cache"
        self.materialize = materialize

        self.cfg = load_config(self.benchmark_dir / "config.yaml")
        self.pairs = read_table(self.benchmark_dir / "pairs.parquet", "pairs")
        if split is not None:
            self.pairs = self.pairs[self.pairs["split"] == split]
        self.pairs = self.pairs.sort_values("pair_id").reset_index(drop=True)

        contexts = read_table(self.benchmark_dir / "contexts.parquet", "contexts")
        queries = read_table(self.benchmark_dir / "queries.parquet", "queries")
        slides = read_table(self.benchmark_dir / "slides.parquet", "slides")
        # I2: the canonical per-row extraction spec (target_mpp, output_size,
        # extraction_version) lives in contexts.parquet/queries.parquet — read it from
        # there, not from the live config, and fail loudly if a row disagrees with the
        # config this dataset was constructed against.
        self._ctx = {
            c.context_id: (
                BBoxL0(c.bbox_l0_x, c.bbox_l0_y, c.bbox_l0_w, c.bbox_l0_h),
                c.target_mpp, c.output_size, c.extraction_version,
            )
            for c in contexts.itertuples()
        }
        self._qry_spec = {
            q.query_id: (q.target_mpp, q.output_size, q.extraction_version)
            for q in queries.itertuples()
        }
        self._path = dict(zip(slides["slide_id"], slides["source_relpath"]))
        self._by_id = {p: i for i, p in enumerate(self.pairs["pair_id"])}
        # spec §8: `item_id` is a stable digest of (seed, pair_id) -- computed once
        # here from metadata alone, no pixels touched, so ds.item_ids is as cheap as
        # ds.pair_ids.
        self._item_ids = [make_item_id(self.cfg.seed, p) for p in self.pairs["pair_id"]]
        self._by_item_id = {iid: i for i, iid in enumerate(self._item_ids)}
        self._open: dict[str, Any] = {}
        # Insertion-ordered, so the oldest key is the first one -- a dict is the LRU.
        self._img_cache: dict[str, Image.Image] = {}

    def _check_spec(self, what: str, target_mpp: float, output_size: int,
                     extraction_version: str, cfg_mpp: float, cfg_size: int) -> None:
        assert_spec_matches_config(what, target_mpp, output_size, extraction_version,
                                    cfg_mpp, cfg_size, self.cfg.extraction_version)

    @property
    def pair_ids(self) -> list[str]:
        """The readable id -- `target`-only. Legitimate to hold onto for building
        `pairs.parquet`-keyed tooling or QC overlay filenames; never returned in
        `inputs` (spec §8)."""
        return list(self.pairs["pair_id"])

    @property
    def item_ids(self) -> list[str]:
        """The opaque id returned in `inputs["item_id"]`. Iterate this, not
        `pair_ids`, when writing verifier code -- it never needs the readable form."""
        return list(self._item_ids)

    def __len__(self) -> int:
        return len(self.pairs)

    def _slide(self, slide_id: str):
        if slide_id not in self._open:
            self._open[slide_id] = open_slide(self.gdc_dir / self._path[slide_id], slide_id)
        return self._open[slide_id]

    def slide_handle(self, slide_id: str):
        """An open handle for one slide, from this dataset's own cache.

        Layer C drives a tool server over the slides its items came from, and it reads
        them through this cache rather than opening its own: one episode plus one item
        would otherwise hold two readers on the same file, and two readers opened by
        different backends can disagree about a slide's mpp -- which would make the
        agent's zoom footprint and the item's query patch two different sizes while both
        looked right.

        Read-only: no pixel decision is taken here, so §11's rule that this module
        decides nothing is intact.
        """
        return self._slide(slide_id)

    def _image(self, slide_id: str, box: BBoxL0, target_mpp: float, size: int,
               extraction_version: str, degradation: tuple[str, dict] | None = None) -> Image.Image:
        kind, params = degradation or ("none", {})
        key = cache_key(slide_id, box, target_mpp, size, extraction_version,
                        degradation_digest(kind, params))
        cached = self._img_cache.get(key)
        if cached is not None:
            # Copied, not shared. Every item has always owned its images, and a verifier
            # that drew on its input would otherwise poison every later item sharing that
            # context -- corruption that would read as a model failure. A 2048x2048 copy
            # is under a millisecond against a ~100 ms decode.
            self._img_cache[key] = self._img_cache.pop(key)      # most recently used
            return cached.copy()
        path = cache_path(self.cache_dir, slide_id, key)
        if path.exists():
            img = load_png(path)
        else:
            img, _ = read_at_mpp(self._slide(slide_id), box, target_mpp, size,
                                 extraction_version)
            # Applied after the read and before the cache write, so a cached entry is the
            # pixels a verifier sees -- one decode per item rather than one decode plus a
            # transform, and the entry's hash is comparable with the frozen one.
            img = degrade_apply(img, kind, params)
            if self.materialize:
                save_png(img, path)
        self._img_cache[key] = img
        while len(self._img_cache) > self.IMAGE_CACHE_SIZE:
            self._img_cache.pop(next(iter(self._img_cache)))
        return img.copy()

    def meta(self, key: int | str) -> dict[str, Any]:
        """The item without its pixels: `inputs` minus `context` and `query`.

        `__getitem__` is the item; this is the half of it that costs nothing. A sharded
        evaluation localises in parallel and then decides in one place, and the deciding
        pass needs only the claim and the answer key -- going through `__getitem__` for
        that would re-read two cached patches per row, which for the lung freeze is
        600,000 PNG decodes to recover numbers already sitting in `pairs.parquet`.

        Everything else is identical, the frozen-spec checks included: a record whose
        own resolution disagrees with the live config raises here exactly as it does on
        a full read, so skipping the pixels cannot skip that gate.
        """
        return self._item(key, with_images=False)

    def __getitem__(self, key: int | str) -> dict[str, Any]:
        """Indexed by integer position, by the opaque `item_id` (spec §8 -- what
        verifier code should use), or by the readable `pair_id` (kept working because
        it is cheap and legitimately useful for `pairs.parquet`-keyed tooling)."""
        return self._item(key, with_images=True)

    def _item(self, key: int | str, with_images: bool) -> dict[str, Any]:
        if isinstance(key, str):
            if key in self._by_id:
                idx = self._by_id[key]
            elif key in self._by_item_id:
                idx = self._by_item_id[key]
            else:
                raise KeyError(key)
        else:
            idx = int(key)
        row = self.pairs.iloc[idx]
        item_id = self._item_ids[idx]

        ctx, ctx_mpp, ctx_size, ctx_ver = self._ctx[row["context_id"]]
        self._check_spec("context", ctx_mpp, ctx_size, ctx_ver,
                          self.cfg.context.target_mpp, self.cfg.context.size_px)
        qry_mpp, qry_size, qry_ver = self._qry_spec[row["query_id"]]
        self._check_spec("query", qry_mpp, qry_size, qry_ver,
                          self.cfg.query.target_mpp, self.cfg.query.size_px)
        claimed = BBoxL0(*(int(row[f"claimed_bbox_l0_{k}"]) for k in _BBOX))
        true = BBoxL0(*(int(row[f"true_bbox_l0_{k}"]) for k in _BBOX))

        # spec §8: `inputs` is everything a verifier is allowed to see; `target` is
        # the answer key. `true_bbox` alone answers the recovery task, and
        # `normalized_offset == 0.0`, `iou_claim_true == 1.0`, a non-null
        # `fault_severity` and the null/non-null pattern of `hardness_score` each
        # restate `claim_correct` -- keeping them out of `inputs` means consuming the
        # answer requires explicitly reaching into `target`. `pair_id` restates the
        # label even more directly (see `make_item_id`), so `inputs` carries the
        # opaque `item_id` instead and `pair_id` moves to `target`.
        # `extraction_version` describes how the pixels were made, not the answer, so
        # it stays in `inputs` alongside the images it describes.
        images = {}
        if with_images:
            images = {
                "context": self._image(row["slide_id"], ctx, ctx_mpp, ctx_size, ctx_ver),
                # Read against `true_slide_id`, not `slide_id`: for an out-of-slide
                # pair the served patch lives on another slide, and reading its box
                # against this slide would return whatever tissue happens to sit at
                # those coordinates here -- a wrong image that looks entirely plausible,
                # which is precisely the failure this benchmark studies. The column
                # equals `slide_id` for every other fault type.
                "query": self._image(_true_slide(row), true, qry_mpp, qry_size, qry_ver,
                                     _degradation(row)),
            }

        return {
            "inputs": {
                **images,
                "claimed_bbox": claimed,
                "context_bbox": ctx,
                "item_id": item_id,
                "slide_id": row["slide_id"],
                "split": row["split"],
                "extraction_version": self.cfg.extraction_version,
                # `mc` and `mq` (task formulation §4.1). Taken from the record's own
                # frozen spec, not from the live config -- the same I2 rule the
                # materializer follows, and `_check_spec` above has already established
                # that the two agree, so a record built under a different resolution
                # cannot be served silently under this one's.
                "context_mpp": float(ctx_mpp),
                "query_mpp": float(qry_mpp),
            },
            "target": {
                "claim_correct": bool(row["claim_correct"]),
                "true_bbox": true,
                # The slide `true_bbox` is expressed in -- `slide_id` for every fault
                # type except the two out-of-slide ones, where the served patch lives
                # on another slide entirely. A run harness that keys its localise cache
                # on (context, true_bbox) alone cannot tell two out-of-slide items
                # apart when their contexts and true_bbox coordinates happen to
                # coincide but their source slides differ; exposing this lets it.
                "true_slide_id": _true_slide(row),
                # Same reason as `true_slide_id` above, for the same consumer. Degradation
                # is drawn per item, so a harness keying its localise cache on
                # (context, true_bbox, true_slide_id) cannot tell two bands of one box
                # apart -- it would serve the second item the first's localisation while
                # the pixels differ. This is in `target` and not `inputs` deliberately: the
                # harness needs it, a verifier must never be told which band it got.
                "degradation_digest": degradation_digest(*_degradation(row)),
                "pair_id": row["pair_id"],
                "fault_type": str(row["fault_type"]),
                "fault_severity": row["fault_severity"],
                "normalized_offset": row["normalized_offset"],
                "iou_claim_true": row["iou_claim_true"],
                "hardness_score": row["hardness_score"],
            },
        }

    def close(self) -> None:
        for handle in self._open.values():
            handle.close()
        self._open.clear()
        self._img_cache.clear()

    def __enter__(self) -> "PyraFaultDataset":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
