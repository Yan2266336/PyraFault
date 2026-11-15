# tests/test_build_window_bank.py
"""Bank construction.

The property worth asserting is not the tiling -- Task 2 covers that -- but that the
grid is built independently of the benchmark's own query lattice. Reusing
04_features/ would be cheap and would hand the method the construction's prior.
"""
import numpy as np
import pandas as pd
import pytest
from PIL import Image


def test_bank_origins_are_not_the_query_lattice(tmp_path):
    """query_candidates.parquet holds the lattice. A bank built at half-window stride
    must contain positions that are not lattice points."""
    from pyrafault.verify.bank import window_origins
    lattice = {(0, 0), (128, 0), (256, 0)}
    grid = set(window_origins(512, side=128, stride=64))
    assert grid - lattice, "grid reproduces the lattice exactly"
    assert (64, 0) in grid


def test_build_writes_one_bank_per_context(tmp_path, monkeypatch):
    from scripts import _bank_builder_entry as entry   # thin importable wrapper
    ctx = pd.DataFrame([{"context_id": "c0", "slide_id": "S1"},
                        {"context_id": "c1", "slide_id": "S1"}])
    images = {"c0": Image.new("RGB", (256, 256), "white"),
              "c1": Image.new("RGB", (256, 256), "black")}

    class _E:
        def embed(self, imgs):
            return np.ones((len(imgs), 8), dtype=np.float32)

    out = tmp_path / "bank"
    entry.build(ctx, lambda cid: images[cid], _E(), side=128, out_dir=out)
    assert (out / "c0.npz").exists() and (out / "c1.npz").exists()

    from pyrafault.verify.bank import WindowBank
    b = WindowBank.load(out / "c0.npz")
    assert b.side == 128 and len(b.origins) == len(b.embeddings) == 9


class _CountingExtractor:
    """Counts embed calls, so a skipped context is provably not re-embedded."""

    def __init__(self):
        self.calls = 0

    def embed(self, imgs):
        self.calls += 1
        return np.ones((len(imgs), 8), dtype=np.float32)


def _frame(ids):
    return pd.DataFrame([{"context_id": c, "slide_id": "S1"} for c in ids])


def _images(ids):
    return {c: Image.new("RGB", (256, 256), "white") for c in ids}


class TestResume:
    """A full-scale build is 7,469 contexts at about eight seconds each, so a shard that
    times out part way through must not start over. Stage 13's own error message cites
    "stage 12 has no resume" as the reason an interrupted build is unusable; this is that
    gap closed.
    """

    def test_an_existing_bank_at_the_same_side_is_not_rebuilt(self, tmp_path):
        from scripts import _bank_builder_entry as entry
        ids = ["c0", "c1"]
        out = tmp_path / "bank"
        entry.build(_frame(ids), lambda c: _images(ids)[c], _CountingExtractor(),
                    side=128, out_dir=out)
        first = sorted(p.stat().st_mtime_ns for p in out.glob("*.npz"))

        ext = _CountingExtractor()
        written = entry.build(_frame(ids), lambda c: _images(ids)[c], ext,
                              side=128, out_dir=out)

        assert ext.calls == 0, "a resumed build re-embedded a context it already had"
        assert written == 0
        assert sorted(p.stat().st_mtime_ns for p in out.glob("*.npz")) == first

    def test_only_the_missing_contexts_are_built(self, tmp_path):
        from scripts import _bank_builder_entry as entry
        ids = ["c0", "c1", "c2"]
        out = tmp_path / "bank"
        entry.build(_frame(["c0"]), lambda c: _images(ids)[c], _CountingExtractor(),
                    side=128, out_dir=out)

        ext = _CountingExtractor()
        written = entry.build(_frame(ids), lambda c: _images(ids)[c], ext,
                             side=128, out_dir=out)

        assert written == 2
        assert {p.stem for p in out.glob("*.npz")} == {"c0", "c1", "c2"}

    def test_a_bank_built_at_a_different_side_is_rebuilt_not_reused(self, tmp_path):
        """The side comes from the config. Reusing a bank built under a different one
        would hand the verifier windows of the wrong size with no indication."""
        from scripts import _bank_builder_entry as entry
        from pyrafault.verify.bank import WindowBank
        out = tmp_path / "bank"
        entry.build(_frame(["c0"]), lambda c: _images(["c0"])[c], _CountingExtractor(),
                    side=64, out_dir=out)

        ext = _CountingExtractor()
        written = entry.build(_frame(["c0"]), lambda c: _images(["c0"])[c], ext,
                             side=128, out_dir=out)

        assert written == 1 and ext.calls > 0
        assert WindowBank.load(out / "c0.npz").side == 128

    def test_resume_can_be_turned_off(self, tmp_path):
        from scripts import _bank_builder_entry as entry
        out = tmp_path / "bank"
        entry.build(_frame(["c0"]), lambda c: _images(["c0"])[c], _CountingExtractor(),
                    side=128, out_dir=out)

        ext = _CountingExtractor()
        assert entry.build(_frame(["c0"]), lambda c: _images(["c0"])[c], ext,
                           side=128, out_dir=out, resume=False) == 1
        assert ext.calls > 0
