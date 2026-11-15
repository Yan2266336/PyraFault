"""The cache under concurrent materialisation on NFS.

A sharded warm-up has many processes reading one cache and writing into it, and two
shards legitimately want the same key (a cross-slide pair's patch belongs to both
slides' pairs). `save_png` used to finish with `os.replace`, which unlinks whatever
inode sat under the name. On a local filesystem a reader holding that inode open keeps
reading it happily; on NFS the reader's file handle is now stale and its next `read()`
raises `OSError: [Errno 116] Stale file handle`. That is what killed 74 of 80 warm-up
shards of the lung freeze, hours in, after the pid-temp-name fix had already landed --
the crash moved from the writer to the readers.

The cache is content-addressed: an entry that exists is already the right pixels. So a
writer that finds the name taken has nothing to contribute and must leave the existing
inode alone, which is what these pin.
"""
from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest
from PIL import Image

from pyrafault import cache
from pyrafault.cache import load_png, save_png


def _img(v: int) -> Image.Image:
    return Image.new("RGB", (8, 8), (v, v, v))


class TestAnExistingEntryIsNeverReplaced:
    def test_the_inode_under_the_name_survives_a_second_write(self, tmp_path):
        p = tmp_path / "patch.png"
        save_png(_img(10), p)
        before = p.stat().st_ino

        save_png(_img(200), p)

        assert p.stat().st_ino == before, (
            "the name was re-pointed at a new inode; every reader holding the old one "
            "gets ESTALE on NFS"
        )

    def test_the_first_writer_s_pixels_are_what_stays(self, tmp_path):
        """Not a preference between the two -- for a content-addressed key they are the
        same pixels. It pins that the second write is dropped rather than applied."""
        p = tmp_path / "patch.png"
        save_png(_img(10), p)

        save_png(_img(200), p)

        assert load_png(p).getpixel((0, 0)) == (10, 10, 10)

    def test_the_dropped_write_leaves_no_temp_file(self, tmp_path):
        p = tmp_path / "patch.png"
        save_png(_img(10), p)

        save_png(_img(200), p)

        assert not list(tmp_path.glob("*.tmp"))

    def test_a_reader_holding_the_file_open_is_unaffected_by_a_concurrent_write(
        self, tmp_path
    ):
        p = tmp_path / "patch.png"
        save_png(_img(10), p)
        held = open(p, "rb")
        try:
            save_png(_img(200), p)
            held.seek(0)
            assert len(held.read()) > 0
        finally:
            held.close()


class TestLoadSurvivesAStaleHandle:
    """A bounded retry, because the window is not only `save_png`'s.

    Entries written by the version that used `os.replace` are still being written by
    jobs already queued, and an NFS client can hand back a stale handle on its own when
    a server-side rename lands between lookup and read. Re-opening resolves the name
    again, which is all that is needed; the retry is bounded so a genuinely broken
    mount still fails rather than spinning.
    """

    def test_a_stale_handle_on_the_first_open_is_retried(self, tmp_path, monkeypatch):
        p = tmp_path / "patch.png"
        save_png(_img(42), p)
        real_open = Image.open
        calls = {"n": 0}

        def flaky(*a, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError(errno.ESTALE, "Stale file handle")
            return real_open(*a, **kw)

        monkeypatch.setattr(cache.Image, "open", flaky)

        assert load_png(p).getpixel((0, 0)) == (42, 42, 42)
        assert calls["n"] == 2

    def test_a_stale_handle_raised_during_decode_is_retried(self, tmp_path, monkeypatch):
        """Where it actually surfaced: `Image.open` is lazy, so the ESTALE came out of
        `convert`, past the point a naive try around the open would see it."""
        p = tmp_path / "patch.png"
        save_png(_img(42), p)
        real_open = Image.open
        calls = {"n": 0}

        class Flaky:
            def __init__(self, inner):
                self._inner = inner

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return self._inner.__exit__(*exc)

            def convert(self, mode):
                if calls["n"] == 1:
                    raise OSError(errno.ESTALE, "Stale file handle")
                return self._inner.convert(mode)

        def flaky(*a, **kw):
            calls["n"] += 1
            return Flaky(real_open(*a, **kw))

        monkeypatch.setattr(cache.Image, "open", flaky)

        assert load_png(p).getpixel((0, 0)) == (42, 42, 42)
        assert calls["n"] == 2

    def test_a_persistently_stale_path_still_raises(self, tmp_path, monkeypatch):
        p = tmp_path / "patch.png"
        save_png(_img(42), p)

        def always(*a, **kw):
            raise OSError(errno.ESTALE, "Stale file handle")

        monkeypatch.setattr(cache.Image, "open", always)

        with pytest.raises(OSError) as exc:
            load_png(p)
        assert exc.value.errno == errno.ESTALE

    def test_an_unrelated_oserror_is_not_retried(self, tmp_path, monkeypatch):
        """A missing file is a cache bug, not a transient mount condition. Retrying it
        would turn a clear failure into a slow one."""
        p = tmp_path / "patch.png"
        save_png(_img(42), p)
        calls = {"n": 0}

        def missing(*a, **kw):
            calls["n"] += 1
            raise FileNotFoundError(errno.ENOENT, "No such file or directory")

        monkeypatch.setattr(cache.Image, "open", missing)

        with pytest.raises(FileNotFoundError):
            load_png(p)
        assert calls["n"] == 1
