"""The window bank belongs to the extractor that built it.

A bank file holds `context_id`, `side`, `origins` and `embeddings` and nothing that says
which extractor produced those embeddings, so a bank built with UNI and one built with
DINOv2 are indistinguishable on disk. With one hard-coded directory, running an ablation
over extractors either overwrites the previous one's banks or -- worse -- silently
retrieves against them: every candidate score is then a dot product between a DINOv2
query and UNI windows, which is a number, is not an error, and is meaningless.

Two things prevent that. The directory is keyed by extractor, so the three sets coexist;
and the directory carries a manifest naming its extractor, which the run checks. The
manifest is the one that matters -- a path can be pointed anywhere, and the check has to
survive that.
"""
from __future__ import annotations

import json

import pytest

from pyrafault.verify.bank import bank_dir, read_bank_manifest, write_bank_manifest


class TestBankDirIsKeyedByExtractor:
    def test_two_extractors_get_two_directories(self, tmp_path):
        assert bank_dir(tmp_path, "uni") != bank_dir(tmp_path, "dinov2")

    def test_the_name_carries_the_extractor(self, tmp_path):
        assert "uni" in bank_dir(tmp_path, "uni").name

    def test_it_sits_under_06_verify(self, tmp_path):
        assert bank_dir(tmp_path, "uni").parent.name == "06_verify"

    def test_the_same_extractor_always_resolves_to_the_same_path(self, tmp_path):
        assert bank_dir(tmp_path, "uni") == bank_dir(tmp_path, "uni")

    def test_an_extractor_name_that_would_escape_the_directory_is_refused(self, tmp_path):
        """`--extractor ../../etc` must not write outside the freeze."""
        for bad in ("../uni", "a/b", ""):
            with pytest.raises(ValueError):
                bank_dir(tmp_path, bad)


class TestTheManifestNamesTheExtractor:
    def test_a_written_manifest_reads_back(self, tmp_path):
        d = bank_dir(tmp_path, "uni")
        d.mkdir(parents=True)
        write_bank_manifest(d, extractor="uni", n_banks=7)
        assert read_bank_manifest(d)["extractor"] == "uni"

    def test_it_records_how_many_banks_it_describes(self, tmp_path):
        d = bank_dir(tmp_path, "dinov2")
        d.mkdir(parents=True)
        write_bank_manifest(d, extractor="dinov2", n_banks=7469)
        assert read_bank_manifest(d)["n_banks"] == 7469

    def test_the_manifest_is_not_mistaken_for_a_bank(self, tmp_path):
        """`_load_banks` globs `*.npz`; the manifest must not match it."""
        d = bank_dir(tmp_path, "uni")
        d.mkdir(parents=True)
        write_bank_manifest(d, extractor="uni", n_banks=0)
        assert list(d.glob("*.npz")) == []

    def test_a_directory_with_no_manifest_reads_as_none(self, tmp_path):
        d = bank_dir(tmp_path, "uni")
        d.mkdir(parents=True)
        assert read_bank_manifest(d) is None

    def test_an_unreadable_manifest_raises_rather_than_reading_as_absent(self, tmp_path):
        """Absent means "never written"; corrupt means something is wrong. Collapsing
        the two would let a damaged freeze pass as an un-migrated one."""
        d = bank_dir(tmp_path, "uni")
        d.mkdir(parents=True)
        (d / "_manifest.json").write_text("{not json")
        with pytest.raises(ValueError):
            read_bank_manifest(d)

    def test_rewriting_updates_the_count(self, tmp_path):
        d = bank_dir(tmp_path, "uni")
        d.mkdir(parents=True)
        write_bank_manifest(d, extractor="uni", n_banks=1)
        write_bank_manifest(d, extractor="uni", n_banks=2)
        assert read_bank_manifest(d)["n_banks"] == 2
