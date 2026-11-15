import hashlib

import pandas as pd
import pytest

from pyrafault.source_directory import (
    build_slides_frame, discover_slides, resolve_case_ids, slide_id_from_path,
)
from pyrafault.source import SourceError


def _write(tmp_path, relpath: str, content: bytes = b"stub-wsi-bytes"):
    p = tmp_path / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    return p


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def test_discover_slides_finds_files_matching_pattern_recursively(tmp_path):
    _write(tmp_path, "caseA/slide1.svs")
    _write(tmp_path, "caseB/nested/slide2.svs")
    _write(tmp_path, "caseB/notes.txt")

    found = discover_slides(tmp_path, "*.svs")
    names = sorted(p.name for p in found)
    assert names == ["slide1.svs", "slide2.svs"]


def test_discover_slides_raises_when_pattern_matches_nothing(tmp_path):
    _write(tmp_path, "caseA/slide1.svs")
    with pytest.raises(SourceError, match="no files"):
        discover_slides(tmp_path, "*.ndpi")


def test_discover_slides_empty_directory_raises(tmp_path):
    with pytest.raises(SourceError, match="no files"):
        discover_slides(tmp_path, "*.svs")


# ---------------------------------------------------------------------------
# slide_id derivation
# ---------------------------------------------------------------------------

def test_slide_id_from_path_strips_final_extension_only():
    assert slide_id_from_path("caseA/slide1.svs") == "slide1"
    assert slide_id_from_path("TCGA-44-2655-01Z-00-DX1.svs") == "TCGA-44-2655-01Z-00-DX1"


# ---------------------------------------------------------------------------
# case_id_from: three forms
# ---------------------------------------------------------------------------

def test_case_id_from_slide_id_copies_slide_id():
    resolved = resolve_case_ids(["slideA", "slideB"], "slide_id")
    assert resolved == {"slideA": "slideA", "slideB": "slideB"}


def test_case_id_from_regex_extracts_capture_group():
    resolved = resolve_case_ids(
        ["TCGA-44-2655-01Z-00-DX1", "TCGA-44-2655-02Z-00-DX2", "TCGA-55-8094-01Z-00-DX1"],
        r"^(TCGA-[^-]+-[^-]+)",
    )
    assert resolved == {
        "TCGA-44-2655-01Z-00-DX1": "TCGA-44-2655",
        "TCGA-44-2655-02Z-00-DX2": "TCGA-44-2655",
        "TCGA-55-8094-01Z-00-DX1": "TCGA-55-8094",
    }


def test_case_id_from_csv_maps_slide_id_to_case_id(tmp_path):
    csv = tmp_path / "case_map.csv"
    csv.write_text("slide_id,case_id\nslideA,patient1\nslideB,patient2\n")
    resolved = resolve_case_ids(["slideA", "slideB"], str(csv))
    assert resolved == {"slideA": "patient1", "slideB": "patient2"}


# ---------------------------------------------------------------------------
# case_id_from: the two error paths, message must name the offending slide
# ---------------------------------------------------------------------------

def test_case_id_from_regex_raises_and_names_the_unmatched_slide():
    with pytest.raises(SourceError) as exc:
        resolve_case_ids(
            ["TCGA-44-2655-01Z-00-DX1", "not-a-tcga-barcode"],
            r"^(TCGA-[^-]+-[^-]+)",
        )
    assert "not-a-tcga-barcode" in str(exc.value)
    assert "TCGA-44-2655-01Z-00-DX1" not in str(exc.value)


def test_case_id_from_csv_raises_and_names_the_missing_slide(tmp_path):
    csv = tmp_path / "case_map.csv"
    csv.write_text("slide_id,case_id\nslideA,patient1\n")
    with pytest.raises(SourceError) as exc:
        resolve_case_ids(["slideA", "slideB"], str(csv))
    assert "slideB" in str(exc.value)


def test_case_id_from_regex_with_zero_capture_groups_is_rejected():
    with pytest.raises(SourceError, match="capture group"):
        resolve_case_ids(["slideA"], r"^TCGA-[^-]+-[^-]+")


def test_case_id_from_regex_with_two_capture_groups_is_rejected():
    with pytest.raises(SourceError, match="capture group"):
        resolve_case_ids(["TCGA-44-2655-01Z"], r"^(TCGA)-([^-]+)")


def test_case_id_from_csv_missing_file_raises():
    with pytest.raises(SourceError, match="not found"):
        resolve_case_ids(["slideA"], "does_not_exist.csv")


# ---------------------------------------------------------------------------
# build_slides_frame: end-to-end discovery + source_relpath + sha256 baseline
# ---------------------------------------------------------------------------

def test_build_slides_frame_source_relpath_is_relative_to_slides_dir(tmp_path):
    _write(tmp_path, "caseA/TCGA-44-2655-01Z-00-DX1.svs", b"A" * 10)
    _write(tmp_path, "caseB/nested/TCGA-55-8094-01Z-00-DX1.svs", b"B" * 20)

    df = build_slides_frame(tmp_path, "**/*.svs", "slide_id")
    df = df.set_index("slide_id")

    assert df.loc["TCGA-44-2655-01Z-00-DX1", "source_relpath"] == "caseA/TCGA-44-2655-01Z-00-DX1.svs"
    assert df.loc["TCGA-55-8094-01Z-00-DX1", "source_relpath"] == (
        "caseB/nested/TCGA-55-8094-01Z-00-DX1.svs"
    )
    assert df.loc["TCGA-44-2655-01Z-00-DX1", "file_size"] == 10
    assert df.loc["TCGA-55-8094-01Z-00-DX1", "file_size"] == 20
    assert df.loc["TCGA-44-2655-01Z-00-DX1", "sha256_local"] == hashlib.sha256(b"A" * 10).hexdigest()


def test_build_slides_frame_with_regex_case_id_from(tmp_path):
    _write(tmp_path, "TCGA-44-2655-01Z-00-DX1.svs")
    _write(tmp_path, "TCGA-44-2655-02Z-00-DX2.svs")
    _write(tmp_path, "TCGA-55-8094-01Z-00-DX1.svs")

    df = build_slides_frame(tmp_path, "*.svs", r"^(TCGA-[^-]+-[^-]+)")
    mapping = dict(zip(df["slide_id"], df["case_id"]))
    assert mapping["TCGA-44-2655-01Z-00-DX1"] == "TCGA-44-2655"
    assert mapping["TCGA-44-2655-02Z-00-DX2"] == "TCGA-44-2655"
    assert mapping["TCGA-55-8094-01Z-00-DX1"] == "TCGA-55-8094"


def test_build_slides_frame_with_csv_case_id_from(tmp_path):
    _write(tmp_path, "slideA.svs")
    _write(tmp_path, "slideB.svs")
    csv = tmp_path / "case_map.csv"
    csv.write_text("slide_id,case_id\nslideA,patient1\nslideB,patient2\n")

    df = build_slides_frame(tmp_path, "*.svs", str(csv))
    mapping = dict(zip(df["slide_id"], df["case_id"]))
    assert mapping == {"slideA": "patient1", "slideB": "patient2"}


def test_build_slides_frame_raises_on_unmatched_regex_naming_the_slide(tmp_path):
    _write(tmp_path, "TCGA-44-2655-01Z-00-DX1.svs")
    _write(tmp_path, "not-a-tcga-barcode.svs")
    with pytest.raises(SourceError) as exc:
        build_slides_frame(tmp_path, "*.svs", r"^(TCGA-[^-]+-[^-]+)")
    assert "not-a-tcga-barcode" in str(exc.value)


def test_build_slides_frame_raises_on_csv_missing_slide_naming_it(tmp_path):
    _write(tmp_path, "slideA.svs")
    _write(tmp_path, "slideB.svs")
    csv = tmp_path / "case_map.csv"
    csv.write_text("slide_id,case_id\nslideA,patient1\n")
    with pytest.raises(SourceError) as exc:
        build_slides_frame(tmp_path, "*.svs", str(csv))
    assert "slideB" in str(exc.value)


def test_build_slides_frame_pattern_matching_nothing_raises(tmp_path):
    _write(tmp_path, "slideA.svs")
    with pytest.raises(SourceError, match="no files"):
        build_slides_frame(tmp_path, "*.ndpi", "slide_id")


def test_build_slides_frame_duplicate_slide_id_raises(tmp_path):
    _write(tmp_path, "caseA/slide1.svs")
    _write(tmp_path, "caseB/slide1.svs")
    with pytest.raises(SourceError, match="duplicate"):
        build_slides_frame(tmp_path, "**/*.svs", "slide_id")
