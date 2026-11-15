import hashlib
import json

import pandas as pd
import pytest

from pyrafault.source import (
    SourceError, join_sources, load_cart_metadata, load_manifest, md5_file,
    resolve_source_file, slide_id_from_filename, verify_files,
)


def _manifest():
    return pd.DataFrame({
        "id": ["u1", "u2"],
        "filename": ["TCGA-55-8094-01Z-00-DX1.aaa.svs", "TCGA-44-2659-01Z-00-DX1.bbb.svs"],
        "source_relpath": ["u1/TCGA-55-8094-01Z-00-DX1.aaa.svs", "u2/TCGA-44-2659-01Z-00-DX1.bbb.svs"],
        "md5": ["m1", "m2"], "size": [10, 20], "state": ["validated", "validated"],
    })


def _sheet():
    return pd.DataFrame({
        "File ID": ["u1", "u2"],
        "File Name": ["TCGA-55-8094-01Z-00-DX1.aaa.svs", "TCGA-44-2659-01Z-00-DX1.bbb.svs"],
        "Project ID": ["TCGA-LUAD"] * 2,
        "Case ID": ["TCGA-55-8094", "TCGA-44-2659"],
        "Sample ID": ["TCGA-55-8094-01Z", "TCGA-44-2659-01Z"],
        "Tissue Type": ["Tumor"] * 2, "Tumor Descriptor": ["Primary"] * 2,
        "Specimen Type": ["Solid Tissue"] * 2, "Preservation Method": ["FFPE"] * 2,
    })


def _meta():
    return pd.DataFrame({
        "file_id": ["u1", "u2"],
        "gdc_case_uuid": ["cu1", "cu2"],
        "experimental_strategy": ["Diagnostic Slide"] * 2,
        "access": ["open"] * 2,
    })


def test_slide_id_strips_uuid_and_extension():
    assert slide_id_from_filename("TCGA-55-8094-01Z-00-DX1.8dc29615.svs") == "TCGA-55-8094-01Z-00-DX1"
    assert slide_id_from_filename("u1/TCGA-44-2659-01Z-00-DX1.b33f.svs") == "TCGA-44-2659-01Z-00-DX1"


def test_join_keeps_case_id_and_uuid_separate():
    j = join_sources(_manifest(), _sheet(), _meta())
    assert len(j) == 2
    assert set(j["case_id"]) == {"TCGA-55-8094", "TCGA-44-2659"}
    assert set(j["gdc_case_uuid"]) == {"cu1", "cu2"}
    assert "case_id" in j and "gdc_case_uuid" in j


def test_join_rejects_a_row_missing_from_the_sheet():
    sheet = _sheet().iloc[:1]
    with pytest.raises(SourceError, match="sample sheet"):
        join_sources(_manifest(), sheet, _meta())


def test_join_rejects_a_row_missing_from_metadata():
    with pytest.raises(SourceError, match="metadata"):
        join_sources(_manifest(), _sheet(), _meta().iloc[:1])


def test_md5_file(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello world")
    assert md5_file(p) == hashlib.md5(b"hello world").hexdigest()


def test_load_manifest_keeps_filename_and_adds_source_relpath(tmp_path):
    """`filename` must stay exactly what GDC published; `source_relpath` is the
    separately-named on-disk `<uuid>/<basename>` path used to actually find the
    file. Conflating the two would quietly cost the provenance audit trail."""
    p = tmp_path / "manifest.txt"
    p.write_text(
        "id\tfilename\tmd5\tsize\tstate\n"
        "61b92a3c-9098-4b5f-96c8-fa21f1861095\t"
        "TCGA-44-2659-01Z-00-DX1.b33f40e6-bc18-48b0-8dad-bdb0a6a200b5.svs\t"
        "c6aa3bc3e54ef3f16d63b86393606874\t99663537\treleased\n"
    )
    df = load_manifest(p)
    assert df.loc[0, "filename"] == "TCGA-44-2659-01Z-00-DX1.b33f40e6-bc18-48b0-8dad-bdb0a6a200b5.svs"
    assert df.loc[0, "source_relpath"] == (
        "61b92a3c-9098-4b5f-96c8-fa21f1861095/"
        "TCGA-44-2659-01Z-00-DX1.b33f40e6-bc18-48b0-8dad-bdb0a6a200b5.svs"
    )
    assert df["size"].dtype == "int64"


def test_load_cart_metadata_reads_case_uuid_from_associated_entities(tmp_path):
    """Real GDC cart exports nest the case UUID under `associated_entities[0].case_id`,
    not a top-level `cases` key. A row with neither is left null so `join_sources`
    is the thing that aborts, rather than this function silently defaulting it."""
    p = tmp_path / "metadata.cart.json"
    p.write_text(json.dumps([
        {
            "file_id": "u1",
            "associated_entities": [{
                "entity_submitter_id": "TCGA-55-8094-01Z-00-DX1",
                "entity_type": "slide",
                "case_id": "case-uuid-1",
                "entity_id": "e1",
            }],
            "experimental_strategy": "Diagnostic Slide",
            "access": "open",
        },
        {
            "file_id": "u2",
            "experimental_strategy": "Diagnostic Slide",
            "access": "open",
        },
    ]))
    df = load_cart_metadata(p)
    row1 = df.loc[df["file_id"] == "u1"].iloc[0]
    assert row1["gdc_case_uuid"] == "case-uuid-1"
    row2 = df.loc[df["file_id"] == "u2"].iloc[0]
    assert pd.isna(row2["gdc_case_uuid"])


def _fixture_tree(tmp_path, payloads):
    joined = join_sources(_manifest(), _sheet(), _meta())
    for rel, data in payloads.items():
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data)
    joined = joined.copy()
    joined["size"] = [len(payloads[r]) for r in joined["source_relpath"]]
    joined["md5"] = [hashlib.md5(payloads[r]).hexdigest() for r in joined["source_relpath"]]
    return joined


def test_verify_files_passes_on_correct_tree(tmp_path):
    payloads = {"u1/TCGA-55-8094-01Z-00-DX1.aaa.svs": b"A" * 10,
                "u2/TCGA-44-2659-01Z-00-DX1.bbb.svs": b"B" * 20}
    joined = _fixture_tree(tmp_path, payloads)
    out = verify_files(joined, tmp_path, verify_md5=True)
    assert list(out["source_relpath"]) == list(joined["source_relpath"])
    assert out["sha256_local"].str.len().eq(64).all()


def test_verify_files_aborts_on_size_mismatch(tmp_path):
    payloads = {"u1/TCGA-55-8094-01Z-00-DX1.aaa.svs": b"A" * 10,
                "u2/TCGA-44-2659-01Z-00-DX1.bbb.svs": b"B" * 20}
    joined = _fixture_tree(tmp_path, payloads)
    joined.loc[0, "size"] = 999
    with pytest.raises(SourceError, match="size"):
        verify_files(joined, tmp_path, verify_md5=True)


def test_verify_files_aborts_on_md5_mismatch(tmp_path):
    payloads = {"u1/TCGA-55-8094-01Z-00-DX1.aaa.svs": b"A" * 10,
                "u2/TCGA-44-2659-01Z-00-DX1.bbb.svs": b"B" * 20}
    joined = _fixture_tree(tmp_path, payloads)
    joined.loc[1, "md5"] = "deadbeef"
    with pytest.raises(SourceError, match="MD5"):
        verify_files(joined, tmp_path, verify_md5=True)


# ---------------------------------------------------------------------------
# resolve_source_file: exact filename or glob (spec §12)
# ---------------------------------------------------------------------------

def test_resolve_source_file_exact_name(tmp_path):
    (tmp_path / "gdc_manifest.2026-08-12.143046.txt").write_text("x")
    got = resolve_source_file(tmp_path, "gdc_manifest.2026-08-12.143046.txt")
    assert got == tmp_path / "gdc_manifest.2026-08-12.143046.txt"


def test_resolve_source_file_glob_matching_exactly_one(tmp_path):
    (tmp_path / "gdc_manifest.2026-09-01.091500.txt").write_text("x")
    got = resolve_source_file(tmp_path, "gdc_manifest.*.txt")
    assert got == tmp_path / "gdc_manifest.2026-09-01.091500.txt"


def test_resolve_source_file_zero_matches_raises_naming_pattern(tmp_path):
    with pytest.raises(SourceError, match="no file"):
        resolve_source_file(tmp_path, "gdc_manifest.*.txt")


def test_resolve_source_file_multiple_matches_raises_naming_both(tmp_path):
    (tmp_path / "gdc_manifest.2026-08-12.143046.txt").write_text("x")
    (tmp_path / "gdc_manifest.2026-09-01.091500.txt").write_text("x")
    with pytest.raises(SourceError) as exc_info:
        resolve_source_file(tmp_path, "gdc_manifest.*.txt")
    msg = str(exc_info.value)
    assert "gdc_manifest.2026-08-12.143046.txt" in msg
    assert "gdc_manifest.2026-09-01.091500.txt" in msg


def test_verify_files_aborts_on_missing_file(tmp_path):
    payloads = {"u1/TCGA-55-8094-01Z-00-DX1.aaa.svs": b"A" * 10}
    joined = _fixture_tree(tmp_path, {**payloads, "u2/TCGA-44-2659-01Z-00-DX1.bbb.svs": b"B" * 20})
    (tmp_path / "u2/TCGA-44-2659-01Z-00-DX1.bbb.svs").unlink()
    with pytest.raises(SourceError, match="missing"):
        verify_files(joined, tmp_path, verify_md5=False)
