import pandas as pd
import pyarrow as pa
import pytest

from pyrafault.schema import SCHEMAS, SchemaError, read_table, write_table

REQUIRED = {
    "slides": ["slide_id", "gdc_file_id", "case_id", "gdc_case_uuid", "md5_gdc",
               "mpp_x", "level_downsamples", "width_l0", "height_l0", "qc_status"],
    "contexts": ["context_id", "slide_id", "bbox_l0_x", "bbox_l0_y", "bbox_l0_w",
                 "bbox_l0_h", "target_mpp", "output_size", "tissue_fraction",
                 "quality_flags", "extraction_version", "split"],
    "queries": ["query_id", "context_id", "slide_id", "true_bbox_l0_x", "true_bbox_l0_y",
                "true_bbox_l0_w", "true_bbox_l0_h", "morphology_cluster",
                "diversity_score", "self_similarity", "extraction_version", "split"],
    "pairs": ["pair_id", "slide_id", "context_id", "query_id",
              "claimed_bbox_l0_x", "claimed_bbox_l0_y", "claimed_bbox_l0_w", "claimed_bbox_l0_h",
              "true_bbox_l0_x", "true_bbox_l0_y", "true_bbox_l0_w", "true_bbox_l0_h",
              "claim_correct", "fault_type", "fault_severity", "normalized_offset",
              "offset_frac_context", "iou_claim_true", "true_within_context",
              "local_recovery_possible", "hardness_score", "hard_negative_miner",
              "miner_model", "miner_revision", "candidate_rank", "hard_pool",
              "construction_seed", "pipeline_version", "split", "qc_status"],
    "splits": ["case_id", "slide_id", "split"],
}


@pytest.mark.parametrize("name,cols", REQUIRED.items())
def test_schema_has_the_spec_columns(name, cols):
    have = set(SCHEMAS[name].names)
    assert set(cols) <= have, f"{name} missing {set(cols) - have}"


def test_all_schemas_registered():
    assert set(SCHEMAS) == {"slides", "context_candidates", "contexts",
                            "query_candidates", "queries", "pairs", "splits"}


def test_ids_are_strings_and_bboxes_are_int64():
    p = SCHEMAS["pairs"]
    assert p.field("pair_id").type == pa.string()
    assert p.field("claimed_bbox_l0_x").type == pa.int64()
    assert p.field("claim_correct").type == pa.bool_()
    assert "quality_flags" not in p.names          # pairs carry flags per-region, not per-pair
    c = SCHEMAS["contexts"]
    assert c.field("quality_flags").type == pa.list_(pa.string())


def test_round_trip_preserves_values(tmp_path):
    df = pd.DataFrame({"case_id": ["TCGA-44-2659"], "slide_id": ["S1"], "split": ["dev"]})
    p = tmp_path / "splits.parquet"
    write_table(df, p, "splits")
    back = read_table(p, "splits")
    pd.testing.assert_frame_equal(df, back)


def test_missing_column_is_rejected(tmp_path):
    df = pd.DataFrame({"case_id": ["c"], "slide_id": ["s"]})   # no `split`
    with pytest.raises(SchemaError, match="split"):
        write_table(df, tmp_path / "x.parquet", "splits")


def test_extra_column_is_rejected(tmp_path):
    df = pd.DataFrame({"case_id": ["c"], "slide_id": ["s"], "split": ["dev"], "oops": [1]})
    with pytest.raises(SchemaError, match="oops"):
        write_table(df, tmp_path / "x.parquet", "splits")


def test_write_is_atomic_on_failure(tmp_path):
    p = tmp_path / "splits.parquet"
    df = pd.DataFrame({"case_id": ["c"], "slide_id": ["s"]})
    with pytest.raises(SchemaError):
        write_table(df, p, "splits")
    assert not p.exists()
    assert list(tmp_path.iterdir()) == []
