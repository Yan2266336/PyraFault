"""Parquet schemas. Declaring them once here is what keeps stage N and stage N+1
from silently disagreeing about a column name or dtype."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

S = pa.string()
I = pa.int64()
F = pa.float64()
B = pa.bool_()
LS = pa.list_(pa.string())


class SchemaError(ValueError):
    """The DataFrame does not match the declared schema."""


def _bbox(prefix: str) -> list[pa.Field]:
    return [pa.field(f"{prefix}_{k}", I) for k in ("x", "y", "w", "h")]


SLIDES = pa.schema([
    pa.field("slide_id", S), pa.field("gdc_file_id", S), pa.field("case_id", S),
    pa.field("gdc_case_uuid", S), pa.field("sample_id", S), pa.field("source_relpath", S),
    pa.field("file_name", S), pa.field("md5_gdc", S), pa.field("sha256_local", S),
    pa.field("file_size", I), pa.field("project_id", S), pa.field("tissue_type", S),
    pa.field("tumor_descriptor", S), pa.field("specimen_type", S),
    pa.field("preservation_method", S), pa.field("experimental_strategy", S),
    pa.field("mpp_x", F), pa.field("mpp_y", F), pa.field("objective_power", F),
    pa.field("vendor", S), pa.field("level_count", I),
    pa.field("level_dimensions", pa.list_(pa.list_(I))),
    pa.field("level_downsamples", pa.list_(F)),
    pa.field("width_l0", I), pa.field("height_l0", I),
    pa.field("qc_status", S), pa.field("exclusion_reason", S),
])

_EXTRACTION = [
    pa.field("target_mpp", F), pa.field("output_size", I),
    pa.field("native_level", I), pa.field("native_downsample", F),
    pa.field("resize_required", B), pa.field("extraction_version", S),
]

CONTEXT_CANDIDATES = pa.schema([
    pa.field("candidate_id", S), pa.field("slide_id", S), *_bbox("bbox_l0"),
    pa.field("tissue_fraction", F), pa.field("quality_flags", LS),
    pa.field("blur_var", F), pa.field("mean_intensity", F),
    pa.field("selected", B), pa.field("selection_rank", I),
])

CONTEXTS = pa.schema([
    pa.field("context_id", S), pa.field("slide_id", S), *_bbox("bbox_l0"),
    *_EXTRACTION,
    pa.field("tissue_fraction", F), pa.field("quality_flags", LS),
    pa.field("blur_var", F), pa.field("mean_intensity", F),
    pa.field("morphology_summary", S), pa.field("self_similarity", F),
    pa.field("materialized_path", S), pa.field("image_sha256", S),
    pa.field("split", S),
])

QUERY_CANDIDATES = pa.schema([
    pa.field("candidate_id", S), pa.field("context_id", S), pa.field("slide_id", S),
    *_bbox("bbox_l0"),
    pa.field("tissue_fraction", F), pa.field("quality_flags", LS),
    pa.field("blur_var", F), pa.field("mean_intensity", F),
    pa.field("morphology_cluster", I), pa.field("selected", B),
])

QUERIES = pa.schema([
    pa.field("query_id", S), pa.field("context_id", S), pa.field("slide_id", S),
    pa.field("candidate_id", S), *_bbox("true_bbox_l0"),
    *_EXTRACTION,
    pa.field("tissue_fraction", F), pa.field("quality_flags", LS),
    pa.field("blur_var", F), pa.field("mean_intensity", F),
    pa.field("morphology_cluster", I), pa.field("diversity_score", F),
    pa.field("self_similarity", F),
    pa.field("materialized_path", S), pa.field("image_sha256", S),
    pa.field("split", S),
])

PAIRS = pa.schema([
    pa.field("pair_id", S), pa.field("slide_id", S), pa.field("context_id", S),
    pa.field("query_id", S),
    *_bbox("claimed_bbox_l0"), *_bbox("true_bbox_l0"),
    pa.field("claim_correct", B), pa.field("fault_type", S),
    pa.field("fault_severity", S), pa.field("normalized_offset", F),
    pa.field("offset_frac_context", F), pa.field("iou_claim_true", F),
    pa.field("true_within_context", B), pa.field("local_recovery_possible", B),
    pa.field("hardness_score", F), pa.field("hard_negative_miner", S),
    pa.field("miner_model", S), pa.field("miner_revision", S),
    pa.field("candidate_rank", I), pa.field("hard_pool", S),
    # Where the served patch actually came from. Equal to `slide_id` for every fault
    # type except `out_of_slide`, whose whole point is that it is not -- the record
    # would otherwise have no way to say which slide `true_bbox_l0` indexes, and the
    # loader would read the substituted box against the wrong slide's coordinate frame.
    # `same_slide` is the proposal's stratification marker, stored rather than derived
    # so a pooled table keeps it after the slide columns are dropped.
    pa.field("true_slide_id", S), pa.field("same_slide", B),
    # The magnification combination this record was built under. Constant within one
    # freeze and therefore redundant with its config -- but three freezes pooled into
    # one analysis are not one config, and stratifying by scale ratio is exactly what
    # the proposal asks for. Note ratio alone does not identify a combination: 5x->20x
    # and 10x->40x both have ratio 4.0 and differ only in absolute resolution, which is
    # why both mpps are stored and not just their quotient.
    pa.field("context_mpp", F), pa.field("query_mpp", F), pa.field("scale_ratio", F),
    # The imaging degradation applied to the served patch, frozen per row. `kind` is a
    # name in `pyrafault.degrade.KINDS`; `params` is its JSON, replayed verbatim by the
    # loader, never re-drawn; `band` is the reportable label the analysis groups by.
    # A degradation is not a fault: a rotated clean patch still came from the claimed
    # box, so `claim_correct` is unaffected and these columns say nothing about it.
    pa.field("degradation_kind", S), pa.field("degradation_params", S),
    pa.field("degradation_band", S),
    # sha256 of the RGB buffer of the patch actually served, for the rows stage 09
    # sampled; "" for the rest. Hashing the buffer and not the file follows
    # `cache.image_sha256`: PNG bytes are not stable across Pillow/zlib versions, and a
    # library upgrade must not look like a corrupted freeze. Declared here with the
    # other three so Task 3's migration can write it; Task 6 adds the check that reads it.
    pa.field("served_pixel_sha256", S),
    pa.field("construction_seed", I), pa.field("pipeline_version", S),
    pa.field("split", S), pa.field("qc_status", S),
])

SPLITS = pa.schema([
    pa.field("case_id", S), pa.field("slide_id", S), pa.field("split", S),
])

SCHEMAS: dict[str, pa.Schema] = {
    "slides": SLIDES,
    "context_candidates": CONTEXT_CANDIDATES,
    "contexts": CONTEXTS,
    "query_candidates": QUERY_CANDIDATES,
    "queries": QUERIES,
    "pairs": PAIRS,
    "splits": SPLITS,
}


def _check(df: pd.DataFrame, schema: pa.Schema, name: str) -> None:
    want, have = set(schema.names), set(df.columns)
    missing, extra = want - have, have - want
    if missing:
        raise SchemaError(f"{name}: missing columns {sorted(missing)}")
    if extra:
        raise SchemaError(f"{name}: unexpected columns {sorted(extra)}")


def write_table(df: pd.DataFrame, path: str | Path, name: str) -> None:
    """Validate against the declared schema and write atomically.

    Atomic because a half-written Parquet that still parses is worse than no file:
    a later stage would silently consume a truncated benchmark.
    """
    if name not in SCHEMAS:
        raise SchemaError(f"unknown table {name!r}; known: {sorted(SCHEMAS)}")
    schema = SCHEMAS[name]
    _check(df, schema, name)

    table = pa.Table.from_pandas(df[schema.names], schema=schema, preserve_index=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp, compression="snappy")
    os.replace(tmp, path)


def read_table(path: str | Path, name: str) -> pd.DataFrame:
    if name not in SCHEMAS:
        raise SchemaError(f"unknown table {name!r}; known: {sorted(SCHEMAS)}")
    table = pq.read_table(path)
    _check(table.to_pandas(), SCHEMAS[name], name)
    return table.to_pandas()
