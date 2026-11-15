#!/usr/bin/env python
"""Stage 08 — render frozen contexts and queries to PNG (derived data, deletable)."""
from __future__ import annotations

import argparse
from pathlib import Path

from pyrafault.cache import cache_key, cache_path, image_sha256, save_png
from pyrafault.config import load_config
from pyrafault.dataset import assert_spec_matches_config
from pyrafault.geom import BBoxL0
from pyrafault.io.read import read_at_mpp
from pyrafault.io.slide import open_slide
from pyrafault.faults import OUT_OF_CONTEXT_TYPES
from pyrafault.schema import read_table, write_table


def _materialize(slide, df, bbox_cols, cfg_target_mpp, cfg_output_size, cfg, out_root, id_col):
    """I2: target_mpp/output_size/extraction_version are read from each row (the
    canonical per-record extraction spec, not the live config); a row that disagrees
    with the config this run is using is a provenance failure, not a value to silently
    prefer either way."""
    paths, hashes = [], []
    for row in df.itertuples():
        box = BBoxL0(*(getattr(row, c) for c in bbox_cols))
        target_mpp, output_size, extraction_version = (
            row.target_mpp, row.output_size, row.extraction_version,
        )
        assert_spec_matches_config(f"{id_col}={getattr(row, id_col)}", target_mpp, output_size,
                                    extraction_version, cfg_target_mpp, cfg_output_size,
                                    cfg.extraction_version)
        key = cache_key(row.slide_id, box, target_mpp, output_size, extraction_version)
        path = cache_path(out_root, row.slide_id, key)
        if path.exists():
            from pyrafault.cache import load_png
            img = load_png(path)
        else:
            img, _ = read_at_mpp(slide, box, target_mpp, output_size, extraction_version)
            save_png(img, path)
        paths.append(str(path))
        hashes.append(image_sha256(img))
    return paths, hashes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/example10.yaml")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--slides", nargs="*", default=None)
    ap.add_argument("--gather", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(args.data_root)
    out_root = root / "03_materialized"
    shard = root / "05_qc" / "stage08"
    shard.mkdir(parents=True, exist_ok=True)

    ctx_path = root / "01_benchmark" / "contexts.parquet"
    qry_path = root / "01_benchmark" / "queries.parquet"

    if args.gather:
        import pandas as pd
        contexts = read_table(ctx_path, "contexts")
        queries = read_table(qry_path, "queries")
        for name, df, idc in (("ctx", contexts, "context_id"), ("qry", queries, "query_id")):
            parts = pd.concat([pd.read_parquet(p) for p in sorted(shard.glob(f"*__{name}.parquet"))])
            m = dict(zip(parts[idc], parts["materialized_path"]))
            h = dict(zip(parts[idc], parts["image_sha256"]))
            df["materialized_path"] = df[idc].map(m)
            df["image_sha256"] = df[idc].map(h)
        write_table(contexts, ctx_path, "contexts")
        write_table(queries, qry_path, "queries")
        print(f"materialized {contexts['image_sha256'].notna().sum()} contexts, "
              f"{queries['image_sha256'].notna().sum()} queries")
        return

    import pandas as pd
    slides = read_table(root / "01_benchmark" / "slides.parquet", "slides")
    contexts = read_table(ctx_path, "contexts")
    queries = read_table(qry_path, "queries")
    targets = args.slides or sorted(contexts["slide_id"].unique())

    # Out-of-context pairs serve a patch the queries table never lists -- its source is
    # outside the parent context, so `queries.true_bbox_l0` does not cover it. The cache
    # is content-addressed by bbox, so leaving these out is not a correctness bug (the
    # loader just misses and reads the slide), but it means every evaluation run pays a
    # WSI read for a third of its items. Materialize them alongside the queries.
    pairs_path = root / "01_benchmark" / "pairs.parquet"
    ooc = pd.DataFrame()
    if pairs_path.exists():
        allp = read_table(pairs_path, "pairs")
        ooc = allp[allp["fault_type"].isin(OUT_OF_CONTEXT_TYPES)].drop_duplicates(
            subset=["slide_id", "true_bbox_l0_x", "true_bbox_l0_y",
                    "true_bbox_l0_w", "true_bbox_l0_h"]
        )
        # _materialize reads the extraction spec off each row rather than off the live
        # config (I2), so these rows must carry one. It comes from the query the pair is
        # a request for -- the substituted patch is served at the query's own mpp, size
        # and extraction_version -- which keeps the spec a per-record fact, not a value
        # this stage invents from whatever config happens to be loaded.
        if len(ooc):
            ooc = ooc.merge(
                queries[["query_id", "target_mpp", "output_size", "extraction_version"]],
                on="query_id", how="left", validate="many_to_one",
            )
            missing_spec = ooc["target_mpp"].isna()
            if missing_spec.any():
                raise SystemExit(
                    "out-of-context pairs reference query_ids absent from queries.parquet: "
                    f"{sorted(ooc.loc[missing_spec, 'pair_id'])[:5]}"
                )

    for slide_id in targets:
        srow = slides[slides["slide_id"] == slide_id].iloc[0]
        sc = contexts[contexts["slide_id"] == slide_id]
        sq = queries[queries["slide_id"] == slide_id]
        with open_slide(Path(cfg.source.root_dir) / srow["source_relpath"], slide_id) as slide:
            cp, ch = _materialize(slide, sc, ("bbox_l0_x", "bbox_l0_y", "bbox_l0_w", "bbox_l0_h"),
                                  cfg.context.target_mpp, cfg.context.size_px, cfg,
                                  out_root, "context_id")
            qp, qh = _materialize(slide, sq,
                                  ("true_bbox_l0_x", "true_bbox_l0_y",
                                   "true_bbox_l0_w", "true_bbox_l0_h"),
                                  cfg.query.target_mpp, cfg.query.size_px, cfg,
                                  out_root, "query_id")
            # Written to the same content-addressed cache the loader reads; no shard
            # column is emitted for them because no table keys on these boxes.
            so = ooc[ooc["slide_id"] == slide_id] if len(ooc) else ooc
            if len(so):
                _materialize(slide, so,
                             ("true_bbox_l0_x", "true_bbox_l0_y",
                              "true_bbox_l0_w", "true_bbox_l0_h"),
                             cfg.query.target_mpp, cfg.query.size_px, cfg,
                             out_root, "pair_id")
        pd.DataFrame({"context_id": sc["context_id"], "materialized_path": cp,
                      "image_sha256": ch}).to_parquet(shard / f"{slide_id}__ctx.parquet")
        pd.DataFrame({"query_id": sq["query_id"], "materialized_path": qp,
                      "image_sha256": qh}).to_parquet(shard / f"{slide_id}__qry.parquet")
        n_ooc = len(ooc[ooc["slide_id"] == slide_id]) if len(ooc) else 0
        print(f"{slide_id}: {len(cp)} contexts + {len(qp)} queries"
              + (f" + {n_ooc} out-of-context sources" if n_ooc else "") + " materialized")


if __name__ == "__main__":
    main()
