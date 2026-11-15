#!/usr/bin/env python
"""Stage 05 — embed query candidates, drop invalid ones, freeze 4 diverse queries per context."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pyrafault.config import load_config
from pyrafault.features.registry import describe, get_extractor
from pyrafault.geom import BBoxL0
from pyrafault.io.read import read_at_mpp
from pyrafault.io.slide import open_slide
from pyrafault.quality import evaluate
from pyrafault.sampling.diversity import cluster_stratified, self_similarity
from pyrafault.schema import read_table, write_table


def _colour_stats(arr: np.ndarray) -> dict[str, object]:
    """Descriptive colour statistics for morphology_summary — never a medical label (§8.2)."""
    a = arr.astype(np.float32) / 255.0
    mx, mn = a.max(axis=2), a.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    flat = a.reshape(-1, 3)
    return {"mean_rgb": [round(float(v), 4) for v in flat.mean(axis=0)],
            "std_rgb": [round(float(v), 4) for v in flat.std(axis=0)],
            "mean_saturation": round(float(sat.mean()), 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/example10.yaml")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--slides", nargs="*", default=None)
    ap.add_argument("--extractor", default=None, help="override visual_sampling.extractor")
    ap.add_argument("--gather", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(args.data_root)
    shard_dir = root / "05_qc" / "stage05"
    shard_dir.mkdir(parents=True, exist_ok=True)

    if args.gather:
        # I6: feature_manifest.json used to be written unconditionally by every array
        # task to one shared path -- ten concurrent non-atomic writers racing on the
        # same file. Each array task now writes its own per-slide descriptor
        # (shard_dir/{slide_id}__manifest.json); the gather phase is the one place
        # that reads all of them and produces the single canonical manifest, so the
        # README's "mixed-extractor tables are impossible to produce by accident"
        # claim is actually enforced rather than merely true by convention.
        manifest_paths = sorted(shard_dir.glob("*__manifest.json"))
        if not manifest_paths:
            raise RuntimeError(
                f"no per-slide feature manifests found under {shard_dir} "
                "-- run stage 05 (non-gather) for at least one slide first"
            )
        shard_manifests = {p.stem.removesuffix("__manifest"): json.loads(p.read_text())
                           for p in manifest_paths}
        first_slide, first_desc = next(iter(shard_manifests.items()))
        for slide_id, desc in shard_manifests.items():
            if desc != first_desc:
                raise RuntimeError(
                    "feature manifest disagreement across array tasks: "
                    f"{first_slide} recorded {first_desc}, but {slide_id} recorded {desc} "
                    "-- a mixed-extractor benchmark must never be produced silently"
                )
        (root / "04_features").mkdir(parents=True, exist_ok=True)
        (root / "04_features" / "feature_manifest.json").write_text(
            json.dumps(first_desc, indent=2), encoding="utf-8")
        print(f"feature manifest: {len(shard_manifests)} shard(s) agree, "
              f"extractor={first_desc.get('name')} revision={first_desc.get('revision')}")

        qs = pd.concat([pd.read_parquet(p) for p in sorted(shard_dir.glob("*__q.parquet"))])
        cands = pd.concat([pd.read_parquet(p) for p in sorted(shard_dir.glob("*__qc.parquet"))])
        write_table(qs, root / "01_benchmark" / "queries.parquet", "queries")
        write_table(cands, root / "01_benchmark" / "query_candidates.parquet", "query_candidates")

        ctxs = pd.concat([pd.read_parquet(p) for p in sorted(shard_dir.glob("*__ctx.parquet"))])
        contexts = read_table(root / "01_benchmark" / "contexts.parquet", "contexts")
        # A partial shard set (e.g. --slides limiting this gather to fewer slides than
        # were originally frozen) must never silently overwrite an already-populated
        # descriptor with NaN -- .map() on a smaller dict does exactly that. Refuse to
        # shrink the populated count instead of writing a regression quietly.
        before = {c: int(contexts[c].notna().sum()) for c in ("self_similarity", "morphology_summary")}
        for col in ("self_similarity", "morphology_summary"):
            contexts[col] = contexts["context_id"].map(dict(zip(ctxs["context_id"], ctxs[col])))
        after = {c: int(contexts[c].notna().sum()) for c in ("self_similarity", "morphology_summary")}
        shrunk = {c: (before[c], after[c]) for c in before if after[c] < before[c]}
        if shrunk:
            raise RuntimeError(
                f"stage 05 gather would shrink populated context descriptors: {shrunk} "
                "(before -> after) -- refusing to overwrite already-populated rows with "
                "NaN under a partial shard set"
            )
        write_table(contexts, root / "01_benchmark" / "contexts.parquet", "contexts")
        print(f"context descriptors: {int(contexts['self_similarity'].notna().sum())}"
              f"/{len(contexts)} contexts")
        per = qs.groupby("context_id").size()
        print(f"{len(qs)} queries across {len(per)} contexts (expect {cfg.query.num_per_context} each)")
        bad = per[per != cfg.query.num_per_context]
        if len(bad):
            print(f"WARNING: {len(bad)} contexts have the wrong query count: {bad.to_dict()}")
        return

    (root / "04_features").mkdir(parents=True, exist_ok=True)
    name = args.extractor or cfg.visual_sampling.extractor
    extractor = get_extractor(name, cfg.features)
    print(f"extractor={extractor.name} revision={extractor.revision} dim={extractor.dim}")

    slides = read_table(root / "01_benchmark" / "slides.parquet", "slides")
    contexts = read_table(root / "01_benchmark" / "contexts.parquet", "contexts")
    cands = read_table(root / "01_benchmark" / "query_candidates.parquet", "query_candidates")
    targets = args.slides or sorted(contexts["slide_id"].unique())

    for slide_id in targets:
        srow = slides[slides["slide_id"] == slide_id].iloc[0]
        sctx = contexts[contexts["slide_id"] == slide_id]
        scand = cands[cands["slide_id"] == slide_id].reset_index(drop=True)

        keep_mask = np.zeros(len(scand), dtype=bool)
        embeds = np.zeros((len(scand), extractor.dim), dtype=np.float32)
        blur = np.full(len(scand), np.nan)
        meanint = np.full(len(scand), np.nan)
        flags: list[list[str]] = [[] for _ in range(len(scand))]

        with open_slide(Path(cfg.source.root_dir) / srow["source_relpath"], slide_id) as slide:
            # Context pass first: the same open handle serves both, and the context images
            # give morphology_summary its colour half at no extra read (spec §10).
            ctx_imgs = [
                read_at_mpp(slide,
                            BBoxL0(c.bbox_l0_x, c.bbox_l0_y, c.bbox_l0_w, c.bbox_l0_h),
                            cfg.context.target_mpp, cfg.context.size_px,
                            cfg.extraction_version)[0]
                for c in sctx.itertuples()
            ]
            ctx_colour = [_colour_stats(np.asarray(im)) for im in ctx_imgs]
            ctx_embeds = (extractor.embed(ctx_imgs) if ctx_imgs
                          else np.zeros((0, extractor.dim), np.float32))
            del ctx_imgs

            batch_idx, batch_img = [], []
            qrec = None                 # the queries' own ExtractionRecord, not the context's

            def flush():
                if not batch_idx:
                    return
                vecs = extractor.embed(batch_img)
                for j, i in enumerate(batch_idx):
                    embeds[i] = vecs[j]
                batch_idx.clear()
                batch_img.clear()

            for i, row in scand.iterrows():
                box = BBoxL0(row.bbox_l0_x, row.bbox_l0_y, row.bbox_l0_w, row.bbox_l0_h)
                try:
                    img, rec = read_at_mpp(slide, box, cfg.query.target_mpp,
                                           cfg.query.size_px, cfg.extraction_version)
                except Exception as exc:
                    flags[i] = ["corrupt_read"]
                    print(f"  drop {row.candidate_id}: {type(exc).__name__}")
                    continue
                qrec = rec          # constant within a slide: same mpp, same output_size
                q = evaluate(np.asarray(img), cfg.quality)
                blur[i], meanint[i], flags[i] = q.blur_var, q.mean_intensity, q.flags
                if q.hard_fail:
                    continue
                keep_mask[i] = True
                batch_idx.append(i)
                batch_img.append(img)
                if len(batch_idx) >= cfg.features.batch_size:
                    flush()
            flush()

        scand["quality_flags"] = flags
        scand["blur_var"] = blur
        scand["mean_intensity"] = meanint
        scand["morphology_cluster"] = None
        scand["selected"] = False

        q_rows = []
        for ctx in sctx.itertuples():
            local = np.flatnonzero(
                (scand["context_id"].values == ctx.context_id) & keep_mask
            )
            if len(local) < cfg.query.num_per_context:
                print(f"  WARNING {ctx.context_id}: only {len(local)} valid candidates")
                continue
            sub = embeds[local]
            picks, labels, dist = cluster_stratified(
                sub, cfg.visual_sampling.n_clusters, cfg.seed
            )
            for li, lab in zip(local, labels):
                scand.at[li, "morphology_cluster"] = int(lab)
            ss = self_similarity(embeds[keep_mask],
                                 indices=[int(np.flatnonzero(np.flatnonzero(keep_mask) == local[p])[0])
                                          for p in picks],
                                 k=cfg.visual_sampling.self_similarity_k)
            for rank, p in enumerate(picks):
                gi = int(local[p])
                scand.at[gi, "selected"] = True
                row = scand.loc[gi]
                q_rows.append({
                    "query_id": f"{ctx.context_id}__q{rank}",
                    "context_id": ctx.context_id, "slide_id": slide_id,
                    "candidate_id": row.candidate_id,
                    "true_bbox_l0_x": row.bbox_l0_x, "true_bbox_l0_y": row.bbox_l0_y,
                    "true_bbox_l0_w": row.bbox_l0_w, "true_bbox_l0_h": row.bbox_l0_h,
                    "target_mpp": cfg.query.target_mpp, "output_size": cfg.query.size_px,
                    "native_level": qrec.native_level,
                    "native_downsample": qrec.native_downsample,
                    "resize_required": qrec.resize_required,
                    "extraction_version": cfg.extraction_version,
                    "tissue_fraction": row.tissue_fraction, "quality_flags": row.quality_flags,
                    "blur_var": row.blur_var, "mean_intensity": row.mean_intensity,
                    "morphology_cluster": int(labels[p]), "diversity_score": float(dist[p]),
                    "self_similarity": float(ss[rank]),
                    "materialized_path": None, "image_sha256": None,
                    "split": ctx.split,
                })

        # Context descriptors. The neighbour pool is this slide's frozen contexts:
        # unselected candidates carry no embedding, and reading them would double the
        # stage's context reads for a field that is descriptive only (spec §10, §8.3).
        k_ctx = max(1, min(cfg.visual_sampling.self_similarity_k, len(sctx) - 1))
        ss_ctx = (self_similarity(ctx_embeds, indices=list(range(len(sctx))), k=k_ctx)
                  if len(sctx) > 1 else np.zeros(len(sctx), dtype=np.float32))
        ctx_rows = []
        for i, c in enumerate(sctx.itertuples()):
            hist: dict[str, int] = {}
            for r in q_rows:
                if r["context_id"] == c.context_id:
                    key = str(r["morphology_cluster"])
                    hist[key] = hist.get(key, 0) + 1
            ctx_rows.append({
                "context_id": c.context_id,
                "self_similarity": float(ss_ctx[i]),
                "morphology_summary": json.dumps({**ctx_colour[i], "query_clusters": hist},
                                                 sort_keys=True),
            })
        pd.DataFrame(ctx_rows).to_parquet(shard_dir / f"{slide_id}__ctx.parquet")

        np.savez_compressed(
            root / "04_features" / f"{slide_id}.npz",
            candidate_id=scand["candidate_id"].to_numpy(),
            embedding=embeds, valid=keep_mask,
        )
        pd.DataFrame(q_rows).to_parquet(shard_dir / f"{slide_id}__q.parquet")
        scand.to_parquet(shard_dir / f"{slide_id}__qc.parquet")
        print(f"{slide_id}: {int(keep_mask.sum())}/{len(scand)} valid candidates "
              f"-> {len(q_rows)} queries")

        # I6: record the extractor this array task actually used, per slide -- one
        # small file per shard rather than every concurrent array task racing to
        # (over)write the single shared manifest. The gather phase reads these back,
        # requires unanimous agreement, and writes the one canonical manifest.
        (shard_dir / f"{slide_id}__manifest.json").write_text(
            json.dumps({**describe(extractor),
                        "pipeline_version": cfg.pipeline_version,
                        "config_digest": cfg.digest(),
                        # Per-section digests, so stage 09's config_drift check can be
                        # scoped to just the sections a stage reads instead of faulting
                        # on any change anywhere -- see check_config_provenance.
                        "section_digests": cfg.section_digests()}, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
