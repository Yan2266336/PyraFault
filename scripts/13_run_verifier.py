#!/usr/bin/env python
"""Stage 13 — run a verifier over a split and write predictions.

    python scripts/13_run_verifier.py --verifier zoomtrace --data-root data --split test
    python scripts/13_run_verifier.py --verifier my_pkg.my_module:MyVerifier --split test

`--verifier` is either `zoomtrace` (built here from its window banks and a matcher)
or `module:Class` naming any importable `Verifier` subclass, instantiated with no
arguments. Everything after construction -- fitting, deciding, writing -- is the same
code for both, which is what makes their outputs comparable.

`--fit-thresholds` fits tau and delta on the given split (calibration, per the
protocol) before deciding: it localises every item without deciding, fits the two
thresholds from those localisations, writes thresholds.json, and only then decides
and writes predictions -- so the same invocation that produces thresholds.json also
produces a predictions file for the split it was fitted on. Running a later, unfitted
split (e.g. test) omits the flag and loads the thresholds.json this step wrote.

A full-scale freeze does not fit in one process: ZoomTrace localises at ~240 ms and the
NCC control at ~360 ms, so the lung freeze's 298,080 calibration+test rows are 20 to 30
hours per method, past the partition's wall. `--localise-only --shard k --n-shards K`
runs the expensive half for one shard and writes it under
`06_verify/<method>/localisations_<split>/`; `--from-localisations` then reads every
shard at once and does the cheap half -- fit, decide, write -- in one process, reading
no pixels and building no matcher. The two routes produce the same predictions table,
which tests/test_verify_runner_sharded.py pins.

    for k in $(seq 0 15); do  # as array jobs
      python scripts/13_run_verifier.py --data-root data_lung --split calibration \
          --localise-only --shard "$k" --n-shards 16
    done
    python scripts/13_run_verifier.py --data-root data_lung --split calibration \
        --from-localisations --fit-thresholds
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import pandas as pd

import pyarrow  # noqa: F401  (must precede kornia)

from pyrafault.config import load_config
from pyrafault.dataset import PyraFaultDataset
from pyrafault.features.registry import get_extractor
from pyrafault.geom import BBoxL0
from pyrafault.schema import read_table
from pyrafault.verify.bank import WindowBank, bank_dir, bank_key, read_bank_manifest
from pyrafault.verify.base import Verifier
from pyrafault.verify.matchers import LoftrMatcher, RomaMatcher, SiftRansacMatcher
from pyrafault.verify.runner import (  # noqa: F401 (fit_thresholds re-exported for tests)
    RecordedVerifier, fit_thresholds, load_localisations, localisation_provenance,
    localise_items, run_verifier, shard_of,
)
from pyrafault.verify.thresholds import Thresholds
from pyrafault.verify.zoomtrace import ZoomTrace

# `roma` is listed here on every machine although RoMa itself lives in an optional
# overlay (requirements-roma.txt): `RomaMatcher` defers its import into `__init__`,
# so selecting it without the overlay fails at construction with a message saying
# how to get it, rather than at import time on machines that will never use it.
MATCHERS = {"loftr": LoftrMatcher, "roma": RomaMatcher, "sift": SiftRansacMatcher}


def _load_banks(bank_dir: Path, ctx_meta: dict) -> dict:
    """Load a window bank for every context in `ctx_meta`, and only those.

    The direction that matters is a context with no bank: without this check the run
    dies hours in with a bare `KeyError` from `zoomtrace.py` the moment `localise` looks
    one up, so it is caught here, at startup, as a named failure.

    A bank file naming a context outside `ctx_meta` used to raise too. It no longer does,
    because `ctx_meta` is now scoped to the split being run while the bank directory holds
    whatever has been built: the lung freeze has 24,113 contexts of which 16,644 are
    `dev`, so one directory legitimately carries banks for contexts this run never reads.
    Rejecting those would force a separate bank directory per split for no gain.
    """
    present = {p.stem for p in bank_dir.glob("*.npz")}
    missing = set(ctx_meta) - present
    if missing:
        sample = sorted(missing)[:10]
        raise SystemExit(
            f"{len(missing)} context(s) in this freeze have no window bank under "
            f"{bank_dir} (e.g. {sample}); stage 12 has no resume, so an interrupted "
            "build must be rerun from scratch before stage 13 can run"
        )
    bank = {}
    for cid in sorted(ctx_meta):
        bank[bank_key(*ctx_meta[cid])] = WindowBank.load(bank_dir / f"{cid}.npz")
    if not bank:
        raise SystemExit(f"no window banks under {bank_dir}; run stage 12 first")
    return bank


def load_plugin(spec: str) -> Verifier:
    """`module:Class` -> an instance of that `Verifier` subclass, built with no arguments.

    A verifier that needs configuration reads it itself (environment, a file it knows
    about) -- the harness passing method-specific arguments would make the command line
    per-method, which is exactly what this script exists not to be.
    """
    module_name, sep, class_name = spec.partition(":")
    if not sep or not module_name or not class_name:
        raise SystemExit(f"--verifier {spec!r}: expected 'zoomtrace' or 'module:Class'")
    cls = getattr(importlib.import_module(module_name), class_name, None)
    if not (isinstance(cls, type) and issubclass(cls, Verifier)):
        raise SystemExit(f"--verifier {spec!r}: {class_name} is not a Verifier subclass "
                         f"in {module_name}")
    return cls()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/example10.yaml")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--verifier", default="zoomtrace",
                    help="'zoomtrace', or 'module:Class' for any Verifier subclass")
    ap.add_argument("--matcher", default="loftr", choices=sorted(MATCHERS),
                    help="ZoomTrace only.")
    ap.add_argument("--extractor", default=None,
                    help="ZoomTrace only. Which feature extractor to retrieve with; "
                         "defaults to the config's `visual_sampling.extractor`. Banks "
                         "are per-extractor, so this must match the one stage 12 built "
                         "them with -- the run refuses if it does not.")
    ap.add_argument("--reliability", default="margin",
                    choices=sorted(ZoomTrace.RELIABILITY_MODES),
                    help="ZoomTrace only. 'margin' is the shipped statistic; the two "
                         "decoy modes reproduce the empirical null it replaced, for the "
                         "ablation. Each is a separate method: fit its own thresholds.")
    ap.add_argument("--top-k", type=int, default=None,
                    help="ZoomTrace only. How many retrieved windows the matcher sees. "
                         "The coarse-retrieval ablation of §5.1: a separate method with "
                         "its own fitted thresholds, not a tuning knob.")
    ap.add_argument("--no-scale-alignment", action="store_true",
                    help="ZoomTrace only, §5.1 ablation. Stop resizing the query to the "
                         "bank's window side, so retrieval and matching compare across "
                         "physical scales.")
    ap.add_argument("--no-refine", action="store_true",
                    help="ZoomTrace only, §5.1 ablation. Report the winning window's own "
                         "box instead of a continuous position from the "
                         "correspondences.")
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--audit-independence", action="store_true")
    ap.add_argument("--fit-thresholds", action="store_true",
                    help="Fit tau and delta on --split (calibration, per the "
                         "protocol) and write thresholds.json before deciding.")
    ap.add_argument("--out-dir", default=None,
                    help="Decide into this directory instead of the one derived from "
                         "the method flags. For a method whose identity is NOT "
                         "expressible as flags -- the reliability variants, whose "
                         "`reliability` column was substituted offline by "
                         "32_reliability_variants.py -- there is no flag combination "
                         "that names them, and deciding them without this would resolve "
                         "to the shipped directory and overwrite its predictions. Only "
                         "with --from-localisations: it names where to read and write, "
                         "not what to compute.")
    ap.add_argument("--reliability-cache", default=None,
                    choices=("candidates", "all"),
                    help="Record, per localisation, the inputs every reliability "
                         "statistic is computed from, into candidates_<split>/. "
                         "`candidates` keeps the top-k match scores and geometry, "
                         "which is everything the margin, the ordinary top-two margin "
                         "and the raw winning score need. `all` additionally scores "
                         "the 128-window decoy pool, which costs what the decoy modes "
                         "cost and MOVES this run's own estimates (RANSAC is "
                         "randomised and the call sequence changes), so it writes to "
                         "its own directory. Only with --localise-only.")
    ap.add_argument("--localise-only", action="store_true",
                    help="Run only the expensive half -- localise this shard's items "
                         "and write them under localisations_<split>/. Fits nothing "
                         "and decides nothing: no threshold exists yet.")
    ap.add_argument("--shard", type=int, default=0,
                    help="With --localise-only: which shard of the split to localise.")
    ap.add_argument("--n-shards", type=int, default=1,
                    help="With --localise-only: how many shards the split is cut into.")
    ap.add_argument("--no-resume", action="store_true",
                    help="With --localise-only: redo this shard even if a complete "
                         "file for it is already on disk.")
    ap.add_argument("--from-localisations", action="store_true",
                    help="Skip localising: read every shard under "
                         "localisations_<split>/ and fit/decide from those. Reads no "
                         "pixels and builds no matcher.")
    args = ap.parse_args()

    if args.out_dir and not args.from_localisations:
        raise SystemExit("--out-dir redirects the deciding half; it is not a way to "
                         "localise into an arbitrary directory, which would detach a "
                         "method's estimates from its name")
    if args.reliability_cache and not args.localise_only:
        raise SystemExit("--reliability-cache records what localisation computed, so "
                         "it belongs on the --localise-only half")
    if args.localise_only and args.from_localisations:
        raise SystemExit("--localise-only and --from-localisations are the two halves "
                         "of a sharded run; pass one or the other")
    if (args.shard, args.n_shards) != (0, 1) and not args.localise_only:
        raise SystemExit("--shard/--n-shards apply to --localise-only; the deciding "
                         "pass reads every shard at once and cannot be sharded")
    if args.from_localisations and args.audit_independence:
        # The audit calls localise a second time under a perturbed claim. There is no
        # localise here, and silently skipping an audit the caller asked for would be
        # worse than refusing: it belongs on the --localise-only half, where it runs.
        raise SystemExit("--audit-independence calls localise again, so it belongs on "
                         "the --localise-only half, not on --from-localisations")

    cfg = load_config(args.config)
    root = Path(args.data_root)

    if args.verifier == "zoomtrace":
        # Keyed by matcher as well as verifier name. tau and delta are fitted per
        # method, and a matcher change is a method change: LoFTR and SIFT reach
        # different score distributions, so their thresholds are not interchangeable.
        # Writing both into one directory would silently have the second run overwrite
        # the first's calibration and then decide against it.
        # Every ablation is a separate method, so every one gets its own directory and
        # fits its own tau and delta. Sharing a directory would let the second run
        # overwrite the first's calibration and then decide against it -- the same
        # failure the matcher and extractor suffixes already exist to prevent.
        suffix = "" if args.reliability == "margin" else f"_{args.reliability}"
        # A cached run is its own method. `candidates` only keeps numbers the shipped
        # run already computed and discarded, so its estimates should match -- but
        # "should" is not evidence, and the shipped predictions are not a thing to
        # overwrite on the strength of it. `all` certainly differs; see `record_decoys`.
        if args.reliability_cache:
            suffix += "_relcache" if args.reliability_cache == "candidates" else "_relcachez"
        if args.top_k is not None:
            suffix += f"_k{args.top_k}"
        if args.no_scale_alignment:
            suffix += "_noscale"
        if args.no_refine:
            suffix += "_norefine"
        # The extractor is part of the method, exactly as the matcher is: UNI and DINOv2
        # reach different score distributions, so their thresholds are not
        # interchangeable and their predictions must not land in one directory. Left out
        # of the name when it is the config's own, so existing paths do not move.
        ext = args.extractor or cfg.visual_sampling.extractor
        ext_suffix = "" if ext == cfg.visual_sampling.extractor else f"_{ext}"
        out_dir = (root / "06_verify"
                   / f"{ZoomTrace.name}_{args.matcher}{ext_suffix}{suffix}")
        expected_name = ZoomTrace.name
        verifier = None            # built below, once the shard's contexts are known
    else:
        verifier = load_plugin(args.verifier)
        if args.audit_independence and verifier.reads_claim:
            # Refused here rather than on the first item: the dataset below opens the
            # slide readers, and a run that cannot succeed should not pay for them.
            raise SystemExit(
                f"--verifier {args.verifier!r} declares reads_claim = True, so the "
                "independence audit cannot hold for it by construction; drop "
                "--audit-independence")
        if verifier.name == "window_bank" or verifier.name.startswith("zoomtrace"):
            raise SystemExit(f"--verifier {args.verifier!r}: name {verifier.name!r} "
                             "would write into a directory stage 12/13 own for ZoomTrace")
        out_dir = root / "06_verify" / verifier.name
        expected_name = verifier.name

    if args.out_dir:
        out_dir = Path(args.out_dir)
    loc_dir = out_dir / f"localisations_{args.split}"

    if args.localise_only:
        with PyraFaultDataset(root / "01_benchmark", cfg.source.root_dir,
                              split=args.split, materialize=True) as ds:
            # Grouped by context, not strided by item. `ds.pairs` is sorted the same way
            # `ds.item_ids` is, so this is the context of each item in order. Two
            # consequences, both load-bearing at full scale: the runner's dedupe survives
            # sharding (items sharing a served patch share a context), and a ZoomTrace
            # shard reads only its own contexts' banks instead of all 12.8 GB of them.
            context_of = dict(zip(ds.item_ids, ds.pairs["context_id"]))
            ids = shard_of(ds.item_ids, args.shard, args.n_shards,
                           groups=[context_of[i] for i in ds.item_ids])
            if not ids:
                raise SystemExit(
                    f"shard {args.shard}/{args.n_shards} of --split {args.split} is "
                    "empty -- there are fewer contexts than shards"
                )
            print(f"shard {args.shard}/{args.n_shards}: {len(ids)} of "
                  f"{len(ds.item_ids)} items in --split {args.split}", flush=True)
            out = loc_dir / f"{args.shard:05d}of{args.n_shards:05d}.parquet"
            cache_out = (out_dir / f"candidates_{args.split}"
                         / f"{args.shard:05d}of{args.n_shards:05d}.parquet"
                         if args.reliability_cache else None)
            # A shard that has its localisations but not its cache is not complete for
            # this invocation: resuming past it would leave a hole the variant script
            # cannot fill without the GPU back.
            if (not args.no_resume and _shard_is_complete(out, ids, expected_name)
                    and (cache_out is None or cache_out.exists())):
                print(f"already complete -> {out}")
                return
            if verifier is None:
                verifier = _build_zoomtrace(cfg, root, args,
                                            context_ids={context_of[i] for i in ids})
            rel_cache = [] if args.reliability_cache else None
            df = localise_items(verifier, ds, item_ids=ids,
                                audit_independence=args.audit_independence,
                                reliability_cache=rel_cache)
        loc_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        print(f"{len(df)} localisations -> {out}")
        if cache_out is not None:
            cache_out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rel_cache).to_parquet(cache_out, index=False)
            print(f"{len(rel_cache)} localisation caches -> {cache_out}")
        return

    precomputed = None
    if args.from_localisations:
        files = sorted(loc_dir.glob("*of*.parquet"))
        if not files:
            raise SystemExit(f"no localisation shards in {loc_dir} -- run "
                             "--localise-only for every shard first")
        # One partition at a time. A directory can hold shards from two runs -- a smoke
        # pass at --n-shards 400 leaves 00000of00400.parquet, then the real pass at 20
        # writes its own -- and reading both would overlap. `load_localisations` does
        # catch that, but "the shards overlap" is a confusing way to say "there is a file
        # here from a different run", so name the partitions instead.
        partitions = sorted({f.stem.split("of")[1] for f in files})
        if len(partitions) > 1:
            raise SystemExit(
                f"{loc_dir} holds shards from {len(partitions)} different partitions "
                f"({', '.join(partitions)}) -- delete the ones that are not from this "
                "run before deciding"
            )
        # Every shard the partition names must be present. The partition check above
        # catches shards from two different runs; it says nothing about one run's shards
        # being all there, and `pd.concat` of 78 files out of 80 raises nothing. The
        # result is a method's row computed on 97.5% of the split, reported as if it were
        # the split, with the shard count in a print line nothing downstream reads. One
        # task on a slow GPU hitting the wall, or one cancelled by hand, is enough.
        #
        # Named indices rather than a count, because the next question is always which
        # ones -- they are the shards to resubmit.
        want = int(partitions[0])
        have = {int(f.stem.split("of")[0]) for f in files}
        gaps = sorted(set(range(want)) - have)
        if gaps:
            raise SystemExit(
                f"{loc_dir} has {len(have)}/{want} shards; missing "
                f"{gaps if len(gaps) <= 20 else str(gaps[:20]) + ' …'}. Deciding now "
                f"would report this method on {len(have) / want:.1%} of --split "
                f"{args.split!r} as though it were the whole split. Rerun those shards "
                "with --localise-only --shard k --n-shards " + str(want) + "."
            )
        loc_df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        precomputed = load_localisations(loc_df)
        name, revision = localisation_provenance(loc_df)
        if name != expected_name:
            raise SystemExit(
                f"{loc_dir} holds localisations from {name!r}, but this invocation "
                f"names {expected_name!r} -- the directory is stale"
            )
        verifier = RecordedVerifier(name, revision)
        print(f"{len(precomputed)} localisations from {len(files)} shard(s) in "
              f"{loc_dir} ({name} rev {revision})")

    if verifier is None:
        # ZoomTrace deciding a split it has no shards for: it has to localise after all,
        # so the heavy half is built here over every context the split references.
        verifier = _build_zoomtrace(cfg, root, args)

    tpath = Path(args.thresholds or out_dir / "thresholds.json")

    if args.fit_thresholds:
        with PyraFaultDataset(root / "01_benchmark", cfg.source.root_dir,
                              split=args.split, materialize=True) as fit_ds:
            thresholds = fit_thresholds(verifier, fit_ds, args.split,
                                        precomputed=precomputed)
        out_dir.mkdir(parents=True, exist_ok=True)
        thresholds.save(tpath)
        print(f"fitted on {args.split} ({thresholds.n_items} items): "
             f"tau={thresholds.tau:.6f} delta={thresholds.delta:.6f} -> {tpath}")

    thresholds = Thresholds.load(tpath)

    with PyraFaultDataset(root / "01_benchmark", cfg.source.root_dir,
                          split=args.split, materialize=True) as ds:
        df = run_verifier(verifier, ds, thresholds,
                          audit_independence=args.audit_independence,
                          precomputed=precomputed)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"predictions_{args.split}.parquet"
    df.to_parquet(out, index=False)
    print(f"{len(df)} predictions -> {out}")
    print(df["decision"].value_counts().to_string())


def _check_bank_extractor(banks_at: Path, extractor: str) -> None:
    """Refuse to retrieve against banks built by a different extractor.

    A bank file records origins and embeddings and nothing about the model that produced
    them, so this is the only place the mismatch can be caught. Getting it wrong is
    silent: every candidate score becomes a dot product between this extractor's query
    vector and another's window vectors, which is a number and not an error.

    An unlabelled directory is refused too, not trusted. Banks built before the manifest
    existed are indistinguishable from another extractor's by construction -- that is the
    whole problem -- so they have to be relabelled deliberately rather than assumed.
    """
    manifest = read_bank_manifest(banks_at)
    if manifest is None:
        raise SystemExit(
            f"{banks_at} carries no {'_manifest.json'} saying which extractor built it. "
            f"Run stage 12 with --extractor {extractor}, or if these banks are known to "
            f"be {extractor}'s, write the manifest for them deliberately."
        )
    if manifest.get("extractor") != extractor:
        raise SystemExit(
            f"{banks_at} was built with extractor {manifest.get('extractor')!r} but this "
            f"run asks for {extractor!r}. Retrieval across two extractors is silent and "
            "meaningless; build the banks for this one with stage 12."
        )


def _shard_is_complete(path: Path, ids: list, expected_name: str) -> bool:
    """Whether `path` already holds this shard's localisations, in full.

    An array that hits the queue's wall clock is resubmitted for the indices that died,
    and nothing is gained by recomputing the ones that finished. "In full" is checked
    against the item ids this invocation would localise and against the method name, so
    a truncated file, a file from a different partition, or one from another method is
    recomputed rather than trusted. A readable-but-wrong file is the thing to fear here;
    an unreadable one is simply recomputed.
    """
    if not path.exists():
        return False
    try:
        df = pd.read_parquet(path, columns=["item_id", "verifier_name"])
    except Exception:
        return False
    return (set(df["item_id"]) == set(ids)
            and len(df) == len(ids)
            and set(df["verifier_name"]) == {expected_name})


def _build_zoomtrace(cfg, root: Path, args, context_ids: set | None = None) -> ZoomTrace:
    """The heavy half of a ZoomTrace run: the banks it needs, plus a matcher on the GPU.

    Skipped entirely when the run only decides from precomputed shards. `context_ids`
    narrows it further to one shard's contexts -- the whole split's banks are 12.8 GB on
    the lung freeze, and a twenty-way array that each read all of them would spend longer
    on that than on localising.
    """
    extractor_name = args.extractor or cfg.visual_sampling.extractor
    banks_at = bank_dir(root, extractor_name)
    _check_bank_extractor(banks_at, extractor_name)
    contexts = read_table(root / "01_benchmark" / "contexts.parquet", "contexts")
    # Only the contexts this split's pairs reference. Loading every context's bank
    # would demand banks for `dev` -- two thirds of them in the lung freeze -- which
    # no evaluation reads, and would hold them all in memory for nothing.
    pairs_all = read_table(root / "01_benchmark" / "pairs.parquet", "pairs")
    used = set(pairs_all.loc[pairs_all["split"] == args.split, "context_id"])
    if context_ids is not None:
        used &= set(context_ids)
    contexts = contexts[contexts["context_id"].isin(used)]
    if contexts.empty:
        raise SystemExit(f"no contexts for --split {args.split!r} in {root}")
    ctx_meta = {r.context_id: (r.slide_id, BBoxL0(r.bbox_l0_x, r.bbox_l0_y,
                                                  r.bbox_l0_w, r.bbox_l0_h))
                for r in contexts.itertuples()}
    print(f"{len(ctx_meta)} contexts in --split {args.split}")
    kw = {} if args.top_k is None else {"top_k": args.top_k}
    if getattr(args, "reliability_cache", None) == "all":
        kw["record_decoys"] = True
    return ZoomTrace(bank=_load_banks(banks_at, ctx_meta),
                     extractor=get_extractor(extractor_name, cfg.features),
                     matcher=MATCHERS[args.matcher](),
                     reliability=args.reliability,
                     scale_alignment=not args.no_scale_alignment,
                     refine=not args.no_refine, **kw)


if __name__ == "__main__":
    main()
