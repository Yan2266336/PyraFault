"""Running any verifier over a split and emitting the shared prediction record.

The work splits in two at a seam the task definition already draws. `localise` sees one
(C, Q) and nothing else, so it parallelises without coordination and costs essentially
everything; the decision is a pure function of (localisation, claim, tau, delta) and
costs nothing. `localise_items` is the first half and shards; `run_verifier` and
`fit_thresholds` are the second and take its output via `precomputed`. With no
`precomputed` they localise as they go, which is the single-process route and still the
only one a small freeze needs.
"""
from __future__ import annotations

import hashlib
import time

import pandas as pd

from pyrafault.geom import BBoxL0, center_distance
from pyrafault.verify.base import Localisation, decide
from pyrafault.verify.thresholds import Thresholds, fit_delta, fit_tau, residual_report

PREDICTION_COLUMNS = (
    "item_id",
    "estimated_bbox_l0_x", "estimated_bbox_l0_y",
    "estimated_bbox_l0_w", "estimated_bbox_l0_h",
    "reliability", "decision",
    "verifier_name", "verifier_revision", "thresholds_digest", "runtime_ms",
)

#: `PREDICTION_COLUMNS` minus everything a threshold determines. A shard writes this
#: before tau and delta exist, so `decision` and `thresholds_digest` cannot appear here:
#: a decision column in a localisation file would be a decision taken under thresholds
#: nobody had fitted yet.
LOCALISATION_COLUMNS = (
    "item_id",
    "estimated_bbox_l0_x", "estimated_bbox_l0_y",
    "estimated_bbox_l0_w", "estimated_bbox_l0_h",
    "reliability",
    "verifier_name", "verifier_revision", "runtime_ms",
    # Which (context, query) produced this row. Items are deduped by `cache_key`, so
    # many rows share one localisation; this names the one they share. It is what joins
    # a shard to the per-localisation reliability cache, where keeping a 128-decoy score
    # vector per *item* would be 500 GB and per *localisation* is 63 MB.
    "loc_key",
)

_BOX_COLUMNS = ("estimated_bbox_l0_x", "estimated_bbox_l0_y",
                "estimated_bbox_l0_w", "estimated_bbox_l0_h")


def loc_key(key: tuple) -> str:
    """A short stable name for a `cache_key`.

    The tuple itself is not something to put in a parquet column and join on; its
    digest is. Truncated to 16 hex characters: at 122,660 localisations per freeze a
    64-bit space leaves a collision probability around 4e-10, and a collision would
    merge two localisations' candidate scores, which the variant script would then
    notice as a reliability that does not reproduce the recorded one.
    """
    return hashlib.sha1(repr(key).encode()).hexdigest()[:16]


def cache_key(item, reads_claim: bool = False) -> tuple:
    """What `localise` actually depends on: the context and the patch served.

    Not the claim, unless the verifier declares `reads_claim` -- the task defines the
    estimate as independent of it, so for every method the task is about, including the
    claim here would cache nothing and including the item id would cache wrongly. A
    claim-conditioned baseline is the one case where the claim is part of the input, and
    there omitting it hands one item's answer to a different claim; see `Verifier.reads_claim`.

    `slide_id` (from `inputs`) stands in for the context's identity -- together with
    `context_bbox` it is what `inputs` exposes in place of a `context_id` it does not
    carry. `true_slide_id` (from `target`) is the *other* half: for an out-of-slide
    pair the served patch lives on a different slide than the context, so two
    out-of-slide items whose context and `true_bbox` coincide but whose source slides
    differ are still different (context, query) pairs and must not collide.

    `degradation_digest` is the third such half, and the one this key was missing.
    Degradation is drawn per item, so two items can share the slide, the context and the
    true box and still be served a 5 degree rotation and a 15 degree one. Without it the
    second item received the first's localisation, and the run looked healthy: one
    estimate per item, plausible reliabilities, no error anywhere. On the freezes on disk
    that was 1,784 of 8,697 keys for the degraded sample and 37,922 of 185,388 for the
    degraded full freeze, up to five parameter sets under a single key. An absent field
    reads as the empty string, which is also `none`'s digest, so an undegraded freeze's
    keys are exactly what they were.
    """
    target = item["target"]
    t = target["true_bbox"]
    i = item["inputs"]
    c = i["context_bbox"]
    base = (i["slide_id"], c.x, c.y, c.w, c.h, target["true_slide_id"],
            t.x, t.y, t.w, t.h, target.get("degradation_digest", ""))
    if not reads_claim:
        return base
    # A claim-conditioned baseline depends on the claim, so the claim is part of what
    # identifies its localisation. Without this, two items that differ only in where
    # they claim the query came from share one cached answer: on the five-cohort test
    # split that is 19,959 of 503,320 items silently scored against someone else's claim.
    b = i["claimed_bbox"]
    return base + (b.x, b.y, b.w, b.h)


def _with_perturbed_claim(inputs: dict) -> dict:
    """A copy of `inputs` whose `claimed_bbox` is shifted.

    Used to audit that `localise` really is independent of the claim: compute once,
    recompute under a different claim, and assert the two estimates agree. The shift
    is an arbitrary, data-independent offset -- large enough that a verifier which
    merely echoes the claim (or derives its estimate from it) cannot coincidentally
    still agree.
    """
    c = inputs["claimed_bbox"]
    return {**inputs, "claimed_bbox": BBoxL0(c.x + 99991, c.y + 99991, c.w, c.h)}


def _assert_claim_independent(verifier, fresh, loc) -> None:
    """Raises, rather than a bare `assert` -- the protocol promises this audit fails
    loudly, and a bare `assert` is stripped under `python -O`, which would silently
    turn the promise off."""
    if not (fresh.estimated_bbox == loc.estimated_bbox
            and fresh.reliability == loc.reliability):
        raise AssertionError(
            f"{verifier.name} produced a different estimate for the same "
            "(context, query) under a different claim: localise must not read "
            "the claim"
        )


def _meta(dataset, item_id):
    """The item's metadata, by the cheap path when the dataset has one.

    `PyraFaultDataset.meta` returns the item without decoding its two patches, which is
    what makes the deciding pass free. A dataset-like object without it -- the fakes in
    the tests, or a third party's own loader -- still works: the metadata is identical
    either way, it just arrives with pixels attached that nothing reads.
    """
    getter = getattr(dataset, "meta", None)
    return getter(item_id) if getter is not None else dataset[item_id]


def shard_of(item_ids, shard: int, n_shards: int, groups=None) -> list:
    """One shard of `item_ids`, partitioning the split exactly.

    Strided rather than contiguous so every shard draws from the whole split: blocks
    would hand one shard all of a single slide, and slides differ enough in tissue and
    patch size that the slowest block would decide the wall clock.

    `groups` is a parallel sequence of group keys, and when given it is the groups that
    are strided -- every item sharing a key lands in the same shard. Passing the context
    id buys two things at once on a real freeze. The runner's dedupe survives: items
    sharing one (C, Q) share a context by construction, so each distinct key is still
    localised once rather than up to 1.6 times. And a shard needs only its own
    contexts' window banks, which is what makes a ZoomTrace shard affordable -- the lung
    freeze's banks are 12.8 GB, and twenty jobs each reading all of them would spend
    more time on that than on localising.
    """
    if n_shards < 1:
        raise ValueError(f"n_shards must be >= 1, got {n_shards}")
    if not 0 <= shard < n_shards:
        raise ValueError(f"shard must be in [0, {n_shards}), got {shard}")
    item_ids = list(item_ids)
    if groups is None:
        return item_ids[shard::n_shards]
    groups = list(groups)
    if len(groups) != len(item_ids):
        raise ValueError(f"groups has {len(groups)} entries for {len(item_ids)} items")
    # Sorted before striding so the partition depends on the set of groups and not on
    # the order the table happened to arrive in: a shard must mean the same items on
    # every job of the array, and on a rerun.
    mine = set(sorted(set(groups))[shard::n_shards])
    return [i for i, g in zip(item_ids, groups) if g in mine]


def _refuse_audit_if_claim_conditioned(verifier, audit_independence: bool) -> None:
    """A claim-conditioned verifier contradicts the audit by construction.

    `_assert_claim_independent` would fire on the first item, and that failure reads as a
    bug in the verifier rather than as the one thing it declared about itself. Refusing
    at the entry point says which of the two the caller got wrong: the audit is not a
    property this verifier can be asked about.
    """
    if audit_independence and getattr(verifier, "reads_claim", False):
        raise ValueError(
            f"{verifier.name} declares reads_claim = True, so the independence audit "
            "cannot hold for it by construction. Drop --audit-independence for this "
            "verifier; it is a claim-conditioned baseline, not a method solving the task"
        )


def localise_items(verifier, dataset, item_ids=None,
                   audit_independence: bool = False,
                   reliability_cache: list | None = None) -> pd.DataFrame:
    """Localise `item_ids` (default: the whole split) and emit `LOCALISATION_COLUMNS`.

    The expensive half, with no threshold anywhere in it. Dedupes within the call by the
    same `(context, query)` key `run_verifier` uses, and runs the same independence
    audit when asked -- a verifier that reads the claim has to be caught here too, since
    this is where `localise` is actually called in a sharded run.
    """
    _refuse_audit_if_claim_conditioned(verifier, audit_independence)
    cache: dict = {}
    rows = []
    for item_id in (dataset.item_ids if item_ids is None else item_ids):
        item = dataset[item_id]
        inputs = item["inputs"]
        key = cache_key(item, verifier.reads_claim)
        if key in cache:
            loc, runtime_ms = cache[key]
            if audit_independence:
                _assert_claim_independent(verifier, verifier.localise(inputs), loc)
        else:
            t0 = time.perf_counter()
            loc = verifier.localise(inputs)
            runtime_ms = (time.perf_counter() - t0) * 1000
            if audit_independence:
                _assert_claim_independent(
                    verifier, verifier.localise(_with_perturbed_claim(inputs)), loc)
            cache[key] = (loc, runtime_ms)
            # One row per localisation, written on the miss -- the hits are the same
            # numbers again, and this vector is 128 floats wide.
            if reliability_cache is not None:
                d = loc.diagnostics or {}
                reliability_cache.append({
                    "loc_key": loc_key(key),
                    "s_star": float(d["s_star"]),
                    "candidate_scores": [float(x) for x in d["candidate_scores"]],
                    "candidate_origin_x": [int(o[0]) for o in d["candidate_origins"]],
                    "candidate_origin_y": [int(o[1]) for o in d["candidate_origins"]],
                    "best_index": int(d["best_index"]),
                    "bank_side": int(d["bank_side"]),
                    "decoy_scores": [float(x) for x in d["decoy_scores"]],
                    "n_competitors": int(d["n_competitors"]),
                    # The statistic this run actually shipped, kept beside its inputs so
                    # the offline recomputation can be checked against it rather than
                    # trusted.
                    "reliability_recorded": float(loc.reliability),
                    "reliability_mode": str(d["reliability_mode"]),
                })
        b = loc.estimated_bbox
        rows.append({
            "item_id": item_id,
            "loc_key": loc_key(key),
            "estimated_bbox_l0_x": None if b is None else b.x,
            "estimated_bbox_l0_y": None if b is None else b.y,
            "estimated_bbox_l0_w": None if b is None else b.w,
            "estimated_bbox_l0_h": None if b is None else b.h,
            "reliability": float(loc.reliability),
            "verifier_name": verifier.name,
            "verifier_revision": verifier.revision,
            "runtime_ms": runtime_ms,
        })
    return pd.DataFrame(rows, columns=list(LOCALISATION_COLUMNS))


def load_localisations(df: pd.DataFrame) -> dict:
    """One or more shards' `localise_items` output, as the `precomputed` map.

    Refuses a duplicated `item_id`: two overlapping shards would mean one item decided
    twice, silently, under whichever row happened to land last. The strided partition
    cannot produce that, so this only ever fires on a set of files assembled by hand.
    """
    dup = df["item_id"][df["item_id"].duplicated()].unique()
    if len(dup):
        raise SystemExit(
            f"{len(dup)} duplicate item_id(s) across the localisation shards, "
            f"e.g. {sorted(dup)[:3]} -- the shards overlap, so some items would be "
            "decided twice"
        )
    out = {}
    for r in df.itertuples():
        x = getattr(r, "estimated_bbox_l0_x")
        box = None if pd.isna(x) else BBoxL0(
            *(int(getattr(r, c)) for c in _BOX_COLUMNS))
        out[r.item_id] = (Localisation(box, float(r.reliability)),
                          float(r.runtime_ms))
    return out


def localisation_provenance(df: pd.DataFrame) -> tuple[str, str]:
    """The (name, revision) every shard agrees on, or a refusal.

    The deciding pass takes these from the shards rather than from a freshly built
    verifier, so a predictions table always records the revision that actually produced
    its estimates. Mixed values mean the directory holds shards from two different
    methods or two different revisions of one, which no amount of care downstream can
    make into a comparable row.
    """
    pairs = sorted(set(zip(df["verifier_name"], df["verifier_revision"])))
    if len(pairs) != 1:
        raise SystemExit(
            f"localisation shards disagree on which verifier produced them: {pairs}. "
            "Delete the stale shards and rerun that half"
        )
    return pairs[0]


class RecordedVerifier:
    """A verifier's identity without the verifier.

    What `run_verifier` and `fit_thresholds` need from one when every localisation is
    already computed: a name and a revision for the output rows. `localise` raises, so
    a code path that still expected to compute something fails here rather than
    quietly producing rows under a half-built method.
    """

    reads_claim = False          # 它不 localise，但属性要在，否则缓存分支会 AttributeError

    def __init__(self, name: str, revision: str) -> None:
        self.name, self.revision = name, revision

    def localise(self, inputs):
        raise AssertionError(
            f"{self.name}: localise was called on a recorded identity -- this run was "
            "supposed to decide from precomputed localisations"
        )


def _precomputed_or_raise(precomputed: dict, item_id: str):
    try:
        return precomputed[item_id]
    except KeyError:
        raise SystemExit(
            f"no localisation for item {item_id!r} -- the shards do not cover this "
            "split. Rerun the missing shard rather than deciding on a subset"
        ) from None


def run_verifier(verifier, dataset, thresholds: Thresholds,
                 audit_independence: bool = False,
                 precomputed: dict | None = None) -> pd.DataFrame:
    """Decide every item in `dataset`.

    With `precomputed` (a `load_localisations` map) this is the second half of a sharded
    run: it calls no `localise`, reads no pixels when the dataset offers `meta`, and
    carries each shard's `runtime_ms` through unchanged so the cost column still means
    what it means in a single-process run.
    """
    _refuse_audit_if_claim_conditioned(verifier, audit_independence)
    digest = thresholds.digest()
    cache: dict = {}
    rows = []

    for item_id in dataset.item_ids:
        if precomputed is not None:
            item = _meta(dataset, item_id)
            inputs = item["inputs"]
            loc, runtime_ms = _precomputed_or_raise(precomputed, item_id)
            rows.append(_prediction_row(item_id, loc, runtime_ms, inputs, thresholds,
                                        digest, verifier))
            continue
        item = dataset[item_id]
        inputs = item["inputs"]
        key = cache_key(item, verifier.reads_claim)

        if key in cache:
            loc, runtime_ms = cache[key]
            if audit_independence:
                # This (context, query) was already computed for another item under a
                # different real claim -- recomputing here and comparing is itself an
                # independence check, on top of the perturbed-claim check every first
                # computation gets below.
                fresh = verifier.localise(inputs)
                _assert_claim_independent(verifier, fresh, loc)
        else:
            t0 = time.perf_counter()
            loc = verifier.localise(inputs)
            runtime_ms = (time.perf_counter() - t0) * 1000
            if audit_independence:
                # A (context, query) that appears exactly once in the split would
                # otherwise never be audited -- the cache-hit check above only fires on
                # a collision. Recompute under a deliberately different claim so every
                # first computation is checked too.
                fresh = verifier.localise(_with_perturbed_claim(inputs))
                _assert_claim_independent(verifier, fresh, loc)
            cache[key] = (loc, runtime_ms)

        rows.append(_prediction_row(item_id, loc, runtime_ms, inputs, thresholds,
                                    digest, verifier))

    return pd.DataFrame(rows, columns=list(PREDICTION_COLUMNS))


def _prediction_row(item_id, loc, runtime_ms, inputs, thresholds, digest, verifier):
    query_side = inputs["claimed_bbox"].w
    b = loc.estimated_bbox
    return {
        "item_id": item_id,
        "estimated_bbox_l0_x": None if b is None else b.x,
        "estimated_bbox_l0_y": None if b is None else b.y,
        "estimated_bbox_l0_w": None if b is None else b.w,
        "estimated_bbox_l0_h": None if b is None else b.h,
        "reliability": float(loc.reliability),
        "decision": decide(loc, inputs["claimed_bbox"], query_side,
                           thresholds.tau, thresholds.delta),
        "verifier_name": verifier.name,
        "verifier_revision": verifier.revision,
        "thresholds_digest": digest,
        # The cost of *computing* this answer -- reused verbatim on every cache hit for
        # the same key, and carried through from the shard that computed it in a sharded
        # run, so this always means "what it cost to compute this answer", not "how long
        # this row's branch happened to take". Summing over distinct cache keys gives
        # total compute.
        "runtime_ms": runtime_ms,
    }


def fit_thresholds(verifier, dataset, split: str,
                   precomputed: dict | None = None) -> Thresholds:
    """Localise every item in `dataset` without deciding, and fit tau/delta from it.

    Dedupes by the same `(context, query)` cache key `run_verifier` uses -- `localise`
    is defined as independent of the claim, so several items backed by the same served
    patch (e.g. `clean`/`adjacent_drift`/`hard_mismatch`/`random_same_context` sharing
    one query) would otherwise be localised once each for no reason.

    Reads `target["claim_correct"]` to split items into clean (delta) and faulty
    (tau); reads `target["true_bbox"]` for the clean items' normalised localisation
    residual, `center_distance(estimated_bbox, true_bbox) / true_bbox.w`, skipping any
    clean item whose `estimated_bbox` is None. `fit_tau` raises if there are no faulty
    items in `dataset`; `fit_delta` raises if there are no clean ones -- both left to
    propagate, per the protocol.

    Asserts, for every clean item that contributes a residual, that `true_bbox.w`
    (what this residual is normalised by) equals `claimed_bbox.w` (what `decide`
    normalises by, via `query_side`) -- they are equal for every current fault type,
    but nothing checked that invariant before. If it ever stops holding, `delta` would
    mean a different tolerance at fit time than at decide time, silently.
    """
    scores, is_clean, clean_residuals = [], [], []
    estimates, claims = [], []           # for the agreement test tau needs
    cache: dict = {}
    for item_id in dataset.item_ids:
        if precomputed is not None:
            item = _meta(dataset, item_id)
            loc = _precomputed_or_raise(precomputed, item_id)[0]
        else:
            item = dataset[item_id]
            key = cache_key(item, verifier.reads_claim)
            if key in cache:
                loc = cache[key]
            else:
                loc = verifier.localise(item["inputs"])
                cache[key] = loc
        target = item["target"]
        clean = bool(target["claim_correct"])
        scores.append(loc.reliability)
        is_clean.append(clean)
        estimates.append(loc.estimated_bbox)
        claims.append(item["inputs"]["claimed_bbox"])
        if clean and loc.estimated_bbox is not None:
            true_bbox = target["true_bbox"]
            claimed_bbox = item["inputs"]["claimed_bbox"]
            if true_bbox.w != claimed_bbox.w:
                raise AssertionError(
                    f"{item_id}: fit_thresholds normalises the residual by "
                    f"true_bbox.w ({true_bbox.w}) but decide normalises by "
                    f"claimed_bbox.w ({claimed_bbox.w}) -- these must agree or "
                    "delta means a different tolerance at fit time than at decide time"
                )
            clean_residuals.append(center_distance(loc.estimated_bbox, true_bbox) / true_bbox.w)

    # Delta first: tau's false-verification rate counts only wrong items whose estimate
    # AGREES with the claim, and agreement is defined against delta. Fitting tau on
    # admission alone makes a method pay for its own correct rejections -- measured on
    # the LoFTR run, an admitted rate of 10.3% against an accepted rate of 0.00%, which
    # drove tau above every score and abstained on everything.
    delta = fit_delta(clean_residuals, n_clean=sum(is_clean))
    agrees = [
        est is not None and center_distance(est, claim) <= delta * claim.w
        for est, claim in zip(estimates, claims)
    ]
    tau = fit_tau(scores, is_clean, agrees)
    residuals = residual_report(clean_residuals)
    return Thresholds(tau=tau, delta=delta, fitted_on=split, n_items=len(scores),
                      residuals=residuals)
