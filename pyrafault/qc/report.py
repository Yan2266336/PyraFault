"""Stage 09 checks: must-be-zero invariants (spec §8/09) and §13.4 statistics.

Nothing here mutates the benchmark. It reads the frozen tables, states what is
wrong, and lets the caller decide the exit code.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pyrafault.cache import image_sha256
from pyrafault.dataset import (
    INPUT_SCALAR_KEYS, INPUT_STRING_KEYS, assert_spec_matches_config, make_item_id,
)
from pyrafault.geom import BBoxL0, center_distance, centred_inset, fov_l0, iou
from pyrafault.io.read import read_at_mpp

CTX_PREFIX = "bbox_l0"
TRUE_PREFIX = "true_bbox_l0"
CLAIM_PREFIX = "claimed_bbox_l0"

# §13.5's string-field audit: the label vocabulary `pyrafault.faults` produces.
# Kept in sync by hand (there is no single source-of-truth constant upstream) --
# a fifth fault type added later must be added here too, or the audit is blind to it.
# Imported, not restated: a second hand-written copy of this tuple is exactly how a
# new fault type gets missed by the leakage audit without anything failing.
from pyrafault.faults import FAULT_TYPES, OUT_OF_CONTEXT_TYPES, OUT_OF_SLIDE_TYPES  # noqa: E402

# I3(c) fallback: spec D3's fixed severity bands (query widths). check_invariants
# accepts the live cfg.faults.severity_bands from stage 09 so a config edit is
# reflected in the check; this default only covers callers (tests) that don't pass one.
DEFAULT_SEVERITY_BANDS = {"small": (0.25, 0.75), "medium": (0.75, 2.0), "large": (2.0, 6.0)}


@dataclass(frozen=True)
class Violation:
    check: str
    table: str
    row_id: str
    detail: str


def _box(row: Any, prefix: str) -> BBoxL0:
    return BBoxL0(int(row[f"{prefix}_x"]), int(row[f"{prefix}_y"]),
                  int(row[f"{prefix}_w"]), int(row[f"{prefix}_h"]))


def _text(value: Any) -> str:
    """A cell as a string, with null (`None` or NaN) collapsing to `""`.

    The same null idiom `pyrafault.dataset._degradation` uses, because the two have to
    agree about what a partially-written row says: a column the loader reads as "no
    degradation" must not read here as a band named "nan".
    """
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return str(value)


def _degradation_violations(row: dict, pid: str) -> list[Violation]:
    """Does `degradation_band` describe the parameters it labels?

    The counterpart of `severity_band_mismatch` below, and for the same reason: a band
    is a *label* for a parameter set, and the spec's acceptance criterion ("the curve
    is monotone in band") is computed on the label, never on the angle. Nothing else
    would notice a row labelled `severe` that carries mild's 5 degrees -- the pixels,
    the frozen hash and the cache key would all be self-consistent with the parameters,
    and the row would simply be plotted on the wrong curve.

    `BAND_SPEC` is imported from `pyrafault.degrade_assign` rather than restating
    5/10/15 here: a second copy of the angles is how the check and the assignment drift
    apart. `KINDS` likewise, so a kind that `pyrafault.degrade.apply` would raise on at
    read time is caught here, where the freeze is, instead of mid-evaluation.
    """
    from pyrafault.degrade import KINDS
    from pyrafault.degrade_assign import BAND_SPEC

    v: list[Violation] = []
    band, kind = _text(row.get("degradation_band")), _text(row.get("degradation_kind"))
    raw = _text(row.get("degradation_params"))

    if kind not in KINDS:
        v.append(Violation("degradation_kind_unknown", "pairs", pid,
                           f"degradation_kind={kind!r} is not one of {list(KINDS)}"))
    params: dict | None
    try:
        params = json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        params = None
        v.append(Violation("degradation_params_unparseable", "pairs", pid,
                           f"degradation_params={raw!r} does not parse: {exc}"))

    spec = BAND_SPEC.get(band)
    if spec is None:
        v.append(Violation("degradation_band_unknown", "pairs", pid,
                           f"degradation_band={band!r} is not one of {sorted(BAND_SPEC)}"))
        return v
    if params is None:
        return v

    spec_kind, strength = spec
    if kind != spec_kind:
        v.append(Violation("degradation_band_mismatch", "pairs", pid,
                           f"band {band!r} is {spec_kind!r}, row says {kind!r}"))
    elif spec_kind == "none":
        # `none` is the in-freeze control; the whole claim made for it in the README is
        # that such a row is byte-identical to an undegraded build. Any parameter at all
        # on it contradicts that claim even when it is never read.
        if params:
            v.append(Violation("degradation_band_mismatch", "pairs", pid,
                               f"band 'none' is the identity transform but the row "
                               f"carries parameters {params}"))
    else:
        angle = params.get("angle_deg")
        if angle is None or not np.isclose(abs(float(angle)), strength, atol=1e-9):
            v.append(Violation("degradation_band_mismatch", "pairs", pid,
                               f"band {band!r} is {strength} degrees, row carries "
                               f"angle_deg={angle!r}"))
    return v


def _dupes(df: pd.DataFrame, id_col: str, table: str) -> list[Violation]:
    """Rows sharing an id are a violation even when that shared id is null.

    `value_counts()` defaults to `dropna=True`, which would make several rows with a
    null id invisible to this check -- exactly the "the check cannot see its own worst
    case" pattern already hit elsewhere on this plan (stage 04's zero-candidate blind
    spot, the extractor's "unknown" revision). `dropna=False` keeps a null id subject
    to the same duplicate check as any other value.
    """
    counts = df[id_col].value_counts(dropna=False)
    return [Violation("duplicate_id", table, str(i), f"appears {int(n)} times")
            for i, n in counts[counts > 1].items()]


def check_invariants(
    slides: pd.DataFrame,
    contexts: pd.DataFrame,
    queries: pd.DataFrame,
    pairs: pd.DataFrame,
    splits: pd.DataFrame,
    source_root: Path,
    require_hashes: bool = False,
    severity_bands: dict[str, tuple[float, float]] | None = None,
) -> list[Violation]:
    """Every violation the benchmark must have zero of. Order is check-by-check, not row-by-row."""
    v: list[Violation] = []
    source_root = Path(source_root)

    # --- identity -------------------------------------------------------------
    for df, col, name in ((slides, "slide_id", "slides"), (contexts, "context_id", "contexts"),
                          (queries, "query_id", "queries"), (pairs, "pair_id", "pairs")):
        v += _dupes(df, col, name)

    # --- source files ---------------------------------------------------------
    for s in slides.itertuples():
        if not (source_root / s.source_relpath).exists():
            v.append(Violation("missing_source", "slides", s.slide_id, s.source_relpath))

    # --- referential integrity ------------------------------------------------
    slide_ids, ctx_ids, qry_ids = (set(slides["slide_id"]), set(contexts["context_id"]),
                                   set(queries["query_id"]))
    for c in contexts.itertuples():
        if c.slide_id not in slide_ids:
            v.append(Violation("dangling_reference", "contexts", c.context_id,
                               f"slide_id={c.slide_id}"))
    for q in queries.itertuples():
        if q.context_id not in ctx_ids:
            v.append(Violation("dangling_reference", "queries", q.query_id,
                               f"context_id={q.context_id}"))
    for p in pairs.itertuples():
        for ref, pool, label in ((p.context_id, ctx_ids, "context_id"),
                                 (p.query_id, qry_ids, "query_id"),
                                 (p.slide_id, slide_ids, "slide_id")):
            if ref not in pool:
                v.append(Violation("dangling_reference", "pairs", p.pair_id, f"{label}={ref}"))

    # --- bounds ---------------------------------------------------------------
    dims = {s.slide_id: (int(s.width_l0), int(s.height_l0)) for s in slides.itertuples()}
    for df, id_col, prefix, table in ((contexts, "context_id", CTX_PREFIX, "contexts"),
                                      (queries, "query_id", TRUE_PREFIX, "queries")):
        for row in df.to_dict("records"):
            if row["slide_id"] not in dims:
                continue
            w, h = dims[row["slide_id"]]
            box = _box(row, prefix)
            if not box.within_bounds(w, h):
                v.append(Violation("out_of_bounds", table, row[id_col],
                                   f"{box} outside {w}x{h}"))

    # --- I3(a): bbox size must match its own magnification claim --------------
    # bbox_w == round(output_size * target_mpp / mpp_x) is the one invariant tying a
    # stored coordinate to its magnification claim. Reconstruction cannot catch a
    # violation here because it re-reads from the same bbox — this check is the only
    # one that would notice e.g. a context frozen with the wrong fov.
    mpp_x = {s.slide_id: float(s.mpp_x) for s in slides.itertuples()}
    for df, id_col, prefix, table in ((contexts, "context_id", CTX_PREFIX, "contexts"),
                                      (queries, "query_id", TRUE_PREFIX, "queries")):
        for row in df.to_dict("records"):
            mx = mpp_x.get(row["slide_id"])
            if mx is None or mx != mx:  # missing slide or NaN mpp
                continue
            box = _box(row, prefix)
            expected_w = fov_l0(int(row["output_size"]), float(row["target_mpp"]), mx)
            if box.w != expected_w:
                v.append(Violation("bbox_mpp_mismatch", table, row[id_col],
                                   f"bbox_w={box.w} but round(output_size*target_mpp/mpp_x)="
                                   f"{expected_w}"))

    # --- containment and labels ----------------------------------------------
    ctx_boxes = {c["context_id"]: _box(c, CTX_PREFIX) for c in contexts.to_dict("records")}
    ctx_split = dict(zip(contexts["context_id"], contexts["split"]))
    qry_split = dict(zip(queries["query_id"], queries["split"]))
    slide_split = dict(zip(splits["slide_id"], splits["split"]))
    # I3(b): the query's own frozen true_bbox_l0 is the only reference today; a pair's
    # copy of that box has never been checked against it.
    qry_true = {q["query_id"]: _box(q, TRUE_PREFIX) for q in queries.to_dict("records")}
    bands = severity_bands or DEFAULT_SEVERITY_BANDS

    for q in queries.to_dict("records"):
        parent = ctx_boxes.get(q["context_id"])
        if parent is not None and not parent.contains(_box(q, TRUE_PREFIX)):
            v.append(Violation("query_outside_context", "queries", q["query_id"],
                               f"{_box(q, TRUE_PREFIX)} not inside {parent}"))

    for p in pairs.to_dict("records"):
        pid = p["pair_id"]
        claimed, true = _box(p, CLAIM_PREFIX), _box(p, TRUE_PREFIX)
        same = claimed == true
        correct = bool(p["claim_correct"])
        clean = p["fault_type"] == "clean"

        cross_slide = p["fault_type"] in OUT_OF_SLIDE_TYPES
        # A table predating the provenance columns cannot contain an out-of-slide pair
        # (the type did not exist), so their absence is answerable rather than a reason
        # to skip work: every row is same-slide. `write_table` still rejects such a
        # table, so this only ever sees one on a direct in-memory call.
        has_prov = "true_slide_id" in p
        if cross_slide and not has_prov:
            v.append(Violation("missing_provenance_columns", "pairs", pid,
                               "out-of-slide fault type in a table with no "
                               "true_slide_id column"))
        if cross_slide and has_prov:
            # The served patch is on another slide, so `true_bbox_l0` indexes a frame
            # this context knows nothing about. What must hold instead is that the
            # record says so, in both columns that describe it, and that the substitute
            # stayed inside the split -- a dev slide's pixels inside a test item would
            # undo the case-level split as surely as putting the slide itself there.
            if p["true_slide_id"] == p["slide_id"]:
                v.append(Violation("out_of_slide_same_slide", "pairs", pid,
                                   f"true_slide_id={p['true_slide_id']} equals slide_id"))
            if bool(p["same_slide"]):
                v.append(Violation("same_slide_flag", "pairs", pid,
                                   "same_slide is True on an out-of-slide pair"))
            src_split = slide_split.get(p["true_slide_id"])
            if src_split is not None and src_split != p["split"]:
                v.append(Violation("cross_split_substitution", "pairs", pid,
                                   f"patch taken from a {src_split} slide "
                                   f"({p['true_slide_id']}) for a {p['split']} pair"))
        elif has_prov and p["true_slide_id"] != p["slide_id"]:
            v.append(Violation("true_slide_id", "pairs", pid,
                               f"true_slide_id={p['true_slide_id']} != slide_id="
                               f"{p['slide_id']} on a same-slide fault type"))


        if correct != same:
            v.append(Violation("claim_correct", "pairs", pid,
                               f"claim_correct={correct} but claimed=={true if same else claimed}"))
        if correct != clean:
            v.append(Violation("claim_correct", "pairs", pid,
                               f"claim_correct={correct} but fault_type={p['fault_type']}"))

        # Both of these compare two boxes as if they shared a coordinate frame. Across
        # slides they do not: two level-0 boxes on different slides can overlap
        # numerically while denoting unrelated tissue, so recomputing IoU or a centre
        # distance would produce a number that means nothing and, worse, one that could
        # be averaged in with the same-slide values. The stored columns are null/zero by
        # construction for these rows and are checked as such instead.
        if cross_slide:
            if float(p["iou_claim_true"]) != 0.0:
                v.append(Violation("iou_mismatch", "pairs", pid,
                                   "iou_claim_true must be 0.0 across slides, got "
                                   f"{p['iou_claim_true']}"))
            for col in ("normalized_offset", "offset_frac_context"):
                if pd.notna(p[col]):
                    v.append(Violation(f"{col}_across_slides", "pairs", pid,
                                       f"{col}={p[col]} is a distance between boxes in "
                                       "two different slides' frames; must be null"))
        else:
            observed_iou = iou(claimed, true)
            if not np.isclose(observed_iou, float(p["iou_claim_true"]), atol=1e-6):
                v.append(Violation("iou_mismatch", "pairs", pid,
                                   f"stored {p['iou_claim_true']}, recomputed {observed_iou:.6f}"))

        # I3(b): for every in-context fault type the pair's true_bbox must equal the
        # query's own. An out-of-context pair breaks that by construction -- the patch
        # served is not the query's patch -- so the check inverts instead of lapsing:
        # its CLAIM must equal the query's true bbox (the request the agent made), and
        # its true_bbox must not. That keeps both halves of the record pinned to the
        # query, which is what I3(b) was protecting.
        q_true = qry_true.get(p["query_id"])
        if q_true is not None and not cross_slide:
            if p["fault_type"] in OUT_OF_CONTEXT_TYPES:
                if claimed != q_true:
                    v.append(Violation("claim_not_request", "pairs", pid,
                                       f"pairs.claimed_bbox_l0={claimed} != "
                                       f"queries.true_bbox_l0={q_true} — an out-of-context "
                                       "claim must be the query's own requested location"))
                if true == q_true:
                    v.append(Violation("true_bbox_mismatch", "pairs", pid,
                                       f"out-of-context pair serves the query's own patch {true}"))
            elif true != q_true:
                v.append(Violation("true_bbox_mismatch", "pairs", pid,
                                   f"pairs.true_bbox_l0={true} != "
                                   f"queries.true_bbox_l0={q_true} for query_id={p['query_id']}"))

        # I3(c): normalized_offset recomputed from the stored boxes, and for
        # adjacent_drift, that it falls inside its own recorded fault_severity band.
        recomputed_offset = (0.0 if cross_slide else
                             (center_distance(claimed, true) / true.w if true.w else 0.0))
        if not cross_slide and not np.isclose(
                recomputed_offset, float(p["normalized_offset"]), atol=1e-6):
            v.append(Violation("normalized_offset_mismatch", "pairs", pid,
                               f"stored {p['normalized_offset']}, "
                               f"recomputed {recomputed_offset:.6f}"))
        if p["fault_type"] == "adjacent_drift":
            severity = p.get("fault_severity")
            band = bands.get(severity)
            if band is not None:
                lo, hi = band
                if not (lo <= recomputed_offset < hi):
                    v.append(Violation(
                        "severity_band_mismatch", "pairs", pid,
                        f"normalized_offset={recomputed_offset:.6f} not in band "
                        f"[{lo}, {hi}) for fault_severity={severity!r}"))

        # Task 2: the same rule for the degradation label. Only for a table that
        # carries the columns -- a freeze built before them must still pass QC
        # unchanged, the same backward-compatibility rule `has_prov` follows above.
        if "degradation_band" in p:
            v += _degradation_violations(p, pid)

        parent = ctx_boxes.get(p["context_id"])
        if parent is not None and cross_slide and has_prov:
            # Only the claim is comparable; it lives in this slide's frame and must
            # still be inside the context, exactly as for every other fault type.
            if not parent.contains(claimed):
                v.append(Violation("claim_outside_context", "pairs", pid,
                                   f"{claimed} not inside {parent}"))
            for col in ("true_within_context", "local_recovery_possible"):
                if bool(p[col]):
                    v.append(Violation(col, "pairs", pid,
                                       f"{col} is True on an out-of-slide pair"))
        elif parent is not None:
            if not parent.contains(claimed):
                v.append(Violation("claim_outside_context", "pairs", pid,
                                   f"{claimed} not inside {parent}"))
            within = parent.contains(true)
            if bool(p["true_within_context"]) != within:
                v.append(Violation("true_within_context", "pairs", pid,
                                   f"flag={p['true_within_context']}, geometry={within}"))
            if bool(p["local_recovery_possible"]) != within:
                v.append(Violation("local_recovery_possible", "pairs", pid,
                                   f"flag={p['local_recovery_possible']}, geometry={within}"))

        expected = slide_split.get(p["slide_id"])
        for label, actual in (("pair", p["split"]),
                              ("context", ctx_split.get(p["context_id"])),
                              ("query", qry_split.get(p["query_id"]))):
            if actual is not None and expected is not None and actual != expected:
                v.append(Violation("split_mismatch", "pairs", pid,
                                   f"{label} split={actual}, slide split={expected}"))

    # --- materialization ------------------------------------------------------
    if require_hashes:
        for df, id_col, table in ((contexts, "context_id", "contexts"),
                                  (queries, "query_id", "queries")):
            missing = df[df["image_sha256"].isna() | (df["image_sha256"] == "")]
            v += [Violation("missing_hash", table, str(r[id_col]), "image_sha256 is null")
                  for r in missing.to_dict("records")]
    return v


# §13.5: a signal a verifier could compute from `inputs` alone (never `true_bbox`)
# must not beat this balanced accuracy at separating clean from not-clean.
LEAKAGE_THRESHOLD = 0.55


def _best_balanced_accuracy(values: Any, is_clean: Any) -> float:
    """The best balanced accuracy any single threshold on `values` achieves at
    separating `is_clean` from not, fit on the same data it scores.

    This is deliberately optimistic, not an estimate: the threshold is chosen after
    seeing the labels it is then scored against, so the result upper-bounds what an
    attacker gets out of sample rather than approximating it. A signal that cannot
    beat 0.55 even fit in-sample cannot be exploited out of sample. Do NOT "fix" this
    into a train/test split later -- that produces a smaller, more comforting number
    that no longer bounds the attack §13.5 exists to catch (spec §13.5).

    For a boolean signal (`values` in {0.0, 1.0}) this reduces to "the two ways
    round": the two candidate thresholds pick out "predict clean iff true" and its
    complement, and the max over both directions below covers each.
    """
    values = np.asarray(values, dtype=float)
    is_clean = np.asarray(is_clean, dtype=bool)
    n_pos, n_neg = int(is_clean.sum()), int((~is_clean).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    best = 0.5
    for t in np.unique(values):
        pred_clean = values <= t
        tpr = (pred_clean & is_clean).sum() / n_pos
        tnr = (~pred_clean & ~is_clean).sum() / n_neg
        ba = 0.5 * (tpr + tnr)
        best = max(best, ba, 1.0 - ba)
    return float(best)


def _string_leak_score(values: Iterable[Any], is_clean: Any,
                       candidates: Iterable[str] = FAULT_TYPES) -> float:
    """The best balanced accuracy any fault-type substring achieves as a boolean
    "field contains substring" predictor of `is_clean`.

    Deliberately substring-based, not `_best_balanced_accuracy`'s numeric threshold
    sweep: a high-cardinality field where (almost) every row's value is unique --
    `item_id` is the motivating case -- has no meaningful numeric ordering, and
    sweeping a threshold over an arbitrary per-row-unique encoding lets it overfit
    almost any label sequence purely by chance (with n distinct values you get n
    candidate splits of the data, no different from bisecting a shuffled deck until
    it happens to line up with the labels). Testing literal substring containment
    instead is exactly the property that mattered for the actual defect this audit
    exists to catch: `"__clean" in pair_id`.
    """
    values = [str(v) for v in values]
    is_clean = np.asarray(is_clean, dtype=bool)
    best = 0.5
    for sub in candidates:
        contains = np.array([sub in v for v in values], dtype=float)
        best = max(best, _best_balanced_accuracy(contains, is_clean))
    return best


def _reconstruct_input_string_field(
    name: str, pairs: pd.DataFrame, seed: int, extraction_version: str,
) -> list[str]:
    """Per-row values for a string field of `inputs`, reconstructed from the frozen
    tables (stage 09 never instantiates a `PyraFaultDataset`, so nothing here reads
    pixels or calls `__getitem__`).

    Deliberately a closed `if/elif` over `pyrafault.dataset.INPUT_STRING_KEYS` rather
    than a lookup table with a default: a field this function does not recognise
    raises rather than being silently skipped. `INPUT_STRING_KEYS` is the same
    constant `PyraFaultDataset` itself is checked against (spec §8), so a field added
    to `inputs` without a case added here fails loudly the next time stage 09 runs --
    a field the audit cannot see is exactly the one that will leak.
    """
    if name == "item_id":
        return [make_item_id(seed, pid) for pid in pairs["pair_id"]]
    if name == "slide_id":
        return list(pairs["slide_id"])
    if name == "split":
        return list(pairs["split"])
    if name == "extraction_version":
        return [extraction_version] * len(pairs)
    raise NotImplementedError(
        f"check_leakage does not know how to reconstruct inputs[{name!r}] from the "
        "frozen tables. It is listed in pyrafault.dataset.INPUT_STRING_KEYS but has "
        "no case in _reconstruct_input_string_field -- add one before shipping the "
        "field, or it goes unaudited (spec §13.5)."
    )


def _reconstruct_input_scalar_field(
    name: str, pairs: pd.DataFrame, contexts: pd.DataFrame, queries: pd.DataFrame,
) -> np.ndarray:
    """Per-row values for a numeric field of `inputs`, from the frozen tables.

    Closed `if/elif` for the same reason `_reconstruct_input_string_field` is: a scalar
    added to `inputs` without a case here fails the next stage-09 run rather than
    shipping unaudited. Values come from the per-record spec columns, which is what the
    loader serves -- auditing the live config instead would test the config against
    itself and miss a table whose records disagree with it.
    """
    if name == "context_mpp":
        by_ctx = dict(zip(contexts["context_id"], contexts["target_mpp"]))
        return np.asarray([float(by_ctx.get(c, np.nan)) for c in pairs["context_id"]])
    if name == "query_mpp":
        by_qry = dict(zip(queries["query_id"], queries["target_mpp"]))
        return np.asarray([float(by_qry.get(q, np.nan)) for q in pairs["query_id"]])
    raise NotImplementedError(
        f"{name!r} is in pyrafault.dataset.INPUT_SCALAR_KEYS but has no case in "
        "_reconstruct_input_scalar_field -- add one before shipping the field, or it "
        "goes unaudited (spec §13.5)."
    )


def check_leakage(
    contexts: pd.DataFrame,
    queries: pd.DataFrame,
    pairs: pd.DataFrame,
    slides: pd.DataFrame,
    query_target_mpp: float,
    query_size_px: int,
    seed: int,
    extraction_version: str,
    threshold: float = LEAKAGE_THRESHOLD,
) -> tuple[list[Violation], dict[str, float]]:
    """§13.5: the leakage audit. Every quantity here is computable from `inputs`
    alone -- never from `target`'s `true_bbox` or `pair_id`. Returns (violations,
    per-signal best balanced accuracy), so the caller can print/record the numbers
    regardless of whether any exceed threshold.

    Geometry signals (spec §13.5), fit with `_best_balanced_accuracy`:
      - on-grid: `(ℓ - C.origin - inset) mod q_fov == 0` under the *centred* inset
        (`centred_inset`) -- the grid-origin assumption an attacker without knowledge
        of the per-context jitter would make (spec §15.1). This is exactly the pre-fix
        defect, re-measured against the post-fix benchmark every freeze.
      - `‖centre(ℓ) − centre(C)‖ / ctx_fov`
      - shortest distance from ℓ to the C border, normalized by ctx_fov
      - ℓ's width and height -- constant by construction; a non-constant value would
        itself surface here as a defect, not just as an out-of-place statistic.

    String signals, fit with `_string_leak_score` (a fault-type substring, never a
    numeric threshold -- see that function's docstring for why): every name in
    `pyrafault.dataset.INPUT_STRING_KEYS` (currently `item_id`, `slide_id`, `split`,
    `extraction_version`), each reconstructed from the frozen tables by
    `_reconstruct_input_string_field` -- which raises rather than silently skipping a
    name it does not recognise, so a field added to `inputs` without teaching this
    audit how to reconstruct it fails the next stage-09 run instead of shipping
    unaudited. This is the check that would have caught `inputs["pair_id"]` -- the
    shipped defect this whole audit responds to -- geometry-only auditing is blind to
    a leak that lives in a string.

    This exists because §15.1's defect was found by measuring lift over the base rate,
    once, by hand -- a construction change that reintroduces a regularity (a different
    grid, a new fault type, a changed severity band, an identifier that embeds the
    label) would sail past every must-be-zero invariant in `check_invariants`, all of
    which the leaking benchmark passed. Measuring it every freeze is what makes the
    class of defect visible.
    """
    ctx_by_id = {c.context_id: BBoxL0(int(c.bbox_l0_x), int(c.bbox_l0_y),
                                      int(c.bbox_l0_w), int(c.bbox_l0_h))
                for c in contexts.itertuples()}
    mpp_x = {s.slide_id: float(s.mpp_x) for s in slides.itertuples()}

    is_clean = (pairs["fault_type"] == "clean").to_numpy()
    on_grid, centre_dist, border_dist, widths, heights = [], [], [], [], []
    for row in pairs.to_dict("records"):
        ctx = ctx_by_id.get(row["context_id"])
        claimed = _box(row, CLAIM_PREFIX)
        if ctx is None:
            on_grid.append(0.0); centre_dist.append(0.0); border_dist.append(0.0)
            widths.append(float(claimed.w)); heights.append(float(claimed.h))
            continue

        mx = mpp_x.get(row["slide_id"])
        q_fov = fov_l0(query_size_px, query_target_mpp, mx) if mx else claimed.w

        if q_fov > 0 and q_fov <= ctx.w and q_fov <= ctx.h:
            inset_x, inset_y = centred_inset(ctx.w, ctx.h, q_fov)
            row_on_grid = ((claimed.x - ctx.x - inset_x) % q_fov == 0
                          and (claimed.y - ctx.y - inset_y) % q_fov == 0)
        else:
            row_on_grid = False
        on_grid.append(1.0 if row_on_grid else 0.0)

        centre_dist.append(center_distance(claimed, ctx) / ctx.w if ctx.w else 0.0)

        border = min(claimed.x - ctx.x, ctx.x1 - claimed.x1,
                    claimed.y - ctx.y, ctx.y1 - claimed.y1)
        border_dist.append(border / ctx.w if ctx.w else 0.0)

        widths.append(float(claimed.w))
        heights.append(float(claimed.h))

    signals = {
        "on_grid": np.asarray(on_grid),
        "centre_distance": np.asarray(centre_dist),
        "border_distance": np.asarray(border_dist),
        "claimed_width": np.asarray(widths),
        "claimed_height": np.asarray(heights),
    }
    scores = {name: _best_balanced_accuracy(vals, is_clean) for name, vals in signals.items()}

    # String fields of `inputs` (spec §13.5): audit every name in
    # pyrafault.dataset.INPUT_STRING_KEYS, not a separately hand-copied list -- see
    # _reconstruct_input_string_field's docstring for why this indirection matters.
    # Numeric scalars of `inputs` (mc, mq): fit with the same best-threshold search the
    # geometry signals use. Constant within one freeze -> 0.500; the check earns its
    # keep the moment several magnification combinations share a table.
    scores.update({
        name: _best_balanced_accuracy(
            _reconstruct_input_scalar_field(name, pairs, contexts, queries), is_clean)
        for name in sorted(INPUT_SCALAR_KEYS)
    })

    scores.update({
        f"string_{name}": _string_leak_score(
            _reconstruct_input_string_field(name, pairs, seed, extraction_version),
            is_clean)
        for name in sorted(INPUT_STRING_KEYS)
    })

    # The degradation columns (Task 2). A band is not in `inputs` -- a verifier is
    # never told which one it got -- but it decides the pixels it receives, so a band
    # correlated with the label is recoverable without reading the answer key, which
    # is the same class of defect as the derivable lattice. Scored as an ordinal index
    # (not a string) so the threshold search sees the ordering the bands actually have;
    # BANDS is imported from pyrafault.degrade_assign rather than restated so the audit
    # can never drift out of sync with the ordering assign_bands actually uses. The
    # *kind* has no such ordering, so it is scored as a string signal like the other
    # string fields of `inputs`. Added only when the columns are present, so a freeze
    # built before them keeps the score key set it had.
    if "degradation_band" in pairs.columns:
        from pyrafault.degrade_assign import BANDS
        band_index = {band: i for i, band in enumerate(BANDS)}
        # An out-of-vocabulary band (a typo, a NaN, a half-written migration) is
        # excluded from this signal rather than mapped to a sentinel ordinal. Mapping it
        # to -1.0, as this did, hands the threshold search a fourth "band" that is
        # perfectly separable from the three real ones, so one corrupt cell can produce
        # a leakage violation that says nothing about the band assignment and sends the
        # reader looking at `assign_bands`. Such a cell is not silently tolerated: it is
        # a `degradation_band_unknown` violation from `check_invariants`, which fails the
        # same run -- the diagnosis belongs there, where the row id is named, not here in
        # an aggregate score. If NO row is in vocabulary the score falls back to 0.500.
        kept = np.asarray([str(b) in band_index for b in pairs["degradation_band"]],
                          dtype=bool)   # dtype pinned: an empty list is float64, which cannot mask
        band_values = np.asarray(
            [float(band_index[str(b)]) for b in pairs["degradation_band"][kept]])
        scores["degradation_band"] = _best_balanced_accuracy(band_values, is_clean[kept])
        scores["string_degradation_kind"] = _string_leak_score(
            [str(k) for k in pairs["degradation_kind"]], is_clean)

    base_rate_not_clean = float((~is_clean).mean()) if len(is_clean) else 0.0
    violations = [
        Violation("leakage", "pairs", name,
                  f"best-threshold balanced accuracy {score:.3f} exceeds "
                  f"{threshold} separating clean from not-clean "
                  f"(base rate not-clean={base_rate_not_clean:.3f})")
        for name, score in scores.items() if score > threshold
    ]
    return violations, scores


def check_config_provenance(archived_config_path: str | Path,
                            feature_manifest_path: str | Path,
                            live_sections: dict[str, str],
                            stage_sections: Sequence[str],
                            ) -> tuple[list[Violation], list[str]]:
    """Did the parts of the config a stage reads change since the freeze was built?

    The archived `config.yaml`, the feature manifest, and the config stage 09 is
    running under are three independent records of "what config built this
    benchmark" -- stage 00's copy, stage 05's snapshot, and whatever is live right
    now. They disagreeing is what task 22 calls config drift: a resize from
    num_per_slide=25/num_per_context=4/n_clusters=4 to 30/5/5 once landed after stage
    00 had already archived the old parameters, so every stage from 03 onward ran
    under numbers the archive never recorded.

    Scoped PER SECTION rather than over the whole config, because one digest cannot
    tell "the config changed" from "the config that mattered to this stage changed"
    -- and under a whole-config rule, adding a section stages 00-05 never read (e.g.
    a `degradation:` block) forces a rebuild from stage 00, re-embedding every query
    candidate with a vision transformer to move zero coordinates. `stage_sections`
    is the caller's declaration of which sections it actually depends on; a section
    outside that list changing anywhere else is none of this call's business.

    A section the live config has LOST -- one the archive and the manifest both record,
    but `live_sections` does not carry -- is a violation on the same footing as one that
    changed. It has to be checked explicitly: the three-way comparison below only
    compares the legs that are present, so a missing leg would otherwise read as one
    fewer opinion rather than as a record that disappeared. Note that this is only
    reachable for a section the CALLER declares: a caller that derives `stage_sections`
    from its own live config can never name a section that config has already lost,
    which is why stage 09 declares a static list (`scripts/09_run_qc.py`).

    `live_sections` is `Config.section_digests()`, NOT `Config.digest()`. A whole
    digest is a single opaque hash -- by design it cannot say *which* section
    changed, only that *something* did, so it cannot support "flag section X but not
    section Y" without either flagging everything (defeating the whole point of
    scoping) or nothing (silently dropping I7's live-vs-archived check). Per-section
    digests are the only representation that lets stage 09's own live config
    participate in a genuinely per-section three-way comparison, the same way it
    already did at the whole-config level before this task.

    Any input being unavailable is itself a provenance failure, not a reason to skip:
    if the archived config or the manifest that recorded what actually ran is
    missing, nobody can tell what built this benchmark, which is precisely the
    "cannot verify" state this check exists to turn into a violation rather than a
    silent pass. Mirrors `missing_source` above, which raises rather than skips when
    a referenced slide file is absent.

    Returns `(violations, unchecked)`. `unchecked` names sections in `stage_sections`
    that this call cannot vouch for either because the archived config never had
    them (a section added after the archive was frozen) or the manifest never
    recorded them (written before section digests existed -- a legacy manifest, or
    a "modern" one from before this particular section landed, i.e. it has
    `section_digests` but not yet an entry for this one). Those sections are
    reported, never silently treated as passing: re-running stage 05 to obtain a
    missing section digest is exactly the GPU cost this whole mechanism exists to
    avoid, and silence would be indistinguishable from an actual pass.

    A section CAN appear in both `violations` and `unchecked` at once: if `recorded`
    never captured it but `archived` and `live` disagree, that disagreement is real
    and worth flagging on its own terms -- two sources we DO have genuinely
    disagree -- even though the manifest never vouched for the section, so
    `unchecked` also names it. Being flagged does not mean "verified"; it means
    "caught anyway, despite one leg of the three-way comparison being unavailable".

    A LEGACY manifest (one with no `section_digests` at all -- which is every
    manifest this repository has on disk) is not handled by reporting every section
    as unchecked. Its whole-config digest is compared against the archive on its own
    terms, and the per-section comparison then runs with the manifest leg standing in
    as "stage 05 saw the archived config", so archived-vs-live is still checked
    section by section. `unchecked` therefore names only the sections the ARCHIVE
    never had: for every other section, two of the three legs are genuinely present
    and genuinely compared.
    """
    from pyrafault.config import load_config

    archived_config_path = Path(archived_config_path)
    feature_manifest_path = Path(feature_manifest_path)

    # Both are checked before returning, rather than short-circuiting on the first
    # miss, so a run where NEITHER exists is reported as two distinct problems, not
    # one -- each is independently "nobody can tell what built this benchmark".
    missing: list[Violation] = []
    if not archived_config_path.exists():
        missing.append(Violation(
            "config_drift", "config", str(archived_config_path),
            f"archived config not found at {archived_config_path}; "
            "cannot verify what built this benchmark"))
    if not feature_manifest_path.exists():
        missing.append(Violation(
            "config_drift", "config", str(feature_manifest_path),
            f"feature manifest not found at {feature_manifest_path}; "
            "cannot verify what built this benchmark"))
    if missing:
        return missing, []

    archived = load_config(archived_config_path)
    manifest = json.loads(feature_manifest_path.read_text())
    archived_sections = archived.section_digests()
    recorded = manifest.get("section_digests")

    violations: list[Violation] = []
    if recorded is None:
        # Legacy manifest: stage 05 ran before section digests existed, so all we
        # have is one whole-config digest. It is compared here on its own terms --
        # and then this falls THROUGH into the per-section loop below rather than
        # returning, with the manifest leg standing in as "stage 05 saw the archived
        # config, for every section it had". Returning here dropped the live leg
        # entirely: the whole-config comparison above only relates the archive to
        # stage 05, both of which are frozen records, so a config edited after the
        # freeze (the drift this check exists to catch, and the state every manifest
        # in this repository is in) passed silently with `unchecked` empty, i.e. with
        # the report saying nothing either.
        if manifest.get("config_digest") != archived.digest():
            violations.append(Violation(
                "config_drift", "config", str(feature_manifest_path),
                "legacy config_digest disagrees with the archived config"))
        recorded = archived_sections

    for section in stage_sections:
        values = {
            "archived": archived_sections.get(section),
            "feature_manifest": recorded.get(section),
            "live (stage 09)": live_sections.get(section) if live_sections else None,
        }
        present = {label: v for label, v in values.items() if v is not None}
        if len(present) > 1 and len(set(present.values())) > 1:
            detail = ", ".join(f"{label}={v[:12]}..." for label, v in present.items())
            violations.append(Violation(
                "config_drift", "config", str(feature_manifest_path),
                f"section {section!r}: {detail}"))
        elif live_sections and section not in live_sections and present:
            # A REMOVAL. Comparing only the legs that are present treats a section the
            # live config no longer has as "one fewer opinion", which is silence: the
            # archive and the manifest both remember the section, so two records say the
            # config stage 09 is running under is missing something they were built with.
            # Under the whole-config digest this faulted; per-section scoping lost it,
            # because a deleted section is also absent from the digests of the live
            # config that `stage_sections` used to be derived from.
            violations.append(Violation(
                "config_drift", "config", str(feature_manifest_path),
                f"section {section!r} is recorded by {', '.join(sorted(present))} but "
                "absent from the live config this stage is running under"))
    unchecked = sorted(s for s in stage_sections
                       if s not in archived_sections or s not in recorded)
    return violations, unchecked


def reconstruction_sample(df: pd.DataFrame, id_col: str, frac: float, seed: int) -> pd.DataFrame:
    """A seeded per-slide stratified sample — never a sample that misses a whole slide."""
    if df.empty or frac <= 0:
        return df.iloc[:0]
    rng = np.random.default_rng(seed)
    picks = []
    for slide_id, group in df.groupby("slide_id", sort=True):
        n = max(1, int(round(len(group) * frac)))
        order = rng.permutation(len(group))[:n]
        picks.append(group.iloc[np.sort(order)])
    return pd.concat(picks).sort_values(id_col).reset_index(drop=True)


def verify_reconstruction(
    slide,
    rows: Iterable[dict],
    bbox_prefix: str,
    id_col: str,
    table: str,
    cfg_target_mpp: float,
    cfg_output_size: int,
    cfg_extraction_version: str,
) -> list[Violation]:
    """Re-read pixels from coordinates alone and compare to the stored hash.

    I2: `target_mpp`/`output_size`/`extraction_version` are read from each row — the
    per-record extraction spec the row itself carries — not passed in as one uniform
    value for the whole batch. A row whose spec disagrees with the live config is a
    provenance failure and raises, rather than silently reading with whichever value
    happened to be passed by the caller.
    """
    v: list[Violation] = []
    for row in rows:
        target_mpp, output_size, extraction_version = (
            row["target_mpp"], row["output_size"], row["extraction_version"],
        )
        assert_spec_matches_config(f"{table} {row[id_col]}", target_mpp, output_size,
                                    extraction_version, cfg_target_mpp, cfg_output_size,
                                    cfg_extraction_version)
        stored = row.get("image_sha256")
        if not stored:
            continue
        img, _ = read_at_mpp(slide, _box(row, bbox_prefix), target_mpp,
                            output_size, extraction_version)
        got = image_sha256(img)
        if got != stored:
            v.append(Violation("hash_mismatch", table, str(row[id_col]),
                               f"stored {stored[:12]}..., re-read {got[:12]}..."))
    return v


def check_served_pixels(rows: Iterable[tuple[dict, Any]]) -> list[Violation]:
    """Compare each sampled row's frozen pixel hash against the patch served now.

    This is the hand-off guarantee in one function. A collaborator regenerates these
    pixels on their own machine; without this check a Pillow or numpy difference there
    produces a slightly different Q and both sides report numbers computed on different
    images, at a precision (median IoU 0.99) where that matters and with nothing raising.

    An empty hash is `served_pixel_unverified`, not a pass: a freeze with no hashes at
    all must not read as fully verified. Stage 09 does not call this on a row whose
    hash it is about to fill for the first time (see `scripts/09_run_qc.py`) -- this
    branch exists for a row whose hash SHOULD already be frozen and is not, which a
    fresh, never-yet-sampled row is not.

    That makes the branch DEFENSIVE ONLY: stage 09 partitions its sample on the same
    predicate, so no production caller can reach it. The guarantee it describes -- that a
    freeze with no hashes never reads as fully verified -- is actually delivered by
    `render_markdown`'s "Served pixels" section, which states how much of the sample was
    compared against a pre-existing hash. The branch stays for a direct caller (a test, a
    notebook, a future stage) that does not partition first.
    """
    violations = []
    for row, served in rows:
        frozen = str(row.get("served_pixel_sha256") or "")
        if not frozen:
            violations.append(Violation(
                "served_pixel_unverified", "pairs", str(row.get("pair_id")),
                "no frozen pixel hash to compare against"))
            continue
        got = image_sha256(served)
        if got != frozen:
            violations.append(Violation(
                "served_pixel_mismatch", "pairs", str(row.get("pair_id")),
                f"frozen={frozen[:12]}... served={got[:12]}... "
                f"band={row.get('degradation_band')}"))
    return violations


def _dist(series: pd.Series) -> dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"n": 0}
    q = s.quantile([0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    return {"n": int(s.size), "mean": float(s.mean()), "std": float(s.std(ddof=0)),
            "min": float(q[0.0]), "p05": float(q[0.05]), "p25": float(q[0.25]),
            "p50": float(q[0.5]), "p75": float(q[0.75]), "p95": float(q[0.95]),
            "max": float(q[1.0])}


def _flags(series: pd.Series) -> dict[str, int]:
    """Count occurrences of each quality flag.

    `quality_flags` is schema `list<string>` (`LS` in `pyrafault.schema`): each cell
    is a list, or a numpy object array from a parquet round-trip, of flag name
    strings — never a delimited string. A missing cell comes back as `None` or a
    scalar float NaN, either of which means "no flags", not the four-character
    string "None" or "nan". Splitting a stringified repr on "|" (the previous
    implementation) silently invents a flag named "[]" for every empty cell and
    mangles a real one into "['pen_mark']" — it never matched anything real.
    """
    out: dict[str, int] = {}
    for value in series:
        if value is None or isinstance(value, float):  # null cell (None or NaN)
            continue
        for flag in value:
            if flag:
                out[flag] = out.get(flag, 0) + 1
    return dict(sorted(out.items()))


def _counts(series: pd.Series) -> dict[str, int]:
    """value_counts with str keys — numpy scalar keys are not JSON-serialisable."""
    return {str(k): int(v) for k, v in series.value_counts().sort_index().items()}


def summarize(contexts: pd.DataFrame, queries: pd.DataFrame,
              pairs: pd.DataFrame, splits: pd.DataFrame) -> dict[str, Any]:
    """The §13.4 statistics block. Descriptive only — nothing here gates the run."""
    clusters = (queries.groupby("context_id")["morphology_cluster"].nunique()
                if "morphology_cluster" in queries else pd.Series(dtype=int))
    severity = pairs["fault_severity"].fillna("n/a")
    # Task 2: the freeze's band balance. Recorded here because `qc_stats.json` is the
    # persisted record of what a freeze contains -- before this, the four-band split
    # existed only in the assigning migration's stdout, so a re-banded or partially
    # migrated table could not be told apart from a balanced one after the fact. Added
    # only when the column exists, so a pre-Task-2 freeze keeps the stats keys it had.
    band = ({"degradation_band": _counts(pairs["degradation_band"].fillna("n/a"))}
            if "degradation_band" in pairs else {})
    return {**band,
        "counts": {"slides": int(contexts["slide_id"].nunique()),
                   "contexts": int(len(contexts)), "queries": int(len(queries)),
                   "pairs": int(len(pairs))},
        "fault_type": _counts(pairs["fault_type"]),
        "fault_severity": _counts(severity),
        "fault_type_x_severity": {f"{a}/{b}": int(n) for (a, b), n
                                  in pairs.groupby(["fault_type", severity]).size().items()},
        "tissue_fraction": {"contexts": _dist(contexts["tissue_fraction"]),
                            "queries": _dist(queries["tissue_fraction"])},
        "quality_flags": {"contexts": _flags(contexts["quality_flags"]),
                          "queries": _flags(queries["quality_flags"])},
        "morphology_clusters": {
            "distinct_per_context": _dist(clusters),
            "global_histogram": (_counts(queries["morphology_cluster"])
                                 if "morphology_cluster" in queries else {}),
        },
        "self_similarity": {"contexts": _dist(contexts["self_similarity"]),
                            "queries": _dist(queries["self_similarity"])},
        "hardness_score": _dist(pairs["hardness_score"]),
        "normalized_offset": {
            "all": _dist(pairs["normalized_offset"]),
            "by_severity": {str(k): _dist(g["normalized_offset"])
                            for k, g in pairs.groupby(pairs["fault_severity"].fillna("n/a"))},
        },
        "per_slide": {str(k): {"contexts": int(len(g)),
                               "queries": int((queries["slide_id"] == k).sum()),
                               "pairs": int((pairs["slide_id"] == k).sum())}
                      for k, g in contexts.groupby("slide_id")},
        "per_split": {str(k): {"slides": int(g["slide_id"].nunique()),
                               "contexts": int((contexts["split"] == k).sum()),
                               "queries": int((queries["split"] == k).sum()),
                               "pairs": int((pairs["split"] == k).sum())}
                      for k, g in splits.groupby("split")},
    }


def render_markdown(stats: dict[str, Any], violations: list[Violation], cfg_digest: str,
                    unchecked_sections: list[str] | None = None,
                    served_pixel_summary: dict[str, int] | None = None) -> str:
    """A report a human reads top-down: verdict, then what failed, then the numbers.

    `unchecked_sections` (default `None`/empty, so callers that predate this
    parameter keep producing the same report) names config sections that
    `check_config_provenance` could not vouch for -- typically because the feature
    manifest is legacy and only carries a whole-config digest, not per-section ones.
    Rendered as its own heading, separate from `## Violations`, because "unverifiable"
    and "verified and passing" must never look the same on the page: a config section
    that silently drops out of the check is exactly the failure mode this exists to
    prevent.

    `served_pixel_summary` (Task 6 fix round 1, same default-`None` backward-compat
    rule) is `{"sampled": n, "compared": n, "newly_hashed": n}` from stage 09's
    served-pixel sample. Right after a freeze is banded every `served_pixel_sha256`
    is `""`, so the whole sample is newly hashed and none of it is compared --
    `check_served_pixels` never runs, so nothing about that would show up in
    `## Violations`, and `Verdict: PASS` on its own would read exactly like a run
    that verified every sampled pair. This section exists so the persisted report
    (not just a stdout `print`, which nothing here captures) always states how much
    of the sample was actually checked against a pre-existing hash this run, the
    same reason `unchecked_sections` gets its own heading rather than silence.
    """
    verdict = "PASS" if not violations else "FAIL"
    lines = ["# PyraFault QC report", "",
             f"**Verdict: {verdict}** — {len(violations)} violation(s)", "",
             f"Config digest: `{cfg_digest}`", ""]

    if unchecked_sections:
        lines += ["## Config sections not verified", "",
                  "The feature manifest cannot vouch for these sections (most likely "
                  "written before per-section config digests existed); they were NOT "
                  "checked for drift, and a drift there would pass silently:", "",
                  ", ".join(f"`{s}`" for s in unchecked_sections), ""]

    if served_pixel_summary is not None:
        sampled = int(served_pixel_summary.get("sampled", 0))
        compared = int(served_pixel_summary.get("compared", 0))
        newly_hashed = int(served_pixel_summary.get("newly_hashed", 0))
        lines += ["## Served pixels", ""]
        if sampled == 0:
            lines += ["No pairs were sampled for served-pixel verification this run.", ""]
        elif compared == 0:
            lines += [
                f"**0 of {sampled} sampled pair(s) were verified this run.** All "
                f"{newly_hashed} had an empty `served_pixel_sha256` and were frozen "
                "just now instead of compared -- this run has NOT confirmed that the "
                "served pixels reproduce; only a later run (this pipeline's, or a "
                "collaborator's) that re-samples these same rows will actually "
                "verify them.", ""]
        else:
            lines += [
                f"{compared} of {sampled} sampled pair(s) were compared against an "
                f"already-frozen `served_pixel_sha256` this run; {newly_hashed} had "
                "none yet and were frozen just now instead (verified only once "
                "something re-samples them).", ""]

    if violations:
        lines += ["## Violations", "", "| check | table | row | detail |", "|---|---|---|---|"]
        by_check: dict[str, int] = {}
        for x in violations:
            by_check[x.check] = by_check.get(x.check, 0) + 1
        for x in violations[:200]:
            lines.append(f"| `{x.check}` | {x.table} | `{x.row_id}` | {x.detail} |")
        if len(violations) > 200:
            lines.append(f"| … | | | {len(violations) - 200} more suppressed |")
        lines += ["", "Counts by check: " +
                  ", ".join(f"`{k}` × {n}" for k, n in sorted(by_check.items())), ""]

    lines += ["## Statistics", "", "```json", json.dumps(stats, indent=2, default=str), "```", ""]
    return "\n".join(lines)


def benchmark_manifest(bench_dir: str | Path, cfg_digest: str,
                       pipeline_version: str) -> dict[str, Any]:
    """I9: `data/` is gitignored, so the canonical benchmark otherwise exists only as
    unversioned files with no checksum. This is the sha256 + row count of every frozen
    `01_benchmark` parquet table, written to `benchmark_manifest.json` at the end of
    stage 09 and (unlike the rest of `data/`) committed -- it is what lets anyone
    verify a copy of the unversioned data directory against the git history that
    claims to have produced it.
    """
    bench_dir = Path(bench_dir)
    tables: dict[str, dict[str, Any]] = {}
    for path in sorted(bench_dir.glob("*.parquet")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        n_rows = pq.ParquetFile(path).metadata.num_rows
        tables[path.name] = {"sha256": digest, "rows": int(n_rows)}
    return {
        "pipeline_version": pipeline_version,
        "config_digest": cfg_digest,
        "tables": tables,
    }
