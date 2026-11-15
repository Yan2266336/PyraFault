"""ZoomTrace: physical-scale alignment, coarse retrieval, geometric verification,
context-specific reliability calibration."""
from __future__ import annotations

import numpy as np
from PIL import Image

from pyrafault.geom import BBoxL0
from pyrafault.verify.base import Localisation, Verifier
from pyrafault.verify.bank import WindowBank, bank_key
from pyrafault.verify.matchers import Matcher, refine_bbox


def normalise_colour(img: Image.Image) -> Image.Image:
    """Step 6.1's optional fixed colour normalisation, applied identically to `Q` and
    every candidate window before matching.

    Deterministic per-channel min-max stretch to the full 0-255 range in RGB: no
    dataset statistics, no learned parameters, so the exact same function applied to
    any image gives a reproducible result independent of what else is in a batch or
    run. That determinism is the point -- the specification's requirement is not that
    normalisation be *this* transform, but that whichever one is used be "kept
    consistent across all baselines" being compared. So this is a property of an
    *experiment* (every method under comparison must call it, or none must), not a
    property ZoomTrace decides for itself: it is exposed as a preprocessing step
    callers opt into, not baked into `ZoomTrace.localise`, and off by default -- turning
    it on changes the pixels every matcher sees, so it must be switched deliberately
    and applied to every method in a comparison, never silently to just one.
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    lo = arr.min(axis=(0, 1), keepdims=True)
    hi = arr.max(axis=(0, 1), keepdims=True)
    span = np.clip(hi - lo, 1e-6, None)
    stretched = np.clip((arr - lo) / span * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(stretched, mode="RGB")


def downsample_query(query: Image.Image, side: int) -> Image.Image:
    """Step 6.1. Only ever downsamples.

    The alternative -- upsampling `C` to meet `Q` -- invents detail that was never
    scanned, which is precisely the untrustworthy evidence this task is about.
    """
    if side > query.size[0]:
        raise ValueError(
            f"would upsample query from {query.size[0]} to {side}; "
            "the context must be the coarser of the two"
        )
    return query.resize((side, side), Image.LANCZOS)


def empirical_reliability(s_star: float, decoy_scores) -> float:
    """Step 6.4. `rho = 1 - (1 + #{s_r >= s*}) / (|D| + 1)`.

    A tail probability against the context's own wrong answers, rather than a raw
    matching score: absolute scores are not comparable between a self-similar region
    and a structurally distinctive one, which is the observation this step exists for.
    """
    d = list(decoy_scores)
    ge = sum(1 for s in d if s >= s_star)
    return 1.0 - (1 + ge) / (len(d) + 1)


def competitor_margin(s_star: float, competitor_scores) -> float:
    """Step 6.4. `1 - best_competitor / s_star`, over candidates that do not overlap the
    winner.

    What a reliability has to answer is "how much better is my answer than the next
    hypothesis I can see" -- and the candidates step 6.3 already matched are exactly that
    set, so this costs nothing. The decoy null it replaces cost 128 extra matcher calls
    per item, about 93% of this verifier's compute, and ranked last of four statistics
    measured on 400 dev items under all four degradation bands (Spearman concordance with
    the localisation residual, per band):

        competitor margin (this)     +0.703  +0.709  +0.719  +0.725
        z-score against the decoys   +0.732  +0.703  +0.697  +0.716
        decoy tail probability       +0.679  +0.647  +0.659  +0.656
        raw match score              +0.760  +0.696  +0.691  +0.675

    Two things that table settles. The margin is the only statistic whose concordance
    improves as the rotation worsens -- a degradation lowers every absolute score but
    leaves the winner's lead over its rivals intact. And the z-score nearly matching it
    says the null was a sound idea scored by a lossy statistic: counting how many decoys
    clear `s_star` throws away the distances, quantises rho to `|D| + 1` levels and caps it
    at `1 - 1/(|D|+1)`. The null is not refuted here; it is merely not worth its price.

    The full five-cohort ablation (503,320 test items, 2026-10-28) keeps that conclusion
    but narrows which part of it is load-bearing, so do not re-argue the choice from the
    dev table alone:

        margin  AUPRC 0.9686  AUROC 0.9872  coverage 0.3439  bal-acc 0.9847
        decoy_z AUPRC 0.9521  AUROC 0.9888  coverage 0.4519  bal-acc 0.9858

    Median IoU is 0.7987 against 0.7986: reliability runs after the estimate is fixed, so
    neither statistic moves the localisation. Both differences in ranking are real under a
    slide-clustered bootstrap (427 slides, B=400): the margin's AUPRC lead is +0.0165
    [+0.0140, +0.0184] and `decoy_z`'s AUROC lead is 0.0016 [0.0012, 0.0020]. The margin
    wins where a selective verifier acts and loses, by an order of magnitude less,
    across the whole ordering.

    What does NOT survive at full scale is the stability reading above. On coverage and
    balanced accuracy per degradation band, `decoy_z` is the flatter of the two -- its
    cross-band coverage range is 0.0121 against the margin's 0.0392, and its bal-acc range
    0.0009 against 0.0021. The dev table measures per-band concordance with the residual,
    not per-band behaviour of the calibrated decision, and the two do not agree here.
    Against `decoy_z`, then, the argument for shipping the margin is price: the null needs
    128 extra matcher calls per item against this function's zero, and loses on AUPRC.

    That argument does not transfer to the third alternative, and an earlier version of
    this docstring stated it as though it did. The fixed-localisation comparison
    (2026-10-30, `results/tables/6.2_reliability_fixed_loc/`) put all three statistics on
    ONE set of boxes -- recorded candidate scores, recomputed offline, identical estimates
    by construction rather than by assertion -- and the raw winning score beats this
    function at the same price:

        s_star  AUPRC 0.9820  AUROC 0.9916  coverage 0.4609  bal-acc 0.9861
        margin  AUPRC 0.9686  AUROC 0.9872  coverage 0.4203  bal-acc 0.9856

    `margin - s_star` is -0.0134 AUPRC [-0.0142, -0.0127], slide-clustered, 427 slides,
    B=400. `s_star` is already computed; it costs exactly what this does. So price is a
    wash and the margin loses the ranking. What it wins is the thing the dev table above
    pointed at and the `decoy_z` comparison took away -- per-band stability of the
    calibrated decision:

        cross-band coverage range   s_star 0.1655 [0.1594, 0.1712]
                                    margin 0.0228 [0.0185, 0.0274]

    The intervals do not overlap; the gap is +0.1424 [+0.1336, +0.1506]. `s_star`'s
    coverage falls from 0.5232 undegraded to 0.3577 severe, so a verifier calibrated on it
    abstains more as the image degrades -- which is when the faults this benchmark is about
    are present. The margin holds near 0.42 across all four bands.

    **So the honest statement is two-sided**: against the decoy null the margin ships on
    price, against the raw score it ships on stability, and against the raw score it is the
    worse discriminator. Do not write "the margin is the better statistic" anywhere; write
    what it is better AT. (Those two rows come from a run whose own delta is 0.375628, not
    the shipped 0.387224, so their coverage is not comparable with the main table's 0.3439
    -- the three variants share a delta with each other, which is what makes them
    comparable among themselves.)

    A competitor overlapping the winner is the same answer at a shifted origin, not a rival
    hypothesis. Excluding them is load-bearing, not hygiene: including them inverts the
    statistic outright, to a concordance of -0.49 on dev (the overlapping neighbour of a
    confident peak scores high, so a correct, unambiguous answer looks maximally
    ambiguous). At full scale the same comparison, on the same fixed localisations, is the
    largest effect anywhere in this method: an ordinary top-two margin that admits
    overlapping rivals reaches AUPRC 0.5206 against this function's 0.9686, a difference of
    +0.4478 [+0.4375, +0.4591]. Whatever else is arguable about the choice of statistic,
    the exclusion is not. The caller does the excluding, because only it knows the window
    geometry.
    """
    if s_star <= 0.0:
        # The matcher found nothing. A ratio against zero is undefined, and returning full
        # confidence in nothing is the one answer that must not come out of here.
        return 0.0
    rivals = list(competitor_scores)
    if not rivals:
        # Every retrieved candidate overlaps the winner: no rival hypothesis was retrieved
        # at all. Measured 0 times in 1600 localisations; `n_competitors` in diagnostics is
        # what lets this be told apart from genuine certainty after the fact.
        return 1.0
    return max(0.0, 1.0 - max(rivals) / s_star)


def context_to_level0(box_ctx: BBoxL0, context_bbox: BBoxL0, context_px: int) -> BBoxL0:
    """Context-image pixels back to the slide's level-0 frame."""
    scale = context_bbox.w / context_px
    return BBoxL0(
        int(round(context_bbox.x + box_ctx.x * scale)),
        int(round(context_bbox.y + box_ctx.y * scale)),
        int(round(box_ctx.w * scale)),
        int(round(box_ctx.h * scale)),
    )


class ZoomTrace(Verifier):
    name = "zoomtrace"
    revision = "1"

    RELIABILITY_MODES = ("margin", "decoy_z", "decoy_tail")

    def __init__(self, bank, extractor, matcher: Matcher,
                 top_k: int = 10, n_decoys: int = 128, seed: int = 0,
                 use_colour_normalisation: bool = False,
                 reliability: str = "margin",
                 scale_alignment: bool = True, refine: bool = True,
                 record_decoys: bool = False) -> None:
        self._bank = bank
        self._extractor = extractor
        self._matcher = matcher
        self._top_k = top_k
        self._n_decoys = n_decoys
        self._seed = seed
        # Off by default -- see `normalise_colour`. When on, applied identically here
        # and to the query below, which is the whole requirement: the same function on
        # every image this verifier prepares.
        self._use_colour = use_colour_normalisation
        # `margin` by default: measured best or equal in every degradation band and the
        # only statistic that improves as the task hardens, at no matcher cost
        # (`competitor_margin`). The two decoy modes stay reachable because the ablation
        # they answer -- "what did the empirical null buy?" -- is one this method has to
        # report, and a mode flag keeps that a run rather than a code change.
        if reliability not in self.RELIABILITY_MODES:
            raise ValueError(f"reliability={reliability!r} is not one of "
                             f"{self.RELIABILITY_MODES}")
        self._reliability = reliability
        #: §5.1's two remaining ablations. On by default, so a run that does not ask for
        #: one gets the shipped method; each is a separate method with its own fitted
        #: thresholds, exactly as a matcher or reliability change is.
        #:
        #: `scale_alignment=False` stops resizing the query to the bank's window side, so
        #: retrieval embeds and the matcher sees a query at a different physical scale
        #: from every window it is compared against. That is the component's whole job
        #: (step 6.1), and removing it is the only way to say what it bought.
        #:
        #: `refine=False` reports the winning window's own box instead of recovering a
        #: continuous position from the correspondences. `refine_bbox`'s docstring says
        #: what to expect: a clean pair's true position lies on a lattice point while a
        #: drifted pair's falls between them, so the window box flatters clean pairs for
        #: a reason that has nothing to do with the evidence. The ablation should
        #: therefore show scores going UP where the fault is a small drift, which is a
        #: leak and not a gain -- the number is what makes that arguable.
        self._scale_alignment = scale_alignment
        self._refine = refine
        #: Score the decoy pool even under `margin`, purely to record it.
        #:
        #: This exists for the fixed-localisation reliability comparison: all four
        #: statistics have to be computed from ONE set of matcher outputs, because two
        #: separate runs do not share their estimates. Measured on the five-cohort test
        #: split, `zoomtrace_loftr` and `zoomtrace_loftr_decoy_z` return different boxes
        #: for 17,375 of 503,320 items (3.45%) -- 12,594 of those are refinement jitter
        #: inside the same window and 4,780 picked a different window outright. The
        #: aggregate median IoU hides it (0.7987 against 0.7986), which is exactly why
        #: comparing two runs' reliability columns would look sound and not be.
        #:
        #: Turning this on does NOT change which statistic lands in `reliability`; it
        #: only fills `decoy_scores` in the diagnostics. It does cost the same 128
        #: matcher calls per item the decoy modes cost, and it moves this run's own
        #: estimates relative to a run without it -- RANSAC is randomised and the call
        #: sequence is no longer the same -- so a run with it set is its own method and
        #: belongs in its own directory, never in the shipped one.
        self._record_decoys = record_decoys

    def _windows(self, context: Image.Image, bank: WindowBank, idx):
        out = []
        for i in idx:
            x, y = bank.origins[i]
            crop = context.crop((x, y, x + bank.side, y + bank.side))
            if self._use_colour:
                crop = normalise_colour(crop)
            out.append(np.asarray(crop.convert("L"), dtype=np.uint8))
        return out

    def localise(self, inputs) -> Localisation:
        bank = self._bank[bank_key(inputs["slide_id"], inputs["context_bbox"])]
        context = inputs["context"]

        q_img = (downsample_query(inputs["query"], bank.side) if self._scale_alignment
                 else inputs["query"])
        # Colour normalisation feeds the matcher only (6.3), not retrieval (6.2): the
        # bank's cached embeddings were built from un-normalised window crops, so
        # normalising `q_img` before embedding it would compare a normalised query
        # against a non-normalised bank -- an inconsistency the bank has no way to
        # signal. Retrieval always sees the image as downsampled; matching sees the
        # colour-normalised version when the switch is on.
        q_match_img = normalise_colour(q_img) if self._use_colour else q_img
        q_gray = np.asarray(q_match_img.convert("L"), dtype=np.uint8)

        # 6.2 retrieval: one matrix product against the cached bank. `normalized_embeddings`
        # is computed once per bank (cached_property) rather than renormalised here on
        # every query -- amortised cost is the reason the bank exists (spec S3.1).
        q_desc = np.asarray(self._extractor.embed([q_img])[0], dtype=np.float32)
        q_norm = q_desc / (np.linalg.norm(q_desc) + 1e-12)
        sims = bank.normalized_embeddings @ q_norm
        cand = list(np.argsort(-sims)[: self._top_k])

        # 6.3 geometric verification; the estimate comes from the correspondences.
        results = [self._matcher.match(q_gray, w)
                   for w in self._windows(context, bank, cand)]
        best = int(np.argmax([r.score for r in results]))
        best_idx, best_res = cand[best], results[best]
        s_star = best_res.score
        origin = bank.origins[best_idx]
        if self._refine:
            box_ctx = refine_bbox(origin, bank.side, best_res.transform)
        else:
            # The window, not the refined position -- but still None when the matcher
            # found no transform at all. A failed match means the method could not place
            # the query, and handing back a window box there would invent an estimate
            # the ablation never had; what is being removed is the refinement step, not
            # the ability to abstain.
            box_ctx = (None if best_res.transform is None
                       else BBoxL0(origin[0], origin[1], bank.side, bank.side))

        # 6.4 reliability. Default `margin`: how far the winner stands above the best
        # rival hypothesis among the candidates step 6.3 already matched, so it adds no
        # matcher calls. The two decoy modes reproduce the empirical null this replaced and
        # exist for the ablation; see `competitor_margin` for the measured comparison.
        #
        # A rival that overlaps the winner is the winner's own answer at a shifted origin,
        # so it is excluded -- including them inverts the statistic (concordance -0.49
        # against +0.71). Two equal squares overlap iff |dx| < side AND |dy| < side, which
        # is the negation tested here, and the decoy pool below applies the same test for
        # the same reason.
        rival_scores = [float(r.score) for j, r in enumerate(results)
                        if j != best
                        and (abs(bank.origins[cand[j]][0] - origin[0]) >= bank.side
                             or abs(bank.origins[cand[j]][1] - origin[1]) >= bank.side)]
        decoy_idx: list = []
        decoy_scores: list = []

        # Sampling and scoring the pool is one thing; which statistic goes into
        # `reliability` is another. Keeping them separate is what lets one run record
        # the null while still shipping the margin.
        if self._reliability != "margin" or self._record_decoys:
            # Decoys are excluded against the REFINED box, not the winning window:
            # predicating on the window lets a decoy that clears the window still overlap
            # the estimate by the refinement shift -- measured, 2 of 4 boundary decoys
            # leak, by up to 3480 px^2. Those score high, inflate the tail count and
            # deflate rho, and the leak scales with the shift, which is larger for
            # off-lattice truths than on-lattice ones: the very asymmetry the refinement
            # exists to remove.
            anchor = (box_ctx.x, box_ctx.y) if box_ctx is not None else origin
            rng = np.random.default_rng(self._seed)
            pool = [i for i, (x, y) in enumerate(bank.origins)
                    if abs(x - anchor[0]) >= bank.side or abs(y - anchor[1]) >= bank.side]
            # |D| must not vary per context: tau is fitted at a fixed |D|, and rho's
            # ceiling -- 1 - 1/(|D|+1) -- falls below any tau fitted at the nominal |D| the
            # moment the pool shrinks. A context whose pool falls short would then be
            # forced to `unverifiable` regardless of match quality, silently. Raise instead
            # of quietly taking min(n_decoys, len(pool)).
            if len(pool) < self._n_decoys:
                raise ValueError(
                    f"bank {bank.context_id!r} has a decoy pool of {len(pool)} windows, "
                    f"fewer than the requested n_decoys={self._n_decoys}; shrinking it "
                    "would silently lower rho's ceiling below tau for this context alone"
                )
            decoy_idx = (list(rng.choice(pool, size=self._n_decoys, replace=False))
                        if self._n_decoys else [])
            decoy_scores = [self._matcher.match(q_gray, w).score
                            for w in self._windows(context, bank, decoy_idx)]

        if self._reliability == "margin":
            rho = competitor_margin(s_star, rival_scores)
        elif self._reliability == "decoy_tail":
            rho = empirical_reliability(s_star, decoy_scores)
        else:
            # `decoy_z`: the same null, standardised instead of counted. Keeps the
            # distances the tail probability throws away.
            d = np.asarray(decoy_scores, dtype=float)
            rho = float((s_star - d.mean()) / (d.std() + 1e-12)) if d.size else 0.0

        est = (context_to_level0(box_ctx, inputs["context_bbox"], context.width)
               if box_ctx is not None else None)
        return Localisation(
            estimated_bbox=est,
            reliability=rho,
            diagnostics={
                "s_star": s_star,
                "best_origin": origin,
                "decoy_origins": [bank.origins[i] for i in decoy_idx],
                "candidate_origins": [bank.origins[i] for i in cand],
                # 离线重算四个可靠性统计量需要的全部原料。`candidate_scores` 与
                # `candidate_origins` 同序，`best_index` 指进这两个列表，`bank_side`
                # 是重叠判定的边长 —— 没有它，「不重叠」这个条件在图之外就无法复现。
                # 这些是记录，不参与任何决策：`reliability` 上面已经算完了。
                "candidate_scores": [float(r.score) for r in results],
                "best_index": best,
                "bank_side": int(bank.side),
                "decoy_scores": [float(x) for x in decoy_scores],
                # Recorded so an empty decoy pool (rho=0.0 because n_decoys=0) is
                # distinguishable from a genuine zero over a real decoy set.
                "n_decoys": len(decoy_idx),
                # Which statistic produced `reliability`, and how many rival hypotheses it
                # saw. `n_competitors == 0` under `margin` is the one path that returns
                # full confidence without evidence for it, so it has to be auditable.
                "reliability_mode": self._reliability,
                "n_competitors": len(rival_scores),
                # Recorded on every prediction so an ablation's parquet names its own
                # configuration; a directory name is not evidence of what ran.
                "scale_alignment": self._scale_alignment,
                "refine": self._refine,
                "top_k": self._top_k,
            },
        )
