"""ZoomTrace's four steps.

The matcher and extractor are injected, so these run without a network: a stub matcher
scores by a rule the test controls, which is what makes the arithmetic assertable.
"""
import numpy as np
import pytest
from PIL import Image

from pyrafault.geom import BBoxL0
from pyrafault.verify.bank import WindowBank, bank_key
from pyrafault.verify.matchers import MatchResult, Matcher, refine_bbox
from pyrafault.verify.zoomtrace import (
    ZoomTrace, competitor_margin, context_to_level0, downsample_query,
    empirical_reliability,
    normalise_colour,
)

#: The runtime bank index. Not `context_id`: a verifier never receives one -- it is not
#: in INPUT_KEYS -- and bbox alone collides across slides (187 contexts, 136 distinct
#: boxes in the pilot freeze).
CTX_BBOX = BBoxL0(0, 0, 16254, 16254)
_KEY = bank_key("S1", CTX_BBOX)


class TestDownsample:
    def test_query_is_reduced_to_the_context_resolution(self):
        q = Image.new("RGB", (512, 512))
        assert downsample_query(q, 128).size == (128, 128)

    def test_upsampling_is_refused(self):
        """Upsampling would invent detail, which is the failure this benchmark studies."""
        with pytest.raises(ValueError, match="upsample"):
            downsample_query(Image.new("RGB", (64, 64)), 128)


class TestNormaliseColour:
    """Step 6.1's optional fixed colour normalisation. Off by default (see
    `ZoomTrace.__init__`); these tests exercise the function itself, which a constant
    (e.g. `lambda img: img` or a fixed grey square) could not pass -- it must actually
    equalise different colour casts of the same content and must not collapse a
    textured image to a flat one.
    """

    def _low_contrast_pattern(self, r_lo, r_hi, g_lo, g_hi, b_lo, b_hi, size=32):
        xs = np.linspace(0.0, 1.0, size)
        ys = np.linspace(0.0, 1.0, size)
        gx, gy = np.meshgrid(xs, ys)
        r = r_lo + (r_hi - r_lo) * gx
        g = g_lo + (g_hi - g_lo) * gy
        b = b_lo + (b_hi - b_lo) * (gx * gy)
        arr = np.stack([r, g, b], axis=-1).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def test_stretches_each_channel_to_the_full_range(self):
        img = self._low_contrast_pattern(100, 150, 60, 90, 40, 70)
        out = np.asarray(normalise_colour(img))
        for c in range(3):
            assert out[..., c].min() <= 2
            assert out[..., c].max() >= 253

    def test_is_deterministic(self):
        img = self._low_contrast_pattern(10, 240, 5, 200, 20, 210)
        a = np.asarray(normalise_colour(img))
        b = np.asarray(normalise_colour(img))
        assert np.array_equal(a, b)

    def test_two_different_colour_casts_of_the_same_pattern_become_equal(self):
        """The whole reason this exists: applied identically to differently colour-cast
        copies of the same content, the copies converge. A no-op stand-in would fail
        this -- the two casts would stay as different as they started."""
        base = self._low_contrast_pattern(60, 150, 40, 130, 50, 140)
        base_arr = np.asarray(base, dtype=np.float32)

        cast_a = np.clip(base_arr * 0.8 + 20, 0, 255).astype(np.uint8)
        cast_b = np.clip(base_arr * 1.15 - 15, 0, 255).astype(np.uint8)
        img_a = Image.fromarray(cast_a, mode="RGB")
        img_b = Image.fromarray(cast_b, mode="RGB")

        assert np.abs(cast_a.astype(int) - cast_b.astype(int)).mean() > 5

        norm_a = np.asarray(normalise_colour(img_a), dtype=np.int16)
        norm_b = np.asarray(normalise_colour(img_b), dtype=np.int16)
        assert np.abs(norm_a - norm_b).mean() < 3

    def test_does_not_collapse_a_textured_image_to_a_flat_one(self):
        """Guards against a degenerate stand-in that "normalises" everything to a
        constant image, which would trivially pass the equalisation test above."""
        img = self._low_contrast_pattern(60, 150, 40, 130, 50, 140)
        out = np.asarray(normalise_colour(img))
        assert out.std() > 20


class TestEmpiricalReliability:
    def test_a_best_score_beating_every_decoy_is_maximally_reliable(self):
        assert empirical_reliability(0.9, [0.1, 0.2, 0.3]) == pytest.approx(3 / 4)

    def test_a_best_score_no_better_than_the_decoys_is_unreliable(self):
        assert empirical_reliability(0.1, [0.5, 0.6, 0.7]) == pytest.approx(0.0)

    def test_the_formula_matches_the_specification(self):
        # p = (1 + #{s_r >= s*}) / (|D| + 1);  rho = 1 - p
        rho = empirical_reliability(0.5, [0.4, 0.6, 0.7, 0.3])
        assert rho == pytest.approx(1 - (1 + 2) / 5)

    def test_reliability_granularity_follows_the_decoy_count(self):
        """|D| = 32 would give rho only 33 levels, visibly stepping risk-coverage and
        ECE. The shipped default is 128 for this reason."""
        vals = {empirical_reliability(s, [0.0] * 32) for s in np.linspace(0, 1, 200)}
        assert len(vals) <= 33


class TestCoordinateMapping:
    def test_a_context_pixel_box_maps_back_to_level_zero(self):
        ctx = BBoxL0(4064, 20320, 16254, 16254)
        box = context_to_level0(BBoxL0(480, 602, 128, 128), ctx, context_px=2048)
        assert box.x == pytest.approx(4064 + 480 / 2048 * 16254, abs=2)
        assert box.w == pytest.approx(128 / 2048 * 16254, abs=2)


class TestLocalise:
    def _bank(self, side=128, n=16):
        origins = [(i * 64, 0) for i in range(n)]
        emb = np.zeros((n, 4), dtype=np.float16)
        emb[3] = 1.0                                   # window 3 is the match
        return WindowBank("ctx0", side, origins, emb)

    def _inputs(self):
        return {
            "context": Image.new("RGB", (2048, 2048), "white"),
            "query": Image.new("RGB", (512, 512), "white"),
            "context_bbox": CTX_BBOX,
            "claimed_bbox": BBoxL0(0, 0, 1016, 1016),
            "context_mpp": 2.0, "query_mpp": 0.5,
            "slide_id": "S1", "item_id": "i0", "split": "test",
            "extraction_version": "v1",
        }

    def test_localise_returns_a_localisation(self):
        """Default mode: the margin statistic, which matches no decoys at all."""
        zt = ZoomTrace(bank={_KEY: self._bank()},
                       extractor=_ones_extractor(), matcher=_ranked_matcher())
        loc = zt.localise(self._inputs())
        assert loc.estimated_bbox is not None
        assert 0.0 <= loc.reliability <= 1.0
        assert loc.diagnostics["reliability_mode"] == "margin"
        assert loc.diagnostics["n_decoys"] == 0

    def test_the_default_mode_calls_the_matcher_only_for_the_candidates(self):
        """The 93%-of-compute claim, made executable: `margin` must not match a decoy.
        A mode that quietly kept matching them would still pass every other test here
        while costing what this change exists to remove."""
        m = _ranked_matcher()
        calls = []
        inner = m.match
        m.match = lambda a, b: (calls.append(1), inner(a, b))[1]
        zt = ZoomTrace(bank={_KEY: self._bank()}, extractor=_ones_extractor(),
                       matcher=m, top_k=3)
        zt.localise(self._inputs())
        assert len(calls) == 3

    def test_an_unknown_reliability_mode_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="reliability="):
            ZoomTrace(bank={_KEY: self._bank()}, extractor=_ones_extractor(),
                      matcher=_ranked_matcher(), reliability="tail")

    def test_the_decoy_modes_still_produce_a_reliability(self):
        """Kept reachable for the ablation the method has to report."""
        for mode in ("decoy_tail", "decoy_z"):
            zt = ZoomTrace(bank={_KEY: self._bank()}, extractor=_ones_extractor(),
                           matcher=_ranked_matcher(), n_decoys=4, reliability=mode)
            loc = zt.localise(self._inputs())
            assert loc.diagnostics["n_decoys"] == 4, mode
            assert loc.diagnostics["reliability_mode"] == mode

    def test_localise_never_reads_the_claim(self):
        """The executable form of the task's independence requirement."""
        zt = ZoomTrace(bank={_KEY: self._bank()},
                       extractor=_ones_extractor(), matcher=_ranked_matcher())
        base = self._inputs()
        a = zt.localise(base)
        b = zt.localise({**base, "claimed_bbox": BBoxL0(9000, 9000, 1016, 1016)})
        assert a.estimated_bbox == b.estimated_bbox
        assert a.reliability == b.reliability

    def test_decoys_do_not_overlap_the_estimate(self):
        """The exclusion must be checked against the REFINED box (`refine_bbox`'s
        output), not the winning window's own origin -- `_ranked_matcher` happens to
        return `np.eye(3)` (zero shift), so here the two coincide; the discriminating
        case is `test_decoys_do_not_overlap_the_refined_estimate_under_a_real_shift`
        below."""
        bank = self._bank()
        m = _ranked_matcher()
        zt = ZoomTrace(bank={_KEY: bank}, extractor=_ones_extractor(),
                       matcher=m, top_k=2, n_decoys=4)
        loc = zt.localise(self._inputs())
        est_origin = loc.diagnostics["best_origin"]
        box_ctx = refine_bbox(est_origin, bank.side, np.eye(3))
        for o in loc.diagnostics["decoy_origins"]:
            assert abs(o[0] - box_ctx.x) >= bank.side or abs(o[1] - box_ctx.y) >= bank.side

    def test_decoys_do_not_overlap_the_refined_estimate_under_a_real_shift(self):
        """Regression for the anchor bug: a decoy that clears the winning WINDOW can
        still overlap the REFINED estimate once the matcher reports a nonzero
        translation. Four decoys sit exactly `side` from the winning window on one
        axis -- they all clear a window-anchored filter -- but with a diagonal 30 px
        refinement shift, two of them fall within `side` of `box_ctx` (the true
        estimate) and must be excluded from the decoy pool."""
        side = 128
        best_origin = (200, 200)
        dx, dy = 30, 30
        box_ctx = refine_bbox(best_origin, side, _translation(dx, dy))
        boundary_decoys = [
            (best_origin[0] + side, best_origin[1]),
            (best_origin[0] - side, best_origin[1]),
            (best_origin[0], best_origin[1] + side),
            (best_origin[0], best_origin[1] - side),
        ]
        origins = [best_origin, *boundary_decoys]
        emb = np.zeros((len(origins), 4), dtype=np.float16)
        emb[0] = 1.0                                   # best_origin is the retrieval hit
        bank = WindowBank("ctx0", side, origins, emb)

        # Of the 4 boundary decoys, exactly 2 (the diagonal ones) clear the refined
        # estimate once the anchor bug is fixed -- so the usable pool here is 2, not
        # len(origins); finding 3 makes requesting more than that raise.
        # A decoy-mode test by construction: it is the decoy pool's anchor that is under
        # test, and the default `margin` mode draws no decoys to filter.
        zt = ZoomTrace(bank={_KEY: bank}, extractor=_ones_extractor(),
                       matcher=_shifted_matcher(dx, dy), top_k=len(origins),
                       n_decoys=2, reliability="decoy_tail")
        loc = zt.localise(self._inputs())

        assert loc.diagnostics["decoy_origins"], "the diagonal decoys must survive filtering"
        for ox, oy in loc.diagnostics["decoy_origins"]:
            assert abs(ox - box_ctx.x) >= side or abs(oy - box_ctx.y) >= side, (
                f"decoy origin {(ox, oy)} overlaps the refined estimate "
                f"{(box_ctx.x, box_ctx.y)} but was not excluded from the decoy pool"
            )


class TestDecoyPoolCompleteness:
    """Finding 3: |D| is load-bearing -- tau was fitted at exactly the ceiling |D|=128
    produces. A context whose pool falls short of n_decoys must raise, not silently
    serve a smaller |D| with a lower rho ceiling."""

    def _bank(self, side=128, n=16):
        origins = [(i * 64, 0) for i in range(n)]
        emb = np.zeros((n, 4), dtype=np.float16)
        emb[3] = 1.0
        return WindowBank("ctx0", side, origins, emb)

    def _inputs(self):
        return {
            "context": Image.new("RGB", (2048, 2048), "white"),
            "query": Image.new("RGB", (512, 512), "white"),
            "context_bbox": CTX_BBOX,
            "claimed_bbox": BBoxL0(0, 0, 1016, 1016),
            "context_mpp": 2.0, "query_mpp": 0.5,
            "slide_id": "S1", "item_id": "i0", "split": "test",
            "extraction_version": "v1",
        }

    def test_a_pool_smaller_than_n_decoys_raises_naming_pool_and_request(self):
        # This bank's non-overlapping pool is 13 (see TestLocalise.test_localise_
        # returns_a_localisation); ask for more than that.
        zt = ZoomTrace(bank={_KEY: self._bank()}, extractor=_ones_extractor(),
                       matcher=_ranked_matcher(), n_decoys=128,
                       reliability="decoy_tail")
        with pytest.raises(ValueError, match="13"):
            zt.localise(self._inputs())

    def test_the_raise_names_the_requested_count_too(self):
        zt = ZoomTrace(bank={_KEY: self._bank()}, extractor=_ones_extractor(),
                       matcher=_ranked_matcher(), n_decoys=128,
                       reliability="decoy_tail")
        with pytest.raises(ValueError, match="128"):
            zt.localise(self._inputs())

    def test_a_pool_exactly_matching_n_decoys_does_not_raise(self):
        zt = ZoomTrace(bank={_KEY: self._bank()}, extractor=_ones_extractor(),
                       matcher=_ranked_matcher(), n_decoys=13,
                       reliability="decoy_tail")
        loc = zt.localise(self._inputs())
        assert loc.diagnostics["n_decoys"] == 13

    def test_n_decoys_used_is_recorded_in_diagnostics(self):
        """Closes the related finding: an empty decoy pool giving rho=0.0 is
        indistinguishable from a genuine zero unless n_decoys actually used is
        recorded."""
        zt = ZoomTrace(bank={_KEY: self._bank()}, extractor=_ones_extractor(),
                       matcher=_ranked_matcher(), n_decoys=0)
        loc = zt.localise(self._inputs())
        assert loc.diagnostics["n_decoys"] == 0
        assert loc.reliability == 0.0


class TestColourNormalisationWiring:
    """Off by default; when on, must be applied identically to the query and every
    window before either reaches the matcher -- that consistency is the entire point
    of the spec's requirement, so this is what a spy matcher is used to check."""

    def _bank(self, side=64, n=2):
        origins = [(0, 0), (64, 0)]
        emb = np.zeros((n, 4), dtype=np.float16)
        emb[0] = 1.0
        return WindowBank("ctx0", side, origins, emb)

    def _low_contrast_rgb(self, seed, size, lo=80, hi=160):
        rng = np.random.default_rng(seed)
        arr = rng.integers(lo, hi, (size, size, 3), dtype=np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def _inputs(self, query):
        return {
            "context": self._low_contrast_rgb(1, 256),
            "query": query,
            "context_bbox": CTX_BBOX,
            "claimed_bbox": BBoxL0(0, 0, 128, 128),
            "context_mpp": 2.0, "query_mpp": 0.5,
            "slide_id": "S1", "item_id": "i0", "split": "test",
            "extraction_version": "v1",
        }

    def test_query_and_window_pixels_are_normalised_when_the_switch_is_on(self):
        query = self._low_contrast_rgb(2, 256)
        spy = _SpyMatcher()
        zt = ZoomTrace(bank={_KEY: self._bank()}, extractor=_ones_extractor(),
                       matcher=spy, top_k=1, n_decoys=0,
                       use_colour_normalisation=True)
        zt.localise(self._inputs(query))

        assert spy.calls, "the matcher was never invoked"
        got_query, got_window = spy.calls[0]

        expected_query = np.asarray(
            normalise_colour(downsample_query(query, 64)).convert("L"), dtype=np.uint8
        )
        expected_window = np.asarray(
            normalise_colour(self._inputs(query)["context"].crop((0, 0, 64, 64))).convert("L"),
            dtype=np.uint8,
        )
        assert np.array_equal(got_query, expected_query)
        assert np.array_equal(got_window, expected_window)

    def test_pixels_are_left_alone_when_the_switch_stays_off(self):
        """Default behaviour: unchanged unless a caller opts in."""
        query = self._low_contrast_rgb(2, 256)
        spy = _SpyMatcher()
        zt = ZoomTrace(bank={_KEY: self._bank()}, extractor=_ones_extractor(),
                       matcher=spy, top_k=1, n_decoys=0)
        zt.localise(self._inputs(query))

        got_query, _ = spy.calls[0]
        plain_query = np.asarray(downsample_query(query, 64).convert("L"), dtype=np.uint8)
        assert np.array_equal(got_query, plain_query)


def _spy_result():
    return MatchResult(1.0, 10, 10, np.eye(3))


class _SpyMatcher(Matcher):
    """Records the exact arrays it is handed, so a caller can check what pixels the
    matcher actually saw without depending on `normalise_colour`'s own correctness."""
    name = "spy"

    def __init__(self):
        self.calls = []

    def match(self, query, window):
        self.calls.append((query.copy(), window.copy()))
        return _spy_result()


def _ones_extractor():
    class E:
        def embed(self, images):
            return np.ones((len(images), 4), dtype=np.float32)
    return E()


def _ranked_matcher():
    class M(Matcher):
        name = "ranked"
        def match(self, query, window):
            return MatchResult(float(window.mean()) / 255.0, 10, 5, np.eye(3))
    return M()


def _translation(dx, dy):
    """A window->query transform carrying a real translation (not `np.eye(3)`), so
    `refine_bbox` moves the estimate away from the winning window's own origin."""
    h = np.eye(3)
    h[0, 2] = dx
    h[1, 2] = dy
    return h


def _shifted_matcher(dx, dy, score=0.9):
    """Every candidate scores identically, so retrieval order alone (not this
    matcher) picks the winner; every match reports the same `(dx, dy)` translation."""
    class M(Matcher):
        name = "shifted"
        def match(self, query, window):
            return MatchResult(score, 10, 5, _translation(dx, dy))
    return M()


class TestCompetitorMargin:
    """Step 6.4's statistic: how far the winner stands above the best rival hypothesis.

    Measured against the alternatives on 400 dev items under all four degradation bands
    (1600 localisations), by Spearman concordance between the statistic and the
    localisation residual -- higher concordance means the statistic ranks bad estimates as
    less reliable, with no threshold chosen for it:

        statistic                     none    mild    moderate  severe
        competitor margin (this)     +0.703  +0.709   +0.719   +0.725
        z-score against 128 decoys   +0.732  +0.703   +0.697   +0.716
        decoy tail probability       +0.679  +0.647   +0.659   +0.656
        raw match score              +0.760  +0.696   +0.691   +0.675

    The margin is the only one that improves as the task gets harder, and it costs
    nothing: the candidates it compares were matched by step 6.3 already. The decoy tail
    -- the shipped statistic -- is last in every band, and the z-score shows why: the
    null is a sound idea scored by a lossy statistic, not a wrong idea.
    """

    def test_a_weak_rival_leaves_the_margin_high(self):
        assert competitor_margin(1.0, [0.1]) == pytest.approx(0.9)

    def test_a_rival_as_strong_as_the_winner_leaves_no_margin(self):
        assert competitor_margin(0.8, [0.8]) == pytest.approx(0.0)

    def test_the_margin_falls_as_the_best_rival_rises(self):
        rising = [competitor_margin(1.0, [s]) for s in (0.1, 0.3, 0.5, 0.7, 0.9)]
        assert rising == sorted(rising, reverse=True)

    def test_only_the_strongest_rival_counts(self):
        """Not the mean: one strong alternative is what makes an answer ambiguous, and
        averaging it against weak ones would hide exactly that case."""
        assert competitor_margin(1.0, [0.9, 0.1, 0.1]) == pytest.approx(0.1)

    def test_no_rival_at_all_is_full_confidence_and_is_recorded(self):
        """Reachable only when every retrieved candidate overlaps the winner. Measured 0
        times in 1600 localisations, so this pins the contract rather than a live path."""
        assert competitor_margin(0.7, []) == 1.0

    def test_a_winner_with_no_match_has_no_confidence(self):
        """s_star <= 0 means the matcher found nothing; a ratio against it is meaningless,
        and returning 1.0 there would read as maximum confidence in nothing."""
        assert competitor_margin(0.0, [0.0]) == 0.0
        assert competitor_margin(-0.1, [0.2]) == 0.0

    def test_the_margin_stays_in_the_unit_interval_when_a_rival_wins(self):
        """A rival scoring above the winner cannot happen while the winner is the argmax,
        but the statistic must not go negative if its caller ever changes."""
        assert competitor_margin(0.5, [0.9]) == 0.0
