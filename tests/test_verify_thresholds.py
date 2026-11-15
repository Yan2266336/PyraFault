"""tau and delta are fitted on calibration and frozen for test.

Fitting them per method in shared code is what stops one author calibrating properly
while another quietly tunes on the evaluation set -- a difference that would not look
like anything in the results.
"""
import json

import numpy as np
import pytest

from pyrafault.verify.thresholds import (TARGET_FVR, Thresholds, fit_delta, fit_tau,
                                         residual_report)


class TestDelta:
    def test_delta_is_the_99th_percentile_of_clean_residuals(self):
        r = list(np.linspace(0.0, 1.0, 101))
        assert fit_delta(r) == pytest.approx(0.99, abs=1e-6)

    def test_percentile_is_configurable_for_the_sensitivity_analysis(self):
        r = list(np.linspace(0.0, 1.0, 101))
        assert fit_delta(r, percentile=50.0) == pytest.approx(0.5, abs=1e-6)

    def test_no_clean_residuals_is_an_error_naming_the_cause(self):
        with pytest.raises(ValueError, match="no clean"):
            fit_delta([])

    def test_no_clean_items_at_all_is_distinguished_from_all_estimates_missing(self):
        """Two different situations used to raise the identical message. `n_clean`
        lets the caller (which knows how many clean items existed before any were
        filtered out for a missing estimate) tell them apart."""
        with pytest.raises(ValueError, match="no clean items in the split"):
            fit_delta([], n_clean=0)
        with pytest.raises(ValueError, match="3 clean item.*estimated_bbox=None"):
            fit_delta([], n_clean=3)


class TestResidualReport:
    def test_records_n_and_every_percentile(self):
        r = list(np.linspace(0.0, 1.0, 101))
        report = residual_report(r)
        assert report["n"] == 101
        assert set(report) == {"n", "p50", "p75", "p90", "p95", "p99"}
        assert report["p50"] == pytest.approx(0.5, abs=1e-6)
        assert report["p99"] == pytest.approx(0.99, abs=1e-6)

    def test_p99_matches_fit_deltas_default(self):
        """delta itself stays the p99 -- the residuals block does not change the
        calibration rule, only records more of the distribution around it."""
        r = list(np.random.default_rng(0).exponential(size=500))
        assert residual_report(r)["p99"] == pytest.approx(fit_delta(r), abs=1e-9)

    def test_empty_input_records_zero_n_and_no_percentiles(self):
        report = residual_report([])
        assert report == {"n": 0, "p50": None, "p75": None, "p90": None,
                          "p95": None, "p99": None}


class TestTau:
    def test_tau_admits_at_most_the_target_false_verification_rate(self):
        scores = [0.9, 0.8, 0.7, 0.6]          # not clean
        clean = [0.95, 0.85]
        # every wrong item here agrees, so admission and false verification coincide
        tau = fit_tau(scores + clean, [False] * 4 + [True] * 2,
                      [True] * 4 + [True] * 2, target_fvr=0.25)
        wrongly_admitted = sum(1 for s in scores if s >= tau) / len(scores)
        assert wrongly_admitted <= 0.25 + 1e-9

    def test_a_wrong_item_that_is_rejected_is_not_a_false_verification(self):
        """The distinction the whole threshold rests on. Admitting a wrong claim is not
        a failure — the verifier goes on to declare it `mismatched`, which is the
        correct answer. A false verification is admitting it AND agreeing with it.
        Conflating the two makes every correctly-rejected fault count against tau, and
        on a method that rejects well the target becomes unreachable at any threshold."""
        scores = [0.9, 0.9, 0.9, 0.9]          # all clear any plausible tau
        is_clean = [False, False, False, False]
        agrees = [False, False, False, False]  # every one correctly disagreed
        assert fit_tau(scores, is_clean, agrees, target_fvr=0.05) <= 0.9

    def test_agreeing_wrong_items_do_count_against_the_threshold(self):
        scores = [0.9, 0.8, 0.7, 0.6]
        is_clean = [False] * 4
        agrees = [True, True, False, False]    # the two highest are genuine false verifications
        tau = fit_tau(scores, is_clean, agrees, target_fvr=0.25)
        admitted = sum(1 for s, a in zip(scores, agrees) if s >= tau and a)
        assert admitted / 4 <= 0.25 + 1e-9

    def test_the_rate_counts_only_agreeing_admissions(self):
        """A method admitting every wrong item but agreeing with none has a false
        verification rate of zero, so no abstention is required of it."""
        scores = list(np.linspace(0.1, 0.9, 9))
        is_clean = [False] * 9
        agrees = [False] * 9
        assert fit_tau(scores, is_clean, agrees, target_fvr=0.0) <= min(scores)

    def test_an_unreachable_target_returns_a_threshold_above_every_score(self):
        tau = fit_tau([0.5, 0.6], [False, False], [True, True], target_fvr=0.0)
        assert tau > 0.6

    def test_no_faulty_items_raises_rather_than_admitting_everything(self):
        """fit_delta raises when its required input is missing; this is the same class
        of missing input, and returning a permissive threshold instead would produce a
        verifier that never abstains, with nothing to show it happened."""
        with pytest.raises(ValueError, match="no faulty"):
            fit_tau([0.5, 0.6], [True, True], [True, True], target_fvr=0.05)


_RESIDUALS = {"n": 100, "p50": 0.01, "p75": 0.05, "p90": 0.2, "p95": 0.4, "p99": 0.7}


class TestPersistence:
    def test_round_trips_and_digests(self, tmp_path):
        t = Thresholds(tau=0.4, delta=0.7, fitted_on="calibration", n_items=2400,
                       residuals=_RESIDUALS)
        p = tmp_path / "thresholds.json"
        t.save(p)
        back = Thresholds.load(p)
        assert back == t
        assert back.digest() == t.digest() and len(back.digest()) == 64

    def test_the_digest_changes_when_a_threshold_does(self, tmp_path):
        a = Thresholds(0.4, 0.7, "calibration", 10, _RESIDUALS)
        b = Thresholds(0.5, 0.7, "calibration", 10, _RESIDUALS)
        assert a.digest() != b.digest()

    def test_the_digest_changes_when_the_residuals_block_does(self, tmp_path):
        """residuals is a real field of the schema, not an afterthought -- it must
        affect the digest recorded on every prediction the same as tau or delta do."""
        a = Thresholds(0.4, 0.7, "calibration", 10, _RESIDUALS)
        b = Thresholds(0.4, 0.7, "calibration", 10, {**_RESIDUALS, "p99": 0.71})
        assert a.digest() != b.digest()

    def test_the_file_records_what_it_was_fitted_on(self, tmp_path):
        p = tmp_path / "t.json"
        Thresholds(0.4, 0.7, "calibration", 2400, _RESIDUALS).save(p)
        assert json.loads(p.read_text())["fitted_on"] == "calibration"

    def test_a_file_from_before_the_residuals_field_fails_to_load(self, tmp_path):
        """Deliberate: the schema change is a breaking one (see the commit that added
        `residuals`). An older thresholds.json silently reporting an empty sensitivity
        analysis that was never measured would be worse than failing to load."""
        p = tmp_path / "old.json"
        p.write_text(json.dumps({"tau": 0.4, "delta": 0.7,
                                 "fitted_on": "calibration", "n_items": 10}))
        # ValueError naming the missing field, not the bare TypeError a dataclass
        # raises on a missing argument: a caller hitting this needs to know which field
        # and how to recover, not that __init__ took the wrong number of arguments.
        with pytest.raises(ValueError, match="residuals"):
            Thresholds.load(p)

    def test_the_saved_file_is_strict_json(self, tmp_path):
        """Non-Python readers are in scope: the protocol invites third parties to
        consume these files. Python's json writes bare Infinity/NaN tokens that RFC 8259
        rejects, so a threshold that could be non-finite would break them silently."""
        import json

        def _reject(token):
            raise ValueError(f"non-standard JSON constant: {token}")

        p = tmp_path / "t.json"
        Thresholds(tau=0.4, delta=0.7, fitted_on="calibration", n_items=10,
                  residuals=_RESIDUALS).save(p)
        json.loads(p.read_text(), parse_constant=_reject)



def test_the_refusal_says_how_to_recover(tmp_path):
    """A refusal that does not say what to do next just moves the confusion."""
    import json

    p = tmp_path / "old.json"
    p.write_text(json.dumps({"tau": 0.9, "delta": 0.5,
                             "fitted_on": "calibration", "n_items": 100}))
    with pytest.raises(ValueError, match="--fit-thresholds"):
        Thresholds.load(p)


class TestTargetFalseVerificationRate:
    """The target is the benchmark's, not a method's: every verifier fits its own tau
    against the same number or the taus are not comparable. It lives here as a named
    constant for that reason, and the protocol states its value.
    """

    def test_the_target_is_one_percent(self):
        assert TARGET_FVR == 0.01

    def test_fit_tau_uses_the_named_target_by_default(self):
        """A method whose false-verification rate at full admission is 1.5% must be made
        to abstain. Measured on the degraded lung freeze: 79 of 5250 wrong items agreed
        within delta, which the old 5% target waved through -- so tau came back at the
        lowest candidate score, no item ever abstained, and rho did not participate in a
        single decision.
        """
        # 100 wrong items, 2 of which would be accepted: a 2% rate at full admission.
        scores = [0.9, 0.8] + [0.1] * 98
        is_clean = [False] * 100
        agrees = [True, True] + [False] * 98

        tau = fit_tau(scores, is_clean, agrees)

        # 2% exceeds 1%, so the lower of the two accepted scores must be excluded.
        assert tau == pytest.approx(0.9)

    def test_a_rate_already_under_the_target_still_requires_no_abstention(self):
        scores = [0.9] + [0.1] * 199
        is_clean = [False] * 200
        agrees = [True] + [False] * 199

        # 1 of 200 = 0.5%, under the 1% target: the lowest candidate suffices.
        assert fit_tau(scores, is_clean, agrees) == pytest.approx(0.9)

    def test_an_explicit_target_still_overrides_it(self):
        """Kept so a sensitivity analysis over the target is a call, not an edit."""
        scores = [0.9, 0.8] + [0.1] * 98
        is_clean = [False] * 100
        agrees = [True, True] + [False] * 98

        assert fit_tau(scores, is_clean, agrees, target_fvr=0.05) == pytest.approx(0.8)
