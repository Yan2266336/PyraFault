# tests/test_verify_base.py
"""The contract every verifier is run and scored through.

`decision` is derived here rather than returned by the implementer: the task
requires that the source estimate be made without the claimed location, and an
interface that cannot pass the claim to `localise` enforces that structurally
instead of relying on each author's discipline.
"""
import pytest

from pyrafault.geom import BBoxL0
from pyrafault.verify.base import DECISIONS, Localisation, Verifier, decide

CLAIM = BBoxL0(1000, 1000, 100, 100)
SIDE = 100.0


def _loc(bbox, rho):
    return Localisation(estimated_bbox=bbox, reliability=rho)


class TestDecisionRule:
    def test_below_tau_abstains_however_good_the_localisation(self):
        assert decide(_loc(CLAIM, 0.1), CLAIM, SIDE, tau=0.5, delta=0.5) == "unverifiable"

    def test_agreeing_estimate_above_tau_verifies(self):
        assert decide(_loc(CLAIM, 0.9), CLAIM, SIDE, tau=0.5, delta=0.5) == "verified"

    def test_disagreeing_estimate_above_tau_mismatches(self):
        far = BBoxL0(5000, 5000, 100, 100)
        assert decide(_loc(far, 0.9), CLAIM, SIDE, tau=0.5, delta=0.5) == "mismatched"

    def test_delta_is_in_units_of_query_side(self):
        near = BBoxL0(1040, 1000, 100, 100)          # centre 40 px away
        assert decide(_loc(near, 0.9), CLAIM, SIDE, tau=0.5, delta=0.5) == "verified"
        assert decide(_loc(near, 0.9), CLAIM, SIDE, tau=0.5, delta=0.3) == "mismatched"

    def test_declining_to_localise_above_tau_is_a_mismatch_not_an_abstention(self):
        """Abstention is a statement about reliability. A method that is confident and
        still cannot find the claimed content is rejecting the claim."""
        assert decide(_loc(None, 0.9), CLAIM, SIDE, tau=0.5, delta=0.5) == "mismatched"

    def test_declining_below_tau_still_abstains(self):
        assert decide(_loc(None, 0.1), CLAIM, SIDE, tau=0.5, delta=0.5) == "unverifiable"

    def test_every_outcome_is_a_declared_decision(self):
        for rho, box in ((0.1, CLAIM), (0.9, CLAIM), (0.9, None)):
            assert decide(_loc(box, rho), CLAIM, SIDE, 0.5, 0.5) in DECISIONS


class TestVerifierInterface:
    def test_localise_cannot_be_handed_the_claim(self):
        """The signature is the guard: one parameter, the inputs mapping."""
        import inspect
        params = list(inspect.signature(Verifier.localise).parameters)
        assert params == ["self", "inputs"], params

    def test_subclass_must_implement_localise(self):
        class Incomplete(Verifier):
            name, revision = "x", "1"
        with pytest.raises(TypeError):
            Incomplete()
