"""A claim-conditioned verifier must not be handed another item's answer.

`runner.cache_key` keys the dedupe on (context, served patch, degradation) because that
is everything a claim-*independent* `localise` can depend on -- and every method the task
is defined around is claim-independent. The claim-conditioned baseline is the one
admitted exception: it exists to show what independent localisation buys, so it reads
`claimed_bbox`, and under the claim-free key two items that differ only in where they
claim the query came from share one cached answer. On the five-cohort test split that is
19,959 of 503,320 items, scored against someone else's claim with no error anywhere.

`Verifier.reads_claim` is the declaration that puts the claim back in the key. It also
makes the independence audit a refusal rather than a failed assertion: such a verifier
contradicts the audit by construction, and `AssertionError` would read as a bug in it
instead of as the one thing it declared about itself.
"""
from __future__ import annotations

import pytest
from PIL import Image

from pyrafault.geom import BBoxL0
from pyrafault.verify.base import Localisation, Verifier
from pyrafault.verify.runner import (
    RecordedVerifier, cache_key, localise_items, run_verifier,
)
from pyrafault.verify.thresholds import Thresholds


def _item(iid, claim_x):
    """Two items of this shape differ *only* in the claim."""
    return {
        "inputs": {
            "context": Image.new("RGB", (64, 64)),
            "query": Image.new("RGB", (16, 16)),
            "context_bbox": BBoxL0(0, 0, 1000, 1000),
            "claimed_bbox": BBoxL0(claim_x, 1000, 100, 100),
            "context_mpp": 2.0, "query_mpp": 0.5, "slide_id": "S1",
            "item_id": iid, "split": "test", "extraction_version": "v1",
        },
        "target": {
            "claim_correct": True, "true_bbox": BBoxL0(1000, 1000, 100, 100),
            "true_slide_id": "S1", "pair_id": iid, "degradation_digest": "",
        },
    }


class TestTheKeyItself:
    def test_the_claim_is_absent_by_default(self):
        """The default is the task's own definition: the estimate does not depend on the
        claim, so two claims over one (context, query) are one localisation."""
        assert cache_key(_item("i0", 1000)) == cache_key(_item("i1", 7000))

    def test_declaring_reads_claim_puts_it_in_the_key(self):
        assert cache_key(_item("i0", 1000), True) != cache_key(_item("i1", 7000), True)

    def test_one_claim_is_still_one_key(self):
        """Only the claim enters -- not the item id, which would cache nothing."""
        assert cache_key(_item("i0", 1000), True) == cache_key(_item("i1", 1000), True)

    def test_the_claim_free_key_is_a_prefix_of_the_claim_bearing_one(self):
        """The two keys agree on everything they share, so declaring `reads_claim` adds
        distinctions and removes none."""
        free = cache_key(_item("i0", 1000))
        bearing = cache_key(_item("i0", 1000), True)
        assert bearing[:len(free)] == free and len(bearing) == len(free) + 4


class _AtTheClaim(Verifier):
    """The claim-conditioned baseline's shape: the estimate *is* the claim."""
    name, revision, reads_claim = "at-the-claim", "1", True

    def __init__(self):
        self.claims = []

    def localise(self, inputs):
        b = inputs["claimed_bbox"]
        self.claims.append(b.x)
        return Localisation(b, b.x / 10000.0)


class _Undeclared(_AtTheClaim):
    """The same verifier, lying about itself -- what the declaration protects against."""
    name, reads_claim = "undeclared", False


class _Fixed(Verifier):
    """Claim-independent, so the audit is a property it has."""
    name, revision = "fixed", "1"

    def localise(self, inputs):
        return Localisation(BBoxL0(1000, 1000, 100, 100), 0.5)


class _TwoClaims:
    def __init__(self):
        self._items = {"i0": _item("i0", 1000), "i1": _item("i1", 7000)}

    @property
    def item_ids(self):
        return list(self._items)

    def __getitem__(self, k):
        return self._items[k]


def _thresholds():
    return Thresholds(tau=0.0, delta=10.0, fitted_on="calibration", n_items=2,
                      residuals={"n": 2, "p50": 0.1, "p75": 0.2, "p90": 0.3,
                                 "p95": 0.4, "p99": 0.5})


class TestTwoClaimsAreTwoLocalisations:
    def test_localise_sees_both_claims(self):
        v = _AtTheClaim()
        localise_items(v, _TwoClaims())
        assert v.claims == [1000, 7000]

    def test_each_item_keeps_its_own_estimate(self):
        df = localise_items(_AtTheClaim(), _TwoClaims()).set_index("item_id")
        assert df.loc["i0", "estimated_bbox_l0_x"] == 1000
        assert df.loc["i1", "estimated_bbox_l0_x"] == 7000

    def test_the_same_holds_through_run_verifier(self):
        df = run_verifier(_AtTheClaim(), _TwoClaims(), _thresholds()).set_index("item_id")
        assert df["reliability"].nunique() == 2

    def test_without_the_declaration_the_second_item_gets_the_first_s_answer(self):
        """The failure this guards, shown rather than described: two rows, a plausible
        reliability on each, no error -- and i1 scored against i0's claim."""
        v = _Undeclared()
        df = localise_items(v, _TwoClaims()).set_index("item_id")
        assert v.claims == [1000], "the second item never reached localise"
        assert len(df) == 2 and df["reliability"].notna().all()
        assert df.loc["i1", "estimated_bbox_l0_x"] == 1000


class TestTheAuditIsRefusedNotFailed:
    def test_localise_items_refuses(self):
        with pytest.raises(ValueError, match="cannot hold for it by construction"):
            localise_items(_AtTheClaim(), _TwoClaims(), audit_independence=True)

    def test_run_verifier_refuses(self):
        with pytest.raises(ValueError, match="reads_claim"):
            run_verifier(_AtTheClaim(), _TwoClaims(), _thresholds(),
                         audit_independence=True)

    def test_it_refuses_before_calling_localise(self):
        """Refusing at the entry point, not on the first item: the message is about the
        caller's flag, so it must not arrive wrapped in a verifier's own work."""
        v = _AtTheClaim()
        with pytest.raises(ValueError):
            localise_items(v, _TwoClaims(), audit_independence=True)
        assert v.claims == []

    def test_a_claim_independent_verifier_is_unaffected(self):
        localise_items(_Fixed(), _TwoClaims(), audit_independence=True)

    def test_an_undeclared_reader_still_fails_the_audit(self):
        """The declaration is not a way past the audit: refused when declared, failed
        when not, and the two messages say which of the two happened."""
        with pytest.raises(AssertionError, match="must not read"):
            localise_items(_Undeclared(), _TwoClaims(), audit_independence=True)


class TestTheDeclarationItself:
    def test_the_default_is_false(self):
        """Every verifier written against the protocol inherits the task's definition
        without naming it."""
        assert Verifier.reads_claim is False

    def test_recorded_verifier_carries_the_attribute(self):
        """It never localises, but `run_verifier` reads the flag off whatever it is
        given, and a from-localisations run passes this stand-in."""
        assert RecordedVerifier("zoomtrace", "1").reads_claim is False
