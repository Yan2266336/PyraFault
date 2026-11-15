"""The contract every verifier is run and scored through."""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from pyrafault.geom import BBoxL0

#: The three states of the task's output. `unverifiable` is the method abstaining --
#: it is not a third ground-truth class, and no benchmark record carries it.
DECISIONS = ("verified", "mismatched", "unverifiable")


@dataclass(frozen=True)
class Localisation:
    """A verifier's claim-independent estimate.

    `reliability` is thresholded against a `tau` fitted per method on `calibration`;
    it is never compared between methods, so an unnormalised correlation and an
    empirical tail probability are equally admissible scales.
    """
    estimated_bbox: BBoxL0 | None
    reliability: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


class Verifier(ABC):
    """Implement `localise` and nothing else.

    The decision rule, threshold fitting, batch running and scoring are shared, which
    is what makes two methods' numbers comparable.
    """

    name: str
    revision: str

    #: Whether `localise` reads `inputs["claimed_bbox"]`. False for every method the
    #: task is defined around, and the default, because the task defines the estimate
    #: as independent of the claim.
    #:
    #: A verifier that sets this True is NOT solving the task as formulated; it is a
    #: baseline that exists to show what claim-independence buys. Setting it changes two
    #: things in the harness, both of which are wrong to skip:
    #:
    #: 1. The dedupe key gains the claim. `cache_key` identifies a localisation by
    #:    (context, query, degradation) because that is everything a claim-independent
    #:    `localise` can depend on. On the five-cohort test split 19,959 of 503,320 items
    #:    share a key with another item, so without this a claim-conditioned verifier
    #:    hands one item's score to a different claim -- silently, with no error anywhere.
    #: 2. The independence audit is refused rather than failed. `--audit-independence`
    #:    asserts that re-running under a shifted claim gives the same answer, which such
    #:    a verifier contradicts by construction; failing that assertion would read as a
    #:    bug rather than as the declared design.
    reads_claim: bool = False

    @abstractmethod
    def localise(self, inputs: Mapping) -> Localisation:
        """Estimate the source of `inputs["query"]` within `inputs["context"]`.

        `inputs` is the loader's `item["inputs"]` and carries no claim-derived field
        beyond `claimed_bbox`, which implementations must not read unless they declare
        `reads_claim = True`: the task defines the estimate as independent of the claim,
        and a run harness asserts it by re-running the same (C, Q) under a different
        claim.
        """


def decide(
    loc: Localisation,
    claimed_bbox: BBoxL0,
    query_side: float,
    tau: float,
    delta: float,
) -> str:
    """Turn a `Localisation` plus the claim into one of `DECISIONS`.

    `delta` is in units of the query side, matching how it is fitted (the 99th
    percentile of the normalised localisation residual over calibration's clean items).
    """
    if loc.reliability < tau:
        return "unverifiable"
    if loc.estimated_bbox is None:
        # Confident, and still unable to place the claimed content: a rejection, not an
        # abstention. Abstention is a statement about reliability, which is above tau here.
        return "mismatched"
    d = math.hypot(loc.estimated_bbox.cx - claimed_bbox.cx,
                   loc.estimated_bbox.cy - claimed_bbox.cy)
    return "verified" if d <= delta * query_side else "mismatched"
