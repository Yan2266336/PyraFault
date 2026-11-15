"""Running any verifier over a split.

The cache key is the assertion that matters. localise depends only on (C, Q), so
caching by (slide_id, context_bbox, true_slide_id, true_bbox) is an identity -- and a
verifier whose result changed with the claim would be violating the task definition,
which the independence test here detects rather than silently caching away.
"""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyrafault.geom import BBoxL0
from pyrafault.verify.base import Localisation, Verifier
from pyrafault.verify.runner import PREDICTION_COLUMNS, run_verifier
from pyrafault.verify.thresholds import Thresholds


class _Counting(Verifier):
    name, revision = "counting", "1"

    def __init__(self):
        self.calls = 0

    def localise(self, inputs):
        self.calls += 1
        return Localisation(BBoxL0(1000, 1000, 100, 100), 0.9)


class _FakeDataset:
    """Two items sharing one (C, Q) and differing only in their claim."""

    def __init__(self):
        self._items = {
            "i0": self._item("i0", BBoxL0(1000, 1000, 100, 100)),
            "i1": self._item("i1", BBoxL0(5000, 5000, 100, 100)),
        }

    @staticmethod
    def _item(iid, claim, true_slide_id="S1"):
        from PIL import Image
        return {
            "inputs": {
                "context": Image.new("RGB", (64, 64)), "query": Image.new("RGB", (16, 16)),
                "context_bbox": BBoxL0(0, 0, 1000, 1000), "claimed_bbox": claim,
                "context_mpp": 2.0, "query_mpp": 0.5, "slide_id": "S1",
                "item_id": iid, "split": "test", "extraction_version": "v1",
            },
            "target": {"true_bbox": BBoxL0(1000, 1000, 100, 100),
                       "true_slide_id": true_slide_id},
        }

    @property
    def item_ids(self):
        return list(self._items)

    def __getitem__(self, k):
        return self._items[k]


def _thresholds():
    return Thresholds(tau=0.5, delta=0.5, fitted_on="calibration", n_items=10,
                      residuals={"n": 10, "p50": 0.1, "p75": 0.2, "p90": 0.3,
                                "p95": 0.4, "p99": 0.5})


def test_emits_one_row_per_item_with_the_declared_columns():
    df = run_verifier(_Counting(), _FakeDataset(), _thresholds())
    assert list(df.columns) == list(PREDICTION_COLUMNS)
    assert len(df) == 2


def test_the_same_context_and_query_is_localised_once():
    v = _Counting()
    run_verifier(v, _FakeDataset(), _thresholds())
    assert v.calls == 1, "localise should be cached across claims on the same (C, Q)"


def test_differing_claims_still_produce_differing_decisions():
    df = run_verifier(_Counting(), _FakeDataset(), _thresholds()).set_index("item_id")
    assert df.loc["i0", "decision"] == "verified"
    assert df.loc["i1", "decision"] == "mismatched"


def test_every_row_records_the_thresholds_it_was_decided_under():
    t = _thresholds()
    df = run_verifier(_Counting(), _FakeDataset(), t)
    assert set(df["thresholds_digest"]) == {t.digest()}


def test_a_verifier_that_reads_the_claim_is_caught():
    """Caching would otherwise hide the violation: the second item would silently
    receive the first item's answer."""
    class Cheating(Verifier):
        name, revision = "cheat", "1"
        def localise(self, inputs):
            return Localisation(inputs["claimed_bbox"], 0.9)

    with pytest.raises(AssertionError, match="claim"):
        run_verifier(Cheating(), _FakeDataset(), _thresholds(), audit_independence=True)


def test_the_audit_still_fires_under_python_dash_o():
    """Cleanup: the audit used to be a bare `assert`, which `python -O` strips --
    silently disabling the check the protocol promises fails loudly. It is now a
    `raise`, which optimisation cannot remove."""
    script = (
        "from pyrafault.verify.base import Localisation, Verifier\n"
        "from pyrafault.verify.runner import run_verifier\n"
        "from pyrafault.verify.thresholds import Thresholds\n"
        "from tests.test_verify_runner import _FakeDataset\n"
        "class Cheating(Verifier):\n"
        "    name, revision = 'cheat', '1'\n"
        "    def localise(self, inputs):\n"
        "        return Localisation(inputs['claimed_bbox'], 0.9)\n"
        "t = Thresholds(tau=0.5, delta=0.5, fitted_on='calibration', n_items=10,\n"
        "               residuals={'n': 1, 'p50': 0, 'p75': 0, 'p90': 0, 'p95': 0, 'p99': 0})\n"
        "run_verifier(Cheating(), _FakeDataset(), t, audit_independence=True)\n"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", script],
        capture_output=True, text=True,
        env={"PYTHONPATH": str(Path.cwd()), "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode != 0
    assert "claim" in result.stderr


class _OutOfSlideDataset:
    """Two items whose context and `true_bbox` coincide but whose served patch comes
    from different slides -- exactly the collision finding 1 describes. Distinguished
    only by `true_slide_id` and by the query image's pixel content (standing in for
    the fact that the two patches are genuinely different images even though their
    coordinates coincide)."""

    def __init__(self):
        self._items = {
            "i0": self._item("i0", "SLIDE_A", 0),
            "i1": self._item("i1", "SLIDE_B", 200),
        }

    @staticmethod
    def _item(iid, true_slide_id, shade):
        from PIL import Image
        return {
            "inputs": {
                "context": Image.new("RGB", (64, 64)),
                "query": Image.new("RGB", (16, 16), color=(shade, shade, shade)),
                "context_bbox": BBoxL0(0, 0, 1000, 1000),
                "claimed_bbox": BBoxL0(1000, 1000, 100, 100),
                "context_mpp": 2.0, "query_mpp": 0.5, "slide_id": "S1",
                "item_id": iid, "split": "test", "extraction_version": "v1",
            },
            "target": {"true_bbox": BBoxL0(1000, 1000, 100, 100),
                       "true_slide_id": true_slide_id},
        }

    @property
    def item_ids(self):
        return list(self._items)

    def __getitem__(self, k):
        return self._items[k]


class _ByQueryShade(Verifier):
    """A verifier whose estimate depends on the served query's pixel content, like a
    real one -- so two items with identical context/`true_bbox` coordinates but
    genuinely different patches get genuinely different estimates, unless the runner's
    cache wrongly collides them first."""
    name, revision = "by-shade", "1"

    def __init__(self):
        self.calls = 0

    def localise(self, inputs):
        self.calls += 1
        shade = inputs["query"].getpixel((0, 0))[0]
        return Localisation(BBoxL0(1000 + shade, 1000, 100, 100), 0.9)


def test_out_of_slide_items_sharing_context_and_true_bbox_are_not_collided():
    """Finding 1: two items differing only in `true_slide_id` are still different
    (context, query) pairs and must be localised independently. Fails against a cache
    key that omits `true_slide_id` -- both items would then collide on
    (slide_id, context_bbox, true_bbox) alone, and i1 would silently receive i0's
    localisation."""
    v = _ByQueryShade()
    df = run_verifier(v, _OutOfSlideDataset(), _thresholds()).set_index("item_id")
    assert v.calls == 2, "two genuinely different (C, Q) pairs must both be localised"
    assert df.loc["i0", "estimated_bbox_l0_x"] != df.loc["i1", "estimated_bbox_l0_x"], (
        "item i1 received item i0's cached localisation"
    )


class _Slow(Verifier):
    name, revision = "slow", "1"

    def localise(self, inputs):
        time.sleep(0.02)
        return Localisation(BBoxL0(1000, 1000, 100, 100), 0.9)


def test_a_cached_rows_runtime_ms_equals_the_computing_rows():
    """Finding 2: runtime_ms must record the cost of *computing* the answer, reused
    verbatim on a cache hit -- not the cost of whichever branch a given row happened
    to take. Fails when a hit times only the near-zero cache lookup instead of
    reusing the real compute cost."""
    df = run_verifier(_Slow(), _FakeDataset(), _thresholds()).set_index("item_id")
    assert df.loc["i0", "runtime_ms"] == df.loc["i1", "runtime_ms"], (
        "a cache hit's runtime_ms must equal the computing row's, not the cost of a "
        "dict lookup"
    )
    assert df.loc["i0", "runtime_ms"] > 10, "sanity: the real localise cost, not ~0"


class _SingleItemDataset:
    """One item -- no (context, query) key can ever collide, so a cache hit never
    happens."""

    def __init__(self):
        from PIL import Image
        self._items = {
            "only": {
                "inputs": {
                    "context": Image.new("RGB", (64, 64)),
                    "query": Image.new("RGB", (16, 16)),
                    "context_bbox": BBoxL0(0, 0, 1000, 1000),
                    "claimed_bbox": BBoxL0(1000, 1000, 100, 100),
                    "context_mpp": 2.0, "query_mpp": 0.5, "slide_id": "S1",
                    "item_id": "only", "split": "test", "extraction_version": "v1",
                },
                "target": {"true_bbox": BBoxL0(1000, 1000, 100, 100),
                           "true_slide_id": "S1"},
            },
        }

    @property
    def item_ids(self):
        return list(self._items)

    def __getitem__(self, k):
        return self._items[k]


def test_a_claim_reading_verifier_is_caught_even_with_no_key_collisions():
    """Finding 3: audit_independence must catch a claim-reading verifier even when
    every (context, patch) appears exactly once in the split -- previously only a
    cache hit triggered the recheck, so a key that never collided was never audited."""
    class Cheating(Verifier):
        name, revision = "cheat", "1"
        def localise(self, inputs):
            return Localisation(inputs["claimed_bbox"], 0.9)

    with pytest.raises(AssertionError, match="claim"):
        run_verifier(Cheating(), _SingleItemDataset(), _thresholds(),
                     audit_independence=True)
