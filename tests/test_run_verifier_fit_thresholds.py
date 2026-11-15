"""`scripts/13_run_verifier.py --fit-thresholds`.

The plan deadlocks without this: `13_run_verifier.py` calls `Thresholds.load()` at
startup, but nothing in the repository calls `fit_tau`/`fit_delta`, so thresholds
would need to already exist before the calibration run that fits them. This mode
breaks the deadlock -- on the fitting split it localises every item without deciding,
fits tau and delta from those localisations, and hands back a `Thresholds` ready to
save. These tests exercise `fit_thresholds` directly (the same function `main()`
calls), alongside `tests/test_verify_runner.py`'s tests of `run_verifier` itself.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from PIL import Image

from pyrafault.geom import BBoxL0
from pyrafault.verify.base import Localisation, Verifier

ROOT = Path(__file__).resolve().parents[1]


def _load_stage13():
    path = ROOT / "scripts" / "13_run_verifier.py"
    spec = importlib.util.spec_from_file_location("stage13_run_verifier_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


stage13 = _load_stage13()


class _FakeVerifier(Verifier):
    name, revision = "fake", "1"

    def __init__(self, answers: dict):
        self.answers = answers

    def localise(self, inputs):
        return self.answers[inputs["item_id"]]


class _FitDataset:
    def __init__(self, items: dict):
        self._items = items

    @property
    def item_ids(self):
        return list(self._items)

    def __getitem__(self, k):
        return self._items[k]


def _item(item_id, claim_correct, true_bbox, ctx_x=0):
    """`ctx_x` shifts `context_bbox`, which the runner's cache key includes -- distinct
    items get distinct (C, Q) identities unless a test deliberately shares `ctx_x` to
    exercise the dedup path."""
    return {
        "inputs": {
            "context": Image.new("RGB", (64, 64)), "query": Image.new("RGB", (16, 16)),
            "context_bbox": BBoxL0(ctx_x, 0, 1000, 1000),
            "claimed_bbox": BBoxL0(1000, 1000, 100, 100),
            "context_mpp": 2.0, "query_mpp": 0.5, "slide_id": "S1",
            "item_id": item_id, "split": "calibration", "extraction_version": "v1",
        },
        "target": {"claim_correct": claim_correct, "true_bbox": true_bbox,
                   "true_slide_id": "S1"},
    }


TRUE_BBOX = BBoxL0(1000, 1000, 100, 100)


def test_fits_tau_and_delta_from_localisations_and_records_the_split():
    items = {
        "clean0": _item("clean0", True, TRUE_BBOX, ctx_x=0),
        "clean1": _item("clean1", True, TRUE_BBOX, ctx_x=1000),
        "fault0": _item("fault0", False, TRUE_BBOX, ctx_x=2000),
        "fault1": _item("fault1", False, TRUE_BBOX, ctx_x=3000),
    }
    answers = {
        # residual = center_distance((1010,1000,100,100), true) / 100 = 10/100 = 0.1
        "clean0": Localisation(BBoxL0(1010, 1000, 100, 100), 0.9),
        "clean1": Localisation(None, 0.9),  # skipped from the residual, kept for tau
        "fault0": Localisation(BBoxL0(2000, 2000, 100, 100), 0.2),
        "fault1": Localisation(BBoxL0(2000, 2000, 100, 100), 0.8),
    }
    t = stage13.fit_thresholds(_FakeVerifier(answers), _FitDataset(items), "calibration")

    assert t.fitted_on == "calibration"
    assert t.n_items == 4
    assert t.delta == pytest.approx(0.1, abs=1e-6)
    # Both faults estimate (2000,2000) against a claim at (1000,1000) — 1414 px apart,
    # far outside delta * claim.w = 10 px — so both are correctly rejected and neither
    # is a false verification. Nothing needs suppressing, so tau falls below every
    # score rather than above the largest wrong one. Under the old definition, which
    # counted admission rather than acceptance, these two correct rejections would have
    # pushed tau to 0.8 and abstained on most of the split.
    assert t.tau == pytest.approx(0.2, abs=1e-6)
    assert t.tau < min(a.reliability for a in answers.values())


def test_a_clean_items_none_estimate_is_excluded_from_delta_but_counted():
    """A clean item that failed to localise must not corrupt the residual, but it is
    still a calibration item and still informs tau (as a clean, never-wrong score)."""
    items = {
        "clean0": _item("clean0", True, TRUE_BBOX, ctx_x=0),
        "clean1": _item("clean1", True, TRUE_BBOX, ctx_x=1000),
        "fault0": _item("fault0", False, TRUE_BBOX, ctx_x=2000),
    }
    answers = {
        "clean0": Localisation(None, 0.9),
        "clean1": Localisation(BBoxL0(1050, 1000, 100, 100), 0.9),  # residual = 0.5
        "fault0": Localisation(BBoxL0(2000, 2000, 100, 100), 0.1),
    }
    t = stage13.fit_thresholds(_FakeVerifier(answers), _FitDataset(items), "calibration")

    assert t.n_items == 3
    assert t.delta == pytest.approx(0.5, abs=1e-6)


def test_no_clean_items_raises_via_fit_delta():
    items = {"fault0": _item("fault0", False, TRUE_BBOX, ctx_x=0),
             "fault1": _item("fault1", False, TRUE_BBOX, ctx_x=1000)}
    answers = {
        "fault0": Localisation(BBoxL0(2000, 2000, 100, 100), 0.2),
        "fault1": Localisation(BBoxL0(2000, 2000, 100, 100), 0.8),
    }
    with pytest.raises(ValueError, match="no clean"):
        stage13.fit_thresholds(_FakeVerifier(answers), _FitDataset(items), "calibration")


def test_no_faulty_items_raises_via_fit_tau():
    items = {"clean0": _item("clean0", True, TRUE_BBOX, ctx_x=0),
             "clean1": _item("clean1", True, TRUE_BBOX, ctx_x=1000)}
    answers = {
        "clean0": Localisation(BBoxL0(1010, 1000, 100, 100), 0.9),
        "clean1": Localisation(BBoxL0(1010, 1000, 100, 100), 0.9),
    }
    with pytest.raises(ValueError, match="no faulty"):
        stage13.fit_thresholds(_FakeVerifier(answers), _FitDataset(items), "calibration")


def test_items_sharing_a_context_and_query_are_localised_once():
    """clean/adjacent_drift/hard_mismatch/random_same_context on one query all share
    (slide_id, context_bbox, true_slide_id, true_bbox) -- localise must run once for
    the pair and be reused, the same dedup run_verifier does for the decide pass."""
    calls = {"n": 0}

    class _Counting(_FakeVerifier):
        def localise(self, inputs):
            calls["n"] += 1
            return super().localise(inputs)

    items = {
        "clean0": _item("clean0", True, TRUE_BBOX),
        "drift0": _item("drift0", False, TRUE_BBOX),  # same true_bbox/context as clean0
    }
    answers = {
        "clean0": Localisation(BBoxL0(1010, 1000, 100, 100), 0.9),
        "drift0": Localisation(BBoxL0(1010, 1000, 100, 100), 0.9),
    }
    v = _Counting(answers)
    t = stage13.fit_thresholds(v, _FitDataset(items), "calibration")

    assert calls["n"] == 1, "localise should be cached across items sharing (C, Q)"
    assert t.n_items == 2, "both items still count toward n_items and the fitted scores"


def test_a_clean_items_true_bbox_width_disagreeing_with_claimed_bbox_width_raises():
    """Cleanup: fit_thresholds normalises the residual by true_bbox.w while `decide`
    (base.py) normalises by claimed_bbox.w. They are equal for every current fault
    type, but nothing asserted it -- if they ever diverged, delta would silently mean
    a different tolerance at fit time than at decide time."""
    item = _item("clean0", True, TRUE_BBOX, ctx_x=0)
    item["inputs"] = {**item["inputs"], "claimed_bbox": BBoxL0(1000, 1000, 999, 100)}
    items = {"clean0": item}
    answers = {"clean0": Localisation(BBoxL0(1010, 1000, 100, 100), 0.9)}
    with pytest.raises(AssertionError, match="true_bbox.w.*claimed_bbox.w"):
        stage13.fit_thresholds(_FakeVerifier(answers), _FitDataset(items), "calibration")


def test_cli_declares_a_fit_thresholds_flag():
    """Guards the CLI surface `main()` parses -- a rename here would silently strand
    the runbook command that passes `--fit-thresholds`."""
    import inspect

    src = inspect.getsource(stage13.main)
    assert "--fit-thresholds" in src
