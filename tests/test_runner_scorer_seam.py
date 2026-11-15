"""The seam between `run_verifier` and `score_predictions`.

Every scorer test in `test_verify_metrics.py` hand-builds a 7-column predictions
frame; the runner actually emits all eleven `PREDICTION_COLUMNS`. `test_verify_runner.py`
only checks `PREDICTION_COLUMNS` against itself (`list(df.columns) == list(PREDICTION_COLUMNS)`)
-- a change that renamed a column consistently everywhere (e.g. `reliability` -> `rho`
in both `runner.py` and its own `PREDICTION_COLUMNS` tuple) would leave every existing
test green and break only the real pipeline, at the one place a hand-built fixture
never exercises: feeding the runner's actual DataFrame into the scorer.

This test runs `run_verifier` over a small fake dataset and feeds its real output
straight into `score_predictions`, so the two modules' contracts are checked against
each other instead of against two independently hand-maintained fixtures.
"""
from __future__ import annotations

import pandas as pd
import pytest
from PIL import Image

from pyrafault.geom import BBoxL0
from pyrafault.verify.base import Localisation, Verifier
from pyrafault.verify.metrics import score_predictions
from pyrafault.verify.runner import run_verifier
from pyrafault.verify.thresholds import Thresholds

TRUE_BBOX = BBoxL0(1000, 1000, 100, 100)
FAR_BBOX = BBoxL0(5000, 5000, 100, 100)


class _DeterministicVerifier(Verifier):
    """Estimates the true source exactly for `item_id`s starting with "clean", and
    somewhere far away otherwise -- so `claim_correct` and the decision correlate,
    giving `score_predictions` something non-degenerate to compute over."""
    name, revision = "seam-fake", "1"

    def localise(self, inputs):
        if inputs["item_id"].startswith("clean"):
            return Localisation(TRUE_BBOX, 0.9)
        return Localisation(FAR_BBOX, 0.9)


class _FakeDataset:
    def __init__(self):
        self._items = {
            "clean0": self._item("clean0", True, ctx_x=0),
            "clean1": self._item("clean1", True, ctx_x=1000),
            "fault0": self._item("fault0", False, ctx_x=2000),
            "fault1": self._item("fault1", False, ctx_x=3000),
        }

    @staticmethod
    def _item(item_id, claim_correct, ctx_x):
        return {
            "inputs": {
                "context": Image.new("RGB", (64, 64)),
                "query": Image.new("RGB", (16, 16)),
                "context_bbox": BBoxL0(ctx_x, 0, 1000, 1000),
                "claimed_bbox": TRUE_BBOX,
                "context_mpp": 2.0, "query_mpp": 0.5, "slide_id": "S1",
                "item_id": item_id, "split": "test", "extraction_version": "v1",
            },
            "target": {"claim_correct": claim_correct, "true_bbox": TRUE_BBOX,
                       "true_slide_id": "S1"},
        }

    @property
    def item_ids(self):
        return list(self._items)

    def __getitem__(self, k):
        return self._items[k]


def _pairs_for(dataset: _FakeDataset) -> pd.DataFrame:
    """The benchmark's `pairs` table, built from the same fake dataset so the two
    sides of the seam describe the same items."""
    rows = []
    for item_id in dataset.item_ids:
        item = dataset[item_id]
        target = item["target"]
        claimed = item["inputs"]["claimed_bbox"]
        true = target["true_bbox"]
        rows.append({
            "item_id": item_id,
            "claim_correct": target["claim_correct"],
            "fault_type": "clean" if target["claim_correct"] else "adjacent_drift",
            "true_within_context": True,
            "true_bbox_l0_x": true.x, "true_bbox_l0_y": true.y,
            "true_bbox_l0_w": true.w, "true_bbox_l0_h": true.h,
            "claimed_bbox_l0_x": claimed.x, "claimed_bbox_l0_y": claimed.y,
            "claimed_bbox_l0_w": claimed.w, "claimed_bbox_l0_h": claimed.h,
            "scale_ratio": 4.0, "hardness_score": 0.5,
        })
    return pd.DataFrame(rows)


def _thresholds() -> Thresholds:
    return Thresholds(tau=0.5, delta=0.5, fitted_on="calibration", n_items=4,
                      residuals={"n": 4, "p50": 0.0, "p75": 0.0, "p90": 0.0,
                                "p95": 0.0, "p99": 0.0})


def test_the_runners_real_output_scores_through_the_real_scorer():
    dataset = _FakeDataset()
    predictions = run_verifier(_DeterministicVerifier(), dataset, _thresholds())
    pairs = _pairs_for(dataset)

    metrics = score_predictions(predictions, pairs)

    assert metrics["n_items"] == 4
    assert metrics["coverage"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["verification_score"]["mean_clean"] is not None
    assert metrics["verification_score"]["mean_fault"] is not None
    assert metrics["verification_score"]["auroc"] is not None
    assert metrics["localisation"]["n"] == 4
    assert set(metrics["by_fault_type"]) == {"clean", "adjacent_drift"}
