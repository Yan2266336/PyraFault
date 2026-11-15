"""Localisation quality, cut by degradation band.

The headline `median_iou` is not comparable between methods when one method's residual
distribution is bimodal. Measured on the degraded sample freeze, the NCC control's median
IoU (0.905) reads *higher* than ZoomTrace's (0.801) while its balanced accuracy is 0.500
against 0.988 -- because NCC either nails a patch to the pixel or misses it entirely, and
a median over a two-humped distribution reports the taller hump, not the method. Per band
the two humps separate, which is the only form in which the number says anything.

The cut has to live in the scorer. `degradation_band` is in neither `inputs` (a verifier
is never told which band it got) nor `target`, so nothing downstream of `pairs` can
produce it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyrafault.verify.metrics import score_predictions

BANDS = ("none", "mild", "moderate", "severe")


def _pairs(offsets: dict) -> pd.DataFrame:
    """One clean pair per band per listed offset, `true_bbox` at the origin."""
    rows = []
    for band, offs in offsets.items():
        for i, _ in enumerate(offs):
            rows.append({
                "item_id": f"{band}-{i}", "claim_correct": True,
                "true_bbox_l0_x": 0, "true_bbox_l0_y": 0,
                "true_bbox_l0_w": 100, "true_bbox_l0_h": 100,
                "claimed_bbox_l0_x": 0, "claimed_bbox_l0_y": 0,
                "claimed_bbox_l0_w": 100, "claimed_bbox_l0_h": 100,
                "true_within_context": True, "fault_type": "clean",
                "degradation_band": band, "scale_ratio": 4.0, "context_mpp": 2.0,
            })
    return pd.DataFrame(rows)


def _predictions(offsets: dict) -> pd.DataFrame:
    rows = []
    for band, offs in offsets.items():
        for i, dx in enumerate(offs):
            rows.append({
                "item_id": f"{band}-{i}", "reliability": 0.9, "decision": "verified",
                "estimated_bbox_l0_x": dx, "estimated_bbox_l0_y": 0,
                "estimated_bbox_l0_w": 100, "estimated_bbox_l0_h": 100,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def scored():
    """Localisation that degrades with band: perfect under `none`, worse each step."""
    offsets = {"none": [0, 0], "mild": [10, 10], "moderate": [30, 30],
               "severe": [60, 60]}
    return score_predictions(_predictions(offsets), _pairs(offsets))


class TestEachBandCarriesItsOwnLocalisation:
    def test_every_band_reports_a_median_iou(self, scored):
        for band in BANDS:
            assert scored["by_degradation_band"][band]["median_iou"] is not None

    def test_every_band_reports_a_median_centre_error(self, scored):
        for band in BANDS:
            got = scored["by_degradation_band"][band]
            assert got["median_normalised_centre_error"] is not None

    def test_the_counts_are_the_localisable_items_not_the_band_size(self, scored):
        """`n` already means the band's item count; the localisation count can differ
        (an item with no estimate contributes to one and not the other), so it needs its
        own name rather than being read off `n`."""
        for band in BANDS:
            assert scored["by_degradation_band"][band]["n_localised"] == 2

    def test_the_numbers_track_the_band(self, scored):
        ious = [scored["by_degradation_band"][b]["median_iou"] for b in BANDS]
        assert ious == sorted(ious, reverse=True), f"not monotone in band: {ious}"
        errs = [scored["by_degradation_band"][b]["median_normalised_centre_error"]
                for b in BANDS]
        assert errs == sorted(errs)

    def test_the_bands_agree_with_the_headline_when_pooled(self, scored):
        """The per-band cut must be the same measurement as the headline one, not a
        second definition of it."""
        total = sum(scored["by_degradation_band"][b]["n_localised"] for b in BANDS)
        assert total == scored["localisation"]["n"]


class TestABandWithNothingToLocalise:
    def test_a_band_whose_every_estimate_is_none_reports_none_not_zero(self):
        """Zero would read as "localised perfectly badly"; the honest answer is that the
        band has no localisation to report."""
        offsets = {"none": [0], "severe": [0]}
        preds = _predictions(offsets)
        preds.loc[preds["item_id"] == "severe-0",
                  ["estimated_bbox_l0_x", "estimated_bbox_l0_y",
                   "estimated_bbox_l0_w", "estimated_bbox_l0_h"]] = None
        got = score_predictions(preds, _pairs(offsets))["by_degradation_band"]
        assert got["severe"]["median_iou"] is None
        assert got["severe"]["n_localised"] == 0
        assert got["none"]["median_iou"] == 1.0

    def test_items_outside_their_context_are_excluded_as_in_the_headline(self):
        """The headline localisation is computed over `true_within_context` only -- an
        out-of-context pair has no in-context answer to recover -- and the per-band cut
        has to use the same filter or the two disagree."""
        offsets = {"none": [0, 0]}
        pairs = _pairs(offsets)
        pairs.loc[0, "true_within_context"] = False
        got = score_predictions(_predictions(offsets), pairs)
        assert got["by_degradation_band"]["none"]["n_localised"] == 1
        assert got["localisation"]["n"] == 1


class TestAFreezeWithoutTheColumn:
    def test_nothing_is_invented_for_a_pre_degradation_freeze(self):
        offsets = {"none": [0]}
        pairs = _pairs(offsets).drop(columns=["degradation_band"])
        got = score_predictions(_predictions(offsets), pairs)
        assert got["by_degradation_band"] == {}
        assert got["localisation"]["median_iou"] == 1.0
