"""Band assignment is balanced, deterministic, and independent of the label."""
from __future__ import annotations

import pandas as pd

from pyrafault.degrade_assign import BANDS, assign_bands


def _pairs(n_per_cell=40):
    rows = []
    for split in ("calibration", "test"):
        for fault in ("clean", "adjacent_drift", "out_of_slide"):
            for i in range(n_per_cell):
                rows.append({"pair_id": f"{split}-{fault}-{i}", "split": split,
                             "fault_type": fault, "claim_correct": fault == "clean"})
    return pd.DataFrame(rows)


def test_every_cell_gets_every_band_in_equal_numbers():
    out = assign_bands(_pairs(), seed=1, bands=BANDS)

    counts = out.groupby(["split", "fault_type", "degradation_band"]).size().unstack()
    assert (counts.nunique(axis=1) == 1).all(), counts


def test_assignment_does_not_correlate_with_the_label():
    """Unequal bands across claim_correct would be a pixel-free predictor of it."""
    out = assign_bands(_pairs(), seed=1, bands=BANDS)

    share = out.groupby("degradation_band")["claim_correct"].mean()
    assert share.max() - share.min() < 1e-9, share


def test_assignment_is_a_function_of_seed_and_pair_id():
    first = assign_bands(_pairs(), seed=1, bands=BANDS)
    shuffled = _pairs().sample(frac=1.0, random_state=7).reset_index(drop=True)
    second = assign_bands(shuffled, seed=1, bands=BANDS)

    merged = first.merge(second, on="pair_id", suffixes=("_a", "_b"))
    assert (merged.degradation_band_a == merged.degradation_band_b).all()
    assert (merged.degradation_params_a == merged.degradation_params_b).all()


def test_a_different_seed_can_move_the_assignment():
    """Otherwise the seed argument would be decorative."""
    first = assign_bands(_pairs(), seed=1, bands=BANDS)
    second = assign_bands(_pairs(), seed=2, bands=BANDS)

    merged = first.merge(second, on="pair_id", suffixes=("_a", "_b"))
    assert (merged.degradation_band_a != merged.degradation_band_b).any()


def test_a_cell_size_not_a_multiple_of_four_still_keeps_the_label_share_uniform():
    """750 % 4 != 0 (data_lung_sample's calibration split): a cell cannot split evenly
    into four bands, but every fault_type cell in one split is the same size by
    construction, so the remainder pattern -- identical per cell size -- keeps the
    aggregate label share exact even though per-cell counts differ by up to one row."""
    rows = []
    for fault in ("clean", "adjacent_drift", "out_of_slide"):
        for i in range(750):
            rows.append({"pair_id": f"calibration-{fault}-{i}", "split": "calibration",
                         "fault_type": fault, "claim_correct": fault == "clean"})
    pairs = pd.DataFrame(rows)

    out = assign_bands(pairs, seed=1, bands=BANDS)

    counts = out.groupby(["split", "fault_type", "degradation_band"]).size().unstack()
    assert (counts.max(axis=1) - counts.min(axis=1) <= 1).all(), counts

    share = out.groupby("degradation_band")["claim_correct"].mean()
    assert share.max() - share.min() < 1e-9, share


def test_bands_map_to_the_measured_angles():
    """Bands are fixed by measurement: none 0deg, mild 5deg, moderate 10deg, severe
    15deg. No other angles."""
    import json

    out = assign_bands(_pairs(n_per_cell=8), seed=1, bands=BANDS)
    angle_by_band = {}
    for row in out.itertuples():
        params = json.loads(row.degradation_params)
        angle_by_band.setdefault(row.degradation_band, set()).add(
            abs(params["angle_deg"]) if row.degradation_band != "none" else 0.0
        )

    assert angle_by_band["none"] == {0.0}
    assert angle_by_band["mild"] == {5.0}
    assert angle_by_band["moderate"] == {10.0}
    assert angle_by_band["severe"] == {15.0}
