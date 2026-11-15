"""Which band each pair gets.

Balanced within every `(split, fault_type)` cell, because a band that correlated with
`claim_correct` would let a verifier predict the label without looking at a pixel --
the leakage audit checks exactly that, and this is the code that has to earn the pass.

Deterministic in `(seed, pair_id)`, not in row order: two runs over the same table in
different orders must assign the same band to the same pair, or a rebuilt freeze is not
the freeze whose numbers were reported.
"""
from __future__ import annotations

import hashlib
import json
from typing import Sequence

import pandas as pd

from pyrafault.degrade import item_rng, sample_params

# Fixed by measurement: 2 degrees barely moves the baseline and 45+ destroys both
# methods, so neither earns a band.
BANDS: tuple[str, ...] = ("none", "mild", "moderate", "severe")
# Public, because it is the only statement of what a band *means*: stage 09's
# `degradation_band` invariant re-derives (kind, |angle_deg|) from a row and compares it
# here rather than restating 5/10/15, so a band label cannot drift away from the
# parameters it names (pyrafault.qc.report.check_invariants).
BAND_SPEC: dict[str, tuple[str, float]] = {
    "none": ("none", 0.0), "mild": ("geometry", 5.0),
    "moderate": ("geometry", 10.0), "severe": ("geometry", 15.0)}


def _rank(seed: int, pair_id: str) -> int:
    """A stable pseudo-random ordering key. Hashing the pair id rather than drawing from
    a stream keeps the result independent of how many rows came before."""
    digest = hashlib.sha256(f"{seed}|{pair_id}|band".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def assign_bands(pairs: pd.DataFrame, seed: int,
                 bands: Sequence[str] = BANDS) -> pd.DataFrame:
    """Assign each row a band, balanced within its `(split, fault_type)` cell.

    Cells are ranked by `_rank(seed, pair_id)` and walked in that order, handing out
    `bands` cyclically. When a cell's size is not a multiple of `len(bands)`, the first
    `size % len(bands)` bands (in `bands` order) get one row more than the rest -- at
    most a one-row difference. That remainder pattern is a function of the cell's size
    alone, so every cell of the same size gets the same split; within one `split`, every
    `fault_type` cell is the same size by construction (one clean/drift/random/... row
    per query), so the *proportion* of each band a fault type contributes is identical
    across fault types even when the per-cell counts are not perfectly equal. That is
    what keeps `degradation_band` uncorrelated with `claim_correct` (which is constant
    within a fault type) even for a cell size the four bands do not divide evenly.
    """
    out = pairs.copy()
    order: dict[str, str] = {}
    for _, group in out.groupby(["split", "fault_type"], sort=True):
        ranked = sorted(group["pair_id"], key=lambda p: _rank(seed, p))
        for i, pair_id in enumerate(ranked):
            order[pair_id] = bands[i % len(bands)]

    kinds, params_json, band_col = [], [], []
    for pair_id in out["pair_id"]:
        band = order[pair_id]
        kind, strength = BAND_SPEC[band]
        params = sample_params(kind, strength, item_rng(seed, pair_id))
        kinds.append(kind)
        params_json.append(json.dumps(params, sort_keys=True))
        band_col.append(band)
    out["degradation_kind"] = kinds
    out["degradation_params"] = params_json
    out["degradation_band"] = band_col
    return out
