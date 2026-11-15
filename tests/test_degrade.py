"""Degradation must replay identically, everywhere, forever.

The collaborator regenerates these pixels on their own machine from the frozen
parameters. A degradation that is not bit-reproducible would not raise anything there --
it would quietly produce a slightly different Q, and two sides would report numbers
computed on different images at a precision (median IoU 0.99) where that matters.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pyrafault.degrade import (KINDS, apply, degradation_digest, item_rng, sample_params)

STRENGTHS = {"none": 0.0, "stain_jitter": 1.0, "blur": 4.0,
             "resolution_loss": 4, "sensor_noise": 25.0, "geometry": 10.0}


def _patch(seed: int = 0, side: int = 64) -> Image.Image:
    rng = np.random.default_rng(seed)
    small = rng.integers(60, 240, size=(16, 16, 3), dtype=np.uint8)
    return Image.fromarray(small).resize((side, side), Image.BILINEAR)


@pytest.mark.parametrize("kind", KINDS)
def test_replaying_frozen_parameters_gives_identical_pixels(kind):
    img = _patch()
    params = sample_params(kind, STRENGTHS[kind], item_rng(7, "pair-x"))

    first = np.asarray(apply(img, kind, params))
    second = np.asarray(apply(img, kind, params))

    assert np.array_equal(first, second)


@pytest.mark.parametrize("kind", KINDS)
def test_parameters_are_a_function_of_the_pair_id_alone(kind):
    """Not of iteration order, and not of how many rows were drawn before this one."""
    a = sample_params(kind, STRENGTHS[kind], item_rng(7, "pair-x"))
    scratch = item_rng(7, "pair-other")
    for _ in range(5):
        scratch.random()
    b = sample_params(kind, STRENGTHS[kind], item_rng(7, "pair-x"))

    assert a == b


def test_none_is_the_identity():
    img = _patch()
    assert apply(img, "none", {}) is img


def test_the_cache_key_term_is_empty_only_for_none():
    assert degradation_digest("none", {}) == ""
    for kind in KINDS:
        if kind == "none":
            continue
        params = sample_params(kind, STRENGTHS[kind], item_rng(7, "p"))
        assert degradation_digest(kind, params), kind


def test_two_severities_of_one_operation_get_different_cache_terms():
    """The bug this prevents: two items serving the same true box under different
    severities share a cache entry, and the second is served the first's pixels."""
    mild = sample_params("geometry", 5.0, item_rng(7, "p"))
    severe = sample_params("geometry", 15.0, item_rng(7, "p"))

    assert degradation_digest("geometry", mild) != degradation_digest("geometry", severe)


def test_geometry_at_zero_rotation_and_unit_scale_changes_nothing():
    img = _patch()
    out = apply(img, "geometry", {"angle_deg": 0.0, "scale": 1.0})
    assert np.array_equal(np.asarray(out), np.asarray(img))


@pytest.mark.parametrize("kind,strength", [("geometry", 10.0), ("blur", 4.0)])
def test_a_degradation_actually_changes_the_pixels(kind, strength):
    """Guards the embarrassing case: an operator that silently no-ops would make a whole
    condition look like it had no effect, which is exactly the claim the sweep tests."""
    img = _patch()
    params = sample_params(kind, strength, item_rng(7, "p"))

    assert not np.array_equal(np.asarray(apply(img, kind, params)), np.asarray(img))
