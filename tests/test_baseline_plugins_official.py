"""Opt-in smoke tests for the eight handover entry points with real local assets.

No test in this file downloads a model or reads a WSI.  NCC always runs; each learned
method is explicitly skipped when none of its asset variables is configured.  Once
any variable for a method is configured, the complete offline preflight is mandatory:
a missing peer path, wrong Git commit, dirty tracked tree, or wrong checkpoint digest
is a failure, never a skip.

The synthetic RGB pair only checks that the new plugin entry point can execute its
published frozen backend and obey the shared localisation contract.  It is not an
accuracy test and does not require the method to recover the planted crop.
"""
from __future__ import annotations

import gc
import math
import os

import numpy as np
import pytest
from PIL import Image

from pyrafault.geom import BBoxL0
from pyrafault.verify.base import Localisation
from scripts import check_baseline_assets as assets


pytestmark = pytest.mark.slow


def _asset_variables(method: str) -> tuple[str, ...]:
    spec = assets._method_assets()[method]
    variables = [source.variable for source in spec.sources]
    variables.extend(file.variable for file in spec.files)
    if spec.dino_snapshot:
        variables.append("PYRAFAULT_DINOV2_SNAPSHOT")
    return tuple(dict.fromkeys(variables))


def _checked_or_skip(method: str):
    """Return successful preflight results, skipping only a wholly absent setup."""
    variables = _asset_variables(method)
    results = assets.check_methods([method])
    if variables and not any(os.environ.get(variable) for variable in variables):
        pytest.skip(
            f"{method}: official assets are not configured; set "
            + ", ".join(variables)
        )
    failures = [result for result in results if not result.passed]
    if failures:
        detail = "\n".join(
            f"- {result.check}: {result.detail}" for result in failures
        )
        pytest.fail(
            f"{method}: configured official assets failed the offline preflight:\n{detail}",
            pytrace=False,
        )
    return results


def _synthetic_inputs():
    rng = np.random.default_rng(20261011)
    pixels = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    # Add stable low-frequency structure as well as texture; methods are allowed to
    # return None, but the smoke input should not be a degenerate constant image.
    pixels[16:112, 60:68] = [240, 30, 90]
    pixels[60:68, 16:112] = [20, 220, 80]
    context = Image.fromarray(pixels, "RGB")
    query = context.crop((32, 32, 96, 96))
    context_bbox = BBoxL0(1000, 2000, 1280, 1280)
    planted = BBoxL0(1320, 2320, 640, 640)
    return {
        "context": context,
        "query": query,
        "context_bbox": context_bbox,
        "claimed_bbox": planted,
        "context_mpp": 1.0,
        "query_mpp": 1.0,
        "slide_id": "synthetic-no-wsi",
        "item_id": "official-plugin-smoke",
        "split": "test",
        "extraction_version": "synthetic-v1",
    }


def _assert_localisation_contract(localisation: Localisation, context_bbox: BBoxL0):
    assert isinstance(localisation, Localisation)
    assert math.isfinite(float(localisation.reliability))
    box = localisation.estimated_bbox
    if box is None:
        return
    assert isinstance(box, BBoxL0)
    assert all(isinstance(value, int) for value in (box.x, box.y, box.w, box.h))
    assert box.w > 0 and box.h > 0
    assert context_bbox.contains(box)


@pytest.mark.parametrize("method", assets.METHODS)
def test_official_plugin_localises_one_synthetic_rgb_pair_offline(method, monkeypatch):
    _checked_or_skip(method)

    # CPU is the safe handover default.  A postdoc can opt into a configured GPU by
    # exporting PYRAFAULT_BASELINE_DEVICE=cuda before selecting this slow test.
    if "PYRAFAULT_BASELINE_DEVICE" not in os.environ:
        monkeypatch.setenv("PYRAFAULT_BASELINE_DEVICE", "cpu")

    torch = None
    previous_precision = None
    plugin = backend = modules = first = second = None
    try:
        if method != "ncc":
            # With assets configured this is an acceptance test, so a missing runtime
            # dependency is a failure rather than another reason to skip.
            import torch as torch_module
            torch = torch_module
            previous_precision = torch.get_float32_matmul_precision()
            torch.set_float32_matmul_precision("highest")

        # This factory is the same method -> new zero-argument plugin mapping used by
        # the handover preflight.  It intentionally does not construct an old registry
        # or instantiate an adapter class directly.
        plugin = assets._plugin_for(method)
        if method == "ncc":
            assert plugin.__class__.__module__ == "pyrafault.verify.ncc_baseline"
        else:
            assert plugin.__class__.__module__ == "pyrafault.verify.plugins"

        inputs = _synthetic_inputs()
        first = plugin.localise(inputs)
        _assert_localisation_contract(first, inputs["context_bbox"])

        changed_claim = BBoxL0(
            inputs["context_bbox"].x,
            inputs["context_bbox"].y,
            inputs["claimed_bbox"].w,
            inputs["claimed_bbox"].h,
        )
        second = plugin.localise({**inputs, "claimed_bbox": changed_claim})
        _assert_localisation_contract(second, inputs["context_bbox"])
        assert second.estimated_bbox == first.estimated_bbox
        assert second.reliability == first.reliability

        if method != "ncc":
            backend = plugin._backend_instance
            assert backend is not None
            modules = assets._expected_torch_modules(method, backend)
            assert modules, f"{method}: no expected torch modules were found"
            frozen = assets._check_frozen_modules(method, modules)
            failures = [result for result in frozen if not result.passed]
            assert not failures, "; ".join(
                f"{result.check}: {result.detail}" for result in failures
            )
    finally:
        # One parameter owns at most one model.  Drop every reference before the next
        # method and return cached accelerator memory to the runtime.
        first = second = modules = backend = plugin = None
        gc.collect()
        if torch is not None:
            if previous_precision is not None:
                torch.set_float32_matmul_precision(previous_precision)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
