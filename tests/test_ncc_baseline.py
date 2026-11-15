"""The protocol's worked skeleton, run the way a baseline author would run it.

VERIFIER_PROTOCOL §11 promises "a complete, runnable" baseline. That promise is only
worth something if the code on the page is the code that runs, and if a verifier the
harness has never seen goes through every public step -- fit on calibration, decide
with the claim-independence audit on, score -- without touching ZoomTrace.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from pyrafault.geom import BBoxL0
from pyrafault.verify.metrics import score_predictions
from pyrafault.verify.ncc_baseline import NccBaseline, _ncc_map
from pyrafault.verify.runner import fit_thresholds, run_verifier

ROOT = Path(__file__).resolve().parents[1]

CTX_PX, Q_PX = 256, 64          # context_mpp / query_mpp = 4 -> a 16 px template
L0_PER_CTX_PX = 8               # context_bbox spans 2048 level-0 px


def _texture(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, size=(40, 40, 3), dtype=np.uint8)
    return Image.fromarray(small).resize((CTX_PX, CTX_PX), Image.BILINEAR)


def _l0(ctx_bbox: BBoxL0, x: int, y: int) -> BBoxL0:
    side = (Q_PX // 4) * L0_PER_CTX_PX
    return BBoxL0(ctx_bbox.x + x * L0_PER_CTX_PX, ctx_bbox.y + y * L0_PER_CTX_PX, side, side)


def _crop_query(img: Image.Image, x: int, y: int) -> Image.Image:
    side = Q_PX // 4
    return img.crop((x, y, x + side, y + side)).resize((Q_PX, Q_PX), Image.LANCZOS)


class _Dataset:
    """Clean, adjacent-drift and out-of-slide items over a few synthetic contexts."""

    def __init__(self, split: str, seed0: int):
        self.items, self.rows = {}, []
        for c in range(3):
            ctx_img = _texture(seed0 + c)
            foreign = _texture(seed0 + 100 + c)
            ctx_bbox = BBoxL0(10_000 + 50_000 * c, 20_000, CTX_PX * L0_PER_CTX_PX,
                              CTX_PX * L0_PER_CTX_PX)
            # Off any coarse grid, on purpose.
            for q, (x, y) in enumerate([(41, 63), (179, 118)]):
                true = _l0(ctx_bbox, x, y)
                drifted = _l0(ctx_bbox, x + 32, y)
                query = _crop_query(ctx_img, x, y)
                self._add(f"{split}-c{c}q{q}-clean", split, ctx_img, query, ctx_bbox,
                          claimed=true, true=true, fault="clean", true_slide="S")
                self._add(f"{split}-c{c}q{q}-drift", split, ctx_img, query, ctx_bbox,
                          claimed=drifted, true=true, fault="adjacent_drift", true_slide="S")
                self._add(f"{split}-c{c}q{q}-oos", split, ctx_img, _crop_query(foreign, x, y),
                          ctx_bbox, claimed=true, true=true, fault="out_of_slide",
                          true_slide="OTHER")

    def _add(self, item_id, split, ctx_img, query, ctx_bbox, *, claimed, true, fault,
             true_slide):
        clean = fault == "clean"
        self.items[item_id] = {
            "inputs": {"context": ctx_img, "query": query, "context_bbox": ctx_bbox,
                       "claimed_bbox": claimed, "context_mpp": 2.0, "query_mpp": 0.5,
                       "slide_id": "S", "item_id": item_id, "split": split,
                       "extraction_version": "v1"},
            "target": {"claim_correct": clean, "true_bbox": true,
                       "true_slide_id": true_slide},
        }
        self.rows.append({
            "item_id": item_id, "claim_correct": clean, "fault_type": fault,
            "true_within_context": true_slide == "S",
            **{f"true_bbox_l0_{k}": getattr(true, k) for k in "xywh"},
            **{f"claimed_bbox_l0_{k}": getattr(claimed, k) for k in "xywh"},
            "scale_ratio": 4.0, "context_mpp": 2.0, "hardness_score": 0.5,
        })

    @property
    def item_ids(self):
        return list(self.items)

    def __getitem__(self, k):
        return self.items[k]


def _section_11_code(doc: str) -> str:
    text = (ROOT / doc).read_text(encoding="utf-8")
    blocks = [b for b in re.findall(r"```python\n(.*?)```", text, re.S)
              if "class NccBaseline" in b]
    assert len(blocks) == 1, f"{doc}: expected exactly one NccBaseline code block"
    return blocks[0]


@pytest.mark.parametrize("doc", ["VERIFIER_PROTOCOL.md", "VERIFIER_PROTOCOL.zh-CN.md"])
def test_the_skeleton_on_the_page_is_the_module_that_runs(doc):
    module = (ROOT / "pyrafault" / "verify" / "ncc_baseline.py").read_text(encoding="utf-8")
    assert _section_11_code(doc) == module


def test_the_ncc_map_is_exhaustive_ncc_at_every_offset():
    rng = np.random.default_rng(0)
    image = rng.random((40, 50)) * 255
    template = image[7:16, 23:34] + rng.random((9, 11))

    def ncc(w, t):
        w, t = w - w.mean(), t - t.mean()
        return (w * t).sum() / np.sqrt((w ** 2).sum() * (t ** 2).sum())

    brute = np.array([[ncc(image[y:y + 9, x:x + 11], template) for x in range(40)]
                      for y in range(32)])
    np.testing.assert_allclose(_ncc_map(image, template), brute, atol=1e-10)


def test_the_skeleton_recovers_a_planted_source():
    """At an offset no coarse stride would sample: an earlier stride-side//4 skeleton
    never landed on the source on real slides, whose query origins are jittered."""
    ds = _Dataset("dev", seed0=0)
    item = ds["dev-c0q1-clean"]
    loc = NccBaseline().localise(item["inputs"])
    assert loc.estimated_bbox == item["target"]["true_bbox"]


def test_a_verifier_the_harness_has_never_seen_runs_fit_decide_score():
    calibration, test = _Dataset("calibration", seed0=1), _Dataset("test", seed0=50)
    verifier = NccBaseline()

    thresholds = fit_thresholds(verifier, calibration, "calibration")
    predictions = run_verifier(verifier, test, thresholds, audit_independence=True)
    metrics = score_predictions(predictions, pd.DataFrame(test.rows))

    assert thresholds.fitted_on == "calibration"
    assert metrics["n_items"] == len(test.item_ids)
    by = metrics["by_fault_type"]
    assert by["clean"]["verified_rate"] == pytest.approx(1.0)
    assert by["adjacent_drift"]["verified_rate"] == pytest.approx(0.0)
    assert by["out_of_slide"]["verified_rate"] == pytest.approx(0.0)
    assert metrics["localisation"]["median_iou"] == pytest.approx(1.0)


def _load_stage13():
    spec = importlib.util.spec_from_file_location(
        "stage13_plugin_test", ROOT / "scripts" / "13_run_verifier.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stage13_loads_a_verifier_by_module_and_class():
    stage13 = _load_stage13()
    assert isinstance(stage13.load_plugin("pyrafault.verify.ncc_baseline:NccBaseline"),
                      NccBaseline)


@pytest.mark.parametrize("spec", [
    "pyrafault.verify.ncc_baseline",            # no class
    "pyrafault.verify.ncc_baseline:Missing",    # no such attribute
    "pyrafault.geom:BBoxL0",                    # not a Verifier
])
def test_stage13_rejects_a_spec_that_is_not_a_verifier(spec):
    with pytest.raises(SystemExit):
        _load_stage13().load_plugin(spec)
