"""EfficientLoFTR adapter tests; ordinary tests use no model or download."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest
from PIL import Image

from pyrafault.geom import BBoxL0
from pyrafault.verify.base import Localisation, Verifier
from pyrafault.verify.efficient_loftr import (
    EfficientLoFTRBaseline,
    Matches,
    OfficialEfficientLoFTR,
)
from pyrafault.verify.runner import run_verifier
from pyrafault.verify.thresholds import Thresholds


def _image(width, height, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(
        rng.integers(0, 256, (height, width, 3), dtype=np.uint8), "RGB"
    )


def _inputs(*, context_size=(64, 48), query_size=(20, 12),
            context_bbox=BBoxL0(1000, 2000, 6400, 2400),
            context_mpp=1.0, query_mpp=0.5):
    return {
        "context": _image(*context_size, seed=1),
        "query": _image(*query_size, seed=2),
        "context_bbox": context_bbox,
        "claimed_bbox": BBoxL0(0, 0, 1, 1),
        "context_mpp": context_mpp,
        "query_mpp": query_mpp,
        "slide_id": "not-an-algorithm-input",
        "item_id": "not-an-algorithm-input",
        "split": "test",
        "extraction_version": "v1",
    }


class _Matcher:
    revision = "fake-1"
    max_input_side = 2048

    def __init__(self, query_points, context_points, confidence=None,
                 expected_shapes=None):
        self.query_points = np.asarray(query_points, dtype=np.float32)
        self.context_points = np.asarray(context_points, dtype=np.float32)
        if confidence is None:
            confidence = np.ones(len(self.query_points), dtype=np.float32)
        self.confidence = np.asarray(confidence, dtype=np.float32)
        self.expected_shapes = expected_shapes
        self.calls = []

    def match(self, query, context):
        assert query.dtype == context.dtype == np.uint8
        assert query.ndim == context.ndim == 2
        if self.expected_shapes is not None:
            assert (query.shape, context.shape) == self.expected_shapes
        self.calls.append((query.copy(), context.copy()))
        return Matches(self.query_points, self.context_points, self.confidence)


def _translated_matches(dx=21, dy=16):
    query = np.array([
        [1.0, 1.0], [3.0, 1.0], [5.0, 2.0],
        [2.0, 3.0], [6.0, 3.0], [8.0, 4.0],
    ])
    return query, query + [dx, dy]


class _GuardedInputs(Mapping):
    allowed = {"context", "query", "context_bbox", "context_mpp", "query_mpp"}

    def __init__(self, values):
        self.values = values
        self.accessed = set()

    def __getitem__(self, key):
        assert key in self.allowed, f"baseline read forbidden input {key!r}"
        self.accessed.add(key)
        return self.values[key]

    def __iter__(self):
        raise AssertionError("baseline must not scan metadata keys")

    def __len__(self):
        return len(self.values)


class TestEfficientLoFTRContract:
    def test_is_a_verifier_and_returns_localisation(self):
        query, context = _translated_matches()
        baseline = EfficientLoFTRBaseline(_Matcher(query, context))
        result = baseline.localise(_inputs())
        assert isinstance(baseline, Verifier)
        assert isinstance(result, Localisation)
        assert baseline.name == "efficient_loftr_baseline"
        assert "fake-1" in baseline.revision

    def test_aligns_mpp_query_first_pads_without_resizing_context(self):
        query, context = _translated_matches()
        matcher = _Matcher(
            query, context,
            expected_shapes=((32, 32), (64, 64)),
        )
        EfficientLoFTRBaseline(matcher).localise(_inputs())
        sent_query, sent_context = matcher.calls[0]
        # Q: 20x12 at 0.5 mpp -> 10x6 at C's 1.0 mpp, then bottom/right pad.
        assert sent_query[:6, :10].shape == (6, 10)
        assert np.all(sent_query[6:, :] == sent_query[5:6, :])
        assert sent_context[:48, :64].shape == (48, 64)

    def test_translation_and_separate_xy_level0_scales(self):
        query, context = _translated_matches(dx=21, dy=16)
        loc = EfficientLoFTRBaseline(_Matcher(query, context)).localise(_inputs())
        # scale_x=100 and scale_y=50; physical footprint is 10x6 C pixels.
        assert loc.estimated_bbox == BBoxL0(3100, 2800, 1000, 300)
        assert loc.reliability == 1.0
        assert loc.diagnostics["translation_xy_context"] == [21.0, 16.0]

    def test_unrounded_physical_footprint_is_used_for_returned_bbox(self):
        # Q physical footprint is 19*.7/2=6.65 by 11*.7/2=3.85 C pixels;
        # only the model raster is rounded to 7x4.
        query = np.array([[1, 1], [2, 1], [3, 1], [4, 2], [5, 2]], dtype=float)
        query_in_context_pixels = query * [6.65 / 7, 3.85 / 4]
        context = query_in_context_pixels + [10, 8]
        loc = EfficientLoFTRBaseline(_Matcher(query, context)).localise(_inputs(
            query_size=(19, 11), context_mpp=2.0, query_mpp=0.7,
            context_bbox=BBoxL0(100, 200, 640, 480),
        ))
        assert loc.estimated_bbox == BBoxL0(200, 280, 66, 38)
        assert loc.diagnostics["aligned_query_size_px"] == [7, 4]
        assert loc.diagnostics["query_to_context_point_scale"] == pytest.approx(
            [6.65 / 7, 3.85 / 4]
        )

    def test_claim_and_label_correlated_metadata_are_never_read(self):
        query, context = _translated_matches()
        guarded = _GuardedInputs(_inputs())
        EfficientLoFTRBaseline(_Matcher(query, context)).localise(guarded)
        assert guarded.accessed == guarded.allowed

    def test_changing_claim_and_metadata_does_not_change_output(self):
        query, context = _translated_matches()
        baseline = EfficientLoFTRBaseline(_Matcher(query, context))
        first = _inputs()
        second = dict(first, claimed_bbox=BBoxL0(999, 888, 7, 6),
                      item_id="different", slide_id="different", split="dev")
        assert baseline.localise(first) == baseline.localise(second)

    def test_is_exactly_deterministic(self):
        query, context = _translated_matches()
        matcher = _Matcher(query, context)
        baseline = EfficientLoFTRBaseline(matcher)
        assert baseline.localise(_inputs()) == baseline.localise(_inputs())

    def test_shared_runner_independence_audit_passes(self):
        query, context = _translated_matches()
        inputs = _inputs()
        true_bbox = BBoxL0(3100, 2800, 1000, 300)
        inputs["claimed_bbox"] = true_bbox
        item = {
            "inputs": inputs,
            "target": {
                "true_bbox": true_bbox,
                "true_slide_id": inputs["slide_id"],
                "claim_correct": True,
            },
        }

        class Dataset:
            item_ids = ["one"]

            def __getitem__(self, key):
                assert key == "one"
                return item

        predictions = run_verifier(
            EfficientLoFTRBaseline(_Matcher(query, context)), Dataset(),
            Thresholds(tau=0.5, delta=0.1, fitted_on="calibration", n_items=1,
                       residuals={}),
            audit_independence=True,
        )
        assert predictions.loc[0, "decision"] == "verified"

    def test_official_wrapper_normalises_query_as_image0_context_as_image1(self):
        torch = pytest.importorskip("torch")
        seen = {}

        class Model:
            def __call__(self, batch):
                seen.update(batch)
                batch["mkpts0_f"] = torch.tensor([[1.0, 2.0]])
                batch["mkpts1_f"] = torch.tensor([[3.0, 4.0]])
                batch["mconf"] = torch.tensor([0.75])

        wrapper = object.__new__(OfficialEfficientLoFTR)
        wrapper._torch = torch
        wrapper._device = "cpu"
        wrapper._model = Model()
        query = np.array([[0, 255], [64, 128]], dtype=np.uint8)
        context = np.full((3, 4), 32, dtype=np.uint8)
        matches = wrapper.match(query, context)

        assert set(seen) >= {"image0", "image1"}
        assert seen["image0"].shape == (1, 1, 2, 2)
        assert seen["image1"].shape == (1, 1, 3, 4)
        assert seen["image0"].dtype == seen["image1"].dtype == torch.float32
        assert seen["image0"].min().item() == 0.0
        assert seen["image0"].max().item() == 1.0
        assert seen["image1"].mean().item() == pytest.approx(32 / 255)
        assert matches.query_points.tolist() == [[1.0, 2.0]]
        assert matches.context_points.tolist() == [[3.0, 4.0]]


class TestEfficientLoFTRGeometry:
    def test_recovers_dominant_translation_with_outliers(self):
        query, context = _translated_matches(dx=21, dy=16)
        query = np.vstack([query, [[1, 4], [7, 1], [4, 4]]])
        context = np.vstack([context, [[50, 40], [12, 30], [60, 2]]])
        confidence = np.array([.8] * 6 + [.9, .9, .9])
        loc = EfficientLoFTRBaseline(
            _Matcher(query, context, confidence)
        ).localise(_inputs())
        assert loc.estimated_bbox == BBoxL0(3100, 2800, 1000, 300)
        assert loc.reliability == pytest.approx(6 / 9)
        assert loc.diagnostics["inlier_matches"] == 6

    def test_padding_nan_inf_and_duplicate_matches_do_not_inflate_support(self):
        query, context = _translated_matches()
        query = np.vstack([
            query,
            query[0],                # exact duplicate
            [12, 2],                 # Q padding: aligned Q is 10x6
            [3, 3],                  # context padding below
            [np.nan, 1],
        ])
        context = np.vstack([
            context,
            context[0],
            [20, 20],
            [10, 55],                # original context height is 48
            [5, 5],
        ])
        loc = EfficientLoFTRBaseline(_Matcher(query, context)).localise(_inputs())
        assert loc.reliability == 1.0
        assert loc.diagnostics["total_matches"] == 10
        assert loc.diagnostics["valid_matches"] == 6

    @pytest.mark.parametrize("count", [0, 1, 2, 3])
    def test_too_few_matches_returns_no_estimate(self, count):
        query, context = _translated_matches()
        loc = EfficientLoFTRBaseline(
            _Matcher(query[:count], context[:count])
        ).localise(_inputs())
        assert loc.estimated_bbox is None
        assert loc.reliability == 0.0
        assert loc.diagnostics["reason"] == "too_few_valid_matches"

    def test_repeated_points_do_not_satisfy_minimum_support(self):
        query = np.repeat([[1.0, 1.0]], 8, axis=0)
        context = np.repeat([[22.0, 17.0]], 8, axis=0)
        loc = EfficientLoFTRBaseline(_Matcher(query, context)).localise(_inputs())
        assert loc.estimated_bbox is None
        assert loc.reliability == 0.0

    def test_inconsistent_matches_return_no_estimate(self):
        query = np.array([[1, 1], [2, 1], [3, 1], [4, 1], [5, 1]], dtype=float)
        context = np.array([[1, 1], [8, 5], [16, 9], [25, 14], [35, 20]], dtype=float)
        loc = EfficientLoFTRBaseline(_Matcher(query, context)).localise(_inputs())
        assert loc.estimated_bbox is None
        assert loc.reliability == 0.0
        assert loc.diagnostics["reason"] == "too_few_geometric_inliers"

    def test_model_limit_fails_cleanly_before_calling_matcher(self):
        query, context = _translated_matches()
        matcher = _Matcher(query, context)
        matcher.max_input_side = 32
        loc = EfficientLoFTRBaseline(matcher).localise(_inputs())
        assert loc.estimated_bbox is None
        assert loc.diagnostics["reason"] == "input_exceeds_model_limit"
        assert matcher.calls == []


class TestEfficientLoFTRInvalidInputs:
    @pytest.mark.parametrize("field", ["context_mpp", "query_mpp"])
    @pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), None, "x"])
    def test_mpp_must_be_finite_and_positive(self, field, value):
        inputs = _inputs()
        inputs[field] = value
        query, context = _translated_matches()
        with pytest.raises(ValueError, match="finite and positive"):
            EfficientLoFTRBaseline(_Matcher(query, context)).localise(inputs)

    def test_query_upsampling_is_refused(self):
        inputs = _inputs(context_mpp=0.5, query_mpp=1.0)
        query, context = _translated_matches()
        with pytest.raises(ValueError, match="upsample"):
            EfficientLoFTRBaseline(_Matcher(query, context)).localise(inputs)

    def test_bad_matcher_shapes_are_rejected(self):
        matcher = _Matcher([[1, 1], [2, 2]], [[3, 3]], [1, 1])
        with pytest.raises(ValueError, match="equally sized"):
            EfficientLoFTRBaseline(matcher).localise(_inputs())

    def test_invalid_baseline_parameters_are_rejected(self):
        query, context = _translated_matches()
        matcher = _Matcher(query, context)
        with pytest.raises(ValueError, match="min_matches"):
            EfficientLoFTRBaseline(matcher, min_matches=1)
        with pytest.raises(ValueError, match="finite and positive"):
            EfficientLoFTRBaseline(matcher, translation_tolerance_px=0)


def test_native_loader_explicitly_freezes_parameters(tmp_path, monkeypatch):
    """Exercise the real loader setup with a tiny synthetic torch module, not weights."""
    import pyrafault.verify.efficient_loftr as native
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"synthetic fixture")
    model = torch.nn.Linear(2, 2)
    monkeypatch.setattr(native, "_git_revision", lambda path: native.OFFICIAL_SOURCE_REVISION)
    monkeypatch.setattr(native, "_sha256", lambda path: native.OFFICIAL_CHECKPOINT_SHA256)
    monkeypatch.setattr(native, "_install_official_compatibility_shims", lambda: dict)
    monkeypatch.setattr(native, "_load_official_symbols", lambda path: (
        lambda config: model, {"coarse": {}}, lambda model: model))

    def load(path, **kwargs):
        assert path == checkpoint
        assert kwargs == {"map_location": "cpu", "weights_only": True}
        return {"state_dict": model.state_dict()}

    monkeypatch.setattr(torch, "load", load)
    matcher = native.OfficialEfficientLoFTR(tmp_path, checkpoint, device="cpu")
    assert matcher._model is model
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
