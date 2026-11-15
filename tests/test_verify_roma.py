"""RoMa v1 geometry/protocol tests without downloading models or WSI data."""
from __future__ import annotations

from collections.abc import Mapping
import builtins
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from pyrafault.geom import BBoxL0
from pyrafault.verify.base import Localisation, Verifier
from pyrafault.verify.efficient_loftr import Matches
from pyrafault.verify import roma
from pyrafault.verify.roma import OfficialRoMa, RoMaBaseline
from pyrafault.verify.runner import PREDICTION_COLUMNS, run_verifier
from pyrafault.verify.thresholds import Thresholds


def _image(width, height, seed):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


def _inputs(*, context_size=(64, 48), query_size=(20, 12),
            context_bbox=BBoxL0(1000, 2000, 6400, 2400),
            context_mpp=1.0, query_mpp=0.5):
    return {
        "context": _image(*context_size, seed=1),
        "query": _image(*query_size, seed=2),
        "context_bbox": context_bbox,
        "context_mpp": context_mpp,
        "query_mpp": query_mpp,
        "claimed_bbox": BBoxL0(0, 0, 1, 1),
        "slide_id": "slide", "item_id": "item", "split": "dev",
        "target": {"true_bbox": BBoxL0(999, 999, 99, 99)},
    }


def _translated_matches(dx=21, dy=16, ratio=.5):
    query = np.array([[2, 2], [6, 2], [10, 4], [4, 6], [12, 6], [16, 8]],
                     dtype=float)
    return query, query * ratio + [dx, dy]


class _Matcher:
    revision = "fake-roma-1"

    def __init__(self, query_points, context_points, confidence=None):
        self.query_points = np.asarray(query_points, dtype=float).reshape(-1, 2)
        self.context_points = np.asarray(context_points, dtype=float).reshape(-1, 2)
        self.confidence = np.asarray(
            np.ones(len(self.query_points)) if confidence is None else confidence,
            dtype=float,
        )
        self.calls = []

    def match(self, query, context):
        assert query.dtype == context.dtype == np.uint8
        assert query.ndim == context.ndim == 3
        assert query.shape[-1] == context.shape[-1] == 3
        self.calls.append((query.copy(), context.copy()))
        return Matches(self.query_points, self.context_points, self.confidence)


class _GuardedInputs(Mapping):
    allowed = {"context", "query", "context_bbox", "context_mpp", "query_mpp"}

    def __init__(self, values):
        self.values, self.accessed = values, set()

    def __getitem__(self, key):
        assert key in self.allowed, f"baseline read forbidden input {key!r}"
        self.accessed.add(key)
        return self.values[key]

    def __iter__(self):
        raise AssertionError("baseline must not scan metadata keys")

    def __len__(self):
        return len(self.values)


class TestRoMaContract:
    def test_is_unified_verifier_with_separate_name_and_revision(self):
        query, context = _translated_matches()
        baseline = RoMaBaseline(_Matcher(query, context))
        assert isinstance(baseline, Verifier)
        assert baseline.name == "roma_baseline"
        assert "fake-roma-1" in baseline.revision
        assert isinstance(baseline.localise(_inputs()), Localisation)

    def test_keeps_original_rgb_rasters_without_grayscale_resize_or_padding(self):
        query, context = _translated_matches()
        matcher = _Matcher(query, context)
        inputs = _inputs()
        result = RoMaBaseline(matcher).localise(inputs)
        sent_query, sent_context = matcher.calls[0]
        np.testing.assert_array_equal(sent_query, np.asarray(inputs["query"]))
        np.testing.assert_array_equal(sent_context, np.asarray(inputs["context"]))
        assert sent_query.shape == (12, 20, 3)
        assert sent_context.shape == (48, 64, 3)
        assert result.estimated_bbox == BBoxL0(3100, 2800, 1000, 300)
        assert result.reliability == 1.0

    @pytest.mark.parametrize("mode", ["L", "RGBA"])
    def test_converts_input_images_to_rgb(self, mode):
        query, context = _translated_matches()
        matcher = _Matcher(query, context)
        inputs = _inputs()
        inputs["query"] = inputs["query"].convert(mode)
        inputs["context"] = inputs["context"].convert(mode)
        RoMaBaseline(matcher).localise(inputs)
        for sent, original in zip(matcher.calls[0], [inputs["query"], inputs["context"]]):
            np.testing.assert_array_equal(sent, np.asarray(original.convert("RGB")))

    @pytest.mark.parametrize("dx,dy", [(0, 0), (21, 16), (54, 42), (12.25, 8.75)])
    def test_mpp_translation_maps_to_anisotropic_level0_context(self, dx, dy):
        query, context = _translated_matches(dx=dx, dy=dy)
        result = RoMaBaseline(_Matcher(query, context)).localise(_inputs())
        assert result.estimated_bbox == BBoxL0(round(1000 + dx * 100),
                                              round(2000 + dy * 50), 1000, 300)

    def test_fractional_mpp_footprint_is_not_rounded_before_geometry(self):
        query, context = _translated_matches(dx=10, dy=8, ratio=.35)
        inputs = _inputs(query_size=(19, 11), context_mpp=2., query_mpp=.7,
                         context_bbox=BBoxL0(100, 200, 640, 480))
        result = RoMaBaseline(_Matcher(query, context)).localise(inputs)
        assert result.estimated_bbox == BBoxL0(200, 280, 66, 38)

    def test_query_may_have_lower_resolution_than_context(self):
        query, context = _translated_matches(dx=10, dy=8, ratio=2.)
        inputs = _inputs(context_mpp=.5, query_mpp=1.)
        result = RoMaBaseline(_Matcher(query, context)).localise(inputs)
        assert result.estimated_bbox == BBoxL0(2000, 2400, 4000, 1200)

    def test_no_claim_id_split_or_ground_truth_is_read(self):
        query, context = _translated_matches()
        guarded = _GuardedInputs(_inputs())
        RoMaBaseline(_Matcher(query, context)).localise(guarded)
        assert guarded.accessed == guarded.allowed

    def test_metadata_changes_cannot_change_result(self):
        query, context = _translated_matches()
        baseline = RoMaBaseline(_Matcher(query, context))
        inputs = _inputs()
        altered = dict(inputs, claimed_bbox=BBoxL0(9999, 9999, 1, 1),
                       split="test", item_id="other", slide_id="other",
                       target={"true_bbox": BBoxL0(0, 0, 999, 999)})
        assert baseline.localise(inputs) == baseline.localise(altered)

    def test_real_runner_scores_claims_without_altering_localisation(self):
        query, context = _translated_matches()
        baseline = RoMaBaseline(_Matcher(query, context))
        inputs = _inputs()
        truth = BBoxL0(3100, 2800, 1000, 300)
        inputs["claimed_bbox"] = truth

        class Dataset:
            item_ids = ["clean", "drift"]

            def __getitem__(self, key):
                claim = truth if key == "clean" else BBoxL0(5000, 3500, 1000, 300)
                return {"inputs": dict(inputs, claimed_bbox=claim),
                        "target": {"true_bbox": truth, "true_slide_id": "slide",
                                   "claim_correct": key == "clean"}}

        predictions = run_verifier(
            baseline, Dataset(),
            Thresholds(tau=.5, delta=.1, fitted_on="calibration", n_items=1,
                       residuals={}),
            audit_independence=True,
        )
        assert tuple(predictions.columns) == PREDICTION_COLUMNS
        assert predictions["decision"].tolist() == ["verified", "mismatched"]
        assert predictions["verifier_name"].tolist() == [baseline.name] * 2
        assert predictions["verifier_revision"].tolist() == [baseline.revision] * 2
        assert predictions["estimated_bbox_l0_x"].tolist() == [3100] * 2

    def test_revision_tracks_backend_and_geometric_settings(self):
        query, context = _translated_matches()
        matcher, other = _Matcher(query, context), _Matcher(query, context)
        other.revision = "different-weights-source"
        variants = [RoMaBaseline(matcher), RoMaBaseline(other),
                    RoMaBaseline(matcher, min_matches=5),
                    RoMaBaseline(matcher, translation_tolerance_px=2.)]
        assert len({v.revision for v in variants}) == len(variants)


class TestRoMaFailuresAndReliability:
    @pytest.mark.parametrize("value", [0, 1, 3, 4.5, True])
    def test_invalid_minimum_support_rejected(self, value):
        query, context = _translated_matches()
        with pytest.raises(ValueError, match="integer at least 4"):
            RoMaBaseline(_Matcher(query, context), min_matches=value)

    @pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
    def test_invalid_translation_tolerance_rejected(self, value):
        query, context = _translated_matches()
        with pytest.raises(ValueError, match="finite and positive"):
            RoMaBaseline(_Matcher(query, context), translation_tolerance_px=value)

    def test_reliability_combines_inlier_fraction_with_mean_model_certainty(self):
        query, context = _translated_matches()
        query = np.vstack([query, [[2, 8], [14, 2], [8, 8]]])
        context = np.vstack([context, [[50, 40], [12, 30], [60, 2]]])
        confidence = [.1, .2, .3, .4, .5, .6, .99, .99, .99]
        result = RoMaBaseline(_Matcher(query, context, confidence)).localise(_inputs())
        assert result.estimated_bbox == BBoxL0(3100, 2800, 1000, 300)
        assert result.reliability == pytest.approx((6 / 9) * .35)
        assert result.diagnostics["inlier_fraction"] == pytest.approx(6 / 9)
        assert result.diagnostics["mean_inlier_certainty"] == pytest.approx(.35)
        assert result.diagnostics["inlier_matches"] == 6

    def test_low_certainty_is_not_thresholded_to_one_or_removed(self):
        query, context = _translated_matches()
        result = RoMaBaseline(_Matcher(query, context, [.001] * 6)).localise(_inputs())
        assert result.estimated_bbox is not None
        assert result.reliability == pytest.approx(.001)
        assert result.diagnostics["valid_matches"] == 6

    def test_duplicate_invalid_points_and_invalid_certainty_do_not_count(self):
        query, context = _translated_matches()
        bad_query = [query[0], [20, 2], [-1, 2], [1, 1], [np.nan, 1],
                     [1, 2], [1, 3], [2, 3], [3, 4], [4, 4]]
        bad_context = [context[0], [30, 20], [30, 20], [64, 2], [5, 5],
                       [5, 5], [5, 5], [5, 5], [5, 5], [5, np.inf]]
        confidence = [.5] * 6 + [.5, .5, .5, .5, .5, np.inf, 0., -.1, 1.1, .5]
        result = RoMaBaseline(_Matcher(np.vstack([query, bad_query]),
                                       np.vstack([context, bad_context]),
                                       confidence)).localise(_inputs())
        assert result.estimated_bbox is not None
        assert result.diagnostics["total_matches"] == 16
        assert result.diagnostics["valid_matches"] == 6
        assert result.reliability == .5

    @pytest.mark.parametrize("count", [0, 1, 2, 3])
    def test_not_enough_unique_valid_pairs_returns_none_zero(self, count):
        query, context = _translated_matches()
        result = RoMaBaseline(_Matcher(query[:count], context[:count])).localise(_inputs())
        assert result.estimated_bbox is None
        assert result.reliability == 0.
        assert result.diagnostics["reason"] == "too_few_valid_matches"

    @pytest.mark.parametrize("side", ["query", "context"])
    def test_repeated_single_endpoint_cannot_fake_support(self, side):
        query, context = _translated_matches()
        if side == "query":
            query[:] = query[0]
        else:
            context[:] = context[0]
        result = RoMaBaseline(_Matcher(query, context)).localise(_inputs())
        assert result.estimated_bbox is None
        assert result.reliability == 0.

    def test_no_four_geometrically_consistent_pairs_abstains(self):
        query, _ = _translated_matches()
        context = np.array([[3, 3], [13, 3], [23, 3], [33, 3], [43, 3], [53, 3]])
        result = RoMaBaseline(_Matcher(query, context)).localise(_inputs())
        assert result.estimated_bbox is None
        assert result.reliability == 0.
        assert result.diagnostics["reason"] == "too_few_geometric_inliers"

    @pytest.mark.parametrize("dx,dy", [(-2., 16.), (56., 16.), (21., -2.), (21., 44.)])
    def test_small_boundary_error_clamps_to_context(self, dx, dy):
        query, context = _translated_matches(dx=dx, dy=dy)
        result = RoMaBaseline(_Matcher(query, context)).localise(_inputs())
        assert result.estimated_bbox == BBoxL0(
            round(1000 + min(max(dx, 0), 54) * 100),
            round(2000 + min(max(dy, 0), 42) * 50), 1000, 300,
        )

    @pytest.mark.parametrize("dx,dy", [(-3.1, 16.), (57.1, 16.),
                                        (21., -3.1), (21., 45.1)])
    def test_large_boundary_error_abstains(self, dx, dy):
        query, context = _translated_matches(dx=dx, dy=dy)
        # Keep corresponding points valid although the implied entire Q footprint is not.
        if dx < 0:
            query[:, 0] += 8
        if dy < 0:
            query[:, 1] = np.linspace(7, 11, len(query))
        if dy > 42:
            query[:, 1] = np.linspace(.1, 1., len(query))
        if dx > 54:
            query[:, 0] = np.linspace(.1, 1., len(query))
        context = query * .5 + [dx, dy]
        result = RoMaBaseline(_Matcher(query, context)).localise(_inputs())
        assert result.estimated_bbox is None
        assert result.reliability == 0.
        assert result.diagnostics["reason"] == "translation_outside_context"

    def test_query_physical_footprint_too_large_skips_matcher(self):
        query, context = _translated_matches()
        matcher = _Matcher(query, context)
        result = RoMaBaseline(matcher).localise(_inputs(query_size=(200, 12)))
        assert result.estimated_bbox is None
        assert result.reliability == 0.
        assert matcher.calls == []

    @pytest.mark.parametrize("field", ["query_mpp", "context_mpp"])
    @pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, "bad", None])
    def test_invalid_mpp_fails_before_matcher(self, field, value):
        query, context = _translated_matches()
        matcher = _Matcher(query, context)
        inputs = _inputs()
        inputs[field] = value
        with pytest.raises(ValueError, match="finite and positive"):
            RoMaBaseline(matcher).localise(inputs)
        assert matcher.calls == []

    @pytest.mark.parametrize("query,context,confidence", [
        (np.zeros((4, 3)), np.zeros((4, 2)), np.ones(4)),
        (np.zeros((4, 2)), np.zeros((3, 2)), np.ones(4)),
        (np.zeros((4, 2)), np.zeros((4, 2)), np.ones(3)),
        (np.zeros(8), np.zeros((4, 2)), np.ones(4)),
    ])
    def test_malformed_match_shapes_raise(self, query, context, confidence):
        class Malformed:
            def match(self, q, c):
                return Matches(query, context, confidence)

        with pytest.raises(ValueError, match="equally sized"):
            RoMaBaseline(Malformed()).localise(_inputs())


class TestRoMaConstruction:
    def test_accepts_injected_matcher_without_official_dependencies(self):
        query, context = _translated_matches()
        result = RoMaBaseline(_Matcher(query, context))
        assert isinstance(result, RoMaBaseline)
        assert result.name == "roma_baseline"


def _native_source(tmp_path):
    source = tmp_path / "RoMa"
    for relative in ("romatch/__init__.py", "romatch/models/model_zoo/__init__.py",
                     "romatch/models/model_zoo/roma_models.py",
                     "romatch/models/matcher.py", "romatch/models/encoders.py"):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    checkpoint, dino = tmp_path / "roma.pth", tmp_path / "dino.pth"
    checkpoint.write_bytes(b"fake RoMa checkpoint")
    dino.write_bytes(b"fake DINO checkpoint")
    return source, checkpoint, dino


def _forbid_torch_import(monkeypatch):
    previous_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name == "torch":
            raise AssertionError("invalid provenance must fail before model imports")
        return previous_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)


def _mock_native_loader(tmp_path, monkeypatch, *, cuda_available=False,
                        matmul_precision="highest"):
    source, checkpoint, dino = _native_source(tmp_path)
    seen = {"loads": []}

    class FrozenModule:
        def __init__(self, name):
            self.name = name

        def eval(self):
            seen[self.name + "_eval"] = True
            return self

        def requires_grad_(self, enabled):
            seen[self.name + "_requires_grad"] = enabled
            return self

        def to(self, device, *args, **kwargs):
            seen[self.name + "_device"] = device
            return self

    model = FrozenModule("model")
    model.encoder = SimpleNamespace(dinov2_vitl14=[FrozenModule("dino")])

    def factory(**kwargs):
        seen["factory"] = kwargs
        return model

    def load(path, **kwargs):
        seen["loads"].append((path, kwargs))
        return {"official_weight": "roma" if path == checkpoint.resolve() else "dino"}

    torch = ModuleType("torch")
    torch.__version__ = "test-torch"
    torch.cuda = SimpleNamespace(is_available=lambda: cuda_available)
    torch.float32, torch.float16 = "float32", "float16"
    torch.get_float32_matmul_precision = lambda: matmul_precision
    torch.load = load
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(roma, "_git_revision", lambda path: roma.OFFICIAL_SOURCE_REVISION)
    monkeypatch.setattr(roma, "_sha256", lambda path: "a" * 64
                        if path == checkpoint.resolve() else "b" * 64)
    monkeypatch.setattr(roma, "_load_official_symbols", lambda path: factory)
    monkeypatch.setattr(roma, "_attention_configuration", lambda:
                        {"backend": "torch-sdpa", "xformers_version": None})
    monkeypatch.setattr(roma, "version", lambda name: f"test-{name}-1")
    return source, checkpoint, dino, seen, torch


class TestOfficialRoMaLoader:
    def test_missing_source_rejected_before_model_import(self, tmp_path, monkeypatch):
        _forbid_torch_import(monkeypatch)
        with pytest.raises(ValueError, match="roma_source"):
            OfficialRoMa(tmp_path / "missing", tmp_path / "roma", tmp_path / "dino")

    @pytest.mark.parametrize("missing", ["checkpoint", "dinov2"])
    def test_missing_weights_rejected_before_model_import(self, tmp_path, monkeypatch,
                                                         missing):
        source, checkpoint, dino = _native_source(tmp_path)
        _forbid_torch_import(monkeypatch)
        with pytest.raises(ValueError, match="does not exist"):
            OfficialRoMa(source, tmp_path / "missing" if missing == "checkpoint" else checkpoint,
                         tmp_path / "missing" if missing == "dinov2" else dino)

    def test_wrong_source_revision_rejected_before_model_or_weights(self, tmp_path,
                                                                  monkeypatch):
        source, checkpoint, dino = _native_source(tmp_path)
        _forbid_torch_import(monkeypatch)
        monkeypatch.setattr(roma, "_git_revision", lambda path: "wrong")
        with pytest.raises(ValueError, match="source revision mismatch"):
            OfficialRoMa(source, checkpoint, dino)

    def test_safe_local_weights_fixed_settings_and_explicit_dino_freezing(self, tmp_path,
                                                                       monkeypatch):
        source, checkpoint, dino, seen, _ = _mock_native_loader(tmp_path, monkeypatch)
        wrapper = OfficialRoMa(source, checkpoint, dino, device="cpu")
        assert seen["loads"] == [
            (checkpoint.resolve(), {"map_location": "cpu", "weights_only": True}),
            (dino.resolve(), {"map_location": "cpu", "weights_only": True}),
        ]
        factory = seen["factory"]
        assert factory["weights"] == {"official_weight": "roma"}
        assert factory["dinov2_weights"] == {"official_weight": "dino"}
        for key, expected in dict(coarse_res=560, upsample_res=864, symmetric=False,
                                  use_custom_corr=False, upsample_preds=True,
                                  with_padding=False, do_compile=False).items():
            assert factory[key] == expected
        assert factory["device"] == "cpu"
        assert seen["model_eval"] is True
        assert seen["model_requires_grad"] is False
        assert seen["dino_eval"] is True
        assert seen["dino_requires_grad"] is False
        assert seen["dino_device"] == "cpu"
        assert wrapper.configuration["source_revision"] == roma.OFFICIAL_SOURCE_REVISION
        assert wrapper.configuration["checkpoint_sha256"] == "a" * 64
        assert wrapper.configuration["dinov2_sha256"] == "b" * 64
        assert wrapper.configuration["device"] == "cpu"
        assert wrapper.configuration["precision"] == "native-cpu-fp32"
        assert wrapper.configuration["attention"] == {
            "backend": "torch-sdpa", "xformers_version": None,
        }
        assert "a" * 12 in wrapper.revision
        assert "b" * 12 in wrapper.revision

    @pytest.mark.parametrize("available,expected", [(False, "cpu"), (True, "cuda")])
    def test_auto_device_is_explicitly_resolved(self, tmp_path, monkeypatch, available,
                                               expected):
        source, checkpoint, dino, seen, _ = _mock_native_loader(
            tmp_path, monkeypatch, cuda_available=available,
        )
        wrapper = OfficialRoMa(source, checkpoint, dino)
        assert seen["factory"]["device"] == expected
        assert wrapper.configuration["device"] == expected
        expected_precision = "native-cpu-fp32" if expected == "cpu" else "native-cuda-mixed-fp16"
        assert wrapper.configuration["precision"] == expected_precision

    @pytest.mark.parametrize("device", ["unknown", "mps", "cuda"])
    def test_unsupported_devices_fail_before_weight_loading(self, tmp_path, monkeypatch,
                                                           device):
        source, checkpoint, dino, seen, _ = _mock_native_loader(tmp_path, monkeypatch)
        with pytest.raises(ValueError):
            OfficialRoMa(source, checkpoint, dino, device=device)
        assert seen["loads"] == []
        assert "factory" not in seen

    @pytest.mark.parametrize("precision", ["high", "medium"])
    def test_non_highest_matmul_precision_is_rejected_not_changed(self, tmp_path,
                                                                 monkeypatch, precision):
        source, checkpoint, dino, seen, torch = _mock_native_loader(
            tmp_path, monkeypatch, matmul_precision=precision,
        )
        with pytest.raises((ValueError, RuntimeError), match="highest"):
            OfficialRoMa(source, checkpoint, dino, device="cpu")
        assert torch.get_float32_matmul_precision() == precision
        assert seen["loads"] == []

    def test_safe_load_failure_is_not_retried_with_pickle_enabled(self, tmp_path, monkeypatch):
        source, checkpoint, dino, _, torch = _mock_native_loader(tmp_path, monkeypatch)
        calls = []

        def reject(path, **kwargs):
            calls.append(kwargs)
            raise RuntimeError("safe load rejection")

        torch.load = reject
        with pytest.raises(RuntimeError, match="safe load rejection"):
            OfficialRoMa(source, checkpoint, dino, device="cpu")
        assert calls == [{"map_location": "cpu", "weights_only": True}]

    def test_incompatible_native_weight_errors_propagate(self, tmp_path, monkeypatch):
        source, checkpoint, dino, _, _ = _mock_native_loader(tmp_path, monkeypatch)

        def reject(**kwargs):
            assert kwargs["weights"] is not None
            assert kwargs["dinov2_weights"] is not None
            raise RuntimeError("strict state dict rejected")

        monkeypatch.setattr(roma, "_load_official_symbols", lambda path: reject)
        with pytest.raises(RuntimeError, match="strict state dict rejected"):
            OfficialRoMa(source, checkpoint, dino, device="cpu")

    @pytest.mark.parametrize("dependency", ["Pillow", "einops", "loguru", "torchvision",
                                            "numpy", "torch"])
    def test_dependency_versions_change_revision_and_are_deterministic(self, tmp_path,
                                                                     monkeypatch, dependency):
        source, checkpoint, dino, _, torch = _mock_native_loader(tmp_path, monkeypatch)
        first = OfficialRoMa(source, checkpoint, dino, device="cpu")
        same = OfficialRoMa(source, checkpoint, dino, device="cpu")
        assert first.configuration == same.configuration
        assert first.revision == same.revision
        if dependency == "numpy":
            monkeypatch.setattr(np, "__version__", "changed-numpy")
        elif dependency == "torch":
            torch.__version__ = "changed-torch"
        else:
            prior_version = roma.version
            monkeypatch.setattr(roma, "version", lambda name:
                                f"changed-{name}" if name == dependency else prior_version(name))
        changed = OfficialRoMa(source, checkpoint, dino, device="cpu")
        assert changed.revision != first.revision
        assert changed.configuration["dependencies"][dependency] != (
            first.configuration["dependencies"][dependency]
        )

    @pytest.mark.parametrize("weight", ["roma", "dino"])
    def test_each_weight_digest_changes_revision(self, tmp_path, monkeypatch, weight):
        source, checkpoint, dino, _, _ = _mock_native_loader(tmp_path, monkeypatch)
        first = OfficialRoMa(source, checkpoint, dino, device="cpu")
        changed_path = checkpoint.resolve() if weight == "roma" else dino.resolve()
        old_hash = roma._sha256
        monkeypatch.setattr(roma, "_sha256", lambda path:
                            "c" * 64 if path == changed_path else old_hash(path))
        changed = OfficialRoMa(source, checkpoint, dino, device="cpu")
        assert changed.revision != first.revision

    def test_model_settings_are_recorded_and_change_revision(self, tmp_path, monkeypatch):
        source, checkpoint, dino, _, _ = _mock_native_loader(tmp_path, monkeypatch)
        first = OfficialRoMa(source, checkpoint, dino, device="cpu")
        assert first.configuration["model_config"] == roma.MODEL_CONFIG
        assert first.configuration["sampling"] == {"method": "query-fixed-grid", "grid_side": 32}
        changed_config = dict(roma.MODEL_CONFIG, coarse_res=448)
        monkeypatch.setattr(roma, "MODEL_CONFIG", changed_config)
        changed = OfficialRoMa(source, checkpoint, dino, device="cpu")
        assert changed.configuration["model_config"]["coarse_res"] == 448
        assert first.configuration["model_config"]["coarse_res"] == 560
        assert changed.revision != first.revision

    def test_sampling_grid_is_recorded_and_changes_revision(self, tmp_path, monkeypatch):
        source, checkpoint, dino, _, _ = _mock_native_loader(tmp_path, monkeypatch)
        first = OfficialRoMa(source, checkpoint, dino, device="cpu")
        monkeypatch.setattr(roma, "GRID_SIDE", 16)
        changed = OfficialRoMa(source, checkpoint, dino, device="cpu")
        assert changed.configuration["sampling"]["grid_side"] == 16
        assert changed.revision != first.revision

    @pytest.mark.parametrize("first_attention,changed_attention", [
        ({"backend": "torch-sdpa", "xformers_version": None},
         {"backend": "xformers", "xformers_version": "0.0.30"}),
        ({"backend": "xformers", "xformers_version": "0.0.30"},
         {"backend": "xformers", "xformers_version": "0.0.31"}),
    ])
    def test_attention_backend_or_xformers_version_changes_revision(
        self, tmp_path, monkeypatch, first_attention, changed_attention,
    ):
        source, checkpoint, dino, _, _ = _mock_native_loader(tmp_path, monkeypatch)
        monkeypatch.setattr(roma, "_attention_configuration", lambda: dict(first_attention))
        first = OfficialRoMa(source, checkpoint, dino, device="cpu")
        same = OfficialRoMa(source, checkpoint, dino, device="cpu")
        assert first.revision == same.revision
        assert first.configuration["attention"] == first_attention
        monkeypatch.setattr(roma, "_attention_configuration", lambda: dict(changed_attention))
        changed = OfficialRoMa(source, checkpoint, dino, device="cpu")
        assert changed.configuration["attention"] == changed_attention
        assert changed.revision != first.revision

    @pytest.mark.parametrize("uses_xformers", [False, True])
    def test_attention_configuration_reads_native_active_backend(self, monkeypatch,
                                                              uses_xformers):
        native_attention = SimpleNamespace(XFORMERS_AVAILABLE=uses_xformers)
        lookups = []

        def import_module(name):
            lookups.append(name)
            return native_attention

        def package_version(name):
            assert name == "xformers"
            if not uses_xformers:
                raise AssertionError("inactive optional backend must not require xformers package")
            return "test-xformers-version"

        monkeypatch.setattr(roma.importlib, "import_module", import_module)
        monkeypatch.setattr(roma, "version", package_version)
        assert roma._attention_configuration() == {
            "backend": "xformers" if uses_xformers else "torch-sdpa",
            "xformers_version": "test-xformers-version" if uses_xformers else None,
        }
        assert lookups == ["romatch.models.transformer.layers.attention"]

    def test_non_mapping_native_weights_rejected_before_factory(self, tmp_path, monkeypatch):
        source, checkpoint, dino, seen, torch = _mock_native_loader(tmp_path, monkeypatch)
        torch.load = lambda *args, **kwargs: ["not a state dict"]
        with pytest.raises(ValueError, match="native state dictionaries"):
            OfficialRoMa(source, checkpoint, dino, device="cpu")
        assert "factory" not in seen

    @pytest.mark.parametrize("clean", [False, True])
    def test_native_package_import_restores_search_path(self, tmp_path, monkeypatch, clean):
        source = tmp_path / "RoMa"
        package = ModuleType("romatch")
        package.__path__ = [str(source / "romatch")]
        package.roma_outdoor = object()
        if clean:
            monkeypatch.delitem(sys.modules, "romatch", raising=False)
        else:
            monkeypatch.setitem(sys.modules, "romatch", package)
        old_path = list(sys.path)

        def import_package(name):
            assert name == "romatch"
            assert sys.path[0] == str(source)
            return package

        monkeypatch.setattr(roma.importlib, "import_module", import_package)
        assert roma._load_official_symbols(source) is package.roma_outdoor
        assert sys.path == old_path

    def test_native_import_failure_restores_search_path(self, tmp_path, monkeypatch):
        source = tmp_path / "RoMa"
        monkeypatch.delitem(sys.modules, "romatch", raising=False)
        old_path = list(sys.path)

        def import_package(name):
            raise ImportError("missing native dependency")

        monkeypatch.setattr(roma.importlib, "import_module", import_package)
        with pytest.raises(ImportError, match="missing native dependency"):
            roma._load_official_symbols(source)
        assert sys.path == old_path

    def test_loaded_unrelated_romatch_is_not_silently_reused(self, tmp_path, monkeypatch):
        source = tmp_path / "RoMa"
        unrelated = ModuleType("romatch")
        unrelated.__path__ = [str(tmp_path / "wrong" / "romatch")]
        monkeypatch.setitem(sys.modules, "romatch", unrelated)
        with pytest.raises(RuntimeError, match="another romatch checkout"):
            roma._load_official_symbols(source)


@pytest.fixture(scope="class")
def _single_torch_thread():
    # Initialize Arrow before torch to avoid previously observed macOS pool conflicts.
    pytest.importorskip("pyarrow")
    torch = pytest.importorskip("torch")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _mock_dense_wrapper(torch, *, shape=(64, 96), bad_output=None):
    seen = {}
    height, width = shape
    ys = (torch.arange(height, dtype=torch.float32) + .5) * 2 / height - 1
    xs = (torch.arange(width, dtype=torch.float32) + .5) * 2 / width - 1
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    query_coords = torch.stack((xx, yy), dim=-1)
    warp = torch.cat((query_coords, query_coords * .5), dim=-1)[None]
    certainty = torch.full((1, height, width), .2)

    class Model:
        def match(self, query, context, **kwargs):
            assert torch.is_inference_mode_enabled()
            assert isinstance(query, Image.Image) and query.mode == "RGB"
            assert isinstance(context, Image.Image) and context.mode == "RGB"
            seen["images"] = (query, context)
            seen["kwargs"] = kwargs
            return (warp, certainty) if bad_output is None else bad_output

        def sample(self, *args, **kwargs):
            raise AssertionError("official random/sample-threshold path must not be used")

    wrapper = object.__new__(OfficialRoMa)
    wrapper._torch, wrapper._device, wrapper._model = torch, "cpu", Model()
    return wrapper, seen, warp, certainty


@pytest.mark.usefixtures("_single_torch_thread")
class TestOfficialRoMaInference:
    def test_rgb_original_shapes_order_fixed_grid_and_original_pixel_coordinates(self):
        torch = pytest.importorskip("torch")
        wrapper, seen, warp, certainty = _mock_dense_wrapper(torch)
        query = np.arange(12 * 20 * 3, dtype=np.uint8).reshape(12, 20, 3)
        context = np.full((48, 64, 3), 32, dtype=np.uint8)
        matches = wrapper.match(query, context)
        assert seen["images"][0].size == (20, 12)
        assert seen["images"][1].size == (64, 48)
        np.testing.assert_array_equal(np.asarray(seen["images"][0]), query)
        np.testing.assert_array_equal(np.asarray(seen["images"][1]), context)
        assert seen["kwargs"] == {"batched": True, "device": "cpu"}
        rows = np.floor((np.arange(32) + .5) * 64 / 32).astype(int)
        columns = np.floor((np.arange(32) + .5) * 96 / 32).astype(int)
        sampled = warp[0].numpy()[rows[:, None], columns[None, :]].reshape(-1, 4)
        assert isinstance(matches, Matches)
        assert matches.query_points.shape == matches.context_points.shape == (1024, 2)
        assert matches.confidence.shape == (1024,)
        np.testing.assert_allclose(matches.query_points,
                                   (sampled[:, :2] + 1) * [20, 12] / 2,
                                   atol=1e-6)
        np.testing.assert_allclose(matches.context_points,
                                   (sampled[:, 2:] + 1) * [64, 48] / 2,
                                   atol=1e-6)
        np.testing.assert_allclose(matches.confidence, .2)
        # Fixed sampling cannot alter the returned raw match certainty.
        np.testing.assert_allclose(certainty.numpy(), .2)

    def test_small_output_does_not_repeat_dense_support(self):
        torch = pytest.importorskip("torch")
        wrapper, _, _, _ = _mock_dense_wrapper(torch, shape=(8, 12))
        matches = wrapper.match(np.zeros((16, 20, 3), dtype=np.uint8),
                                np.zeros((48, 64, 3), dtype=np.uint8))
        assert matches.query_points.shape == (96, 2)
        assert len(np.unique(matches.query_points, axis=0)) == 96

    def test_noncontiguous_rgb_arrays_are_accepted_with_pixels_unchanged(self):
        torch = pytest.importorskip("torch")
        wrapper, seen, _, _ = _mock_dense_wrapper(torch)
        query = np.arange(32 * 40 * 3, dtype=np.uint8).reshape(32, 40, 3)[::2, ::2]
        assert not query.flags.c_contiguous
        wrapper.match(query, np.zeros((48, 64, 3), dtype=np.uint8))
        np.testing.assert_array_equal(np.asarray(seen["images"][0]), query)

    def test_nonfinite_zero_and_out_of_range_outputs_are_removed(self):
        torch = pytest.importorskip("torch")
        wrapper, _, warp, certainty = _mock_dense_wrapper(torch, shape=(8, 12))
        warp[0, 0, 0, 0] = float("nan")
        warp[0, 0, 1, 2] = 1.0
        warp[0, 0, 2, 3] = -1.01
        certainty[0, 0, 3] = float("inf")
        certainty[0, 0, 4] = 0.
        certainty[0, 0, 5] = -.1
        certainty[0, 0, 6] = 1.1
        matches = wrapper.match(np.zeros((16, 20, 3), dtype=np.uint8),
                                np.zeros((48, 64, 3), dtype=np.uint8))
        assert matches.query_points.shape == (89, 2)
        assert np.isfinite(matches.query_points).all()
        assert np.isfinite(matches.context_points).all()
        assert ((matches.confidence > 0) & (matches.confidence <= 1)).all()

    def test_zero_certainty_dense_map_returns_empty_match_arrays(self):
        torch = pytest.importorskip("torch")
        wrapper, _, _, certainty = _mock_dense_wrapper(torch)
        certainty.zero_()
        matches = wrapper.match(np.zeros((16, 20, 3), dtype=np.uint8),
                                np.zeros((48, 64, 3), dtype=np.uint8))
        assert matches.query_points.shape == matches.context_points.shape == (0, 2)
        assert matches.confidence.shape == (0,)

    @pytest.mark.parametrize("warp_shape,certainty_shape", [
        ((64, 96, 4), (64, 96)),
        ((2, 64, 96, 4), (2, 64, 96)),
        ((1, 64, 96, 2), (1, 64, 96)),
        ((1, 64, 96, 4), (1, 64, 95)),
        ((1, 0, 96, 4), (1, 0, 96)),
    ])
    def test_malformed_native_dense_output_raises(self, warp_shape, certainty_shape):
        torch = pytest.importorskip("torch")
        bad_output = (torch.zeros(warp_shape), torch.zeros(certainty_shape))
        wrapper, _, _, _ = _mock_dense_wrapper(torch, bad_output=bad_output)
        with pytest.raises(ValueError, match="warp"):
            wrapper.match(np.zeros((16, 20, 3), dtype=np.uint8),
                          np.zeros((48, 64, 3), dtype=np.uint8))

    @pytest.mark.parametrize("shape,dtype", [((16, 20), np.uint8),
                                            ((16, 20, 4), np.uint8),
                                            ((16, 20, 3), np.float32),
                                            ((0, 20, 3), np.uint8)])
    @pytest.mark.parametrize("side", ["query", "context"])
    def test_invalid_rgb_arrays_raise_before_native_model(self, shape, dtype, side):
        torch = pytest.importorskip("torch")
        wrapper, seen, _, _ = _mock_dense_wrapper(torch)
        valid = np.zeros((16, 20, 3), dtype=np.uint8)
        invalid = np.zeros(shape, dtype=dtype)
        with pytest.raises(ValueError):
            wrapper.match(invalid if side == "query" else valid,
                          invalid if side == "context" else valid)
        assert "images" not in seen
