"""LightGlue adapter contract checks; no weights, downloads or WSI required."""
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
from pyrafault.verify import lightglue
from pyrafault.verify.lightglue import LightGlueBaseline, OfficialSuperPointLightGlue
from pyrafault.verify.runner import PREDICTION_COLUMNS, run_verifier
from pyrafault.verify.thresholds import Thresholds


def _image(width, height, seed):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


def _inputs(*, context_size=(64, 48), query_size=(32, 24),
            context_bbox=BBoxL0(1000, 2000, 6400, 2400),
            context_mpp=1., query_mpp=.5):
    return {
        "context": _image(*context_size, seed=1),
        "query": _image(*query_size, seed=2),
        "context_bbox": context_bbox,
        "context_mpp": context_mpp, "query_mpp": query_mpp,
        "claimed_bbox": BBoxL0(0, 0, 1, 1),
        "item_id": "item", "slide_id": "slide", "split": "dev",
        "target": {"true_bbox": BBoxL0(9999, 9999, 9999, 9999)},
    }


def _translated_matches(dx=21., dy=16., scale=(1., 1.)):
    query = np.array([[1, 1], [3, 1], [5, 2], [2, 3], [6, 3], [8, 4]], dtype=float)
    # SuperPoint keypoint 0 denotes the first pixel centre. Physical raster-edge
    # coordinates are keypoint + .5, including fractional aligned-raster scales.
    context = (query + .5) * np.asarray(scale) + [dx, dy] - .5
    return query, context


class _Matcher:
    revision = "fake-superpoint-lightglue-1"

    def __init__(self, query_points, context_points, confidence=None):
        self.query_points = np.asarray(query_points, dtype=float).reshape(-1, 2)
        self.context_points = np.asarray(context_points, dtype=float).reshape(-1, 2)
        self.confidence = np.asarray(
            np.ones(len(self.query_points)) if confidence is None else confidence, dtype=float,
        )
        self.calls = []

    def match(self, query, context):
        assert query.dtype == context.dtype == np.uint8
        assert query.ndim == context.ndim == 3
        assert query.shape[2] == context.shape[2] == 3
        self.calls.append((query.copy(), context.copy()))
        return Matches(self.query_points, self.context_points, self.confidence)


class _GuardedInputs(Mapping):
    allowed = {"context", "query", "context_bbox", "context_mpp", "query_mpp"}

    def __init__(self, values):
        self.values, self.accessed = values, set()

    def __getitem__(self, key):
        assert key in self.allowed, f"verifier accessed forbidden input {key!r}"
        self.accessed.add(key)
        return self.values[key]

    def __iter__(self):
        raise AssertionError("localise must not inspect the metadata mapping")

    def __len__(self):
        return len(self.values)


class TestLightGlueContract:
    def test_unified_verifier_and_method_identity(self):
        query, context = _translated_matches()
        baseline = LightGlueBaseline(_Matcher(query, context))
        assert isinstance(baseline, Verifier)
        assert baseline.name == "lightglue_baseline"
        assert "fake-superpoint-lightglue-1" in baseline.revision
        assert isinstance(baseline.localise(_inputs()), Localisation)

    def test_mpp_aligned_rgb_query_and_unchanged_context_without_padding(self):
        query, context = _translated_matches()
        matcher = _Matcher(query, context)
        inputs = _inputs()
        result = LightGlueBaseline(matcher).localise(inputs)
        sent_query, sent_context = matcher.calls[0]
        expected = inputs["query"].convert("RGB").resize((16, 12), Image.Resampling.LANCZOS)
        np.testing.assert_array_equal(sent_query, np.asarray(expected))
        np.testing.assert_array_equal(sent_context, np.asarray(inputs["context"]))
        assert sent_query.shape == (12, 16, 3)
        assert sent_context.shape == (48, 64, 3)
        assert result.estimated_bbox == BBoxL0(3100, 2800, 1600, 600)
        assert result.reliability == 1.
        assert result.diagnostics["aligned_query_size_px"] == [16, 12]
        assert result.diagnostics["context_size_px"] == [64, 48]

    def test_same_mpp_retains_original_raster_and_keypoint_offset_cancels(self):
        query, context = _translated_matches()
        matcher = _Matcher(query, context)
        inputs = _inputs(query_mpp=1.)
        result = LightGlueBaseline(matcher).localise(inputs)
        np.testing.assert_array_equal(matcher.calls[0][0], np.asarray(inputs["query"]))
        assert result.estimated_bbox == BBoxL0(3100, 2800, 3200, 1200)

    def test_fractional_footprint_uses_superpoint_pixel_centre_convention(self):
        # A 41x31 query at .66 MPP has a 13.53x10.23 C-pixel footprint,
        # but the network receives a 14x10 raster. +.5 on both point sets is
        # necessary: treating SuperPoint centres as raster edges biases translation.
        scale = [13.53 / 14, 10.23 / 10]
        query, context = _translated_matches(dx=10.2, dy=8.3, scale=scale)
        inputs = _inputs(query_size=(41, 31), query_mpp=.66, context_mpp=2.,
                         context_bbox=BBoxL0(100, 200, 6400, 4800))
        result = LightGlueBaseline(_Matcher(query, context)).localise(inputs)
        assert result.estimated_bbox == BBoxL0(1120, 1030, 1353, 1023)
        assert result.diagnostics["aligned_query_size_px"] == [14, 10]
        assert result.diagnostics["query_to_context_point_scale"] == pytest.approx(scale)
        assert result.diagnostics["translation_xy_context"] == pytest.approx([10.2, 8.3])

    @pytest.mark.parametrize("mode", ["L", "RGBA"])
    def test_non_rgb_inputs_are_converted_before_matching(self, mode):
        query, context = _translated_matches()
        matcher = _Matcher(query, context)
        inputs = _inputs()
        inputs["query"] = inputs["query"].convert(mode)
        inputs["context"] = inputs["context"].convert(mode)
        LightGlueBaseline(matcher).localise(inputs)
        sent_query, sent_context = matcher.calls[0]
        expected = inputs["query"].convert("RGB").resize((16, 12), Image.Resampling.LANCZOS)
        np.testing.assert_array_equal(sent_query, np.asarray(expected))
        np.testing.assert_array_equal(sent_context, np.asarray(inputs["context"].convert("RGB")))

    def test_reads_only_five_algorithm_fields(self):
        query, context = _translated_matches()
        guarded = _GuardedInputs(_inputs())
        LightGlueBaseline(_Matcher(query, context)).localise(guarded)
        assert guarded.accessed == guarded.allowed

    def test_claim_labels_identifiers_and_split_cannot_change_result(self):
        query, context = _translated_matches()
        baseline = LightGlueBaseline(_Matcher(query, context))
        inputs = _inputs()
        altered = dict(inputs, claimed_bbox=BBoxL0(99999, 99999, 1, 1), split="test",
                       item_id="other", slide_id="other", target={"claim_correct": False})
        assert baseline.localise(inputs) == baseline.localise(altered)

    def test_real_shared_runner_scores_distinct_claims_and_audits_independence(self):
        query, context = _translated_matches()
        baseline = LightGlueBaseline(_Matcher(query, context))
        inputs = _inputs()
        truth = BBoxL0(3100, 2800, 1600, 600)

        class Dataset:
            item_ids = ["clean", "drift"]

            def __getitem__(self, key):
                claim = truth if key == "clean" else BBoxL0(5000, 3500, 1600, 600)
                return {"inputs": dict(inputs, claimed_bbox=claim),
                        "target": {"true_bbox": truth, "true_slide_id": "slide",
                                   "claim_correct": key == "clean"}}

        predictions = run_verifier(
            baseline, Dataset(), Thresholds(tau=.5, delta=.1, fitted_on="calibration",
                                           n_items=1, residuals={}),
            audit_independence=True,
        )
        assert tuple(predictions.columns) == PREDICTION_COLUMNS
        assert predictions["decision"].tolist() == ["verified", "mismatched"]
        assert predictions["verifier_name"].tolist() == [baseline.name] * 2
        assert predictions["verifier_revision"].tolist() == [baseline.revision] * 2
        assert predictions["estimated_bbox_l0_x"].tolist() == [3100] * 2

    def test_revision_changes_for_backend_or_adapter_settings(self):
        query, context = _translated_matches()
        matcher, other = _Matcher(query, context), _Matcher(query, context)
        other.revision = "other-lightglue-weights"
        baselines = [LightGlueBaseline(matcher), LightGlueBaseline(other),
                     LightGlueBaseline(matcher, min_matches=5),
                     LightGlueBaseline(matcher, translation_tolerance_px=2.)]
        assert len({baseline.revision for baseline in baselines}) == len(baselines)


class TestLightGlueGeometryFailures:
    def test_reliability_is_geometric_inlier_fraction_times_mean_match_score(self):
        query, context = _translated_matches()
        query = np.vstack([query, [[1, 4], [7, 1], [4, 4]]])
        context = np.vstack([context, [[50, 40], [12, 30], [60, 2]]])
        scores = [.1, .2, .3, .4, .5, .6, .99, .99, .99]
        result = LightGlueBaseline(_Matcher(query, context, scores)).localise(_inputs())
        assert result.estimated_bbox == BBoxL0(3100, 2800, 1600, 600)
        assert result.reliability == pytest.approx((6 / 9) * .35)
        assert result.diagnostics["inlier_fraction"] == pytest.approx(6 / 9)
        assert result.diagnostics["mean_inlier_match_score"] == pytest.approx(.35)

    def test_low_but_positive_scores_remain_low_not_thresholded_or_removed(self):
        query, context = _translated_matches()
        result = LightGlueBaseline(_Matcher(query, context, [.001] * 6)).localise(_inputs())
        assert result.estimated_bbox is not None
        assert result.reliability == pytest.approx(.001)

    def test_duplicates_nonfinite_outside_and_invalid_scores_do_not_count(self):
        query, context = _translated_matches()
        bad_query = [query[0], [16, 2], [-1, 2], [1, 1], [np.nan, 1],
                     [1, 2], [1, 3], [2, 3], [3, 4], [4, 4]]
        bad_context = [context[0], [30, 20], [30, 20], [64, 2], [5, 5],
                       [5, 5], [5, 5], [5, 5], [5, 5], [5, np.inf]]
        scores = [.5] * 6 + [.5, .5, .5, .5, .5, np.inf, 0., -.1, 1.1, .5]
        result = LightGlueBaseline(_Matcher(np.vstack([query, bad_query]),
                                            np.vstack([context, bad_context]),
                                            scores)).localise(_inputs())
        assert result.estimated_bbox is not None
        assert result.diagnostics["total_matches"] == 16
        assert result.diagnostics["valid_matches"] == 6
        assert result.reliability == .5

    @pytest.mark.parametrize("count", [0, 1, 2, 3])
    def test_fewer_than_four_valid_matches_abstains(self, count):
        query, context = _translated_matches()
        result = LightGlueBaseline(_Matcher(query[:count], context[:count])).localise(_inputs())
        assert result.estimated_bbox is None
        assert result.reliability == 0.
        assert result.diagnostics["reason"] == "too_few_valid_matches"

    @pytest.mark.parametrize("side", ["query", "context"])
    def test_many_repeated_endpoint_matches_cannot_fake_four_unique_points(self, side):
        query, context = _translated_matches()
        (query if side == "query" else context)[:] = (query if side == "query" else context)[0]
        result = LightGlueBaseline(_Matcher(query, context)).localise(_inputs())
        assert result.estimated_bbox is None
        assert result.reliability == 0.

    def test_insufficient_geometric_support_abstains(self):
        query, _ = _translated_matches()
        context = np.array([[3, 3], [13, 3], [23, 3], [33, 3], [43, 3], [53, 3]])
        result = LightGlueBaseline(_Matcher(query, context)).localise(_inputs())
        assert result.estimated_bbox is None
        assert result.reliability == 0.
        assert result.diagnostics["reason"] == "too_few_geometric_inliers"

    @pytest.mark.parametrize("dx,dy", [(-2., 16.), (50., 16.), (21., -2.), (21., 38.)])
    def test_boundary_tolerance_clamps_whole_query_footprint(self, dx, dy):
        query, context = _translated_matches(dx=dx, dy=dy)
        # Ensure keypoints remain inside their rasters even when the inferred box
        # extends slightly outside the context.
        query[:, 0] = np.linspace(4, 10, len(query))
        query[:, 1] = np.linspace(4, 9, len(query))
        context = query + [dx, dy]
        result = LightGlueBaseline(_Matcher(query, context)).localise(_inputs())
        assert result.estimated_bbox == BBoxL0(
            round(1000 + min(max(dx, 0), 48) * 100),
            round(2000 + min(max(dy, 0), 36) * 50), 1600, 600,
        )

    @pytest.mark.parametrize("dx,dy", [(-3.1, 16.), (51.1, 16.),
                                        (21., -3.1), (21., 39.1)])
    def test_large_boundary_error_abstains(self, dx, dy):
        query, _ = _translated_matches()
        query[:, 0] = np.linspace(4, 10, len(query))
        query[:, 1] = np.linspace(4, 8, len(query))
        context = query + [dx, dy]
        result = LightGlueBaseline(_Matcher(query, context)).localise(_inputs())
        assert result.estimated_bbox is None
        assert result.reliability == 0.
        assert result.diagnostics["reason"] == "translation_outside_context"

    def test_physically_large_query_skips_matcher(self):
        query, context = _translated_matches()
        matcher = _Matcher(query, context)
        result = LightGlueBaseline(matcher).localise(_inputs(query_size=(200, 24)))
        assert result.estimated_bbox is None
        assert result.reliability == 0.
        assert result.diagnostics["reason"] == "query_footprint_too_large"
        assert matcher.calls == []

    @pytest.mark.parametrize("query_size,context_size", [((20, 14), (64, 48)),
                                                         ((16, 14), (64, 7))])
    def test_superpoint_short_side_guard_skips_matcher(self, query_size, context_size):
        query, context = _translated_matches()
        matcher = _Matcher(query, context)
        result = LightGlueBaseline(matcher).localise(_inputs(query_size=query_size,
                                                            context_size=context_size))
        assert result.estimated_bbox is None
        assert result.reliability == 0.
        assert result.diagnostics["reason"] == "input_too_small_for_superpoint"
        assert matcher.calls == []

    def test_would_upsample_query_is_rejected(self):
        query, context = _translated_matches()
        matcher = _Matcher(query, context)
        with pytest.raises(ValueError, match="upsample"):
            LightGlueBaseline(matcher).localise(_inputs(query_mpp=2.))
        assert matcher.calls == []

    @pytest.mark.parametrize("field,value", [("query_mpp", 0), ("context_mpp", -1),
                                             ("query_mpp", np.nan), ("context_mpp", np.inf),
                                             ("query_mpp", "bad"), ("context_mpp", None)])
    def test_invalid_mpp_is_rejected_before_matching(self, field, value):
        query, context = _translated_matches()
        matcher = _Matcher(query, context)
        inputs = _inputs()
        inputs[field] = value
        with pytest.raises(ValueError, match="finite and positive"):
            LightGlueBaseline(matcher).localise(inputs)
        assert matcher.calls == []

    @pytest.mark.parametrize("settings", [{"min_matches": 3}, {"min_matches": 4.5},
                                          {"min_matches": True}, {"min_matches": None},
                                          {"translation_tolerance_px": 0},
                                          {"translation_tolerance_px": np.nan},
                                          {"translation_tolerance_px": np.inf}])
    def test_invalid_constructor_settings_raise_value_error(self, settings):
        query, context = _translated_matches()
        with pytest.raises(ValueError):
            LightGlueBaseline(_Matcher(query, context), **settings)

    @pytest.mark.parametrize("query,context,scores", [
        (np.zeros((4, 3)), np.zeros((4, 2)), np.ones(4)),
        (np.zeros((4, 2)), np.zeros((3, 2)), np.ones(4)),
        (np.zeros((4, 2)), np.zeros((4, 2)), np.ones(3)),
    ])
    def test_malformed_match_arrays_raise(self, query, context, scores):
        class Malformed:
            def match(self, q, c):
                return Matches(query, context, scores)

        with pytest.raises(ValueError, match="equally sized"):
            LightGlueBaseline(Malformed()).localise(_inputs())


def _native_source(tmp_path):
    source = tmp_path / "LightGlue"
    for name in ("superpoint.py", "lightglue.py"):
        path = source / "lightglue" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    sp, lg = tmp_path / "superpoint.pth", tmp_path / "lightglue.pth"
    sp.write_bytes(b"mock SuperPoint checkpoint")
    lg.write_bytes(b"mock LightGlue checkpoint")
    return source, sp, lg


def _mock_native_loader(tmp_path, monkeypatch, *, cuda_available=False):
    source, sp, lg = _native_source(tmp_path)
    seen = {"loads": []}
    derived_buffer = object()

    class FrozenModel:
        def __init__(self, name):
            self.name = name

        def eval(self):
            seen[self.name + "_eval"] = True
            return self

        def requires_grad_(self, enabled):
            seen[self.name + "_requires_grad"] = enabled
            return self

        def to(self, device):
            seen[self.name + "_device"] = device
            return self

    class SuperPoint(FrozenModel):
        def __init__(self, **kwargs):
            super().__init__("extractor")
            seen["superpoint_config"] = kwargs
            seen["superpoint_weights"] = torch.hub.load_state_dict_from_url(
                lightglue.SUPERPOINT_URL, map_location="cpu",
            )

    class LightGlue(FrozenModel):
        def __init__(self, **kwargs):
            super().__init__("matcher")
            # Mirror the native constructor's global side effect. The wrapper
            # must restore the caller's flash-SDP setting after construction.
            torch.backends.cuda.enable_flash_sdp(False)
            seen["lightglue_config"] = kwargs
            self.confidence_thresholds = SimpleNamespace(
                detach=lambda: SimpleNamespace(clone=lambda: derived_buffer),
            )
            self.pruning_keypoint_thresholds = {"cpu": -1, "cuda": 1024, "flash": 1536}

        def load_state_dict(self, state_dict, *, strict):
            seen["learned_weights"] = state_dict
            seen["strict"] = strict
            expected_keys = {"transformers.0.self_attn.Wqkv.weight",
                             "transformers.8.cross_attn.to_qk.weight",
                             "final_proj.0.weight", "confidence_thresholds"}
            if strict and set(state_dict) != expected_keys:
                raise RuntimeError("native strict state dict rejected keys")

    torch = ModuleType("torch")
    torch.__version__ = "test-torch"
    torch.cuda = SimpleNamespace(is_available=lambda: cuda_available)
    seen["flash_enabled"] = True
    torch.backends = SimpleNamespace(
        cudnn=SimpleNamespace(allow_tf32=True, deterministic=False),
        cuda=SimpleNamespace(flash_sdp_enabled=lambda: seen["flash_enabled"],
                             enable_flash_sdp=lambda value: seen.update(flash_enabled=value)),
    )
    torch.get_float32_matmul_precision = lambda: "highest"

    def original_hub_loader(*args, **kwargs):
        raise AssertionError("tests and adapter must never download model weights")

    torch.hub = SimpleNamespace(load_state_dict_from_url=original_hub_loader)
    sp_weights = {"conv1a.weight": "local SuperPoint weights"}
    lg_weights = {"self_attn.0.Wqkv.weight": "self",
                  "cross_attn.8.to_qk.weight": "cross", "final_proj.0.weight": "final"}
    lg_weights.update({f"log_assignment.{layer}.r": f"obsolete scalar {layer}"
                       for layer in range(9)})

    def load(path, **kwargs):
        seen["loads"].append((path, kwargs))
        return sp_weights if path == sp.resolve() else dict(lg_weights)

    torch.load = load
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(lightglue, "_git_revision", lambda path: lightglue.OFFICIAL_SOURCE_REVISION)
    monkeypatch.setattr(lightglue, "_sha256", lambda path: lightglue.SUPERPOINT_SHA256
                        if path == sp.resolve() else lightglue.LIGHTGLUE_SHA256)
    monkeypatch.setattr(lightglue, "_load_official_symbols", lambda path: (SuperPoint, LightGlue))
    monkeypatch.setattr(lightglue, "version", lambda name: f"test-{name}-1")
    seen["hub_original"], seen["derived_buffer"] = original_hub_loader, derived_buffer
    return source, sp, lg, seen, torch


class TestOfficialLightGlueLoader:
    @pytest.mark.parametrize("missing", ["source", "superpoint", "lightglue"])
    def test_missing_paths_fail_before_torch_import(self, tmp_path, monkeypatch, missing):
        source, sp, lg = _native_source(tmp_path)
        prior_import = builtins.__import__

        def reject_torch(name, *args, **kwargs):
            if name == "torch":
                raise AssertionError("bad paths must fail before model imports")
            return prior_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", reject_torch)
        paths = [source, sp, lg]
        paths[["source", "superpoint", "lightglue"].index(missing)] = tmp_path / "missing"
        with pytest.raises(ValueError):
            OfficialSuperPointLightGlue(*paths)

    def test_wrong_source_pin_rejected_before_loading_weights(self, tmp_path, monkeypatch):
        source, sp, lg, seen, _ = _mock_native_loader(tmp_path, monkeypatch)
        monkeypatch.setattr(lightglue, "_git_revision", lambda path: "wrong")
        with pytest.raises(ValueError, match="source revision mismatch"):
            OfficialSuperPointLightGlue(source, sp, lg)
        assert seen["loads"] == []

    @pytest.mark.parametrize("wrong_weight", ["superpoint", "lightglue"])
    def test_either_weight_hash_mismatch_rejected_before_loading(self, tmp_path, monkeypatch,
                                                              wrong_weight):
        source, sp, lg, seen, _ = _mock_native_loader(tmp_path, monkeypatch)
        original_hash = lightglue._sha256
        wrong_path = sp.resolve() if wrong_weight == "superpoint" else lg.resolve()
        monkeypatch.setattr(lightglue, "_sha256", lambda path:
                            "wrong" if path == wrong_path else original_hash(path))
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            OfficialSuperPointLightGlue(source, sp, lg)
        assert seen["loads"] == []

    def test_safe_local_loading_architecture_and_strict_legacy_key_conversion(self, tmp_path,
                                                                          monkeypatch):
        source, sp, lg, seen, torch = _mock_native_loader(tmp_path, monkeypatch)
        backend = OfficialSuperPointLightGlue(source, sp, lg, device="cpu")
        assert seen["loads"] == [
            (sp.resolve(), {"map_location": "cpu", "weights_only": True}),
            (lg.resolve(), {"map_location": "cpu", "weights_only": True}),
        ]
        assert seen["superpoint_config"] == lightglue.SUPERPOINT_CONFIG
        assert seen["lightglue_config"] == {"features": None, **lightglue.LIGHTGLUE_CONFIG}
        assert seen["superpoint_weights"] == {"conv1a.weight": "local SuperPoint weights"}
        assert seen["strict"] is True
        assert seen["learned_weights"] == {
            "transformers.0.self_attn.Wqkv.weight": "self",
            "transformers.8.cross_attn.to_qk.weight": "cross",
            "final_proj.0.weight": "final",
            "confidence_thresholds": seen["derived_buffer"],
        }
        assert torch.hub.load_state_dict_from_url is seen["hub_original"]
        assert torch.backends.cuda.flash_sdp_enabled() is True
        for model in ("extractor", "matcher"):
            assert seen[model + "_eval"] is True
            assert seen[model + "_requires_grad"] is False
            assert seen[model + "_device"] == "cpu"
        assert backend.configuration["extract_resize"] is None
        assert backend.configuration["superpoint_sha256"] == lightglue.SUPERPOINT_SHA256
        assert backend.configuration["lightglue_sha256"] == lightglue.LIGHTGLUE_SHA256
        assert backend.configuration["lightglue"]["flash"] is False
        assert backend.configuration["lightglue"]["mp"] is False
        assert backend.configuration["precision"] == "fp32-flash-off-matmul-highest"
        assert backend.configuration["attention_backend"] == "torch-sdpa-math"
        assert backend.configuration["cudnn_allow_tf32"] is False
        assert backend.configuration["cudnn_deterministic"] is True

    @pytest.mark.parametrize("available,expected", [(False, "cpu"), (True, "cuda")])
    def test_auto_device_resolution_changes_configuration_and_revision(self, tmp_path,
                                                                    monkeypatch, available,
                                                                    expected):
        source, sp, lg, seen, _ = _mock_native_loader(tmp_path, monkeypatch,
                                                    cuda_available=available)
        backend = OfficialSuperPointLightGlue(source, sp, lg)
        assert backend.configuration["device"] == expected
        assert seen["matcher_device"] == expected
        assert expected in backend.revision

    @pytest.mark.parametrize("device", ["mps", "cuda", "invalid"])
    def test_unavailable_or_unsupported_devices_do_not_load_weights(self, tmp_path,
                                                                 monkeypatch, device):
        source, sp, lg, seen, _ = _mock_native_loader(tmp_path, monkeypatch)
        with pytest.raises(ValueError):
            OfficialSuperPointLightGlue(source, sp, lg, device=device)
        assert seen["loads"] == []

    def test_fp32_matmul_precision_is_required_not_changed(self, tmp_path, monkeypatch):
        source, sp, lg, seen, torch = _mock_native_loader(tmp_path, monkeypatch)
        torch.get_float32_matmul_precision = lambda: "high"
        with pytest.raises(ValueError, match="highest"):
            OfficialSuperPointLightGlue(source, sp, lg)
        assert torch.get_float32_matmul_precision() == "high"
        assert seen["loads"] == []

    def test_safe_load_failure_is_never_retried_using_pickle(self, tmp_path, monkeypatch):
        source, sp, lg, _, torch = _mock_native_loader(tmp_path, monkeypatch)
        calls = []

        def reject(path, **kwargs):
            calls.append(kwargs)
            raise RuntimeError("safe weights rejection")

        torch.load = reject
        with pytest.raises(RuntimeError, match="safe weights rejection"):
            OfficialSuperPointLightGlue(source, sp, lg)
        assert calls == [{"map_location": "cpu", "weights_only": True}]

    @pytest.mark.parametrize("failed_url", [None, "https://example.com/other-model.pth"])
    def test_superpoint_constructor_failure_restores_hub_and_blocks_unexpected_urls(
        self, tmp_path, monkeypatch, failed_url,
    ):
        source, sp, lg, seen, torch = _mock_native_loader(tmp_path, monkeypatch)
        _, model_class = lightglue._load_official_symbols(source)

        def failed_extractor(**kwargs):
            if failed_url is not None:
                torch.hub.load_state_dict_from_url(failed_url)
            raise RuntimeError("extractor construction failed")

        monkeypatch.setattr(lightglue, "_load_official_symbols", lambda path:
                            (failed_extractor, model_class))
        error = "extractor construction failed" if failed_url is None else "unexpected model download"
        with pytest.raises(RuntimeError, match=error):
            OfficialSuperPointLightGlue(source, sp, lg)
        assert torch.hub.load_state_dict_from_url is seen["hub_original"]

    @pytest.mark.parametrize("dependency", ["torch", "numpy", "kornia", "Pillow"])
    def test_changed_dependency_version_invalidates_revision(self, tmp_path, monkeypatch,
                                                           dependency):
        source, sp, lg, _, torch = _mock_native_loader(tmp_path, monkeypatch)
        first = OfficialSuperPointLightGlue(source, sp, lg)
        assert OfficialSuperPointLightGlue(source, sp, lg).revision == first.revision
        if dependency == "torch":
            torch.__version__ = "changed-torch"
        elif dependency == "numpy":
            monkeypatch.setattr(np, "__version__", "changed-numpy")
        else:
            old_version = lightglue.version
            monkeypatch.setattr(lightglue, "version", lambda name:
                                "changed-version" if name == dependency else old_version(name))
        assert OfficialSuperPointLightGlue(source, sp, lg).revision != first.revision

    def test_released_confidence_buffer_if_present_is_not_overwritten(self, tmp_path, monkeypatch):
        source, sp, lg, seen, torch = _mock_native_loader(tmp_path, monkeypatch)
        old_load = torch.load
        existing_buffer = object()

        def load(path, **kwargs):
            result = old_load(path, **kwargs)
            return dict(result, confidence_thresholds=existing_buffer) if path == lg.resolve() else result

        torch.load = load
        OfficialSuperPointLightGlue(source, sp, lg)
        assert seen["learned_weights"]["confidence_thresholds"] is existing_buffer

    @pytest.mark.parametrize("invalid", ["unknown", "missing-current-learned"])
    def test_legacy_r_whitelist_does_not_hide_unknown_or_missing_learned_keys(
        self, tmp_path, monkeypatch, invalid,
    ):
        source, sp, lg, seen, torch = _mock_native_loader(tmp_path, monkeypatch)
        old_load = torch.load

        def changed_load(path, **kwargs):
            result = old_load(path, **kwargs)
            if path == lg.resolve():
                if invalid == "unknown":
                    result["unknown.learned.weight"] = "not in current architecture"
                else:
                    result.pop("final_proj.0.weight")
            return result

        torch.load = changed_load
        with pytest.raises(RuntimeError, match="native strict state dict rejected keys"):
            OfficialSuperPointLightGlue(source, sp, lg)
        assert seen["strict"] is True
        assert not any(f"log_assignment.{layer}.r" in seen["learned_weights"]
                       for layer in range(9))
        assert torch.hub.load_state_dict_from_url is seen["hub_original"]
        assert torch.backends.cuda.flash_sdp_enabled() is True

    def test_missing_learned_keys_are_not_silently_accepted(self, tmp_path, monkeypatch):
        source, sp, lg, seen, torch = _mock_native_loader(tmp_path, monkeypatch)
        _, model_class = lightglue._load_official_symbols(source)

        def reject_state(self, weights, *, strict):
            assert strict is True
            raise RuntimeError("missing learned key")

        monkeypatch.setattr(model_class, "load_state_dict", reject_state)
        with pytest.raises(RuntimeError, match="missing learned key"):
            OfficialSuperPointLightGlue(source, sp, lg)
        assert torch.hub.load_state_dict_from_url is seen["hub_original"]
        assert torch.backends.cuda.flash_sdp_enabled() is True

    def test_failed_lightglue_constructor_restores_flash_setting(self, tmp_path, monkeypatch):
        source, sp, lg, _, torch = _mock_native_loader(tmp_path, monkeypatch)
        extractor, _ = lightglue._load_official_symbols(source)

        def reject_model(**kwargs):
            torch.backends.cuda.enable_flash_sdp(False)
            raise RuntimeError("matcher construction failed")

        monkeypatch.setattr(lightglue, "_load_official_symbols", lambda path: (extractor, reject_model))
        with pytest.raises(RuntimeError, match="matcher construction failed"):
            OfficialSuperPointLightGlue(source, sp, lg)
        assert torch.backends.cuda.flash_sdp_enabled() is True

    @pytest.mark.parametrize("fail", [False, True])
    def test_native_import_restores_cudnn_and_search_path_even_on_failure(self, tmp_path,
                                                                     monkeypatch, fail):
        real_load_symbols = lightglue._load_official_symbols
        source, _, _, _, torch = _mock_native_loader(tmp_path, monkeypatch)
        monkeypatch.setattr(lightglue, "_load_official_symbols", real_load_symbols)
        monkeypatch.delitem(sys.modules, "lightglue", raising=False)
        old_path = list(sys.path)
        package = SimpleNamespace(SuperPoint=object(), LightGlue=object())

        def import_package(name):
            assert name == "lightglue"
            assert sys.path[0] == str(source)
            torch.backends.cudnn.deterministic = True
            if fail:
                raise ImportError("native import failed")
            return package

        monkeypatch.setattr(lightglue.importlib, "import_module", import_package)
        if fail:
            with pytest.raises(ImportError, match="native import failed"):
                real_load_symbols(source)
        else:
            assert real_load_symbols(source) == (package.SuperPoint, package.LightGlue)
        assert torch.backends.cudnn.deterministic is False
        assert sys.path == old_path


@pytest.fixture(scope="class")
def _single_torch_thread():
    pytest.importorskip("pyarrow")
    torch = pytest.importorskip("torch")
    previous = torch.get_num_threads()
    torch.set_num_threads(min(4, previous))
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _mock_inference_backend(torch, *, pairs=None, scores=None, fail=False):
    seen = {"calls": []}
    q_points = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
    c_points = torch.tensor([[11., 12.], [13., 14.], [15., 16.], [17., 18.], [19., 20.]])
    pairs = torch.tensor([[2, 4], [0, 1]]) if pairs is None else pairs
    scores = torch.tensor([.2, .75]) if scores is None else scores

    class Extractor:
        def extract(self, image, *, resize):
            assert resize is None
            assert image.ndim == 3 and image.shape[0] == 3
            assert image.dtype == torch.float32
            assert torch.is_inference_mode_enabled()
            assert not torch.is_autocast_enabled(image.device.type)
            assert torch.backends.cudnn.deterministic is True
            assert torch.backends.cudnn.allow_tf32 is False
            assert torch.backends.cudnn.benchmark is False
            assert torch.backends.cuda.flash_sdp_enabled() is False
            seen["calls"].append(image.clone())
            points = q_points if len(seen["calls"]) == 1 else c_points
            return {"keypoints": points[None], "descriptors": torch.zeros((1, len(points), 256))}

    class Matcher:
        def __call__(self, features):
            assert len(features["image0"]["keypoints"][0]) == 3
            assert len(features["image1"]["keypoints"][0]) == 5
            if fail:
                raise RuntimeError("native inference failed")
            return {"matches": [pairs], "scores": [scores],
                    "token_confidence": torch.ones((1, 3)),
                    "matching_scores0": torch.ones((1, 3))}

    wrapper = object.__new__(OfficialSuperPointLightGlue)
    wrapper._torch, wrapper._device = torch, "cpu"
    wrapper._extractor, wrapper._model = Extractor(), Matcher()
    return wrapper, seen, q_points, c_points


@pytest.mark.usefixtures("_single_torch_thread")
class TestOfficialLightGlueInference:
    def test_native_pair_indices_scores_and_rgb_rasters_with_resize_disabled(self):
        torch = pytest.importorskip("torch")
        wrapper, seen, _, _ = _mock_inference_backend(torch)
        query = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
        context = np.full((48, 64, 3), 32, dtype=np.uint8)
        result = wrapper.match(query, context)
        assert [tuple(image.shape) for image in seen["calls"]] == [(3, 12, 16), (3, 48, 64)]
        np.testing.assert_allclose(seen["calls"][0].numpy(), query.transpose(2, 0, 1) / 255,
                                   atol=1e-7)
        np.testing.assert_allclose(seen["calls"][1].numpy(), context.transpose(2, 0, 1) / 255,
                                   atol=1e-7)
        np.testing.assert_array_equal(result.query_points, [[5., 6.], [1., 2.]])
        np.testing.assert_array_equal(result.context_points, [[19., 20.], [13., 14.]])
        np.testing.assert_allclose(result.confidence, [.2, .75])

    @pytest.mark.parametrize("fail", [False, True])
    def test_external_autocast_is_disabled_locally_and_restored(self, fail):
        torch = pytest.importorskip("torch")
        wrapper, _, _, _ = _mock_inference_backend(torch, fail=fail)
        query, context = np.zeros((12, 16, 3), np.uint8), np.zeros((48, 64, 3), np.uint8)
        previous = torch.is_autocast_enabled("cpu")
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            assert torch.is_autocast_enabled("cpu")
            if fail:
                with pytest.raises(RuntimeError, match="native inference failed"):
                    wrapper.match(query, context)
            else:
                wrapper.match(query, context)
            assert torch.is_autocast_enabled("cpu")
        assert torch.is_autocast_enabled("cpu") == previous

    @pytest.mark.parametrize("precision", ["high", "medium"])
    def test_changed_matmul_precision_fails_before_feature_extraction(self, precision):
        torch = pytest.importorskip("torch")
        wrapper, seen, _, _ = _mock_inference_backend(torch)
        previous = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision(precision)
            with pytest.raises(ValueError, match="matmul precision 'highest'"):
                wrapper.match(np.zeros((12, 16, 3), np.uint8), np.zeros((48, 64, 3), np.uint8))
            assert seen["calls"] == []
        finally:
            torch.set_float32_matmul_precision(previous)

    @pytest.mark.parametrize("fail", [False, True])
    def test_inference_restores_callers_attention_and_cudnn_flags(self, monkeypatch, fail):
        torch = pytest.importorskip("torch")
        original_flash = torch.backends.cuda.flash_sdp_enabled()
        original_flags = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                          torch.backends.cudnn.allow_tf32)
        wrapper, _, _, _ = _mock_inference_backend(torch, fail=fail)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = True
        try:
            query, context = np.zeros((12, 16, 3), np.uint8), np.zeros((48, 64, 3), np.uint8)
            if fail:
                with pytest.raises(RuntimeError, match="native inference failed"):
                    wrapper.match(query, context)
            else:
                wrapper.match(query, context)
            assert torch.backends.cuda.flash_sdp_enabled() is True
            assert torch.backends.cudnn.benchmark is True
            assert torch.backends.cudnn.deterministic is False
            assert torch.backends.cudnn.allow_tf32 is True
        finally:
            torch.backends.cuda.enable_flash_sdp(original_flash)
            (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
             torch.backends.cudnn.allow_tf32) = original_flags

    @pytest.mark.parametrize("pairs,scores", [
        ([[3, 0]], [.5]), ([[0, 5]], [.5]), ([[-1, 0]], [.5]),
        ([[0., 1.]], [.5]), ([[0, 1]], [.5, .6]),
    ])
    def test_malformed_pair_indices_or_score_counts_raise(self, pairs, scores):
        torch = pytest.importorskip("torch")
        wrapper, _, _, _ = _mock_inference_backend(torch, pairs=torch.tensor(pairs),
                                                  scores=torch.tensor(scores))
        with pytest.raises(ValueError, match="malformed match indices or scores"):
            wrapper.match(np.zeros((12, 16, 3), np.uint8), np.zeros((48, 64, 3), np.uint8))

    def test_empty_matching_output_returns_correctly_shaped_arrays(self):
        torch = pytest.importorskip("torch")
        wrapper, _, _, _ = _mock_inference_backend(torch, pairs=torch.empty((0, 2), dtype=torch.long),
                                                  scores=torch.empty(0))
        result = wrapper.match(np.zeros((12, 16, 3), np.uint8), np.zeros((48, 64, 3), np.uint8))
        assert result.query_points.shape == result.context_points.shape == (0, 2)
        assert result.confidence.shape == (0,)

    @pytest.mark.parametrize("shape,dtype", [((12, 16), np.uint8), ((12, 16, 4), np.uint8),
                                            ((12, 16, 3), np.float32), ((7, 16, 3), np.uint8)])
    def test_invalid_rgb_or_superpoint_short_side_fails_before_extraction(self, shape, dtype):
        torch = pytest.importorskip("torch")
        wrapper, seen, _, _ = _mock_inference_backend(torch)
        with pytest.raises(ValueError, match="RGB uint8"):
            wrapper.match(np.zeros(shape, dtype=dtype), np.zeros((48, 64, 3), np.uint8))
        assert seen["calls"] == []
