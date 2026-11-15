"""MINIMA-LightGlue identity and inherited adapter checks without real weights."""
from __future__ import annotations

import re

import numpy as np
import pytest

from pyrafault.geom import BBoxL0
from pyrafault.verify.base import Verifier
from pyrafault.verify import lightglue
from pyrafault.verify.lightglue import LightGlueBaseline, OfficialSuperPointLightGlue
from pyrafault.verify import minima_lightglue
from pyrafault.verify.minima_lightglue import MinimaLightGlueBaseline, OfficialMinimaLightGlue

# Reuse small fake-image/matcher/native-loader helpers, not the parent's test suite.
from tests.test_verify_lightglue import (
    _GuardedInputs, _Matcher, _inputs, _mock_native_loader, _translated_matches,
)


def _minima_loader(tmp_path, monkeypatch):
    source, sp, checkpoint, seen, torch = _mock_native_loader(tmp_path, monkeypatch)
    monkeypatch.setattr(lightglue, "_sha256", lambda path: lightglue.SUPERPOINT_SHA256
                        if path == sp.resolve() else minima_lightglue.MINIMA_LIGHTGLUE_SHA256)
    return source, sp, checkpoint, seen, torch


class TestMinimaLightGlueAdapter:
    def test_new_method_identity_reuses_exactly_parent_localise(self):
        query, context = _translated_matches()
        baseline = MinimaLightGlueBaseline(_Matcher(query, context))
        original = LightGlueBaseline(_Matcher(query, context))
        assert isinstance(baseline, Verifier)
        assert isinstance(baseline, LightGlueBaseline)
        assert MinimaLightGlueBaseline.localise is LightGlueBaseline.localise
        assert baseline.name == "minima_lightglue_baseline"
        assert baseline.name != original.name
        assert baseline.revision.startswith("minima")
        assert baseline.revision != original.revision
        assert baseline.localise(_inputs()) == original.localise(_inputs())

    def test_mpp_alignment_keeps_rgb_context_and_does_not_pad_or_auto_resize(self):
        query, context = _translated_matches()
        matcher = _Matcher(query, context)
        inputs = _inputs()
        result = MinimaLightGlueBaseline(matcher).localise(inputs)
        assert matcher.calls[0][0].shape == (12, 16, 3)
        np.testing.assert_array_equal(matcher.calls[0][1], np.asarray(inputs["context"]))
        assert result.estimated_bbox == BBoxL0(3100, 2800, 1600, 600)
        assert result.diagnostics["aligned_query_size_px"] == [16, 12]
        assert result.diagnostics["context_size_px"] == [64, 48]

    def test_fractional_mpp_preserves_superpoint_pixel_centres_and_footprint(self):
        scale = [13.53 / 14, 10.23 / 10]
        query, context = _translated_matches(dx=10.2, dy=8.3, scale=scale)
        inputs = _inputs(query_size=(41, 31), query_mpp=.66, context_mpp=2.,
                         context_bbox=BBoxL0(100, 200, 6400, 4800))
        result = MinimaLightGlueBaseline(_Matcher(query, context)).localise(inputs)
        assert result.estimated_bbox == BBoxL0(1120, 1030, 1353, 1023)
        assert result.diagnostics["query_to_context_point_scale"] == pytest.approx(scale)
        assert result.diagnostics["translation_xy_context"] == pytest.approx([10.2, 8.3])

    def test_claim_ground_truth_split_and_ids_are_not_read(self):
        query, context = _translated_matches()
        guarded = _GuardedInputs(_inputs())
        baseline = MinimaLightGlueBaseline(_Matcher(query, context))
        baseline.localise(guarded)
        assert guarded.accessed == guarded.allowed
        inputs = _inputs()
        changed = dict(inputs, claimed_bbox=BBoxL0(99999, 99999, 1, 1),
                       split="test", item_id="other", slide_id="other", target={})
        assert baseline.localise(inputs) == baseline.localise(changed)

    def test_shared_reliability_uses_inlier_support_times_mean_match_score(self):
        query, context = _translated_matches()
        query = np.vstack([query, [[1, 4], [7, 1], [4, 4]]])
        context = np.vstack([context, [[50, 40], [12, 30], [60, 2]]])
        result = MinimaLightGlueBaseline(
            _Matcher(query, context, [.1, .2, .3, .4, .5, .6, .99, .99, .99])
        ).localise(_inputs())
        assert result.estimated_bbox == BBoxL0(3100, 2800, 1600, 600)
        assert result.reliability == pytest.approx((6 / 9) * .35)
        assert result.diagnostics["mean_inlier_match_score"] == pytest.approx(.35)

    @pytest.mark.parametrize("count", [0, 3])
    def test_insufficient_matches_keep_none_zero_protocol(self, count):
        query, context = _translated_matches()
        result = MinimaLightGlueBaseline(_Matcher(query[:count], context[:count])).localise(_inputs())
        assert result.estimated_bbox is None
        assert result.reliability == 0.
        assert result.diagnostics["reason"] == "too_few_valid_matches"

    def test_small_query_abstains_and_would_upsample_raises_like_parent(self):
        query, context = _translated_matches()
        matcher = _Matcher(query, context)
        baseline = MinimaLightGlueBaseline(matcher)
        result = baseline.localise(_inputs(query_size=(20, 14)))
        assert result.estimated_bbox is None
        assert result.reliability == 0.
        assert result.diagnostics["reason"] == "input_too_small_for_superpoint"
        with pytest.raises(ValueError, match="upsample"):
            baseline.localise(_inputs(query_mpp=2.))
        assert matcher.calls == []

    def test_separate_method_constructs_without_models_when_matcher_is_injected(self):
        query, context = _translated_matches()
        result = MinimaLightGlueBaseline(_Matcher(query, context))
        assert isinstance(result, MinimaLightGlueBaseline)
        assert result.name == "minima_lightglue_baseline"


class TestOfficialMinimaLightGlue:
    def test_weight_family_is_distinct_without_changing_parent_checksum(self):
        assert lightglue.LIGHTGLUE_SHA256 == (
            "6ff7040d0a497fc6639337946d7538dae07428c18f77a067a0b5a960e7cc551a"
        )
        assert re.fullmatch(r"[0-9a-f]{64}", minima_lightglue.MINIMA_LIGHTGLUE_SHA256)
        assert minima_lightglue.MINIMA_LIGHTGLUE_SHA256 != lightglue.LIGHTGLUE_SHA256
        assert issubclass(OfficialMinimaLightGlue, OfficialSuperPointLightGlue)
        assert OfficialMinimaLightGlue.match is OfficialSuperPointLightGlue.match

    def test_minima_rejects_stock_lightglue_checkpoint_before_deserialization(self, tmp_path,
                                                                         monkeypatch):
        source, sp, checkpoint, seen, _ = _mock_native_loader(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            OfficialMinimaLightGlue(source, sp, checkpoint)
        assert seen["loads"] == []

    def test_stock_lightglue_rejects_minima_checkpoint_before_deserialization(self, tmp_path,
                                                                         monkeypatch):
        source, sp, checkpoint, seen, _ = _minima_loader(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            OfficialSuperPointLightGlue(source, sp, checkpoint)
        assert seen["loads"] == []

    def test_safe_strict_frozen_loader_retains_parent_configuration_with_new_family(self, tmp_path,
                                                                                monkeypatch):
        source, sp, checkpoint, seen, torch = _minima_loader(tmp_path, monkeypatch)
        backend = OfficialMinimaLightGlue(source, sp, checkpoint, device="cpu")
        assert seen["loads"] == [
            (sp.resolve(), {"map_location": "cpu", "weights_only": True}),
            (checkpoint.resolve(), {"map_location": "cpu", "weights_only": True}),
        ]
        assert seen["strict"] is True
        assert seen["superpoint_config"] == lightglue.SUPERPOINT_CONFIG
        assert seen["lightglue_config"] == {"features": None, **lightglue.LIGHTGLUE_CONFIG}
        assert backend.configuration["checkpoint_family"] == "minima_lightglue"
        assert backend.configuration["lightglue_sha256"] == minima_lightglue.MINIMA_LIGHTGLUE_SHA256
        assert backend.configuration["superpoint_sha256"] == lightglue.SUPERPOINT_SHA256
        assert backend.configuration["extract_resize"] is None
        assert backend.configuration["attention_backend"] == "torch-sdpa-math"
        assert backend.configuration["precision"] == "fp32-flash-off-matmul-highest"
        assert backend.revision.startswith("minima")
        for model in ("extractor", "matcher"):
            assert seen[model + "_eval"] is True
            assert seen[model + "_requires_grad"] is False
        assert torch.hub.load_state_dict_from_url is seen["hub_original"]
        assert torch.backends.cuda.flash_sdp_enabled() is True

    def test_rejects_missing_current_learned_keys_even_with_correct_family_hash(self, tmp_path,
                                                                           monkeypatch):
        source, sp, checkpoint, _, torch = _minima_loader(tmp_path, monkeypatch)
        old_load = torch.load

        def missing_key(path, **kwargs):
            result = old_load(path, **kwargs)
            if path == checkpoint.resolve():
                result.pop("final_proj.0.weight")
            return result

        torch.load = missing_key
        with pytest.raises(RuntimeError, match="strict state dict rejected"):
            OfficialMinimaLightGlue(source, sp, checkpoint)

    def test_checkpoint_family_does_not_enable_unsafe_pickle_fallback(self, tmp_path, monkeypatch):
        source, sp, checkpoint, _, torch = _minima_loader(tmp_path, monkeypatch)
        calls = []

        def rejected(path, **kwargs):
            calls.append(kwargs)
            raise RuntimeError("safe checkpoint rejection")

        torch.load = rejected
        with pytest.raises(RuntimeError, match="safe checkpoint rejection"):
            OfficialMinimaLightGlue(source, sp, checkpoint)
        assert calls == [{"map_location": "cpu", "weights_only": True}]
