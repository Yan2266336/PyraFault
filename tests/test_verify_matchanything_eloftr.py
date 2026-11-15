"""MatchAnything-ELoFTR adapter contract tests without models or downloads."""
from __future__ import annotations

from collections.abc import Mapping
import builtins
import hashlib
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from pyrafault.geom import BBoxL0
from pyrafault.verify.base import Localisation, Verifier
from pyrafault.verify.efficient_loftr import EfficientLoFTRBaseline, Matches
from pyrafault.verify import matchanything_eloftr as matchanything
from pyrafault.verify.matchanything_eloftr import (
    MatchAnythingELoFTRBaseline,
    OfficialMatchAnythingELoFTR,
)
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
        "slide_id": "slide",
        "item_id": "item",
        "split": "dev",
        "target": {"true_bbox": BBoxL0(999, 999, 99, 99)},
    }


def _translated_matches(dx=21, dy=16):
    query = np.array([
        [1.0, 1.0], [3.0, 1.0], [5.0, 2.0],
        [2.0, 3.0], [6.0, 3.0], [8.0, 4.0],
    ])
    return query, query + [dx, dy]


class _Matcher:
    revision = "fake-matchanything-1"
    max_input_side = 2048

    def __init__(self, query_points, context_points, confidence=None):
        self.query_points = np.asarray(query_points, dtype=np.float32).reshape(-1, 2)
        self.context_points = np.asarray(context_points, dtype=np.float32).reshape(-1, 2)
        self.confidence = np.asarray(
            np.ones(len(self.query_points)) if confidence is None else confidence,
            dtype=np.float32,
        )
        self.calls = []

    def match(self, query, context):
        assert query.dtype == context.dtype == np.uint8
        assert query.ndim == context.ndim == 2
        self.calls.append((query.copy(), context.copy()))
        return Matches(self.query_points, self.context_points, self.confidence)


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


class TestMatchAnythingELoFTRContract:
    def test_inherits_existing_geometry_and_is_a_verifier(self):
        query, context = _translated_matches()
        baseline = MatchAnythingELoFTRBaseline(_Matcher(query, context))
        assert isinstance(baseline, EfficientLoFTRBaseline)
        assert isinstance(baseline, Verifier)
        assert MatchAnythingELoFTRBaseline.localise is EfficientLoFTRBaseline.localise
        assert isinstance(baseline.localise(_inputs()), Localisation)
        assert baseline.name == "matchanything_eloftr_baseline"
        assert baseline.name != EfficientLoFTRBaseline.name
        assert "fake-matchanything-1" in baseline.revision

    @pytest.mark.parametrize("dx,dy", [(0, 0), (21, 16), (54, 42), (12.25, 8.75)])
    def test_correspondences_produce_exactly_original_eloftr_geometry(self, dx, dy):
        query, context = _translated_matches(dx=dx, dy=dy)
        original = EfficientLoFTRBaseline(_Matcher(query, context))
        new = MatchAnythingELoFTRBaseline(_Matcher(query, context))
        assert new.localise(_inputs()) == original.localise(_inputs())

    def test_query_mpp_alignment_padding_and_image_order(self):
        query, context = _translated_matches()
        matcher = _Matcher(query, context)
        result = MatchAnythingELoFTRBaseline(matcher).localise(_inputs())
        sent_query, sent_context = matcher.calls[0]
        # Q 20x12 at .5 mpp -> 10x6 at C 1 mpp; right/bottom edge padding only.
        assert sent_query.shape == (32, 32)
        assert sent_context.shape == (64, 64)
        assert np.all(sent_query[6:, :] == sent_query[5:6, :])
        assert np.all(sent_query[:, 10:] == sent_query[:, 9:10])
        expected_context = np.asarray(_inputs()["context"].convert("L"))
        np.testing.assert_array_equal(sent_context[:48, :64], expected_context)
        assert result.diagnostics["aligned_query_size_px"] == [10, 6]
        assert result.estimated_bbox == BBoxL0(3100, 2800, 1000, 300)
        assert result.reliability == 1.0

    def test_fractional_physical_footprint_not_rounded_model_raster(self):
        query = np.array([[1, 1], [2, 1], [3, 1], [4, 2], [5, 2]], dtype=float)
        context = query * [6.65 / 7, 3.85 / 4] + [10, 8]
        inputs = _inputs(query_size=(19, 11), context_mpp=2.0, query_mpp=0.7,
                         context_bbox=BBoxL0(100, 200, 640, 480))
        result = MatchAnythingELoFTRBaseline(_Matcher(query, context)).localise(inputs)
        assert result.estimated_bbox == BBoxL0(200, 280, 66, 38)
        assert result.diagnostics["aligned_query_size_px"] == [7, 4]
        assert result.diagnostics["query_to_context_point_scale"] == pytest.approx(
            [6.65 / 7, 3.85 / 4]
        )

    def test_claim_ids_split_and_target_are_not_algorithm_inputs(self):
        query, context = _translated_matches()
        guarded = _GuardedInputs(_inputs())
        MatchAnythingELoFTRBaseline(_Matcher(query, context)).localise(guarded)
        assert guarded.accessed == guarded.allowed

    def test_changing_metadata_and_labels_does_not_change_localisation(self):
        query, context = _translated_matches()
        baseline = MatchAnythingELoFTRBaseline(_Matcher(query, context))
        first = _inputs()
        second = dict(first, claimed_bbox=BBoxL0(99999, 88888, 10, 20),
                      item_id="other", slide_id="other", split="test",
                      target={"true_bbox": BBoxL0(0, 0, 100000, 100000),
                              "claim_correct": False})
        assert baseline.localise(first) == baseline.localise(second)

    def test_real_shared_runner_audits_claim_independence(self):
        query, context = _translated_matches()
        baseline = MatchAnythingELoFTRBaseline(_Matcher(query, context))
        inputs = _inputs()
        true_bbox = BBoxL0(3100, 2800, 1000, 300)
        inputs["claimed_bbox"] = true_bbox
        moved_inputs = dict(inputs, claimed_bbox=BBoxL0(5000, 3500, 1000, 300))

        class Dataset:
            item_ids = ["clean", "drift"]

            def __getitem__(self, key):
                return {"inputs": inputs if key == "clean" else moved_inputs,
                        "target": {"true_bbox": true_bbox,
                                   "true_slide_id": "slide",
                                   "claim_correct": key == "clean"}}

        predictions = run_verifier(
            baseline, Dataset(),
            Thresholds(tau=0.5, delta=0.1, fitted_on="calibration", n_items=1,
                       residuals={}),
            audit_independence=True,
        )
        assert tuple(predictions.columns) == PREDICTION_COLUMNS
        assert predictions["decision"].tolist() == ["verified", "mismatched"]
        assert predictions["verifier_name"].tolist() == [baseline.name] * 2
        assert predictions["verifier_revision"].tolist() == [baseline.revision] * 2
        assert predictions["estimated_bbox_l0_x"].tolist() == [3100] * 2
        assert predictions["reliability"].tolist() == [1.0] * 2
        # First calculation + deliberately perturbed claim + actual second claim.
        assert len(baseline._matcher.calls) == 3

    def test_revision_tracks_backend_weights_or_source_and_adapter_parameters(self):
        query, context = _translated_matches()
        matcher = _Matcher(query, context)
        original = MatchAnythingELoFTRBaseline(matcher)
        matcher2 = _Matcher(query, context)
        matcher2.revision = "fake-matchanything-source2-weights2"
        variants = [
            original,
            MatchAnythingELoFTRBaseline(matcher2),
            MatchAnythingELoFTRBaseline(matcher, min_matches=5),
            MatchAnythingELoFTRBaseline(matcher, translation_tolerance_px=2.0),
        ]
        assert len({baseline.revision for baseline in variants}) == len(variants)


class TestMatchAnythingELoFTRGeometryFailures:
    def test_geometric_inlier_ratio_not_raw_model_confidence(self):
        query, context = _translated_matches()
        query = np.vstack([query, [[1, 4], [7, 1], [4, 4]]])
        context = np.vstack([context, [[50, 40], [12, 30], [60, 2]]])
        matcher = _Matcher(query, context, [.2] * 6 + [.99] * 3)
        result = MatchAnythingELoFTRBaseline(matcher).localise(_inputs())
        assert result.estimated_bbox == BBoxL0(3100, 2800, 1000, 300)
        assert result.reliability == pytest.approx(6 / 9)
        assert result.diagnostics["inlier_matches"] == 6
        assert result.diagnostics["median_inlier_confidence"] == pytest.approx(.2)

    def test_padding_nonfinite_and_duplicates_are_removed(self):
        query, context = _translated_matches()
        query = np.vstack([query, query[0], [12, 2], [3, 3], [np.nan, 1], [1, 2]])
        context = np.vstack([context, context[0], [20, 20], [10, 55], [5, 5], [5, 5]])
        confidence = np.array([1.0] * 10 + [np.inf])
        result = MatchAnythingELoFTRBaseline(
            _Matcher(query, context, confidence)
        ).localise(_inputs())
        assert result.reliability == 1.0
        assert result.diagnostics["total_matches"] == 11
        assert result.diagnostics["valid_matches"] == 6

    @pytest.mark.parametrize("count", [0, 1, 2, 3])
    def test_insufficient_matches_returns_none_and_zero(self, count):
        query, context = _translated_matches()
        result = MatchAnythingELoFTRBaseline(
            _Matcher(query[:count], context[:count])
        ).localise(_inputs())
        assert result.estimated_bbox is None
        assert result.reliability == 0.0
        assert result.diagnostics["reason"] == "too_few_valid_matches"

    def test_repeated_points_cannot_fake_geometric_support(self):
        query = np.repeat([[1.0, 1.0]], 10, axis=0)
        context = np.repeat([[22.0, 17.0]], 10, axis=0)
        result = MatchAnythingELoFTRBaseline(_Matcher(query, context)).localise(_inputs())
        assert result.estimated_bbox is None
        assert result.reliability == 0.0

    def test_inconsistent_matches_return_no_geometric_estimate(self):
        query = np.array([[1, 1], [2, 1], [3, 1], [4, 1], [5, 1]], dtype=float)
        context = np.array([[1, 1], [8, 5], [16, 9], [25, 14], [35, 20]], dtype=float)
        result = MatchAnythingELoFTRBaseline(_Matcher(query, context)).localise(_inputs())
        assert result.estimated_bbox is None
        assert result.reliability == 0.0
        assert result.diagnostics["reason"] == "too_few_geometric_inliers"

    def test_input_limit_and_oversized_query_fail_before_model(self):
        query, context = _translated_matches()
        matcher = _Matcher(query, context)
        matcher.max_input_side = 32
        result = MatchAnythingELoFTRBaseline(matcher).localise(_inputs())
        assert result.estimated_bbox is None
        assert result.diagnostics["reason"] == "input_exceeds_model_limit"
        assert matcher.calls == []
        matcher.max_input_side = 2048
        result = MatchAnythingELoFTRBaseline(matcher).localise(
            _inputs(query_size=(200, 200))
        )
        assert result.estimated_bbox is None
        assert result.diagnostics["reason"] == "query_footprint_too_large"
        assert matcher.calls == []

    @pytest.mark.parametrize("field", ["context_mpp", "query_mpp"])
    @pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), None, "invalid"])
    def test_invalid_mpp_is_an_error_not_false_no_match(self, field, value):
        inputs = _inputs()
        inputs[field] = value
        query, context = _translated_matches()
        with pytest.raises(ValueError, match="finite and positive"):
            MatchAnythingELoFTRBaseline(_Matcher(query, context)).localise(inputs)

    def test_query_upsampling_is_not_silently_allowed(self):
        query, context = _translated_matches()
        with pytest.raises(ValueError, match="upsample"):
            MatchAnythingELoFTRBaseline(_Matcher(query, context)).localise(
                _inputs(context_mpp=.5, query_mpp=1.0)
            )

    def test_malformed_matches_raise_instead_of_returning_a_rejection(self):
        matcher = _Matcher([[1, 1], [2, 2]], [[3, 3]], [1, 1])
        with pytest.raises(ValueError, match="equally sized"):
            MatchAnythingELoFTRBaseline(matcher).localise(_inputs())

    @pytest.mark.parametrize("kwargs,pattern", [
        ({"min_matches": 1}, "min_matches"),
        ({"translation_tolerance_px": 0}, "finite and positive"),
        ({"translation_tolerance_px": float("inf")}, "finite and positive"),
        ({"translation_tolerance_px": float("nan")}, "finite and positive"),
    ])
    def test_invalid_adapter_configuration_raises(self, kwargs, pattern):
        query, context = _translated_matches()
        with pytest.raises(ValueError, match=pattern):
            MatchAnythingELoFTRBaseline(_Matcher(query, context), **kwargs)


class TestMatchAnythingELoFTRConstruction:
    def test_constructs_injected_backend_without_optional_models(self):
        query, context = _translated_matches()
        matcher = _Matcher(query, context)
        baseline = MatchAnythingELoFTRBaseline(matcher)
        assert isinstance(baseline, MatchAnythingELoFTRBaseline)
        assert baseline._matcher is matcher
        assert baseline.localise(_inputs()).estimated_bbox == BBoxL0(3100, 2800, 1000, 300)


def _native_source(tmp_path):
    """Only placeholder paths; no author source is needed for loader unit tests."""
    source = tmp_path / "native-source"
    for relative in ("src/loftr/__init__.py", "src/config/default.py",
                     "configs/models/eloftr_model.py"):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    checkpoint = tmp_path / "fake.ckpt"
    checkpoint.write_bytes(b"not an actual checkpoint")
    return source, checkpoint


def _forbidden_backend_imports(monkeypatch):
    original_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name in {"torch", "kornia"}:
            raise AssertionError("invalid provenance must fail before model imports")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)


def _mock_native_loader(tmp_path, monkeypatch, *, config=None, cuda_available=False):
    source, checkpoint = _native_source(tmp_path)
    seen = {}
    config = {"coarse": {"npe": [832, 832, 2048, 2048]},
              "match_coarse": {"thr": .1}} if config is None else config

    class Model:
        def __init__(self, *, config):
            seen["config"] = config

        def load_state_dict(self, state_dict, *, strict):
            seen["state_dict"] = state_dict
            seen["strict"] = strict

        def eval(self):
            seen["eval"] = True
            return self

        def requires_grad_(self, value):
            seen["requires_grad"] = value
            return self

        def to(self, device):
            seen["device"] = device
            return self

    def load(path, **kwargs):
        seen["load_path"] = path
        seen["load_kwargs"] = kwargs
        return {"state_dict": {"frozen_weights": "test"}}

    torch = ModuleType("torch")
    torch.__version__ = "test-torch"
    torch.cuda = SimpleNamespace(is_available=lambda: cuda_available)
    torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    torch.load = load
    kornia = ModuleType("kornia")
    kornia.__version__ = "test-kornia"
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "kornia", kornia)
    monkeypatch.setattr(matchanything, "_git_revision",
                        lambda path: matchanything.OFFICIAL_SOURCE_REVISION)
    monkeypatch.setattr(matchanything, "_sha256",
                        lambda path: matchanything.OFFICIAL_CHECKPOINT_SHA256)
    monkeypatch.setattr(matchanything, "_load_official_symbols",
                        lambda path: (Model, config))
    monkeypatch.setattr(matchanything, "version", lambda name: f"test-{name}-1")
    return source, checkpoint, seen, torch


class TestOfficialMatchAnythingELoFTRLoader:
    def test_invalid_source_fails_before_model_import(self, tmp_path, monkeypatch):
        _forbidden_backend_imports(monkeypatch)
        with pytest.raises(ValueError, match="matchanything_eloftr_source"):
            OfficialMatchAnythingELoFTR(tmp_path / "missing", tmp_path / "missing.ckpt")

    def test_missing_checkpoint_fails_before_model_import(self, tmp_path, monkeypatch):
        source, _ = _native_source(tmp_path)
        _forbidden_backend_imports(monkeypatch)
        with pytest.raises(ValueError, match="checkpoint does not exist"):
            OfficialMatchAnythingELoFTR(source, tmp_path / "missing.ckpt")

    def test_wrong_git_revision_fails_before_hash_or_model(self, tmp_path, monkeypatch):
        source, checkpoint = _native_source(tmp_path)
        _forbidden_backend_imports(monkeypatch)
        monkeypatch.setattr(matchanything, "_git_revision", lambda path: "wrong-revision")

        def unexpected_hash(path):
            raise AssertionError("wrong source must stop before checkpoint inspection")

        monkeypatch.setattr(matchanything, "_sha256", unexpected_hash)
        with pytest.raises(ValueError, match="source revision mismatch"):
            OfficialMatchAnythingELoFTR(source, checkpoint)

    def test_wrong_checkpoint_hash_fails_before_model_import(self, tmp_path, monkeypatch):
        source, checkpoint = _native_source(tmp_path)
        _forbidden_backend_imports(monkeypatch)
        monkeypatch.setattr(matchanything, "_git_revision",
                            lambda path: matchanything.OFFICIAL_SOURCE_REVISION)
        monkeypatch.setattr(matchanything, "_sha256", lambda path: "wrong-digest")
        with pytest.raises(ValueError, match="checkpoint sha256 mismatch"):
            OfficialMatchAnythingELoFTR(source, checkpoint)

    def test_safe_strict_loading_and_frozen_eval_model(self, tmp_path, monkeypatch):
        source, checkpoint, seen, _ = _mock_native_loader(tmp_path, monkeypatch)
        wrapper = OfficialMatchAnythingELoFTR(source, checkpoint, device="cpu")
        assert seen["load_path"] == checkpoint.resolve()
        assert seen["load_kwargs"] == {"map_location": "cpu", "weights_only": True}
        assert seen["state_dict"] == {"frozen_weights": "test"}
        assert seen["strict"] is True
        assert seen["eval"] is True
        assert seen["requires_grad"] is False
        assert seen["device"] == "cpu"
        assert wrapper.max_input_side == 2048
        assert matchanything.OFFICIAL_SOURCE_REVISION[:12] in wrapper.revision
        assert matchanything.OFFICIAL_CHECKPOINT_SHA256[:12] in wrapper.revision
        assert "fp32-npe2048-thr0.1-unfused-cpu" in wrapper.revision
        assert "test-torch" in wrapper.revision
        assert "test-kornia" in wrapper.revision
        assert wrapper.configuration["source_revision"] == matchanything.OFFICIAL_SOURCE_REVISION
        assert wrapper.configuration["checkpoint_sha256"] == matchanything.OFFICIAL_CHECKPOINT_SHA256
        assert wrapper.configuration["native_model_config"] == seen["config"]
        assert wrapper.configuration["device"] == "cpu"
        assert wrapper.configuration["precision"] == "fp32-native-rope-buffers"
        assert wrapper.configuration["reparameterized"] is False
        assert wrapper.configuration["dependencies"] == {
            **{name: f"test-{name}-1" for name in ("Pillow", "einops", "loguru", "yacs")},
            "numpy": np.__version__, "torch": "test-torch", "kornia": "test-kornia",
        }

    @pytest.mark.parametrize("dependency", ["Pillow", "einops", "loguru", "yacs",
                                            "numpy", "torch", "kornia"])
    def test_dependency_versions_are_deterministic_and_change_revision(self, tmp_path,
                                                                      monkeypatch, dependency):
        source, checkpoint, _, torch = _mock_native_loader(tmp_path, monkeypatch)
        first = OfficialMatchAnythingELoFTR(source, checkpoint, device="cpu")
        same = OfficialMatchAnythingELoFTR(source, checkpoint, device="cpu")
        assert first.configuration == same.configuration
        assert first.revision == same.revision
        if dependency == "numpy":
            monkeypatch.setattr(np, "__version__", "changed-numpy")
        elif dependency == "torch":
            torch.__version__ = "changed-torch"
        elif dependency == "kornia":
            sys.modules["kornia"].__version__ = "changed-kornia"
        else:
            previous_version = matchanything.version
            monkeypatch.setattr(matchanything, "version", lambda name:
                                f"changed-{name}" if name == dependency else previous_version(name))
        changed = OfficialMatchAnythingELoFTR(source, checkpoint, device="cpu")
        assert changed.revision != first.revision
        assert changed.configuration["dependencies"][dependency] != (
            first.configuration["dependencies"][dependency]
        )
        unaffected = set(first.configuration["dependencies"]) - {dependency}
        for name in unaffected:
            assert changed.configuration["dependencies"][name] == (
                first.configuration["dependencies"][name]
            )

    @pytest.mark.parametrize("cuda_available,expected", [(False, "cpu"), (True, "cuda")])
    def test_auto_device_resolves_and_is_recorded(self, tmp_path, monkeypatch,
                                                 cuda_available, expected):
        source, checkpoint, seen, _ = _mock_native_loader(
            tmp_path, monkeypatch, cuda_available=cuda_available
        )
        wrapper = OfficialMatchAnythingELoFTR(source, checkpoint)
        assert seen["device"] == expected
        assert f"unfused-{expected}" in wrapper.revision

    @pytest.mark.parametrize("device,pattern", [
        ("unknown", "device must be"),
        ("cuda", "CUDA was requested"),
        ("mps", "MPS was requested"),
    ])
    def test_unsupported_device_fails_before_weights_load(self, tmp_path, monkeypatch,
                                                          device, pattern):
        source, checkpoint, seen, _ = _mock_native_loader(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match=pattern):
            OfficialMatchAnythingELoFTR(source, checkpoint, device=device)
        assert "load_path" not in seen
        assert "config" not in seen

    def test_safe_load_errors_are_not_retried_with_pickle_enabled(self, tmp_path, monkeypatch):
        source, checkpoint, _, torch = _mock_native_loader(tmp_path, monkeypatch)
        calls = []

        def reject_checkpoint(path, **kwargs):
            calls.append(kwargs)
            raise RuntimeError("safe checkpoint rejection")

        torch.load = reject_checkpoint
        with pytest.raises(RuntimeError, match="safe checkpoint rejection"):
            OfficialMatchAnythingELoFTR(source, checkpoint, device="cpu")
        assert calls == [{"map_location": "cpu", "weights_only": True}]

    def test_strict_state_dict_errors_propagate(self, tmp_path, monkeypatch):
        source, checkpoint, seen, _ = _mock_native_loader(tmp_path, monkeypatch)
        model_type, config = matchanything._load_official_symbols(source)

        def incompatible_state(self, state_dict, *, strict):
            assert strict is True
            raise RuntimeError("incompatible author weights")

        monkeypatch.setattr(model_type, "load_state_dict", incompatible_state)
        with pytest.raises(RuntimeError, match="incompatible author weights"):
            OfficialMatchAnythingELoFTR(source, checkpoint, device="cpu")
        assert "eval" not in seen

    def test_configuration_digest_changes_revision(self, tmp_path, monkeypatch):
        source, checkpoint, _, _ = _mock_native_loader(tmp_path, monkeypatch,
                                                       config={"coarse": {"test": 1}})
        first = OfficialMatchAnythingELoFTR(source, checkpoint, device="cpu")
        model_type, _ = matchanything._load_official_symbols(source)
        monkeypatch.setattr(matchanything, "_load_official_symbols",
                            lambda path: (model_type, {"coarse": {"test": 2}}))
        second = OfficialMatchAnythingELoFTR(source, checkpoint, device="cpu")
        assert first.revision != second.revision

    @pytest.mark.parametrize("existing_aliases", [False, True])
    def test_config_merge_failure_restores_top_level_src_aliases(self, tmp_path, monkeypatch,
                                                               existing_aliases):
        source = tmp_path / "native-source"
        namespace = "_pyrafault_matchanything_" + hashlib.sha256(
            str(source).encode()
        ).hexdigest()[:12]
        package = ModuleType(namespace)
        config_package = ModuleType(f"{namespace}.config")
        defaults = ModuleType(f"{namespace}.config.default")
        aliases = ("src", "src.config", "src.config.default")
        saved = {}
        for name in aliases:
            if existing_aliases:
                saved[name] = ModuleType(name)
                monkeypatch.setitem(sys.modules, name, saved[name])
            else:
                monkeypatch.delitem(sys.modules, name, raising=False)

        class BrokenConfig:
            def merge_from_file(self, path):
                assert sys.modules["src"] is package
                assert sys.modules["src.config"] is config_package
                assert sys.modules["src.config.default"] is defaults
                raise RuntimeError("config merge failed")

        defaults.get_cfg_defaults = BrokenConfig
        monkeypatch.setitem(sys.modules, namespace, package)
        monkeypatch.setitem(sys.modules, f"{namespace}.config", config_package)
        monkeypatch.setitem(sys.modules, f"{namespace}.config.default", defaults)
        with pytest.raises(RuntimeError, match="config merge failed"):
            matchanything._load_official_symbols(source)
        for name in aliases:
            if existing_aliases:
                assert sys.modules[name] is saved[name]
            else:
                assert name not in sys.modules


@pytest.fixture(scope="class")
def _single_torch_thread_for_tensor_contract():
    """Keep tensor-only tests independent of other native CPU thread pools.

    The full suite also exercises scikit-learn/OpenCV. On macOS their prior
    OpenMP use can stall a subsequently initialized multi-threaded torch pool.
    These tests validate conversion, not parallel performance; restore the user's
    original setting when this one class finishes.
    """
    torch = pytest.importorskip("torch")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.usefixtures("_single_torch_thread_for_tensor_contract")
class TestOfficialMatchAnythingELoFTRInference:
    def test_image0_query_image1_context_keep_independent_shapes_and_normalisation(self):
        torch = pytest.importorskip("torch")
        seen = {}

        class Model:
            def __call__(self, batch):
                assert torch.is_inference_mode_enabled()
                seen.update(batch)
                batch["mkpts0_f"] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
                batch["mkpts1_f"] = torch.tensor([[11.0, 12.0], [13.0, 14.0]])
                batch["mconf"] = torch.tensor([.25, .75])

        wrapper = object.__new__(OfficialMatchAnythingELoFTR)
        wrapper._torch = torch
        wrapper._device = "cpu"
        wrapper._model = Model()
        query = np.arange(32 * 64, dtype=np.uint8).reshape(32, 64)
        context = np.full((96, 128), 32, dtype=np.uint8)
        matches = wrapper.match(query, context)
        assert set(seen) == {"image0", "image1"}
        assert seen["image0"].shape == (1, 1, 32, 64)
        assert seen["image1"].shape == (1, 1, 96, 128)
        assert seen["image0"].dtype == seen["image1"].dtype == torch.float32
        assert seen["image0"].min().item() == 0.0
        assert seen["image0"].max().item() == 1.0
        assert seen["image1"].mean().item() == pytest.approx(32 / 255)
        np.testing.assert_allclose(seen["image0"][0, 0].numpy(), query / 255)
        assert isinstance(matches, Matches)
        assert isinstance(matches.query_points, np.ndarray)
        assert isinstance(matches.context_points, np.ndarray)
        assert isinstance(matches.confidence, np.ndarray)
        np.testing.assert_array_equal(matches.query_points, [[1, 2], [3, 4]])
        np.testing.assert_array_equal(matches.context_points, [[11, 12], [13, 14]])
        np.testing.assert_array_equal(matches.confidence, [.25, .75])

    @pytest.mark.parametrize("shape", [(32, 32), (32, 2048), (2048, 32)])
    def test_input_size_boundaries_are_accepted_without_resizing(self, shape):
        torch = pytest.importorskip("torch")
        wrapper = object.__new__(OfficialMatchAnythingELoFTR)
        wrapper._torch = torch
        wrapper._device = "cpu"

        def model(batch):
            assert tuple(batch["image0"].shape[2:]) == shape
            assert tuple(batch["image1"].shape[2:]) == (32, 32)
            batch["mkpts0_f"] = torch.empty((0, 2))
            batch["mkpts1_f"] = torch.empty((0, 2))
            batch["mconf"] = torch.empty(0)

        wrapper._model = model
        matches = wrapper.match(np.zeros(shape, dtype=np.uint8),
                                np.zeros((32, 32), dtype=np.uint8))
        assert matches.query_points.shape == matches.context_points.shape == (0, 2)
        assert matches.confidence.shape == (0,)

    def test_noncontiguous_input_is_copied_to_contiguous_tensor(self):
        torch = pytest.importorskip("torch")
        wrapper = object.__new__(OfficialMatchAnythingELoFTR)
        wrapper._torch = torch
        wrapper._device = "cpu"
        query = np.arange(64 * 128, dtype=np.uint8).reshape(64, 128)[::2, ::2]
        assert not query.flags.c_contiguous

        def model(batch):
            assert batch["image0"].is_contiguous()
            np.testing.assert_allclose(batch["image0"][0, 0].numpy(), query / 255)
            batch["mkpts0_f"] = torch.empty((0, 2))
            batch["mkpts1_f"] = torch.empty((0, 2))
            batch["mconf"] = torch.empty(0)

        wrapper._model = model
        matches = wrapper.match(query, np.zeros((32, 32), dtype=np.uint8))
        assert matches.confidence.size == 0

    @pytest.mark.parametrize("shape,dtype", [
        ((32, 32, 3), np.uint8), ((32, 32), np.float32),
        ((0, 32), np.uint8), ((31, 32), np.uint8),
        ((33, 32), np.uint8), ((32, 2080), np.uint8),
    ])
    @pytest.mark.parametrize("invalid_side", ["query", "context"])
    def test_invalid_input_array_raises_before_model(self, shape, dtype, invalid_side):
        torch = pytest.importorskip("torch")
        wrapper = object.__new__(OfficialMatchAnythingELoFTR)
        wrapper._torch = torch
        wrapper._device = "cpu"

        def unexpected_model(batch):
            raise AssertionError("invalid images must not reach the model")

        wrapper._model = unexpected_model
        valid = np.zeros((32, 32), dtype=np.uint8)
        invalid = np.zeros(shape, dtype=dtype)
        with pytest.raises(ValueError, match="2-D uint8 grayscale arrays"):
            wrapper.match(invalid if invalid_side == "query" else valid,
                          invalid if invalid_side == "context" else valid)
