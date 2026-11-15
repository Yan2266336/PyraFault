"""The external-baseline plugins at the shared stage-13 boundary.

These tests deliberately replace only the lazy backend factory.  They exercise the
real plugin loader, calibration/prediction runner, scorer, claim-independence audit,
and degradation-aware cache key without loading or pretending to validate published
weights.  The method-specific adapter tests inject controlled matchers/extractors and
remain the tests of each algorithm's image and geometry handling.
"""
from __future__ import annotations

import importlib.util
import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from pyrafault.geom import BBoxL0
from pyrafault.verify import plugins
from pyrafault.verify.base import Localisation, Verifier
from pyrafault.verify.metrics import score_predictions
from pyrafault.verify.runner import (
    LOCALISATION_COLUMNS,
    PREDICTION_COLUMNS,
    fit_thresholds,
    run_verifier,
)
from pyrafault.verify.thresholds import Thresholds


ROOT = Path(__file__).resolve().parents[1]
TRUE_BOX = BBoxL0(132, 232, 32, 32)
WRONG_BOX = BBoxL0(500, 600, 32, 32)

PLUGIN_CASES = (
    (plugins.EfficientLoFTRPlugin,
     ("EFFICIENT_LOFTR_SOURCE", "EFFICIENT_LOFTR_CHECKPOINT")),
    (plugins.SUEPlugin, ("PYRAFAULT_DINOV2_SNAPSHOT",)),
    (plugins.StatisticalUncertaintyPlugin, ("PYRAFAULT_DINOV2_SNAPSHOT",)),
    (plugins.MatchAnythingELoFTRPlugin,
     ("MATCHANYTHING_ELOFTR_SOURCE", "MATCHANYTHING_ELOFTR_CHECKPOINT")),
    (plugins.RoMaPlugin,
     ("ROMA_SOURCE", "ROMA_CHECKPOINT", "ROMA_DINOV2_WEIGHTS")),
    (plugins.LightGluePlugin,
     ("LIGHTGLUE_SOURCE", "SUPERPOINT_CHECKPOINT", "LIGHTGLUE_CHECKPOINT")),
    (plugins.MinimaLightGluePlugin,
     ("LIGHTGLUE_SOURCE", "SUPERPOINT_CHECKPOINT", "MINIMA_LIGHTGLUE_CHECKPOINT")),
)


def _case_id(case):
    return case[0].__name__


@pytest.fixture(scope="module")
def stage13():
    spec = importlib.util.spec_from_file_location(
        "stage13_baseline_plugin_test", ROOT / "scripts" / "13_run_verifier.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def baseline_environment(tmp_path, monkeypatch):
    directories = {
        "EFFICIENT_LOFTR_SOURCE",
        "MATCHANYTHING_ELOFTR_SOURCE",
        "ROMA_SOURCE",
        "LIGHTGLUE_SOURCE",
        "PYRAFAULT_DINOV2_SNAPSHOT",
    }
    values = {}
    for variable in sorted({v for _, variables in PLUGIN_CASES for v in variables}):
        if variable == "PYRAFAULT_DINOV2_SNAPSHOT":
            path = tmp_path / "snapshots" / plugins.DINO_REVISION
        else:
            path = tmp_path / variable.lower()
        if variable in directories:
            path.mkdir(parents=True)
        else:
            path.write_bytes(b"test placeholder; never loaded")
        monkeypatch.setenv(variable, str(path))
        values[variable] = path
    monkeypatch.setenv("PYRAFAULT_BASELINE_DEVICE", "cpu")
    return values


class _SyntheticBackend(Verifier):
    revision = "synthetic-backend-v1"

    def __init__(self, name, localisation=None):
        self.name = name
        self.localisation = localisation or Localisation(TRUE_BOX, 0.9)
        self.calls = 0

    def localise(self, inputs):
        self.calls += 1
        # Deliberately touch only claim-independent inputs.  The adapter-specific
        # tests make the stronger assertion about the exact allowed key set.
        assert inputs["context"].size and inputs["query"].size
        return self.localisation


class _SyntheticDataset:
    def __init__(self, *, split="calibration", digests=("none", "none"),
                 claims=(TRUE_BOX, WRONG_BOX)):
        context = Image.new("RGB", (48, 48), (20, 30, 40))
        query = Image.new("RGB", (16, 16), (80, 90, 100))
        self._items = {}
        for index, (digest, claim) in enumerate(zip(digests, claims)):
            item_id = f"item-{index}"
            self._items[item_id] = {
                "inputs": {
                    "context": context,
                    "query": query,
                    "context_bbox": BBoxL0(100, 200, 96, 96),
                    "claimed_bbox": claim,
                    "context_mpp": 1.0,
                    "query_mpp": 1.0,
                    "slide_id": "synthetic-slide",
                    "item_id": item_id,
                    "split": split,
                    "extraction_version": "v1",
                },
                "target": {
                    "claim_correct": claim == TRUE_BOX,
                    "true_bbox": TRUE_BOX,
                    "true_slide_id": "synthetic-slide",
                    "degradation_digest": digest,
                },
            }

    @property
    def item_ids(self):
        return list(self._items)

    def __getitem__(self, item_id):
        return self._items[item_id]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


def _pairs(dataset):
    rows = []
    for item_id in dataset.item_ids:
        item = dataset[item_id]
        claim = item["inputs"]["claimed_bbox"]
        target = item["target"]
        rows.append({
            "item_id": item_id,
            "claim_correct": target["claim_correct"],
            "fault_type": "clean" if target["claim_correct"] else "adjacent_drift",
            "true_within_context": True,
            "true_bbox_l0_x": TRUE_BOX.x,
            "true_bbox_l0_y": TRUE_BOX.y,
            "true_bbox_l0_w": TRUE_BOX.w,
            "true_bbox_l0_h": TRUE_BOX.h,
            "claimed_bbox_l0_x": claim.x,
            "claimed_bbox_l0_y": claim.y,
            "claimed_bbox_l0_w": claim.w,
            "claimed_bbox_l0_h": claim.h,
            "degradation_band": "none",
            "context_mpp": 1.0,
            "scale_ratio": 1.0,
        })
    return pd.DataFrame(rows)


@pytest.mark.parametrize("plugin_class,expected_variables", PLUGIN_CASES,
                         ids=[_case_id(case) for case in PLUGIN_CASES])
def test_module_class_plugins_are_lazy_and_use_the_shared_harness(
        stage13, monkeypatch, baseline_environment, plugin_class, expected_variables):
    assert plugin_class.path_variables == expected_variables
    built = []

    def build(instance):
        backend = _SyntheticBackend(instance.name)
        built.append(backend)
        return backend

    monkeypatch.setattr(plugin_class, "_build_backend", build)
    verifier = stage13.load_plugin(
        f"pyrafault.verify.plugins:{plugin_class.__name__}"
    )
    assert isinstance(verifier, Verifier)
    assert built == [], "module:Class construction must not load a model"

    calibration = _SyntheticDataset()
    thresholds = fit_thresholds(verifier, calibration, "calibration")
    predictions = run_verifier(
        verifier, calibration, thresholds, audit_independence=True
    )
    metrics = score_predictions(predictions, _pairs(calibration))

    assert len(built) == 1, "first localisation builds once and later calls reuse it"
    assert list(predictions.columns) == list(PREDICTION_COLUMNS)
    assert predictions.set_index("item_id")["decision"].to_dict() == {
        "item-0": "verified",
        "item-1": "mismatched",
    }
    assert predictions["estimated_bbox_l0_x"].nunique() == 1
    assert predictions["estimated_bbox_l0_x"].iloc[0] == TRUE_BOX.x
    assert predictions["verifier_name"].eq(verifier.name).all()
    assert predictions["verifier_revision"].eq(_SyntheticBackend.revision).all()
    assert metrics["n_items"] == 2
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["verification_score"]["auroc"] == pytest.approx(1.0)


def test_upstream_ncc_is_already_a_direct_zero_argument_plugin(stage13):
    from pyrafault.verify.ncc_baseline import NccBaseline

    verifier = stage13.load_plugin("pyrafault.verify.ncc_baseline:NccBaseline")
    assert isinstance(verifier, NccBaseline)


@pytest.mark.parametrize("plugin_class,expected_variables", PLUGIN_CASES,
                         ids=[_case_id(case) for case in PLUGIN_CASES])
def test_environment_is_frozen_at_zero_argument_construction(
        monkeypatch, baseline_environment, plugin_class, expected_variables):
    verifier = plugin_class()
    captured = {key: str(baseline_environment[key]) for key in expected_variables}
    for variable in expected_variables:
        monkeypatch.setenv(variable, str(baseline_environment[variable]) + "-changed")
    monkeypatch.setenv("PYRAFAULT_BASELINE_DEVICE", "cuda")

    assert verifier._paths == captured
    assert verifier.device == "cpu"


@pytest.mark.parametrize(
    "plugin_class,missing",
    [(plugin_class, variable)
     for plugin_class, variables in PLUGIN_CASES for variable in variables],
    ids=[f"{plugin_class.__name__}-{variable}"
         for plugin_class, variables in PLUGIN_CASES for variable in variables],
)
def test_every_missing_backend_path_has_an_actionable_lazy_error(
        monkeypatch, baseline_environment, plugin_class, missing):
    monkeypatch.delenv(missing)
    verifier = plugin_class()
    with pytest.raises(ValueError, match=rf"{missing}.*BASELINE_HANDOVER\.zh-CN\.md"):
        verifier.localise({})


def test_invalid_shared_device_is_rejected_before_any_backend_load(monkeypatch):
    monkeypatch.setenv("PYRAFAULT_BASELINE_DEVICE", "mps")
    with pytest.raises(ValueError, match="auto, cpu, or cuda"):
        plugins.EfficientLoFTRPlugin()


MATCHER_ROUTE_CASES = (
    (
        plugins.EfficientLoFTRPlugin,
        "pyrafault.verify.efficient_loftr",
        "OfficialEfficientLoFTR",
        "EfficientLoFTRBaseline",
        ("EFFICIENT_LOFTR_SOURCE", "EFFICIENT_LOFTR_CHECKPOINT"),
    ),
    (
        plugins.MatchAnythingELoFTRPlugin,
        "pyrafault.verify.matchanything_eloftr",
        "OfficialMatchAnythingELoFTR",
        "MatchAnythingELoFTRBaseline",
        ("MATCHANYTHING_ELOFTR_SOURCE", "MATCHANYTHING_ELOFTR_CHECKPOINT"),
    ),
    (
        plugins.RoMaPlugin,
        "pyrafault.verify.roma",
        "OfficialRoMa",
        "RoMaBaseline",
        ("ROMA_SOURCE", "ROMA_CHECKPOINT", "ROMA_DINOV2_WEIGHTS"),
    ),
    (
        plugins.LightGluePlugin,
        "pyrafault.verify.lightglue",
        "OfficialSuperPointLightGlue",
        "LightGlueBaseline",
        ("LIGHTGLUE_SOURCE", "SUPERPOINT_CHECKPOINT", "LIGHTGLUE_CHECKPOINT"),
    ),
    (
        plugins.MinimaLightGluePlugin,
        "pyrafault.verify.minima_lightglue",
        "OfficialMinimaLightGlue",
        "MinimaLightGlueBaseline",
        ("LIGHTGLUE_SOURCE", "SUPERPOINT_CHECKPOINT", "MINIMA_LIGHTGLUE_CHECKPOINT"),
    ),
)


class _RoutedMatcher:
    revision = "routed-matcher-v1"


@pytest.mark.parametrize(
    "plugin_class,module_name,official_name,adapter_name,path_variables",
    MATCHER_ROUTE_CASES,
    ids=[case[0].__name__ for case in MATCHER_ROUTE_CASES],
)
def test_matcher_plugins_route_frozen_paths_and_device_into_the_real_adapter(
        monkeypatch, baseline_environment, plugin_class, module_name, official_name,
        adapter_name, path_variables):
    """Only the official neural wrapper is replaced; the actual adapter is built."""
    module = importlib.import_module(module_name)
    adapter_class = getattr(module, adapter_name)
    matcher = _RoutedMatcher()
    seen = []

    def official(*args, **kwargs):
        seen.append((args, kwargs))
        return matcher

    monkeypatch.setattr(module, official_name, official)
    backend = plugin_class()._build_backend()

    assert isinstance(backend, adapter_class)
    assert backend._matcher is matcher
    assert seen == [(
        tuple(baseline_environment[key].resolve() for key in path_variables),
        {"device": "cpu"},
    )]


class _RoutedExtractor:
    name = "dinov2"
    revision = plugins.DINO_REVISION
    preprocessing = "controlled-test-extractor"
    dim = 768
    input_size = 224
    device = "cpu"
    batch_size = 2
    hf_repo = plugins.DINO_REPO
    timm_model = None

    def embed(self, images):
        raise AssertionError("backend routing must not perform inference")


@pytest.mark.parametrize(
    "plugin_class,module_name,adapter_name",
    (
        (plugins.SUEPlugin, "pyrafault.verify.sue", "SUEBaseline"),
        (plugins.StatisticalUncertaintyPlugin,
         "pyrafault.verify.statistical_uncertainty",
         "StatisticalUncertaintyBaseline"),
    ),
    ids=("SUEPlugin", "StatisticalUncertaintyPlugin"),
)
def test_retrieval_plugins_route_snapshot_and_device_into_the_real_adapter(
        monkeypatch, baseline_environment, plugin_class, module_name, adapter_name):
    module = importlib.import_module(module_name)
    adapter_class = getattr(module, adapter_name)
    extractor = _RoutedExtractor()
    seen = []

    def construct(snapshot, device):
        seen.append((snapshot, device))
        return extractor

    monkeypatch.setattr(plugins, "_PinnedDinoV2Extractor", construct)
    backend = plugin_class()._build_backend()

    assert isinstance(backend, adapter_class)
    assert backend._extractor is extractor
    assert seen == [(
        baseline_environment["PYRAFAULT_DINOV2_SNAPSHOT"].resolve(), "cpu"
    )]


@pytest.fixture
def _single_torch_thread():
    """Keep the real-torch loader doubles isolated from native OpenMP test pools."""
    torch = pytest.importorskip("torch")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield torch
    finally:
        torch.set_num_threads(previous)


def _install_fake_transformers(monkeypatch, torch, *, processor_size=(224, 224),
                               model_type="dinov2", hidden_size=768):
    """Install a deterministic loader double around a real torch.nn.Module."""
    seen = {}

    class Batch(dict):
        def to(self, device):
            seen.setdefault("batch_devices", []).append(device)
            for key, value in list(self.items()):
                self[key] = value.to(device)
            return self

    class Processor:
        def __call__(self, *, images, return_tensors):
            assert return_tensors == "pt"
            seen.setdefault("batch_sizes", []).append(len(images))
            height, width = processor_size
            return Batch(pixel_values=torch.ones(len(images), 3, height, width))

        def __str__(self):
            return "fake-processor-224"

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vector = torch.nn.Parameter(
                torch.arange(1, hidden_size + 1, dtype=torch.float32)
            )
            self.config = SimpleNamespace(
                model_type=model_type, hidden_size=hidden_size
            )

        def forward(self, pixel_values):
            batch = pixel_values.shape[0]
            tokens = self.vector.reshape(1, 1, -1).expand(batch, 2, -1)
            return SimpleNamespace(last_hidden_state=tokens)

    processor = Processor()
    model = Model()

    def load_processor(path, **kwargs):
        seen["processor_load"] = (path, kwargs)
        return processor

    def load_model(path, **kwargs):
        seen["model_load"] = (path, kwargs)
        return model

    transformers = ModuleType("transformers")
    transformers.AutoImageProcessor = SimpleNamespace(from_pretrained=load_processor)
    transformers.AutoModel = SimpleNamespace(from_pretrained=load_model)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return seen, model


def test_pinned_dinov2_uses_only_the_named_local_snapshot_and_freezes_real_torch_module(
        tmp_path, monkeypatch, _single_torch_thread):
    """The model is synthetic; this proves loader/freeze/embed plumbing, not weights."""
    torch = _single_torch_thread
    snapshot = tmp_path / "snapshots" / plugins.DINO_REVISION
    snapshot.mkdir(parents=True)
    seen, model = _install_fake_transformers(monkeypatch, torch)

    extractor = plugins._PinnedDinoV2Extractor(snapshot, "cpu")
    output = extractor.embed([
        Image.new("RGB", (31, 27), "red"),
        Image.new("RGB", (19, 23), "blue"),
    ])

    expected_load = (str(snapshot), {"local_files_only": True})
    assert seen["processor_load"] == expected_load
    assert seen["model_load"] == expected_load
    assert seen["batch_sizes"] == [1, 2]  # 224x224 probe, then the embed call.
    assert seen["batch_devices"] == ["cpu"]
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert output.shape == (2, 768)
    assert output.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(output, axis=1), 1.0, atol=1e-6)


def test_pinned_dinov2_rejects_a_different_snapshot_directory_name(
        tmp_path, monkeypatch, _single_torch_thread):
    torch = _single_torch_thread
    _install_fake_transformers(monkeypatch, torch)
    wrong = tmp_path / "snapshots" / "moving-main"
    wrong.mkdir(parents=True)

    with pytest.raises(ValueError, match="must be the pinned snapshots"):
        plugins._PinnedDinoV2Extractor(wrong, "cpu")


def test_pinned_dinov2_rejects_the_wrong_model_family(
        tmp_path, monkeypatch, _single_torch_thread):
    torch = _single_torch_thread
    snapshot = tmp_path / plugins.DINO_REVISION
    snapshot.mkdir()
    _install_fake_transformers(monkeypatch, torch, model_type="vit")

    with pytest.raises(ValueError, match="facebook/dinov2-base"):
        plugins._PinnedDinoV2Extractor(snapshot, "cpu")


def test_pinned_dinov2_requires_the_frozen_224_processor(
        tmp_path, monkeypatch, _single_torch_thread):
    torch = _single_torch_thread
    snapshot = tmp_path / plugins.DINO_REVISION
    snapshot.mkdir()
    _install_fake_transformers(monkeypatch, torch, processor_size=(256, 256))

    with pytest.raises(ValueError, match="224x224"):
        plugins._PinnedDinoV2Extractor(snapshot, "cpu")


def _thresholds():
    return Thresholds(
        tau=0.5,
        delta=0.0,
        fitted_on="calibration",
        n_items=2,
        residuals={"n": 1, "p50": 0.0, "p75": 0.0, "p90": 0.0,
                   "p95": 0.0, "p99": 0.0},
    )


def test_rotation_degradation_digest_keeps_localisation_cache_entries_separate(
        monkeypatch):
    built = []

    def build(instance):
        backend = _SyntheticBackend(instance.name)
        built.append(backend)
        return backend

    monkeypatch.setattr(plugins.EfficientLoFTRPlugin, "_build_backend", build)
    verifier = plugins.EfficientLoFTRPlugin()
    dataset = _SyntheticDataset(
        split="test", digests=("rotation-5deg", "rotation-15deg"),
        claims=(TRUE_BOX, TRUE_BOX),
    )
    run_verifier(verifier, dataset, _thresholds())

    assert len(built) == 1
    assert built[0].calls == 2, "different rotations must not share a cached answer"


def test_none_is_a_legal_plugin_localisation_and_shared_runner_output(monkeypatch):
    def build(instance):
        return _SyntheticBackend(instance.name, Localisation(None, 0.9))

    monkeypatch.setattr(plugins.SUEPlugin, "_build_backend", build)
    verifier = plugins.SUEPlugin()
    dataset = _SyntheticDataset(
        split="test", digests=("none",), claims=(TRUE_BOX,)
    )
    predictions = run_verifier(verifier, dataset, _thresholds())

    assert predictions.loc[0, "decision"] == "mismatched"
    assert predictions.loc[0, [
        "estimated_bbox_l0_x", "estimated_bbox_l0_y",
        "estimated_bbox_l0_w", "estimated_bbox_l0_h",
    ]].isna().all()
    metrics = score_predictions(predictions, _pairs(dataset))
    assert metrics["localisation"]["n"] == 0
    assert metrics["localisation"]["median_iou"] is None


def test_from_localisations_uses_recorded_identity_without_building_backend(
        tmp_path, monkeypatch, stage13):
    def forbidden_build(instance):
        pytest.fail("--from-localisations must not build a model backend")

    monkeypatch.setattr(
        plugins.EfficientLoFTRPlugin, "_build_backend", forbidden_build
    )
    for variable in plugins.EfficientLoFTRPlugin.path_variables:
        monkeypatch.delenv(variable, raising=False)

    dataset = _SyntheticDataset(split="test")
    method_dir = tmp_path / "06_verify" / plugins.EfficientLoFTRPlugin.name
    localisation_dir = method_dir / "localisations_test"
    localisation_dir.mkdir(parents=True)
    rows = []
    for item_id in dataset.item_ids:
        rows.append({
            "item_id": item_id,
            "estimated_bbox_l0_x": TRUE_BOX.x,
            "estimated_bbox_l0_y": TRUE_BOX.y,
            "estimated_bbox_l0_w": TRUE_BOX.w,
            "estimated_bbox_l0_h": TRUE_BOX.h,
            "reliability": 0.9,
            "verifier_name": plugins.EfficientLoFTRPlugin.name,
            "verifier_revision": "recorded-revision-v1",
            "runtime_ms": 12.5,
        })
    pd.DataFrame(rows, columns=list(LOCALISATION_COLUMNS)).to_parquet(
        localisation_dir / "00000of00001.parquet", index=False
    )
    _thresholds().save(method_dir / "thresholds.json")

    monkeypatch.setattr(
        stage13, "load_config",
        lambda path: SimpleNamespace(source=SimpleNamespace(root_dir=tmp_path / "unused")),
    )
    monkeypatch.setattr(stage13, "PyraFaultDataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(sys, "argv", [
        "13_run_verifier.py",
        "--data-root", str(tmp_path),
        "--split", "test",
        "--verifier", "pyrafault.verify.plugins:EfficientLoFTRPlugin",
        "--from-localisations",
    ])
    stage13.main()

    predictions = pd.read_parquet(method_dir / "predictions_test.parquet")
    assert predictions["verifier_revision"].eq("recorded-revision-v1").all()
    assert predictions["runtime_ms"].eq(12.5).all()
    assert predictions.set_index("item_id")["decision"].to_dict() == {
        "item-0": "verified",
        "item-1": "mismatched",
    }
