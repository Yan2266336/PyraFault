"""SUE's mathematics and verifier contract without weights, network, or WSI.

The descriptor fakes make geometry and uncertainty independently checkable. These
tests are not an assertion that a pretrained retriever works on pathology images.
"""
from collections.abc import Mapping
from copy import deepcopy

import numpy as np
import pytest
from PIL import Image

from pyrafault.geom import BBoxL0
from pyrafault.verify.base import Localisation, Verifier
from pyrafault.verify.sue import SUEBaseline, spatial_uncertainty


class _Extractor:
    name = "test-extractor"
    revision = "fixed-v1"
    preprocessing = "test-only-no-model"
    input_size = 16
    dim = 2
    device = "cpu"

    def __init__(self, output=None, *, winner=0):
        self.output = output
        self.winner = winner
        self.calls = []

    def embed(self, images):
        self.calls.append(list(images))
        if callable(self.output):
            return self.output(images)
        if self.output is not None:
            return np.asarray(self.output)
        result = np.tile([0.0, 1.0], (len(images), 1))
        result[0] = [1.0, 0.0]
        result[1 + self.winner] = [1.0, 0.0]
        return result


def _inputs(context_size=(10, 8), query_size=(4, 4), *, bbox=None,
            context_mpp=1.0, query_mpp=1.0):
    # The coordinate image exposes window origin and crop ordering directly.
    width, height = context_size
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[:, :, 0] = np.arange(width)[None, :]
    pixels[:, :, 1] = np.arange(height)[:, None]
    return {
        "context": Image.fromarray(pixels),
        "query": Image.new("RGB", query_size, (110, 40, 70)),
        "context_bbox": bbox or BBoxL0(0, 0, width, height),
        "context_mpp": context_mpp,
        "query_mpp": query_mpp,
        "claimed_bbox": BBoxL0(98765, 98765, 1, 1),
        "item_id": "unrelated-id", "slide_id": "unrelated-slide",
        "split": "test", "extraction_version": "v1",
    }


class _ClaimBlindInputs(Mapping):
    allowed = {"context", "query", "context_bbox", "context_mpp", "query_mpp"}

    def __init__(self, values):
        self.values = values
        self.accessed = set()

    def __getitem__(self, key):
        assert key in self.allowed, f"SUE read forbidden input {key!r}"
        self.accessed.add(key)
        return self.values[key]

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)


class TestSpatialUncertainty:
    def test_uniform_two_point_distribution_uses_population_variance(self):
        uncertainty, weights = spatial_uncertainty(
            [[0.0, 0.0], [2.0, 4.0]], [0.5, 0.5]
        )
        assert weights == pytest.approx([0.5, 0.5])
        # The mean is (1, 2): trace of population covariance = 1 + 4.
        assert uncertainty == pytest.approx(5.0)

    def test_distance_weighting_uses_distance_not_squared_distance(self):
        uncertainty, weights = spatial_uncertainty(
            [[0.0, 0.0], [3.0, 0.0]], [0.2, 0.6], distance_decay=2.0
        )
        expected = np.exp(-2.0 * np.array([0.2, 0.6]))
        expected /= expected.sum()
        assert weights == pytest.approx(expected)
        assert uncertainty == pytest.approx(9.0 * expected[0] * expected[1])
        squared = np.exp(-2.0 * np.array([0.2, 0.6]) ** 2)
        squared /= squared.sum()
        assert not np.allclose(weights, squared)

    def test_large_common_distance_does_not_underflow_all_weights(self):
        uncertainty, weights = spatial_uncertainty(
            [[0.0, 0.0], [4.0, 0.0]], [1e6, 1e6], distance_decay=350.0
        )
        assert weights == pytest.approx([0.5, 0.5])
        assert uncertainty == pytest.approx(4.0)

    def test_extreme_gap_can_safely_underflow_inferior_weights(self):
        uncertainty, weights = spatial_uncertainty(
            [[0.0, 0.0], [4.0, 3.0]], [0.0, 1e6], distance_decay=350.0
        )
        assert np.isfinite(uncertainty)
        assert np.all(np.isfinite(weights))
        assert weights == pytest.approx([1.0, 0.0])
        assert uncertainty == pytest.approx(0.0)

    def test_spatial_translation_and_common_distance_offset_do_not_change_score(self):
        positions = np.array([[0.0, 1.0], [2.0, 4.0], [-3.0, 0.0]])
        distances = np.array([0.1, 0.2, 0.4])
        u0, w0 = spatial_uncertainty(positions, distances, distance_decay=3.0)
        u1, w1 = spatial_uncertainty(
            positions + [100.0, -200.0], distances + 17.0, distance_decay=3.0
        )
        assert u1 == pytest.approx(u0)
        assert w1 == pytest.approx(w0)

    def test_zero_decay_makes_distances_irrelevant(self):
        uncertainty, weights = spatial_uncertainty(
            [[0.0, 0.0], [2.0, 0.0]], [0.0, 1000.0], distance_decay=0.0
        )
        assert weights == pytest.approx([0.5, 0.5])
        assert uncertainty == pytest.approx(1.0)

    @pytest.mark.parametrize("positions,distances", [
        ([], []), ([[1.0, 2.0, 3.0]], [1.0]),
        ([[1.0, 2.0]], [1.0]),
        ([[1.0, 2.0], [3.0, 4.0]], [1.0]),
        ([[1.0, 2.0]], [[1.0]]),
        ([[float("nan"), 2.0]], [1.0]),
        ([[1.0, 2.0]], [float("inf")]),
        ([[1.0, 2.0]], [-0.1]),
    ])
    def test_invalid_inputs_raise(self, positions, distances):
        with pytest.raises(ValueError):
            spatial_uncertainty(positions, distances)

    @pytest.mark.parametrize("decay", [-1.0, float("nan"), float("inf")])
    def test_invalid_decay_raises(self, decay):
        with pytest.raises(ValueError):
            spatial_uncertainty([[0.0, 0.0], [1.0, 0.0]], [0.1, 0.2], decay)


class TestSUEContract:
    def test_pixel_dependent_retrieval_locates_a_planted_grid_patch(self):
        class PixelExtractor(_Extractor):
            dim = 4

            def embed(self, images):
                # Not a competitive descriptor: a controlled image-dependent fake
                # verifies that the query and actual candidate crops reach retrieval.
                return np.asarray([(*im.getpixel((0, 0)), 1.0) for im in images])

        inputs = _inputs()
        inputs["query"] = inputs["context"].crop((6, 2, 10, 6))
        loc = SUEBaseline(PixelExtractor()).localise(inputs)
        assert loc.estimated_bbox == BBoxL0(6, 2, 4, 4)
        inputs["query"] = inputs["context"].crop((0, 4, 4, 8))
        assert SUEBaseline(PixelExtractor()).localise(inputs).estimated_bbox == BBoxL0(0, 4, 4, 4)

    def test_implements_shared_contract_and_top_one_provides_position(self):
        baseline = SUEBaseline(_Extractor(winner=5), top_k=2)
        assert isinstance(baseline, Verifier)
        assert baseline.name == "sue_baseline"
        loc = baseline.localise(_inputs(bbox=BBoxL0(1000, 2000, 100, 160)))
        assert isinstance(loc, Localisation)
        # Grid: x=[0,2,4,6], y=[0,2,4]. Index 5 is (2,2).
        assert loc.estimated_bbox == BBoxL0(1020, 2040, 40, 80)
        assert 0.0 < loc.reliability <= 1.0

    def test_grid_is_row_major_and_includes_last_in_bounds_origin(self):
        extractor = _Extractor(winner=11)
        loc = SUEBaseline(extractor).localise(
            _inputs(context_size=(11, 9), query_size=(4, 4))
        )
        images = extractor.calls[0]
        origins = [image.getpixel((0, 0))[:2] for image in images[1:]]
        assert origins == [(x, y) for y in [0, 2, 4, 5] for x in [0, 2, 4, 6, 7]]
        assert all(image.size == (4, 4) for image in images)
        assert loc.estimated_bbox == BBoxL0(2, 4, 4, 4)

    def test_ties_are_deterministically_resolved_by_row_major_order(self):
        def identical(images):
            return np.tile([3.0, 4.0], (len(images), 1))
        inputs = _inputs()
        baseline = SUEBaseline(_Extractor(identical), top_k=2)
        first = baseline.localise(inputs)
        second = baseline.localise(inputs)
        assert first == second
        assert first.estimated_bbox == BBoxL0(0, 0, 4, 4)
        # First two candidates are one half-query-width apart.
        assert first.reliability == pytest.approx(1.0 / (1.0 + 0.0625))

    def test_descriptor_norm_is_removed_before_l2_retrieval(self):
        def scaled(images):
            descriptors = np.tile([0.0, 1.0], (len(images), 1))
            descriptors[0] = [10.0, 0.0]
            descriptors[1] = [1.0, 0.0]
            descriptors[2] = [10.0, 1.0]
            return descriptors
        loc = SUEBaseline(_Extractor(scaled)).localise(_inputs())
        # Unnormalised Euclidean distance would wrongly select candidate 1.
        assert loc.estimated_bbox == BBoxL0(0, 0, 4, 4)

    def test_dimensionless_uncertainty_and_inverse_mapping(self):
        def equal(images):
            return np.tile([1.0, 0.0], (len(images), 1))
        baseline = SUEBaseline(_Extractor(equal), top_k=2, distance_decay=0.0)
        loc = baseline.localise(_inputs(context_size=(12, 8), query_size=(4, 4)))
        # top-2 positions (0,0) and (2/4,0): U=0.0625.
        assert loc.reliability == pytest.approx(1.0 / (1.0 + 0.0625))
        loc2 = baseline.localise(_inputs(context_size=(24, 16), query_size=(8, 8)))
        assert loc2.reliability == pytest.approx(loc.reliability)

    def test_top_k_is_clipped_to_available_windows(self):
        extractor = _Extractor(lambda images: np.tile([1.0, 0.0], (len(images), 1)))
        loc = SUEBaseline(extractor, top_k=100).localise(
            _inputs(context_size=(6, 4), query_size=(4, 4))
        )
        assert loc.diagnostics["n_candidates"] == 2
        assert loc.diagnostics["top_k_used"] == 2
        assert loc.reliability == pytest.approx(1.0 / (1.0 + 0.0625))

    def test_top_k_does_not_include_lower_ranked_far_windows(self):
        extractor = _Extractor(lambda images: np.tile([1.0, 0.0], (len(images), 1)))
        loc = SUEBaseline(extractor, top_k=2).localise(_inputs())
        assert loc.diagnostics["candidate_origins_context"] == [[0.0, 0.0], [2.0, 0.0]]
        assert loc.diagnostics["top_k_used"] == 2

    def test_small_stride_rounding_has_minimum_one_pixel(self):
        extractor = _Extractor()
        SUEBaseline(extractor, stride_fraction=0.01).localise(_inputs())
        origins = [im.getpixel((0, 0))[:2] for im in extractor.calls[0][1:]]
        assert origins == [(x, y) for y in range(5) for x in range(7)]

    def test_query_is_downsampled_before_joint_descriptor_batch(self):
        extractor = _Extractor(winner=5)
        loc = SUEBaseline(extractor).localise(_inputs(
            context_size=(10, 8), query_size=(16, 12),
            context_mpp=2.0, query_mpp=0.5,
            bbox=BBoxL0(100, 200, 100, 160),
        ))
        assert len(extractor.calls) == 1
        assert all(image.size == (4, 3) for image in extractor.calls[0])
        # x=[0,2,4,6], y=[0,2,4,5], fifth zero-based origin is (2,2).
        assert loc.estimated_bbox == BBoxL0(120, 240, 40, 60)

    def test_noninteger_footprint_is_preserved_and_boundary_is_physically_safe(self):
        extractor = _Extractor(winner=15)
        loc = SUEBaseline(extractor).localise(_inputs(
            context_size=(16, 12), query_size=(19, 13),
            context_mpp=2.0, query_mpp=0.7,
            bbox=BBoxL0(1000, 2000, 1600, 960),
        ))
        # True footprint 6.65x4.55; integer crops 7x5. x=[0,4,8,9],
        # y=[0,2,4,6,7]. candidate 15=(9,6).
        assert all(image.size == (7, 5) for image in extractor.calls[0])
        assert loc.estimated_bbox == BBoxL0(1900, 2480, 665, 364)
        assert _inputs(context_size=(16, 12), bbox=BBoxL0(1000, 2000, 1600, 960))[
            "context_bbox"
        ].contains(loc.estimated_bbox)

    def test_rounding_down_crop_never_lets_physical_bbox_leave_context(self):
        extractor = _Extractor(winner=7)
        loc = SUEBaseline(extractor).localise(_inputs(
            context_size=(8, 6), query_size=(7, 5),
            context_mpp=2.0, query_mpp=0.9,
            bbox=BBoxL0(0, 0, 800, 600),
        ))
        # Footprint 3.15x2.25 rounds to3x2; max origin is4x3, not5x4.
        origins = [im.getpixel((0, 0))[:2] for im in extractor.calls[0][1:]]
        assert max(x for x, _ in origins) == 4
        assert max(y for _, y in origins) == 3
        assert loc.estimated_bbox.w == 315
        assert loc.estimated_bbox.h == 225
        assert BBoxL0(0, 0, 800, 600).contains(loc.estimated_bbox)

    def test_localise_never_reads_claim_ids_split_or_target(self):
        guarded = _ClaimBlindInputs(_inputs())
        SUEBaseline(_Extractor()).localise(guarded)
        assert guarded.accessed == guarded.allowed

    def test_changed_claim_and_metadata_do_not_change_result(self):
        original = _inputs()
        alternate = dict(original, claimed_bbox=BBoxL0(0, 0, 2, 2),
                         item_id="other", slide_id="other", split="calibration")
        baseline = SUEBaseline(_Extractor())
        assert baseline.localise(original) == baseline.localise(alternate)

    def test_shared_runner_claim_independence_audit(self):
        from pyrafault.verify.runner import run_verifier
        from pyrafault.verify.thresholds import Thresholds

        class Dataset:
            item_ids = ["i0", "i1"]

            def __getitem__(self, item_id):
                inputs = _inputs()
                inputs["item_id"] = item_id
                inputs["claimed_bbox"] = (
                    BBoxL0(0, 0, 4, 4) if item_id == "i0"
                    else BBoxL0(200, 200, 4, 4)
                )
                return {"inputs": inputs,
                        "target": {"true_bbox": BBoxL0(0, 0, 4, 4),
                                   "true_slide_id": inputs["slide_id"]}}

        thresholds = Thresholds(0.0, 0.5, "calibration", 2, {})
        rows = run_verifier(SUEBaseline(_Extractor()), Dataset(), thresholds,
                            audit_independence=True).set_index("item_id")
        assert rows.loc["i0", "decision"] == "verified"
        assert rows.loc["i1", "decision"] == "mismatched"
        assert rows.loc["i0", "estimated_bbox_l0_x"] == rows.loc["i1", "estimated_bbox_l0_x"]


class TestSUEFailures:
    @pytest.mark.parametrize("query_size", [(11, 4), (4, 9), (11, 9)])
    def test_larger_physical_query_abstains_without_embedding(self, query_size):
        extractor = _Extractor()
        loc = SUEBaseline(extractor).localise(_inputs(query_size=query_size))
        assert loc.estimated_bbox is None
        assert loc.reliability == 0.0
        assert extractor.calls == []

    def test_single_candidate_is_not_reported_with_false_perfect_confidence(self):
        loc = SUEBaseline(_Extractor()).localise(
            _inputs(context_size=(4, 4), query_size=(4, 4))
        )
        assert loc.estimated_bbox is None
        assert loc.reliability == 0.0

    def test_zero_query_descriptor_abstains(self):
        def descriptors(images):
            result = np.tile([1.0, 0.0], (len(images), 1))
            result[0] = 0.0
            return result
        loc = SUEBaseline(_Extractor(descriptors)).localise(_inputs())
        assert loc.estimated_bbox is None
        assert loc.reliability == 0.0

    def test_all_zero_window_descriptors_abstain(self):
        def descriptors(images):
            result = np.zeros((len(images), 2))
            result[0] = [1.0, 0.0]
            return result
        loc = SUEBaseline(_Extractor(descriptors)).localise(_inputs())
        assert loc.estimated_bbox is None
        assert loc.reliability == 0.0

    def test_one_nonzero_candidate_after_filtering_still_abstains(self):
        def descriptors(images):
            result = np.zeros((len(images), 2))
            result[:2] = [1.0, 0.0]
            return result
        loc = SUEBaseline(_Extractor(descriptors)).localise(_inputs())
        assert loc.estimated_bbox is None
        assert loc.reliability == 0.0

    def test_zero_windows_are_filtered_without_shifting_candidate_coordinates(self):
        def descriptors(images):
            result = np.zeros((len(images), 2))
            result[0] = [1.0, 0.0]
            result[6] = [1.0, 0.0]  # zero-based candidate5=(2,2)
            result[7] = [0.0, 1.0]
            return result
        loc = SUEBaseline(_Extractor(descriptors)).localise(_inputs())
        assert loc.estimated_bbox == BBoxL0(2, 2, 4, 4)
        assert 0 < loc.reliability <= 1

    @pytest.mark.parametrize("field", ["context_mpp", "query_mpp"])
    @pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), None, "bad"])
    def test_mpp_must_be_finite_and_positive(self, field, value):
        inputs = _inputs()
        inputs[field] = value
        with pytest.raises(ValueError):
            SUEBaseline(_Extractor()).localise(inputs)

    def test_query_upsampling_is_refused(self):
        with pytest.raises(ValueError):
            SUEBaseline(_Extractor()).localise(
                _inputs(context_mpp=0.5, query_mpp=2.0)
            )

    @pytest.mark.parametrize("output", [
        lambda images: np.zeros(len(images)),
        lambda images: np.zeros((len(images) - 1, 2)),
        lambda images: np.zeros((len(images), 0)),
        lambda images: np.full((len(images), 2), np.nan),
        lambda images: np.full((len(images), 2), np.inf),
    ])
    def test_invalid_descriptor_arrays_fail_loudly(self, output):
        with pytest.raises(ValueError):
            SUEBaseline(_Extractor(output)).localise(_inputs())

    def test_overflowing_descriptor_norm_fails_instead_of_abstaining(self):
        extractor = _Extractor(lambda images: np.full((len(images), 2), 1e308))
        with np.errstate(over="ignore"):
            with pytest.raises(ValueError, match="norm"):
                SUEBaseline(extractor).localise(_inputs())

    def test_extractor_error_is_not_hidden_by_a_fallback(self):
        def broken(images):
            raise RuntimeError("model failed")
        with pytest.raises(RuntimeError, match="model failed"):
            SUEBaseline(_Extractor(broken)).localise(_inputs())

    @pytest.mark.parametrize("params", [
        {"top_k": 0}, {"top_k": 1}, {"top_k": -1}, {"top_k": 2.5},
        {"top_k": True},
        {"distance_decay": -1}, {"distance_decay": float("nan")},
        {"distance_decay": float("inf")},
        {"stride_fraction": 0}, {"stride_fraction": -0.1},
        {"stride_fraction": 1.01},
        {"stride_fraction": float("nan")}, {"stride_fraction": float("inf")},
    ])
    def test_invalid_hyperparameters_fail_at_construction(self, params):
        with pytest.raises(ValueError):
            SUEBaseline(_Extractor(), **params)

    @pytest.mark.parametrize("field", ["distance_decay", "stride_fraction"])
    @pytest.mark.parametrize("value", [None, "not-a-number"])
    def test_nonnumeric_hyperparameters_raise_value_error(self, field, value):
        with pytest.raises(ValueError, match="numeric"):
            SUEBaseline(_Extractor(), **{field: value})

    def test_numeric_strings_are_normalized_before_validation_and_revision(self):
        numeric = SUEBaseline(_Extractor(), distance_decay=350.0, stride_fraction=0.5)
        strings = SUEBaseline(_Extractor(), distance_decay="350.0", stride_fraction="0.5")
        assert strings.configuration == numeric.configuration
        assert strings.revision == numeric.revision
        assert strings.localise(_inputs()) == numeric.localise(_inputs())


class TestSUERevision:
    def test_same_frozen_configuration_has_same_revision(self):
        assert SUEBaseline(_Extractor()).revision == SUEBaseline(_Extractor()).revision

    @pytest.mark.parametrize("field,value", [
        ("name", "other-extractor"), ("revision", "weights-v2"),
        ("preprocessing", "different-resize"), ("dim", 3),
        ("input_size", 32), ("device", "cuda"), ("batch_size", 32),
    ])
    def test_extractor_identity_is_part_of_revision(self, field, value):
        extractor = _Extractor()
        changed = deepcopy(extractor)
        setattr(changed, field, value)
        assert SUEBaseline(extractor).revision != SUEBaseline(changed).revision

    @pytest.mark.parametrize("params", [
        {"top_k": 5}, {"distance_decay": 1.0}, {"stride_fraction": 0.25},
    ])
    def test_algorithm_hyperparameters_are_part_of_revision(self, params):
        assert SUEBaseline(_Extractor()).revision != SUEBaseline(_Extractor(), **params).revision

    @pytest.mark.parametrize("name,libraries", [
        ("dinov2", ["torch", "transformers", "torchvision"]),
        ("uni", ["torch", "timm", "torchvision"]),
    ])
    def test_model_runtime_versions_are_recorded_without_importing_models(
        self, monkeypatch, name, libraries
    ):
        import builtins
        import pyrafault.verify.sue as sue_module

        original_import = builtins.__import__

        def forbid_model_imports(module_name, *args, **kwargs):
            assert module_name.split(".")[0] not in {
                "torch", "transformers", "torchvision", "timm"
            }, "constructing SUE imported a neural model dependency"
            return original_import(module_name, *args, **kwargs)

        queried = []

        def frozen_version(library):
            queried.append(library)
            return "test-1.2.3"

        extractor = _Extractor()
        extractor.name = name
        monkeypatch.setattr(builtins, "__import__", forbid_model_imports)
        monkeypatch.setattr(sue_module, "version", frozen_version)
        baseline = SUEBaseline(extractor)
        assert queried == libraries
        for library in libraries:
            assert baseline.configuration[library] == "test-1.2.3"

    @pytest.mark.parametrize("library", ["torch", "transformers", "torchvision"])
    def test_changed_runtime_version_changes_revision(self, monkeypatch, library):
        import pyrafault.verify.sue as sue_module

        extractor = _Extractor()
        extractor.name = "dinov2"
        versions = {"torch": "2.1", "transformers": "4.1", "torchvision": "0.1"}
        monkeypatch.setattr(sue_module, "version", lambda key: versions[key])
        original = SUEBaseline(extractor)
        same = SUEBaseline(extractor)
        assert original.revision == same.revision
        versions[library] = "new-version"
        assert SUEBaseline(extractor).revision != original.revision

    def test_fake_extractor_does_not_query_or_import_torch(self, monkeypatch):
        import builtins
        import pyrafault.verify.sue as sue_module

        original_import = builtins.__import__

        def forbid_torch_imports(module_name, *args, **kwargs):
            assert module_name.split(".")[0] != "torch"
            return original_import(module_name, *args, **kwargs)

        def unexpected_version_query(library):
            raise AssertionError(f"fake extractor unexpectedly queried {library}")

        monkeypatch.setattr(builtins, "__import__", forbid_torch_imports)
        monkeypatch.setattr(sue_module, "version", unexpected_version_query)
        loc = SUEBaseline(_Extractor()).localise(_inputs())
        assert loc.estimated_bbox is not None

    def test_missing_model_package_is_explicit_in_configuration(self, monkeypatch):
        import pyrafault.verify.sue as sue_module

        def missing(library):
            raise sue_module.PackageNotFoundError(library)

        extractor = _Extractor()
        extractor.name = "dinov2"
        monkeypatch.setattr(sue_module, "version", missing)
        configuration = SUEBaseline(extractor).configuration
        assert all(configuration[library] == "not-installed"
                   for library in ("torch", "transformers", "torchvision"))
