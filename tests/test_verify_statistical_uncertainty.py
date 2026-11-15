"""SU formulas and the shared verifier contract, without models or WSI.

Controlled frozen descriptors test the adaptation; they do not establish that
DINOv2 retrieval or SU's score assumptions work on pathology images.
"""
from collections.abc import Mapping
from copy import deepcopy

import numpy as np
import pytest
from PIL import Image

from pyrafault.geom import BBoxL0
from pyrafault.verify.base import Localisation, Verifier
from pyrafault.verify.statistical_uncertainty import (
    MIN_BEST_SIMILARITY,
    StatisticalUncertaintyBaseline,
    statistical_uncertainty,
)
from pyrafault.verify.sue import SUEBaseline


class _Extractor:
    name = "test-extractor"
    revision = "fixed-v1"
    preprocessing = "test-only-no-model"
    input_size = 16
    dim = 2
    device = "cpu"
    batch_size = 8

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
        result = np.tile([0.5, np.sqrt(0.75)], (len(images), 1))
        result[0] = result[1 + self.winner] = [1.0, 0.0]
        return result


def _cosine_descriptors(scores):
    """Choose unit vectors whose raw cosine against the query is controlled."""
    scores = np.asarray(scores, dtype=float)
    return np.vstack(([1.0, 0.0], np.column_stack((scores, np.sqrt(1 - scores ** 2)))))


def _inputs(context_size=(10, 8), query_size=(4, 4), *, bbox=None,
            context_mpp=1.0, query_mpp=1.0):
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
        assert key in self.allowed, f"SU read forbidden input {key!r}"
        self.accessed.add(key)
        return self.values[key]

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)


class TestStatisticalUncertaintyFormula:
    @pytest.mark.parametrize("scores,rs,sd,su", [
        ([0.9, 0.4, 0.3, 0.2], 1 / 3, 7 / 18, 13 / 36),
        ([0.9, 0.89, 0.88, 0.87], 44 / 45, 59 / 60, 353 / 360),
        ([1.0, 1.0], 1.0, 1.0, 1.0),
        ([1.0, 0.0], 0.0, 0.5, 0.25),
        ([0.8, 0.4, 0.0], 0.25, 0.5, 0.375),
    ])
    def test_paper_ratios_use_all_scores_for_sd_but_exclude_best_for_rs(
        self, scores, rs, sd, su
    ):
        result = statistical_uncertainty(scores)
        assert result["ratio_spread"] == pytest.approx(rs)
        assert result["similarity_distribution"] == pytest.approx(sd)
        assert result["statistical_uncertainty"] == pytest.approx(su)

    def test_common_positive_scale_preserves_ratios(self):
        scores = np.array([0.9, 0.4, 0.3, 0.2])
        assert statistical_uncertainty(scores * 10) == pytest.approx(
            statistical_uncertainty(scores)
        )

    def test_common_offset_changes_score_and_is_not_a_safe_domain_repair(self):
        scores = np.array([0.9, 0.4, 0.3, 0.2])
        assert statistical_uncertainty(scores + 1)["statistical_uncertainty"] != pytest.approx(
            statistical_uncertainty(scores)["statistical_uncertainty"]
        )

    @pytest.mark.parametrize("scores", [
        [], [0.9], [[0.9, 0.8]], [0.8, 0.9], [0.9, -0.1],
        [float("nan"), 0.5], [float("inf"), 0.5], [1.0, float("nan")],
        [0.0, 0.0], [MIN_BEST_SIMILARITY, 0.0],
        [MIN_BEST_SIMILARITY / 2, 0.0], ["not-numeric", 0.0],
    ])
    def test_unsupported_domain_is_rejected_not_shifted_or_clipped(self, scores):
        with pytest.raises(ValueError):
            statistical_uncertainty(scores)

    def test_best_score_just_above_domain_threshold_is_accepted(self):
        best = np.nextafter(MIN_BEST_SIMILARITY, np.inf)
        assert statistical_uncertainty([best, 0.0])["statistical_uncertainty"] == pytest.approx(0.25)


class TestSUContract:
    def test_shared_contract_and_top_one_not_an_average_gives_position(self):
        baseline = StatisticalUncertaintyBaseline(_Extractor(winner=5), top_k=2)
        assert isinstance(baseline, Verifier)
        assert baseline.name == "statistical_uncertainty_baseline"
        loc = baseline.localise(_inputs(bbox=BBoxL0(1000, 2000, 100, 160)))
        assert isinstance(loc, Localisation)
        assert loc.estimated_bbox == BBoxL0(1020, 2040, 40, 80)
        assert loc.reliability == pytest.approx(1 - 0.625)
        assert loc.diagnostics["score_domain_supported"] is True

    def test_uses_raw_cosine_not_l2_distance_or_affine_shift(self):
        loc = StatisticalUncertaintyBaseline(
            _Extractor(_cosine_descriptors([0.9, 0.4, 0.3, 0.2])), top_k=4
        ).localise(_inputs(context_size=(10, 4)))
        assert loc.diagnostics["candidate_cosine_similarities"] == pytest.approx([0.9, 0.4, 0.3, 0.2])
        assert loc.diagnostics["statistical_uncertainty"] == pytest.approx(13 / 36)
        assert loc.reliability == pytest.approx(23 / 36)
        distances = np.sqrt(2 - 2 * np.array([0.9, 0.4, 0.3, 0.2]))
        assert loc.diagnostics["candidate_l2_distances"] == pytest.approx(distances)

    def test_equal_scores_are_maximally_uncertain_not_perfectly_reliable(self):
        same = lambda images: np.tile([3.0, 4.0], (len(images), 1))
        baseline = StatisticalUncertaintyBaseline(_Extractor(same), top_k=2)
        first = baseline.localise(_inputs())
        assert first == baseline.localise(_inputs())
        assert first.estimated_bbox == BBoxL0(0, 0, 4, 4)
        assert first.reliability == 0.0
        assert first.diagnostics["score_domain_supported"] is True
        assert first.diagnostics["statistical_uncertainty"] == pytest.approx(1.0)

    def test_true_orthogonal_candidate_is_zero_not_negative_roundoff(self):
        inputs = _inputs(context_size=(6, 4))
        descriptors = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        loc = StatisticalUncertaintyBaseline(_Extractor(descriptors), top_k=2).localise(inputs)
        # Computing cosine via 1 - sqrt(2)**2 / 2 would produce a tiny negative
        # score and wrongly reject this supported nonnegative score domain.
        assert loc.diagnostics["candidate_cosine_similarities"] == [1.0, 0.0]
        assert loc.diagnostics["su_cosine_similarities"] == [1.0, 0.0]
        assert loc.diagnostics["score_domain_supported"] is True
        assert loc.reliability == pytest.approx(0.75)
        assert loc.estimated_bbox == BBoxL0(0, 0, 4, 4)
        assert loc.estimated_bbox == SUEBaseline(_Extractor(descriptors), top_k=2).localise(inputs).estimated_bbox

    def test_near_tie_cosines_are_sorted_for_formula_not_for_location(self):
        # Unit-L2 and direct-dot arithmetic can differ by an ulp. On the tested
        # runtime the L2-ranked scores here are slightly increasing; sorting only
        # the SU input avoids a spurious strict-descending validation failure.
        descriptors = np.array([
            [-0.5878191837142573, 0.6059793441285014, 0.5359642168532294],
            [0.054029264560640486, 0.9223590791005709, 0.38253701490415704],
            [0.0540292645606404, 0.9223590791005709, 0.3825370149041569],
        ])
        inputs = _inputs(context_size=(6, 4))
        loc = StatisticalUncertaintyBaseline(_Extractor(descriptors), top_k=2).localise(inputs)
        scores = loc.diagnostics["candidate_cosine_similarities"]
        sorted_scores = np.sort(scores)[::-1]
        assert loc.diagnostics["su_cosine_similarities"] == pytest.approx(sorted_scores)
        assert loc.diagnostics["score_domain_supported"] is True
        assert 0.0 <= loc.reliability <= 1.0
        assert loc.reliability == pytest.approx(1 - statistical_uncertainty(sorted_scores)["statistical_uncertainty"])
        assert loc.estimated_bbox == SUEBaseline(_Extractor(descriptors), top_k=2).localise(inputs).estimated_bbox

    def test_pixel_dependent_retrieval_locates_actual_planted_patch(self):
        class PixelExtractor(_Extractor):
            dim = 4

            def embed(self, images):
                return np.asarray([(*im.getpixel((0, 0)), 1.0) for im in images])

        inputs = _inputs()
        inputs["query"] = inputs["context"].crop((6, 2, 10, 6))
        assert StatisticalUncertaintyBaseline(PixelExtractor()).localise(inputs).estimated_bbox == BBoxL0(6, 2, 4, 4)
        inputs["query"] = inputs["context"].crop((0, 4, 4, 8))
        assert StatisticalUncertaintyBaseline(PixelExtractor()).localise(inputs).estimated_bbox == BBoxL0(0, 4, 4, 4)

    @pytest.mark.parametrize("winner", [0, 5, 11])
    @pytest.mark.parametrize("top_k", [2, 10, 50])
    def test_same_retrieval_configuration_has_same_position_as_sue(self, winner, top_k):
        inputs = _inputs(bbox=BBoxL0(1000, 2000, 100, 160))
        su = StatisticalUncertaintyBaseline(_Extractor(winner=winner), top_k=top_k).localise(inputs)
        sue = SUEBaseline(_Extractor(winner=winner), top_k=top_k).localise(inputs)
        assert su.estimated_bbox == sue.estimated_bbox
        assert su.diagnostics["candidate_origins_context"] == sue.diagnostics["candidate_origins_context"]
        assert su.diagnostics["candidate_l2_distances"] == sue.diagnostics["candidate_l2_distances"]

    def test_grid_is_row_major_and_includes_right_and_bottom_edges(self):
        extractor = _Extractor(winner=19)
        loc = StatisticalUncertaintyBaseline(extractor).localise(
            _inputs(context_size=(11, 9))
        )
        images = extractor.calls[0]
        origins = [image.getpixel((0, 0))[:2] for image in images[1:]]
        assert origins == [(x, y) for y in [0, 2, 4, 5] for x in [0, 2, 4, 6, 7]]
        assert all(image.size == (4, 4) for image in images)
        assert loc.estimated_bbox == BBoxL0(7, 5, 4, 4)

    def test_top_k_is_clipped_to_valid_windows(self):
        loc = StatisticalUncertaintyBaseline(_Extractor(), top_k=100).localise(
            _inputs(context_size=(6, 4))
        )
        assert loc.diagnostics["n_candidates"] == 2
        assert loc.diagnostics["top_k_used"] == 2

    def test_lower_ranked_negative_score_outside_top_k_does_not_poison_su(self):
        loc = StatisticalUncertaintyBaseline(
            _Extractor(_cosine_descriptors([0.9, 0.4, -0.3, -0.8])), top_k=2
        ).localise(_inputs(context_size=(10, 4)))
        assert loc.diagnostics["score_domain_supported"] is True
        assert loc.diagnostics["candidate_cosine_similarities"] == pytest.approx([0.9, 0.4])
        assert loc.reliability > 0

    def test_normalizes_descriptors_before_l2_ranking(self):
        def scaled(images):
            result = np.tile([0.0, 1.0], (len(images), 1))
            result[0], result[1], result[2] = [10.0, 0.0], [1.0, 0.0], [10.0, 1.0]
            return result
        loc = StatisticalUncertaintyBaseline(_Extractor(scaled), top_k=2).localise(_inputs())
        assert loc.estimated_bbox == BBoxL0(0, 0, 4, 4)
        assert loc.diagnostics["candidate_cosine_similarities"][0] == pytest.approx(1.0)

    def test_mpp_alignment_and_rectangular_l0_conversion(self):
        extractor = _Extractor(winner=5)
        inputs = _inputs(context_size=(10, 8), query_size=(16, 12),
                         context_mpp=2.0, query_mpp=0.5,
                         bbox=BBoxL0(100, 200, 100, 160))
        loc = StatisticalUncertaintyBaseline(extractor).localise(inputs)
        assert len(extractor.calls) == 1
        assert all(image.size == (4, 3) for image in extractor.calls[0])
        assert loc.estimated_bbox == BBoxL0(120, 240, 40, 60)
        assert loc.estimated_bbox == SUEBaseline(_Extractor(winner=5)).localise(inputs).estimated_bbox

    def test_noninteger_footprint_and_edge_crops_match_sue(self):
        extractor = _Extractor(winner=15)
        inputs = _inputs(context_size=(16, 12), query_size=(19, 13),
                         context_mpp=2.0, query_mpp=0.7,
                         bbox=BBoxL0(1000, 2000, 1600, 960))
        loc = StatisticalUncertaintyBaseline(extractor).localise(inputs)
        assert all(image.size == (7, 5) for image in extractor.calls[0])
        assert loc.estimated_bbox == BBoxL0(1900, 2480, 665, 364)
        assert inputs["context_bbox"].contains(loc.estimated_bbox)
        assert loc.estimated_bbox == SUEBaseline(_Extractor(winner=15)).localise(inputs).estimated_bbox

    def test_localise_reads_only_five_algorithm_inputs(self):
        guarded = _ClaimBlindInputs(_inputs())
        StatisticalUncertaintyBaseline(_Extractor()).localise(guarded)
        assert guarded.accessed == guarded.allowed

    def test_changed_claim_ids_split_do_not_change_localisation_or_su(self):
        inputs = _inputs()
        other = dict(inputs, claimed_bbox=BBoxL0(0, 0, 2, 2),
                     item_id="other", slide_id="other", split="calibration")
        baseline = StatisticalUncertaintyBaseline(_Extractor())
        assert baseline.localise(inputs) == baseline.localise(other)

    def test_shared_runner_audit_and_claim_dependent_decisions(self):
        from pyrafault.verify.runner import run_verifier
        from pyrafault.verify.thresholds import Thresholds

        class Dataset:
            item_ids = ["i0", "i1"]

            def __getitem__(self, item_id):
                inputs = _inputs()
                inputs["item_id"] = item_id
                inputs["claimed_bbox"] = (BBoxL0(0, 0, 4, 4) if item_id == "i0"
                                          else BBoxL0(200, 200, 4, 4))
                return {"inputs": inputs,
                        "target": {"true_bbox": BBoxL0(0, 0, 4, 4),
                                   "true_slide_id": inputs["slide_id"]}}

        result = run_verifier(StatisticalUncertaintyBaseline(_Extractor()), Dataset(),
                              Thresholds(0.0, 0.2, "calibration", 2, {}),
                              audit_independence=True).set_index("item_id")
        assert result.loc["i0", "decision"] == "verified"
        assert result.loc["i1", "decision"] == "mismatched"
        assert result.loc["i0", "estimated_bbox_l0_x"] == result.loc["i1", "estimated_bbox_l0_x"]


class TestSUFailureAndDomainHandling:
    @pytest.mark.parametrize("scores", [
        [0.9, 0.4, 0.3, -0.1], [-0.1, -0.2, -0.3, -0.4],
        [MIN_BEST_SIMILARITY / 2, 0.0, 0.0, 0.0],
    ])
    def test_invalid_score_domain_retains_location_but_gives_zero_reliability(self, scores):
        inputs = _inputs(context_size=(10, 4))
        extractor = _Extractor(_cosine_descriptors(scores))
        loc = StatisticalUncertaintyBaseline(extractor, top_k=4).localise(inputs)
        assert loc.estimated_bbox == BBoxL0(0, 0, 4, 4)
        assert loc.reliability == 0.0
        assert loc.diagnostics["score_domain_supported"] is False
        assert loc.diagnostics["reason"]
        assert loc.diagnostics["candidate_cosine_similarities"] == pytest.approx(scores, abs=1e-14)
        assert loc.diagnostics["n_candidates"] == 4
        assert loc.diagnostics["top_k_used"] == 4
        assert all(loc.diagnostics[key] is None for key in (
            "ratio_spread", "similarity_distribution", "statistical_uncertainty"
        ))
        assert loc.estimated_bbox == SUEBaseline(_Extractor(_cosine_descriptors(scores))).localise(inputs).estimated_bbox

    @pytest.mark.parametrize("query_size", [(11, 4), (4, 9), (11, 9)])
    def test_oversized_query_abstains_before_embedding(self, query_size):
        extractor = _Extractor()
        loc = StatisticalUncertaintyBaseline(extractor).localise(_inputs(query_size=query_size))
        assert loc.estimated_bbox is None
        assert loc.reliability == 0.0
        assert extractor.calls == []

    def test_one_grid_window_cannot_support_uncertainty(self):
        loc = StatisticalUncertaintyBaseline(_Extractor()).localise(
            _inputs(context_size=(4, 4))
        )
        assert loc.estimated_bbox is None
        assert loc.reliability == 0.0

    @pytest.mark.parametrize("nonzero", [0, 1])
    def test_fewer_than_two_valid_window_descriptors_abstains(self, nonzero):
        def descriptors(images):
            result = np.zeros((len(images), 2))
            result[:1 + nonzero] = [1.0, 0.0]
            return result
        loc = StatisticalUncertaintyBaseline(_Extractor(descriptors)).localise(_inputs())
        assert loc.estimated_bbox is None
        assert loc.reliability == 0.0

    def test_zero_query_abstains(self):
        def descriptors(images):
            result = np.tile([1.0, 0.0], (len(images), 1))
            result[0] = 0.0
            return result
        loc = StatisticalUncertaintyBaseline(_Extractor(descriptors)).localise(_inputs())
        assert loc.estimated_bbox is None
        assert loc.reliability == 0.0

    def test_filtering_zero_windows_preserves_original_coordinates(self):
        def descriptors(images):
            result = np.zeros((len(images), 2))
            result[0] = result[6] = [1.0, 0.0]
            result[7] = [0.6, 0.8]
            return result
        loc = StatisticalUncertaintyBaseline(_Extractor(descriptors)).localise(_inputs())
        assert loc.estimated_bbox == BBoxL0(2, 2, 4, 4)
        assert loc.diagnostics["n_candidates"] == 2
        assert loc.reliability > 0

    @pytest.mark.parametrize("field", ["context_mpp", "query_mpp"])
    @pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), None, "bad"])
    def test_invalid_mpp_raises(self, field, value):
        inputs = _inputs()
        inputs[field] = value
        with pytest.raises(ValueError):
            StatisticalUncertaintyBaseline(_Extractor()).localise(inputs)

    def test_query_upsampling_is_refused(self):
        with pytest.raises(ValueError):
            StatisticalUncertaintyBaseline(_Extractor()).localise(
                _inputs(context_mpp=0.5, query_mpp=2.0)
            )

    @pytest.mark.parametrize("output", [
        lambda images: np.zeros(len(images)),
        lambda images: np.zeros((len(images) - 1, 2)),
        lambda images: np.zeros((len(images), 0)),
        lambda images: np.full((len(images), 2), np.nan),
        lambda images: np.full((len(images), 2), np.inf),
    ])
    def test_invalid_embedding_output_fails_loudly(self, output):
        with pytest.raises(ValueError):
            StatisticalUncertaintyBaseline(_Extractor(output)).localise(_inputs())

    def test_overflowing_norms_fail_loudly(self):
        with np.errstate(over="ignore"):
            with pytest.raises(ValueError, match="norm"):
                StatisticalUncertaintyBaseline(
                    _Extractor(lambda images: np.full((len(images), 2), 1e308))
                ).localise(_inputs())

    def test_model_failure_is_not_hidden_by_a_fallback(self):
        def failed(images):
            raise RuntimeError("test model failure")
        with pytest.raises(RuntimeError, match="test model failure"):
            StatisticalUncertaintyBaseline(_Extractor(failed)).localise(_inputs())

    @pytest.mark.parametrize("params", [
        {"top_k": 0}, {"top_k": 1}, {"top_k": -1}, {"top_k": 2.5}, {"top_k": True},
        {"stride_fraction": 0}, {"stride_fraction": -0.1}, {"stride_fraction": 1.01},
        {"stride_fraction": float("nan")}, {"stride_fraction": float("inf")},
        {"stride_fraction": None}, {"stride_fraction": "bad"},
    ])
    def test_invalid_parameters_are_rejected(self, params):
        with pytest.raises(ValueError):
            StatisticalUncertaintyBaseline(_Extractor(), **params)


class TestSURevision:
    def test_same_frozen_configuration_has_same_revision(self):
        assert StatisticalUncertaintyBaseline(_Extractor()).revision == StatisticalUncertaintyBaseline(_Extractor()).revision

    @pytest.mark.parametrize("field,value", [
        ("name", "other"), ("revision", "weights-v2"),
        ("preprocessing", "new-resize"), ("dim", 3), ("input_size", 32),
        ("device", "cuda"), ("batch_size", 32),
    ])
    def test_extractor_provenance_changes_revision(self, field, value):
        original = _Extractor()
        changed = deepcopy(original)
        setattr(changed, field, value)
        assert StatisticalUncertaintyBaseline(original).revision != StatisticalUncertaintyBaseline(changed).revision

    @pytest.mark.parametrize("params", [{"top_k": 5}, {"stride_fraction": 0.25}])
    def test_algorithm_configuration_changes_revision(self, params):
        assert StatisticalUncertaintyBaseline(_Extractor()).revision != StatisticalUncertaintyBaseline(_Extractor(), **params).revision

    @pytest.mark.parametrize("attribute", ["name", "revision"])
    def test_frozen_extractor_identity_is_required(self, attribute):
        extractor = _Extractor()
        setattr(extractor, attribute, "")
        with pytest.raises(ValueError):
            StatisticalUncertaintyBaseline(extractor)

    def test_numeric_stride_string_has_same_effect_and_revision(self):
        numeric = StatisticalUncertaintyBaseline(_Extractor(), stride_fraction=0.5)
        string = StatisticalUncertaintyBaseline(_Extractor(), stride_fraction="0.5")
        assert string.configuration == numeric.configuration
        assert string.revision == numeric.revision
        assert string.localise(_inputs()) == numeric.localise(_inputs())

    def test_fixed_formula_and_domain_policy_are_part_of_configuration(self):
        configuration = StatisticalUncertaintyBaseline(_Extractor()).configuration
        assert configuration["alpha"] == 0.5
        assert configuration["formula"] == "0.5*mean(s[1:]/s[0])+0.5*median(s/s[0])"
        assert configuration["minimum_best_similarity"] == MIN_BEST_SIMILARITY
        assert configuration["score_domain_policy"] == "nonnegative-topk-otherwise-rho0-preserve-bbox"
        assert configuration["reliability"] == "1-SU"

    def test_changed_domain_guard_changes_revision(self, monkeypatch):
        import pyrafault.verify.statistical_uncertainty as su_module

        original = StatisticalUncertaintyBaseline(_Extractor())
        monkeypatch.setattr(su_module, "MIN_BEST_SIMILARITY", 1e-6)
        changed = StatisticalUncertaintyBaseline(_Extractor())
        assert original.revision != changed.revision
        assert changed.configuration["minimum_best_similarity"] == 1e-6

    @pytest.mark.parametrize("name,libraries", [
        ("dinov2", ["torch", "transformers", "torchvision"]),
        ("uni", ["torch", "timm", "torchvision"]),
    ])
    def test_runtime_library_versions_recorded_without_importing_neural_models(
        self, monkeypatch, name, libraries
    ):
        import builtins
        import pyrafault.verify.statistical_uncertainty as su_module

        original_import = builtins.__import__

        def forbid_model_imports(module_name, *args, **kwargs):
            assert module_name.split(".")[0] not in {
                "torch", "transformers", "torchvision", "timm"
            }
            return original_import(module_name, *args, **kwargs)

        queried = []

        def fixed_version(library):
            queried.append(library)
            return "test-1.2.3"

        extractor = _Extractor()
        extractor.name = name
        monkeypatch.setattr(builtins, "__import__", forbid_model_imports)
        monkeypatch.setattr(su_module, "version", fixed_version)
        baseline = StatisticalUncertaintyBaseline(extractor)
        assert queried == libraries
        assert all(baseline.configuration[library] == "test-1.2.3" for library in libraries)

    def test_runtime_version_change_changes_revision(self, monkeypatch):
        import pyrafault.verify.statistical_uncertainty as su_module

        extractor = _Extractor()
        extractor.name = "dinov2"
        versions = {"torch": "2.1", "transformers": "4.1", "torchvision": "0.1"}
        monkeypatch.setattr(su_module, "version", lambda key: versions[key])
        original = StatisticalUncertaintyBaseline(extractor)
        versions["transformers"] = "4.2"
        assert StatisticalUncertaintyBaseline(extractor).revision != original.revision

    def test_missing_model_package_is_explicit_not_silently_omitted(self, monkeypatch):
        import pyrafault.verify.statistical_uncertainty as su_module

        def missing(library):
            raise su_module.PackageNotFoundError(library)

        extractor = _Extractor()
        extractor.name = "dinov2"
        monkeypatch.setattr(su_module, "version", missing)
        configuration = StatisticalUncertaintyBaseline(extractor).configuration
        assert all(configuration[library] == "not-installed"
                   for library in ("torch", "transformers", "torchvision"))
