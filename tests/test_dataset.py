import inspect

import pytest

import pyrafault.dataset as ds_mod
from pyrafault.dataset import (
    INPUT_KEYS, TARGET_KEYS, PyraFaultDataset, assert_spec_matches_config, make_item_id,
)

# Imported, not restated: this file used to keep its own copy of the tuple, which meant
# a newly added fault type was silently exempt from the item_id leakage test below.
from pyrafault.faults import FAULT_TYPES


def test_matching_spec_raises_nothing():
    assert_spec_matches_config("context", 2.0, 2048, "v1", 2.0, 2048, "v1") is None


@pytest.mark.parametrize("target_mpp,output_size,version", [
    (999.0, 2048, "v1"),   # I2: a row's target_mpp disagreeing with the live config
    (2.0, 1, "v1"),        # ...or its output_size...
    (2.0, 2048, "v99"),    # ...or its extraction_version must be caught, not read past
])
def test_mismatched_spec_raises(target_mpp, output_size, version):
    with pytest.raises(ValueError):
        assert_spec_matches_config("context", target_mpp, output_size, version, 2.0, 2048, "v1")


def test_evaluation_path_contains_no_randomness():
    """Guideline §11/§17: no random crop, no colour augmentation, no re-sampling."""
    src = inspect.getsource(ds_mod)
    for banned in ("random", "RandomCrop", "ColorJitter", "shuffle", "augment"):
        assert banned not in src, f"{banned!r} must not appear on the evaluation path"


def test_no_tissue_or_quality_recomputation_at_load_time():
    src = inspect.getsource(ds_mod)
    for banned in ("build_mask", "TissueMask", "evaluate(", "candidate_pool"):
        assert banned not in src, f"{banned!r} would re-derive membership at eval time"


@pytest.mark.slow
def test_item_shape_and_content(benchmark_root, gdc_root):
    ds = PyraFaultDataset(benchmark_root, gdc_root, split="test")
    assert len(ds) > 0
    item = ds[0]
    assert set(item) == {"inputs", "target"}
    assert set(item["inputs"]) == INPUT_KEYS
    assert set(item["target"]) == TARGET_KEYS
    assert item["inputs"]["context"].size == (2048, 2048)
    assert item["inputs"]["query"].size == (512, 512)
    assert isinstance(item["target"]["claim_correct"], bool)
    ds.close()


# spec §8: "a test asserts that `inputs` contains no key whose value determines
# `claim_correct`" -- kept as a first line of defense (it is still useful: a future
# leaky field added by NAME, e.g. a "normalized_offset_bucket" key, fails this
# instantly) but it is NOT sufficient on its own. The first version of this loader
# split shipped `inputs["pair_id"] = "..._q3__clean"` -- a key-set assertion never
# catches that, because "pair_id" is a plausible-looking, non-obviously-leaky NAME;
# the leak lived entirely in the VALUE. See the two tests below this one for the
# value-based checks spec §8 now requires alongside this key-set check.
def test_inputs_key_set_excludes_the_named_answer_fields(benchmark_root, gdc_root):
    ds = PyraFaultDataset(benchmark_root, gdc_root, split="test")
    item = ds[0]
    assert set(item["inputs"]) == INPUT_KEYS
    leaky = {"true_bbox", "normalized_offset", "iou_claim_true",
             "fault_severity", "hardness_score", "claim_correct", "pair_id"}
    assert leaky.isdisjoint(item["inputs"])
    ds.close()


def test_item_id_contains_no_fault_type_substring():
    """spec §8: 'item_id contains no fault-type substring.' item_id's hex alphabet
    (0-9a-f) structurally cannot spell any of FAULT_TYPES (each contains a letter
    outside a-f), so this can't fail today -- it pins that property so a future
    change to how item_id is derived (e.g. splicing in a truncated pair_id) would."""
    for seed in (0, 1, 42, 123456):
        for i in range(50):
            for fault_type in FAULT_TYPES:
                pair_id = f"S1_ctx{i:04d}_q0__{fault_type}"
                item_id = make_item_id(seed, pair_id)
                assert not any(ft in item_id for ft in FAULT_TYPES), (seed, pair_id, item_id)


@pytest.mark.slow
def test_no_inputs_field_value_predicts_claim_correct_better_than_base_rate(
    benchmark_root, gdc_root,
):
    """spec §8: the split must be a property of VALUES, not key names. This is the
    test that would have caught `inputs["pair_id"]` -- it did not exist the first
    time this loader shipped, because the only guard was the key-set test above,
    which is blind to what a key's value actually contains.

    Deliberately does NOT hand-list which fields to check (a second review round
    caught exactly that shape of bug one level up: a hardcoded field list next to a
    hardcoded key-set assertion means a field added to BOTH in the same commit sails
    past every guard). Instead it iterates `item["inputs"]` itself and scores every
    string-valued field actually present -- whatever that turns out to be -- against
    `pyrafault.qc.report`'s own leakage-scoring helper (§13.5), so a future field
    gets checked automatically, with no test-file edit required.
    """
    from pyrafault.qc.report import _string_leak_score

    ds = PyraFaultDataset(benchmark_root, gdc_root)
    pids = ds.pair_ids
    step = max(1, len(pids) // 150)
    sample = [ds[pid] for pid in pids[::step]]
    is_clean = [it["target"]["fault_type"] == "clean" for it in sample]

    string_fields = {k for k, v in sample[0]["inputs"].items() if isinstance(v, str)}
    assert string_fields, "sanity: inputs should contain at least one string field"
    for field in sorted(string_fields):
        values = [it["inputs"][field] for it in sample]
        score = _string_leak_score(values, is_clean)
        assert score <= 0.55, f"inputs[{field!r}] predicts claim_correct: {score:.3f}"
    ds.close()


@pytest.mark.slow
def test_clean_pairs_claim_their_true_box(benchmark_root, gdc_root):
    ds = PyraFaultDataset(benchmark_root, gdc_root)
    for pid in [p for p in ds.pair_ids if p.endswith("__clean")][:20]:
        item = ds[pid]
        assert item["target"]["claim_correct"] is True
        assert item["inputs"]["claimed_bbox"] == item["target"]["true_bbox"]
    ds.close()


@pytest.mark.slow
def test_fault_pairs_never_claim_their_true_box(benchmark_root, gdc_root):
    ds = PyraFaultDataset(benchmark_root, gdc_root)
    faults = [p for p in ds.pair_ids if not p.endswith("__clean")]
    for pid in faults[:30]:
        item = ds[pid]
        assert item["target"]["claim_correct"] is False
        assert item["inputs"]["claimed_bbox"] != item["target"]["true_bbox"]
    ds.close()


@pytest.mark.slow
def test_two_instances_return_identical_bytes(benchmark_root, gdc_root):
    from pyrafault.cache import image_sha256
    a = PyraFaultDataset(benchmark_root, gdc_root)
    b = PyraFaultDataset(benchmark_root, gdc_root)
    for pid in a.pair_ids[:10]:
        ia, ib = a[pid], b[pid]
        assert image_sha256(ia["inputs"]["context"]) == image_sha256(ib["inputs"]["context"])
        assert image_sha256(ia["inputs"]["query"]) == image_sha256(ib["inputs"]["query"])
    a.close(); b.close()


@pytest.mark.slow
def test_reconstruction_after_cache_deletion_is_identical(tmp_path, benchmark_root, gdc_root):
    from pyrafault.cache import image_sha256
    warm = PyraFaultDataset(benchmark_root, gdc_root, cache_dir=tmp_path / "c1")
    pid = warm.pair_ids[0]
    first = image_sha256(warm[pid]["inputs"]["query"])
    warm.close()

    cold = PyraFaultDataset(benchmark_root, gdc_root, cache_dir=tmp_path / "c2",
                            materialize=False)
    assert image_sha256(cold[pid]["inputs"]["query"]) == first
    cold.close()
