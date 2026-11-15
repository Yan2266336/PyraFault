import pytest

from pyrafault.config import SplitsCfg
from pyrafault.sampling.splits import (
    SplitError, assign_splits, derive_fraction_counts, resolve_split_counts,
)

CFG = SplitsCfg(
    level="case", dev=6, calibration=2, test=2,
    min_contexts_per_slide={"test": 15, "calibration": 15},
    max_reshuffle_attempts=1000,
    mode="absolute",
)

FRACTION_CFG = SplitsCfg(
    level="case", dev=0.6, calibration=0.2, test=0.2,
    min_contexts_per_slide={"test": 15, "calibration": 15},
    max_reshuffle_attempts=1000,
    mode="fraction",
)

# The real pilot: TCGA-55-A4DG can only host 6 contexts (spec §3 D1).
COUNTS = {
    "TCGA-44-2655-01Z-00-DX1": 25, "TCGA-44-2659-01Z-00-DX1": 25,
    "TCGA-49-4505-01Z-00-DX1": 25, "TCGA-55-6986-01Z-00-DX1": 25,
    "TCGA-55-7281-01Z-00-DX1": 21, "TCGA-55-8094-01Z-00-DX1": 20,
    "TCGA-55-8205-01Z-00-DX1": 25, "TCGA-55-A4DG-01Z-00-DX1": 6,
    "TCGA-64-1680-01Z-00-DX1": 25, "TCGA-99-8025-01Z-00-DX1": 25,
}
CASES = {sid[:12]: [sid] for sid in COUNTS}


def test_partition_is_six_two_two():
    a = assign_splits(CASES, COUNTS, CFG, seed=20260812)
    counts = {s: sum(1 for v in a.values() if v == s) for s in ("dev", "calibration", "test")}
    assert counts == {"dev": 6, "calibration": 2, "test": 2}
    assert set(a) == set(CASES)


def test_starved_slide_is_forced_into_dev():
    a = assign_splits(CASES, COUNTS, CFG, seed=20260812)
    assert a["TCGA-55-A4DG"] == "dev", "6 contexts is below the 15 threshold"


def test_every_eval_slide_clears_the_threshold():
    a = assign_splits(CASES, COUNTS, CFG, seed=20260812)
    for case, split in a.items():
        if split in ("test", "calibration"):
            for sid in CASES[case]:
                assert COUNTS[sid] >= 15


def test_assignment_is_deterministic():
    assert assign_splits(CASES, COUNTS, CFG, 20260812) == assign_splits(CASES, COUNTS, CFG, 20260812)


def test_different_seed_gives_a_different_assignment():
    a = assign_splits(CASES, COUNTS, CFG, 20260812)
    b = assign_splits(CASES, COUNTS, CFG, 999)
    assert a != b


def test_multi_slide_case_stays_together():
    cases = dict(CASES)
    cases["TCGA-44-2655"] = ["TCGA-44-2655-01Z-00-DX1", "TCGA-44-2655-02Z-00-DX1"]
    counts = dict(COUNTS, **{"TCGA-44-2655-02Z-00-DX1": 18})
    a = assign_splits(cases, counts, CFG, 20260812)
    assert len({a["TCGA-44-2655"]}) == 1


def test_raises_when_the_constraint_cannot_be_met():
    starved = {k: 3 for k in COUNTS}
    with pytest.raises(SplitError, match="min_contexts_per_slide"):
        assign_splits(CASES, starved, CFG, 20260812)


def test_raises_when_case_count_does_not_match_config():
    """The mismatch error must name both the configured total and the real cohort
    size (spec §8) so a user can tell at a glance which number to change."""
    fewer = {k: v for k, v in list(CASES.items())[:5]}
    with pytest.raises(SplitError) as exc_info:
        assign_splits(fewer, COUNTS, CFG, 20260812)
    msg = str(exc_info.value)
    assert "10 cases" in msg, msg
    assert "5 cases" in msg, msg


# ---------------------------------------------------------------------------
# Largest-remainder derivation (spec §8)
# ---------------------------------------------------------------------------

def test_derive_fraction_counts_matches_absolute_for_the_pilot():
    """10 cases at 0.6/0.2/0.2 must reproduce the pilot's own 6/2/2 split."""
    counts = derive_fraction_counts(10, 0.6, 0.2, 0.2)
    assert counts == {"dev": 6, "calibration": 2, "test": 2}
    assert sum(counts.values()) == 10


def test_derive_fraction_counts_contested_rounding_seven_cases():
    """7 cases at 0.6/0.2/0.2: raw shares are dev=4.2, calibration=1.4, test=1.4 —
    calibration and test tie on fractional remainder (0.4 each). The tie is broken
    alphabetically ('calibration' < 'test'), so calibration gets the spare case."""
    counts = derive_fraction_counts(7, 0.6, 0.2, 0.2)
    assert counts == {"dev": 4, "calibration": 2, "test": 1}
    assert sum(counts.values()) == 7
    assert counts["test"] >= 1 and counts["calibration"] >= 1


def test_derive_fraction_counts_floor_bites_three_cases():
    """3 cases at 0.6/0.2/0.2: naive largest-remainder gives dev=2, calibration=1,
    test=0 (or the calibration/test-swapped equivalent); the floor-at-1 correction
    must take a case back from dev so every split ends up non-empty."""
    counts = derive_fraction_counts(3, 0.6, 0.2, 0.2)
    assert counts == {"dev": 1, "calibration": 1, "test": 1}
    assert sum(counts.values()) == 3


@pytest.mark.parametrize("n_cases", [3, 4, 5, 6, 8, 9, 10, 13, 20, 50, 100])
def test_derive_fraction_counts_always_sums_to_n_and_never_empties_eval_splits(n_cases):
    counts = derive_fraction_counts(n_cases, 0.6, 0.2, 0.2)
    assert sum(counts.values()) == n_cases
    assert counts["test"] >= 1
    assert counts["calibration"] >= 1
    assert all(v >= 0 for v in counts.values())


def test_derive_fraction_counts_raises_when_cohort_too_small():
    with pytest.raises(SplitError, match="too small"):
        derive_fraction_counts(1, 0.6, 0.2, 0.2)


def test_resolve_split_counts_absolute_mode_matches_cohort():
    assert resolve_split_counts(CFG, 10) == {"dev": 6, "calibration": 2, "test": 2}


def test_resolve_split_counts_absolute_mode_mismatch_names_both_numbers():
    with pytest.raises(SplitError) as exc_info:
        resolve_split_counts(CFG, 50)
    msg = str(exc_info.value)
    assert "10 cases" in msg
    assert "50 cases" in msg


def test_resolve_split_counts_fraction_mode_adapts_to_any_cohort_size():
    assert resolve_split_counts(FRACTION_CFG, 10) == {"dev": 6, "calibration": 2, "test": 2}
    counts_50 = resolve_split_counts(FRACTION_CFG, 50)
    assert sum(counts_50.values()) == 50
    assert counts_50 == {"dev": 30, "calibration": 10, "test": 10}


def test_assign_splits_works_with_fraction_mode_config():
    """End-to-end: a fraction-mode config assigns the same 6/2/2 partition shape
    as the equivalent absolute-mode config, for the pilot's 10 cases."""
    a = assign_splits(CASES, COUNTS, FRACTION_CFG, seed=20260812)
    counts = {s: sum(1 for v in a.values() if v == s) for s in ("dev", "calibration", "test")}
    assert counts == {"dev": 6, "calibration": 2, "test": 2}


class TestManyStarvedCasesStillPartition:
    """The failure that stopped the five-cohort build.

    Relaxing the alpha floor in `io.read` readmitted ~700 slides, and some of them are
    geometrically small: of 2,698 slides with contexts, 228 froze fewer than 15. That
    put 136 of 2,152 cases below the evaluation floor while 2,016 remained eligible --
    comfortably more than the 646 that test and calibration need. The old rejection
    sampler still failed, because it needed all 136 to land in the dev tail by chance
    (~0.7^136), and reported it as "no case-level partition satisfies ...", which reads
    as a cohort problem rather than an algorithm one.
    """

    @staticmethod
    def _cohort(n_eligible: int, n_starved: int):
        counts, cases = {}, {}
        for i in range(n_eligible):
            sid = f"ELIG-{i:04d}-01Z-00-DX1"
            counts[sid] = 30
            cases[f"ELIG-{i:04d}"] = [sid]
        for i in range(n_starved):
            sid = f"POOR-{i:04d}-01Z-00-DX1"
            counts[sid] = 3
            cases[f"POOR-{i:04d}"] = [sid]
        return cases, counts

    def test_a_cohort_the_sampler_could_not_partition_now_partitions(self):
        """136 starved of 2,152, the five-cohort freeze's own shape."""
        cases, counts = self._cohort(2016, 136)
        cfg = SplitsCfg(level="case", dev=0.7, calibration=0.15, test=0.15,
                        min_contexts_per_slide={"test": 15, "calibration": 15},
                        max_reshuffle_attempts=1000, mode="fraction")
        a = assign_splits(cases, counts, cfg, seed=20260812)
        assert len(a) == len(cases)
        for case, split in a.items():
            if split in ("test", "calibration"):
                assert all(counts[s] >= 15 for s in cases[case])

    def test_every_starved_case_lands_in_dev(self):
        cases, counts = self._cohort(2016, 136)
        cfg = SplitsCfg(level="case", dev=0.7, calibration=0.15, test=0.15,
                        min_contexts_per_slide={"test": 15, "calibration": 15},
                        max_reshuffle_attempts=1000, mode="fraction")
        a = assign_splits(cases, counts, cfg, seed=20260812)
        assert {a[c] for c in cases if c.startswith("POOR")} == {"dev"}

    def test_the_split_sizes_are_still_exact(self):
        """Starved cases go to dev, which is where the leftovers were going anyway, so
        the configured shares are untouched as long as enough eligible cases exist."""
        cases, counts = self._cohort(2016, 136)
        cfg = SplitsCfg(level="case", dev=0.7, calibration=0.15, test=0.15,
                        min_contexts_per_slide={"test": 15, "calibration": 15},
                        max_reshuffle_attempts=1000, mode="fraction")
        a = assign_splits(cases, counts, cfg, seed=20260812)
        got = {s: sum(1 for v in a.values() if v == s) for s in ("dev", "calibration", "test")}
        assert got == derive_fraction_counts(len(cases), 0.7, 0.15, 0.15)

    def test_a_slide_with_zero_contexts_is_treated_as_starved(self):
        """67 slides froze no context at all. They cannot be evaluated on, and a
        `.get(s, 0)` default must not quietly read as 'no floor configured'."""
        cases, counts = self._cohort(20, 0)
        cases["EMPTY-0000"] = ["EMPTY-0000-01Z-00-DX1"]        # absent from `counts`
        cfg = SplitsCfg(level="case", dev=0.7, calibration=0.15, test=0.15,
                        min_contexts_per_slide={"test": 15, "calibration": 15},
                        max_reshuffle_attempts=1000, mode="fraction")
        a = assign_splits(cases, counts, cfg, seed=1)
        assert a["EMPTY-0000"] == "dev"

    def test_too_few_eligible_cases_says_which_number_is_wrong(self):
        """Still an error when the cohort genuinely cannot support the floored splits --
        but one that names how many qualify, so the reader can tell a small cohort from
        a stage-03 problem."""
        cases, counts = self._cohort(2, 50)
        cfg = SplitsCfg(level="case", dev=0.7, calibration=0.15, test=0.15,
                        min_contexts_per_slide={"test": 15, "calibration": 15},
                        max_reshuffle_attempts=1000, mode="fraction")
        with pytest.raises(SplitError, match="only 2 of 52 cases qualify"):
            assign_splits(cases, counts, cfg, seed=1)

    def test_it_is_still_a_function_of_the_seed(self):
        cases, counts = self._cohort(200, 30)
        cfg = SplitsCfg(level="case", dev=0.7, calibration=0.15, test=0.15,
                        min_contexts_per_slide={"test": 15, "calibration": 15},
                        max_reshuffle_attempts=1000, mode="fraction")
        a = assign_splits(cases, counts, cfg, 7)
        assert a == assign_splits(cases, counts, cfg, 7)
        assert a != assign_splits(cases, counts, cfg, 8)
