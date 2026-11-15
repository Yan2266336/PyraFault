"""Case-level split assignment (guideline §12).

Splitting by patient rather than by pair is what keeps a verifier from memorising a
slide in dev and being scored on it in test. The minimum-context constraint additionally
stops a geometrically starved slide from halving an evaluation split.

The assignment is *constructed* so that constraint holds, rather than sampled until it
happens to -- see `assign_splits` for why that distinction decided whether a 2,765-slide
freeze could be built at all.
"""
from __future__ import annotations

import numpy as np

from pyrafault.config import SplitsCfg


class SplitError(RuntimeError):
    """No valid case-level partition exists under the configured constraints."""


def derive_fraction_counts(
    n_cases: int, dev_frac: float, calibration_frac: float, test_frac: float
) -> dict[str, int]:
    """Largest-remainder apportionment of `n_cases` into dev/calibration/test.

    Each split's exact share (`n_cases * fraction`) is floored, then the leftover
    cases (at most 2, since three fractions summing to 1.0 leave a deficit smaller
    than the number of splits) are dealt one at a time to whichever split has the
    largest fractional remainder, ties broken alphabetically by split name for
    determinism.

    `test` and `calibration` are then floored at 1 — a split must never be empty —
    by taking one case at a time from whichever split currently holds the most.
    This keeps the total exactly `n_cases` throughout, so a fraction-mode config
    always resolves for any cohort size, with no separate "does this match the
    cohort" check needed (unlike absolute mode, see `resolve_split_counts`).
    """
    fracs = {"dev": dev_frac, "calibration": calibration_frac, "test": test_frac}
    raw = {k: n_cases * v for k, v in fracs.items()}
    floor = {k: int(v) for k, v in raw.items()}
    remainder = n_cases - sum(floor.values())
    order = sorted(fracs, key=lambda k: (-(raw[k] - floor[k]), k))
    for i in range(remainder):
        floor[order[i]] += 1

    for k in ("test", "calibration"):
        while floor[k] < 1:
            donor = max(
                (s for s in floor if s != k and floor[s] > 1),
                key=lambda s: floor[s], default=None,
            )
            if donor is None:
                raise SplitError(
                    f"cannot derive a non-empty '{k}' split from {n_cases} cases at "
                    f"fractions dev={dev_frac}, calibration={calibration_frac}, "
                    f"test={test_frac} — cohort is too small for these fractions"
                )
            floor[donor] -= 1
            floor[k] += 1
    return floor


def resolve_split_counts(cfg: SplitsCfg, n_cases: int) -> dict[str, int]:
    """Turn the configured split sizes into absolute per-split case counts.

    Absolute-count configs (`cfg.mode == "absolute"`) are checked against the real
    cohort size here — stage 03's gather, the earliest point that size is known —
    rather than at config load. Fraction configs (`cfg.mode == "fraction"`) need no
    such check: the largest-remainder derivation always sums to exactly `n_cases`,
    for any cohort size.
    """
    if cfg.mode == "absolute":
        counts = {
            "dev": int(cfg.dev), "calibration": int(cfg.calibration), "test": int(cfg.test),
        }
        total = sum(counts.values())
        if total != n_cases:
            raise SplitError(
                f"splits.dev/calibration/test sum to {total} cases "
                f"(dev={counts['dev']}, calibration={counts['calibration']}, "
                f"test={counts['test']}) but the cohort has {n_cases} cases"
            )
        return counts
    return derive_fraction_counts(n_cases, cfg.dev, cfg.calibration, cfg.test)


def assign_splits(
    case_to_slides: dict[str, list[str]],
    context_counts: dict[str, int],
    cfg: SplitsCfg,
    seed: int,
) -> dict[str, str]:
    cases = sorted(case_to_slides)
    sizes = resolve_split_counts(cfg, len(cases))

    def ok(case: str, split: str) -> bool:
        floor = cfg.min_contexts_per_slide.get(split)
        if floor is None:
            return True
        return all(context_counts.get(s, 0) >= floor for s in case_to_slides[case])

    # Constructed, not sampled. The earlier version drew a random permutation, cut it
    # into test/calibration/dev, and retried until no floored split held a starved
    # case. That is rejection sampling, and it cannot find an assignment that plainly
    # exists: every one of the k starved cases has to land in the dev tail by luck, so
    # the chance is roughly (dev share)^k. The five-cohort freeze has k = 136 starved
    # cases out of 2,152 and 2,016 eligible ones -- far more than the 646 that test and
    # calibration need -- yet 0.7^136 is about 1e-21, so 1,000 attempts failed and the
    # build stopped with "no partition satisfies ...". The floor was never the problem.
    #
    # So: shuffle once, fill each floored split from the eligible cases in that order,
    # and let everything left over -- starved cases included -- fall to dev. The floor
    # is then satisfied by construction rather than by luck, the split sizes are exactly
    # `sizes` whenever enough eligible cases exist, and the result is still a function
    # of the seed alone.
    #
    # Note for anyone re-running an existing freeze: this assigns differently from the
    # old code even where the old code succeeded, so a freeze whose splits are already
    # frozen in its tables must not have its gather re-run casually -- the splits would
    # move under results computed against them.
    rng = np.random.default_rng(seed)
    order = [cases[i] for i in rng.permutation(len(cases))]

    assignment: dict[str, str] = {}
    for split in ("test", "calibration"):
        want = sizes[split]
        taken = [c for c in order if c not in assignment and ok(c, split)][:want]
        if len(taken) < want:
            eligible = sum(1 for c in cases if ok(c, split))
            raise SplitError(
                f"split {split!r} needs {want} cases whose every slide clears "
                f"min_contexts_per_slide={cfg.min_contexts_per_slide.get(split)}, but "
                f"only {eligible} of {len(cases)} cases qualify "
                f"({len(taken)} still free at this point). Either the cohort is too "
                f"small or stage 03 froze too few contexts per slide."
            )
        assignment.update({c: split for c in taken})
    for c in order:
        assignment.setdefault(c, "dev")
    return assignment
