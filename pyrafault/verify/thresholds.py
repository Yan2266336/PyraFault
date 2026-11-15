"""Fitting tau and delta on `calibration`, and freezing them for `test`."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np


#: The false-verification rate `tau` must hold a method under. A property of the
#: benchmark, not of a method: every verifier fits its own `tau`, and two taus fitted
#: against different targets do not describe comparable abstention behaviour.
#:
#: 0.01, not the 0.05 this started at, because 0.05 was measured inert. On the degraded
#: lung freeze 79 of 5250 wrong items agreed with their claim within `delta` -- a rate of
#: 1.50% at full admission, a third of the old target -- so `fit_tau` returned the lowest
#: candidate score, nothing ever abstained, and `rho` did not participate in a single
#: decision. A target no method can breach does not calibrate anything; it only makes the
#: three-state decision two-state while appearing to do more.
TARGET_FVR = 0.01


def fit_delta(clean_residuals, percentile: float = 99.0, n_clean: int = 0) -> float:
    """Spatial tolerance: a percentile of the normalised localisation residual over
    calibration's clean items, per the task formulation.

    `n_clean`, when given, is the number of clean items the caller saw *before*
    filtering out any with no estimate -- it exists only to make the empty-input error
    name the actual cause, which used to collapse two different situations into one
    "no clean residuals" message: no clean items in the split at all, versus clean
    items that all failed to localise (`estimated_bbox is None`). Those call for
    different fixes -- pick a split with clean items, versus fix the verifier -- so a
    caller diagnosing this needs to know which one happened.
    """
    r = list(clean_residuals)
    if not r:
        if n_clean:
            raise ValueError(
                f"no clean residuals to fit delta on: {n_clean} clean item(s) in the "
                "split, but every one had estimated_bbox=None"
            )
        raise ValueError("no clean residuals to fit delta on: no clean items in the split")
    return float(np.percentile(r, percentile))


_RESIDUAL_PERCENTILES = (50.0, 75.0, 90.0, 95.0, 99.0)


def residual_report(clean_residuals) -> dict:
    """The residual distribution `delta` (one percentile) reduces to one number.

    Spec S3.4: "sensitivity over other percentiles is reported" -- nothing recorded
    that before this. `delta` itself keeps meaning whatever `fit_delta` returns (the
    p99, unless a caller deliberately varies `percentile`); this is purely additional
    recording; it does not change the calibration rule.
    """
    r = list(clean_residuals)
    out: dict = {"n": len(r)}
    for p in _RESIDUAL_PERCENTILES:
        out[f"p{int(p)}"] = float(np.percentile(r, p)) if r else None
    return out


def fit_tau(scores, is_clean, agrees, target_fvr: float = TARGET_FVR) -> float:
    """Reliability threshold holding false verifications to at most `target_fvr`.

    A false verification is a wrong claim *accepted*: the item cleared `tau` AND its
    estimate agreed with the claim within `delta`. A wrong item that clears `tau` and
    whose estimate disagrees is declared `mismatched` — the correct answer, and a
    success rather than a failure.

    Counting every admitted wrong item as a false verification, as this once did, makes
    a method pay for its own correct rejections. Measured on the LoFTR run: the admitted
    rate at the ceiling of rho was 10.3% while the accepted rate was 0.00%, so the 5%
    target was unreachable at any threshold and every item abstained — a verifier that
    rejects faults perfectly was calibrated into saying nothing at all.

    `agrees[i]` must be computed against the already-fitted `delta`; fit delta first.

    `target_fvr` defaults to the benchmark's `TARGET_FVR` and stays an argument so a
    sensitivity analysis over it is a call rather than an edit.
    """
    scores = np.asarray(list(scores), dtype=float)
    is_clean = np.asarray(list(is_clean), dtype=bool)
    agrees = np.asarray(list(agrees), dtype=bool)
    if not (len(scores) == len(is_clean) == len(agrees)):
        raise ValueError(
            f"scores, is_clean and agrees must be the same length; got "
            f"{len(scores)}, {len(is_clean)}, {len(agrees)}"
        )

    wrong_mask = ~is_clean
    if not wrong_mask.any():
        raise ValueError(
            "no faulty items to fit tau on: every calibration item's claim is correct, "
            "so there is no false-verification rate to bound. A tau fitted here would "
            "admit everything and the verifier would never abstain."
        )

    n_wrong = int(wrong_mask.sum())
    # Only wrong items that would be ACCEPTED can constitute a false verification.
    accepted = scores[wrong_mask & agrees]

    def rate(tau: float) -> float:
        return float((accepted >= tau).sum()) / n_wrong

    if accepted.size == 0:
        # Nothing this method admits is ever accepted wrongly, so no abstention is
        # required of it. Return a threshold below every score rather than -inf, which
        # is not representable in strict JSON.
        return float(scores.min() - 1e-9) if scores.size else 0.0

    for tau in np.unique(np.concatenate([accepted, [accepted.max() + 1e-9]])):
        if rate(float(tau)) <= target_fvr:
            return float(tau)
    return float(accepted.max() + 1e-9)


@dataclass(frozen=True)
class Thresholds:
    tau: float
    delta: float
    fitted_on: str
    n_items: int
    #: `residual_report`'s output: `{"n": ..., "p50": ..., "p75": ..., "p90": ...,
    #: "p95": ..., "p99": ...}` over calibration's clean residuals. Required, not
    #: defaulted: a `thresholds.json` written before this field existed must fail to
    #: load rather than silently report an empty sensitivity analysis that was never
    #: measured. This is a deliberate breaking change to the schema (see the commit
    #: introducing this field) -- `digest()` changes too, since it hashes every field.
    residuals: dict

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "Thresholds":
        """Read a thresholds file.

        A file written before `residuals` existed is refused by name rather than
        loaded with an empty block. Silently supplying one would hand the caller a
        `Thresholds` that looks complete and reports no residual distribution, which is
        indistinguishable from a fit whose residuals were genuinely degenerate -- and
        the residual block is the only record of the distribution `delta` was drawn
        from. Refitting is cheap next to misreading a sensitivity report that is not
        there.
        """
        raw = json.loads(Path(path).read_text())
        known = {f.name for f in fields(cls)}
        missing = known - set(raw)
        if missing:
            raise ValueError(
                f"{path} is missing {sorted(missing)}. It was written before those "
                "fields existed; refit with `scripts/13_run_verifier.py "
                "--fit-thresholds` rather than editing the file, so the values and the "
                "distribution they came from stay consistent."
            )
        # Keys that are not fields are dropped, not refused. A file may carry
        # provenance a caller has no business in -- which tau policy it encodes, a note
        # on why -- and `**raw` sent those straight into a frozen dataclass: one extra
        # key raised TypeError inside the MCP server's startup, the server never bound
        # its port, and 180 agent episodes came back void with nothing but a port
        # number to say why. Dropping them cannot hide a misspelled field, because a
        # field spelt wrong is also a field absent, and the check above has already
        # fired. `extras` is attached for inspection and deliberately not a dataclass
        # field, so `digest()` keeps naming the numbers that decide and nothing else.
        th = cls(**{k: v for k, v in raw.items() if k in known})
        object.__setattr__(th, "extras", {k: v for k, v in raw.items()
                                          if k not in known})
        return th

    def digest(self) -> str:
        """Recorded on every prediction, so a scored run names its calibration."""
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()
