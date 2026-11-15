# The verifier protocol

## 1. Purpose

Every verifier — ZoomTrace and every baseline that follows it — is run and scored
through this one harness: `pyrafault/verify/base.py`, `pyrafault/verify/runner.py`,
`pyrafault/verify/metrics.py` and `pyrafault/verify/thresholds.py`. That is what makes
two methods' numbers comparable: neither the run nor the scoring step is bespoke per
method, so a difference in the reported metrics reflects a difference in localisation,
not a difference in how it was measured. This document is the contract those four
modules implement, written so that an author who never opens this repository can build
a conforming `predictions.parquet` and get numbers that mean the same thing ZoomTrace's
do. If this document and the code ever disagree, the code is authoritative — but that
disagreement is a bug in this document, and should be reported as one.

## 2. Getting started

Everything below assumes you can load the benchmark. This section is the shortest path
from a clone to a verifier of your own that runs.

**Environment.** `environment.lock.yml` is the exact solve this project's results were
produced with; `environment.yml` is the loose spec to start a fresh one from.

```bash
conda env create -f environment.lock.yml -p /path/to/your/env
export PYRAFAULT_PY=/path/to/your/env/bin/python
$PYRAFAULT_PY -c "import openslide, timm, torch; print('ok')"
```

Your own method may need nothing beyond NumPy — the worked skeleton in §11 does not.

**Loading items.** `PyraFaultDataset` is what turns the frozen coordinate tables into
the `inputs` mapping §4 describes. It needs the benchmark directory and the root your
config's `source:` points at, because the images are read from the slides at the frozen
coordinates:

```python
from pyrafault.dataset import PyraFaultDataset

with PyraFaultDataset("data/01_benchmark", "data/00_source/gdc_download",
                      split="test", materialize=True) as ds:
    item = ds[ds.item_ids[0]]
    inputs = item["inputs"]     # what your localise() receives
    target = item["target"]     # the answer key — do not read it while deciding
```

`materialize=True` serves cached PNGs where they exist and reads the slide otherwise;
both paths produce identical bytes. `ds.item_ids` is the full split, in a stable order.

**Running your verifier over a split.** Once `localise` works on one item, fit your
thresholds on `calibration`, freeze them, and decide `test` against them (§9):

```python
from pyrafault.verify.runner import fit_thresholds, run_verifier

with PyraFaultDataset("data/01_benchmark", "data/00_source/gdc_download",
                      split="calibration", materialize=True) as cal:
    thresholds = fit_thresholds(MyVerifier(), cal, "calibration")
thresholds.save("thresholds.json")

with PyraFaultDataset("data/01_benchmark", "data/00_source/gdc_download",
                      split="test", materialize=True) as ds:
    predictions = run_verifier(MyVerifier(), ds, thresholds, audit_independence=True)
predictions.to_parquet("predictions_test.parquet", index=False)
```

Or from the command line. `scripts/13_run_verifier.py` takes any importable `Verifier`
subclass as `--verifier module:Class`, builds it with no arguments, and runs it through
the same fitting, deciding and writing code ZoomTrace goes through:

```bash
V=my_package.my_verifier:MyVerifier          # must be importable (PYTHONPATH or installed)

python scripts/13_run_verifier.py --config configs/example10.yaml --data-root data \
    --verifier $V --split calibration --fit-thresholds --audit-independence
python scripts/13_run_verifier.py --config configs/example10.yaml --data-root data \
    --verifier $V --split test
python scripts/14_score_verifier.py --data-root data --split test \
    --predictions data/06_verify/<verifier name>/predictions_test.parquet
```

Output goes to `06_verify/<name>/`, keyed by your class's `name`. A verifier that needs
configuration reads it itself — the harness passes no method-specific arguments, which is
what keeps the command line the same for every method. To watch the whole path work before
writing any code, use `V=pyrafault.verify.ncc_baseline:NccBaseline`: the §11 skeleton,
shipped as a module. Thresholds are per-method: fit your own, never reuse another
verifier's.

**Checking your work before you submit.** Two things are worth doing locally:

- Run with `audit_independence=True` at least once. It is off by default, and it is the
  only mechanical check that your `localise` does not read the claim (§3).
- Score your own predictions with `scripts/14_score_verifier.py` (§10) before handing
  them over. The scorer raises on a submission whose item set does not match the split,
  and it is better to find that yourself.

**If your method needs per-context precomputation.** ZoomTrace caches a grid of window
descriptors per context, built by `scripts/12_build_window_bank.py` and stored under
`06_verify/window_bank/`. Nothing in the protocol requires or forbids this — `localise`
is called per item and may consult anything it precomputed, as long as the precomputation
does not depend on the claim. If you build one, keep it out of `06_verify/window_bank/`,
which belongs to ZoomTrace.

**A reference implementation.** `pyrafault/verify/zoomtrace.py` is a complete verifier
in about 140 lines. Read it if you want to see the interfaces used in anger rather than
in the skeleton.

## 3. What you implement

A `Verifier` subclass (`pyrafault/verify/base.py`) with one abstract method:

```python
def localise(self, inputs: Mapping) -> Localisation:
    ...
```

`localise` never receives the claim. `inputs` carries a field named `claimed_bbox`, and
implementations must not read it: the task defines the estimate `ℓ̂` as a function of
the context `C` and the query `Q` alone, independent of what the claim `ℓ` says. A
method that peeks at `claimed_bbox` is not verifying the claim — it is echoing it, and
would trivially score perfectly without doing the task.

This is not merely stated; the run harness *can* audit it, but only when asked.
`run_verifier`'s `audit_independence` flag defaults to `False`
(`pyrafault/verify/runner.py`), and `scripts/13_run_verifier.py` only sets it when
invoked with `--audit-independence` — without that flag, nothing checks this property
at run time. When the flag is on, `run_verifier` computes `localise(inputs)`, then
recomputes it for the same `(C, Q)` under a perturbed `claimed_bbox` — shifted by a
large, data-independent offset — and asserts that `estimated_bbox` and `reliability`
agree between the two calls. A verifier that sets `reads_claim = True` on itself is
refused the audit rather than failed by it — the one admitted exception, used for the
claim-conditioned baseline that exists to measure what independence buys; declaring it
also puts the claim in the harness's dedupe key, without which two items differing only
in their claim share one answer. A submission to the benchmark does not set it.
(Only those two fields: `diagnostics` is not compared and
may differ.) It repeats the check on every cache hit too, using the *other* real claim
that produced the collision, so a `(C, Q)` pair that recurs across items in the split is
audited against a genuine second claim as well as a synthetic one. A verifier run with
the audit on that reads `claimed_bbox` anywhere in `localise` fails this assertion, not
silently. A submitter who never runs this harness — building `predictions.parquet`
outside this repository (§10) is exactly that path — gets no audit at all; do not rely on
being caught. The independence requirement in the paragraph above binds regardless of
whether it is checked.

## 4. What arrives

`inputs` carries exactly these ten fields — nothing else is available, and nothing
here is optional. This mirrors README's "What a verifier receives" exactly; if the two
ever diverge, that is a bug in one of them.

| field | type | meaning |
|---|---|---|
| `context` | `PIL.Image`, RGB, `context.size_px`² (2048² shipped) | the low-magnification context `C` |
| `query` | `PIL.Image`, RGB, `query.size_px`² (512² shipped) | the high-magnification patch `Q` |
| `context_bbox` | `BBoxL0(x, y, w, h)` | where `C` sits on the slide, in level-0 pixels |
| `claimed_bbox` | `BBoxL0(x, y, w, h)` | the claimed source `ℓ`, level-0, always inside `context_bbox` — must not be read by `localise` (§3) |
| `context_mpp` | `float`, µm/px | `mc` |
| `query_mpp` | `float`, µm/px | `mq`; `mc / mq` is the scale factor for aligning `C` to `Q` |
| `slide_id` | `str` | which slide `C` is on |
| `item_id` | `str`, sha256 hex | opaque handle; carries no information about the label |
| `split` | `str` | `dev` / `calibration` / `test` |
| `extraction_version` | `str` | how the pixels were produced |

All four bbox numbers are level-0 pixels of `slide_id`, so `claimed_bbox` and
`context_bbox` are directly comparable, and a level-0 box converts to a position inside
the `context` image as `(box.x - context_bbox.x) / context_bbox.w * context.width`.

**`query` may be degraded, and you are not told whether it was.** The served `Q` carries
a frozen per-row imaging degradation -- a rotation about its centre (`none` 0°, `mild` 5°,
`moderate` 10°, `severe` 15°) plus a small scale error -- so that locating `Q` inside `C`
is not solvable by grayscale template matching alone. The condition is recorded in
`pairs.degradation_band`, which is in neither `inputs` nor `target`: the pixels are the
only evidence, by design. The parameters are frozen into the table and replayed, never
redrawn, so the pixels are reproducible bit-for-bit; a `none` row is the identity
transform and is byte-identical to a build that never knew about degradation (all three
freezes shipped in this repository are entirely `none`). Results are broken out per band
by the scorer -- see `by_degradation_band` in §8.

## 5. What you return

`localise` returns a `Localisation` (`pyrafault/verify/base.py`):

```python
Localisation(
    estimated_bbox: BBoxL0 | None,   # ℓ̂ — level-0 pixels of slide_id; None only
                                      # when the method declines to localise at all
    reliability: float,              # thresholded against a per-method tau
    diagnostics: Mapping[str, Any] = {},
)
```

`reliability` may use any monotone scale. It is thresholded against a `τ` fitted per
method on `calibration` (§7, §9), and it is never compared *between* methods — only
against that method's own `τ`. An unnormalised cross-correlation peak and an empirical
tail probability against a null distribution are equally admissible values for it; there
is no requirement that it be bounded, calibrated as a probability, or comparable across
verifiers. What matters is that within one method, a higher `reliability` means "this
estimate is more trustworthy."

## 6. What the harness produces

`run_verifier` turns every item's `Localisation` into one row of
`pyrafault/verify/runner.py`'s `PREDICTION_COLUMNS`:

| column | type |
|---|---|
| `item_id` | `str` |
| `estimated_bbox_l0_x` | `int`, nullable (`NaN`/`None` when `estimated_bbox is None`; on disk this upcasts the whole column to `float64` the moment any row is null, so a reader must handle a `float64` column here, not a literal `int` one) |
| `estimated_bbox_l0_y` | `int`, nullable |
| `estimated_bbox_l0_w` | `int`, nullable |
| `estimated_bbox_l0_h` | `int`, nullable |
| `reliability` | `float` |
| `decision` | `str`, one of `"verified"`, `"mismatched"`, `"unverifiable"` |
| `verifier_name` | `str` |
| `verifier_revision` | `str` |
| `thresholds_digest` | `str`, sha256 hex — identifies which `(τ, δ)` produced `decision` |
| `runtime_ms` | `float` — cost of *computing* this answer; repeated verbatim on a cache hit for the same `(context, query)`, so summing over distinct `(slide_id, context_bbox, true_slide_id, true_bbox, degradation_digest)` keys gives total compute, not wall time per row. `degradation_digest` is part of that key and not an afterthought: the imaging difference is drawn per item, so two items can share the slide, the context and `true_bbox` and still be served a 5° rotation and a 15° one. A harness that keys without it serves the second item the first's localisation — silently, with one estimate per item and no error anywhere. It is in `target` for exactly this reason, alongside `true_slide_id`, which is there for the same kind of collision. |

A conforming submission is a parquet file with exactly these columns, one row per
`item_id` in the split being scored.

## 6b. Running it at full scale

One process cannot evaluate a full freeze. ZoomTrace localises at about 240 ms per
distinct `(C, Q)` and the NCC control at about 360 ms, so the lung freeze's 143,000
calibration and 155,080 test pairs are 20 to 30 hours per method before counting the
pixel reads — past the wall clock of most queues, and twice over because each method
fits its own thresholds.

Nothing about the task is sequential. `localise` sees one `(C, Q)` and the decision is a
pure function of `(localisation, claim, τ, δ)`, so stage 13 splits at that seam:

```bash
# the expensive half, one array job per shard
sbatch --array=0-19%20 ... scripts/13_run_verifier.py --data-root data_lung \
    --split calibration --localise-only --shard "$SLURM_ARRAY_TASK_ID" --n-shards 20
# the cheap half, once, over every shard on disk
python scripts/13_run_verifier.py --data-root data_lung --split calibration \
    --from-localisations --fit-thresholds
```

`--localise-only` writes `LOCALISATION_COLUMNS` — `PREDICTION_COLUMNS` minus `decision`
and `thresholds_digest`, which cannot exist yet — to
`06_verify/<method>/localisations_<split>/<shard>of<n>.parquet`. `--from-localisations`
reads every such file, fits if asked, decides, and writes the same
`predictions_<split>.parquet` an unsharded run would. That equality is a test, not a
claim (`tests/test_run_verifier_sharded_cli.py`).

Three properties of the partition are worth knowing if you reimplement it:

- **It is by context, not by item.** Items sharing a served patch share a context, so
  grouping by `context_id` keeps the harness's `(C, Q)` dedupe intact across shards —
  and lets a ZoomTrace shard read only its own contexts' window banks instead of all
  12.8 GB of them.
- **The groups are sorted before striding**, so shard *k* means the same items on every
  job of the array and on a rerun.
- **Gaps are refused, not filled.** Deciding with any item missing a localisation exits
  with that item's id rather than writing a predictions table over part of the split,
  and two shards that overlap are refused the same way. An incomplete evaluation whose
  every row is individually correct is the failure mode here.

`--audit-independence` belongs on the `--localise-only` half, where `localise` actually
runs; passing it to the deciding half is refused rather than silently skipped.

## 7. The decision rule

The three values of `DECISIONS` (`pyrafault/verify/base.py`) are `"verified"`,
`"mismatched"` and `"unverifiable"`. `unverifiable` is the method abstaining — it is not
a third ground-truth class, and no benchmark record carries it as a label.

`decide(loc, claimed_bbox, query_side, tau, delta)` applies, in order:

```python
if loc.reliability < tau:
    return "unverifiable"
if loc.estimated_bbox is None:
    return "mismatched"
d = math.hypot(loc.estimated_bbox.cx - claimed_bbox.cx,
               loc.estimated_bbox.cy - claimed_bbox.cy)
return "verified" if d <= delta * query_side else "mismatched"
```

So: below the reliability threshold `τ`, abstain. At or above it, but with no estimate
at all, that is a confident rejection — `mismatched`, not an abstention, because
`reliability` is already above `τ`. At or above it with an estimate, compare the centre
distance between `ℓ̂` and `ℓ` against `δ · query_side`: within tolerance is `verified`,
outside is `mismatched`. `δ` is expressed in units of the query side because that is how
it is fitted (§9) — as a percentile of the *normalised* localisation residual, i.e. the
raw pixel residual divided by `true_bbox.w`.

## 8. Scoring

`score_predictions(predictions, pairs, item_ids=None)` (`pyrafault/verify/metrics.py`)
returns a dict with exactly these keys — nothing else is in `metrics.json`:

| key | what it holds |
|---|---|
| `n_items` | number of scored rows |
| `coverage` | fraction of rows with `decision != "unverifiable"` |
| `decisions` | counts of each `decision` value |
| `balanced_accuracy` | over covered rows, `decision == "verified"` vs. `claim_correct`; `None` if all covered rows share one `claim_correct` value |
| `verification_score` | `{"mean_clean": …, "mean_fault": …, "auroc": …, "auprc": …}` — the mean of `verification_score(ρ, ℓ̂, ℓ)` (below) split by `claim_correct`, plus AUROC and AUPRC of the same per-item scores against `claim_correct` (`None` in the degenerate single-class case) |
| `localisation` | `{"n", "median_normalised_centre_error", "median_iou"}`, computed on items whose `true_bbox` lies inside `context_bbox` |
| `by_fault_type` | per-`fault_type` `n`, `coverage`, `verified_rate` |
| `by_scale_ratio` | per-(`scale_ratio`, `context_mpp`) `n`, `verified_rate` — both, not `scale_ratio` alone: 5×→20× and 10×→40× share a ratio of 4.0 but differ in `context_mpp`, and keying on the ratio alone would pool two strata the calibration is meant to keep apart |
| `by_degradation_band` | per-`degradation_band` `n`, `coverage`, `verified_rate`, `balanced_accuracy` (`None` for a band holding one class only) — the imaging-degradation dimension (§4). Empty `{}` for a freeze whose `pairs` table predates the column; `{"none": …}` for the freezes shipped here, all of whose rows are the identity transform. A method cannot report this for itself: the band is in neither `inputs` nor `target`, so only the scorer, holding `pairs`, can cut the results this way — which is exactly why a pooled number over a multi-band freeze hides the dimension it was built to measure |

`verification_score` is the per-item quantity those means are built from:

```
verification_score(ρ, ℓ̂, ℓ) = ρ · IoU(ℓ̂, ℓ)      (0.0 when ℓ̂ is None)
```

`ρ` alone will not do: it is high whenever the localisation is *discriminative* —
concentrated, confident, low-entropy — which says nothing about whether the estimate
agrees with the claim. A verifier could be highly reliable about a wrong answer. Scaling
by `IoU(ℓ̂, ℓ)` makes the score depend on both quantities that should matter: how sure
the method is, and whether it agrees with what was claimed. The product is parameter-free
and monotone in each factor. It is the continuous quantity `score_predictions` computes
AUROC and AUPRC on: AUROC via the rank-sum (Mann-Whitney U) identity, AUPRC via the
step-wise precision-recall sum, both numpy-only and both against `claim_correct`. Either
is `None`, not a number, when the scored set is single-class — AUROC and AUPRC are
undefined there, not zero. A risk–coverage curve is not computed; only the table above.

Abstentions (`decision == "unverifiable"`) are excluded from the accuracy-style metrics
— `balanced_accuracy` is computed only over items with a decision — but every item,
abstained or not, enters `coverage` (the fraction of items with a non-abstaining
decision). An abstention is not a wrong answer and is not a right one; it is a statement
that the method chose not to answer, and it must be visible as that rather than dropped
from the denominator.

## 9. The two rules

Two properties of the harness — not of the data — are what make results from different
runs, and different methods, comparable.

1. **Score per item.** Each row's output is a function of its own
   `(context, query, claimed_bbox, …)` alone. Tensor batching for throughput is fine:
   under `eval()`, with no cross-sample operation in the model, a batch of N is
   arithmetically identical to N separate forward passes over N single items — the
   run harness itself caches and dedupes by `(context, query)` key but never lets one
   item's score depend on another's. What is not fine is *pooling*: a scoring rule that
   reads other items sharing the same context. `context_bbox` is visible in `inputs` —
   the ~40 pairs a context contributes all share it — and a rule that looks at how the
   drifted-versus-clean items are distributed within one context can recover
   `claim_correct` without reading a single pixel. Per-item scoring is what closes that
   residual; pooling reopens it.
2. **Fit `τ` and `δ` on `calibration`, freeze them for `test`.** `fit_tau` and
   `fit_delta` (`pyrafault/verify/thresholds.py`) both take only `calibration`'s scores.
   Estimating either threshold on `test` would use the evaluation set to tune the
   evaluation — the number reported would no longer measure generalisation. `dev` is for
   development, `calibration` is for fitting exactly these two numbers, and `test` is
   the split the number you report comes from.

## 10. Submitting without this repository

You do not need this repository's code to be scored, only its interfaces. Write a
`predictions.parquet` with exactly `PREDICTION_COLUMNS` (§6), one row per `item_id` of
the split you ran, and score it with:

```bash
python scripts/14_score_verifier.py \
    --predictions path/to/predictions_test.parquet \
    --data-root data \
    --split test
```

`--split` matters. Predictions are produced one split at a time
(`predictions_<split>.parquet`), while the archived `pairs.parquet` spans every split.
`score_predictions(predictions, pairs, item_ids=None)` computes the expected `item_id`
set from `pairs["item_id"]` (or from `item_ids` when given) and **raises** —
`ValueError`, not a warning — if the submitted set differs from it, in either
direction: an item present in `predictions` but not expected, or an item expected but
missing from `predictions`, both fail the call. This is deliberate: a merge that
silently dropped the missing items would still report `coverage == 1.0` and a plausible
`balanced_accuracy`, because every metric is computed only over the rows that matched. A
partial submission would then read as a complete run instead of a broken one. So —
before you build your predictions file — decide which split you are scoring and pass
`--split` (or pre-filter `--pairs` to that split yourself); scoring a per-split
predictions file against the full, unfiltered `pairs.parquet` raises, not silently
scores a subset.

`scripts/14_score_verifier.py` reads no verifier code — only `predictions.parquet` and
the benchmark's `pairs.parquet` — so this path works from outside this repository
entirely, as long as your predictions file conforms to §6.

When you deliberately want to score fewer items than a full split — a debugging run, a
resubmission that only covers the items that previously failed — `item_ids` is the
sanctioned way to do it: call `score_predictions(predictions, pairs, item_ids=my_subset)`
directly with the exact set you intend to score, so that set (not `pairs["item_id"]`)
becomes "expected" and the raise above checks your submission against it rather than
against the whole split. `scripts/14_score_verifier.py` does not expose an `item_ids`
flag — from the CLI, restrict the expected set with `--split`, or point `--pairs` at an
already-filtered table; scoring an intentional subset by any other means (e.g.
hand-editing `pairs.parquet`) is not the sanctioned path.

## 11. A worked skeleton

A complete, runnable normalised-cross-correlation baseline — no learned features, no
external dependency beyond NumPy and Pillow. It downsamples `Q` to `C`'s physical
resolution using `context_mpp`/`query_mpp`, computes NCC of the resampled `Q` against
the full-resolution `C` at every pixel offset (by FFT, well under a second for a 2048²
context), and reports the peak correlation as `reliability`.

It ships as `pyrafault/verify/ncc_baseline.py`, class `NccBaseline`, and
`--verifier pyrafault.verify.ncc_baseline:NccBaseline` runs it unmodified. **Start your own
method from this file, not from `ZoomTrace`.** ZoomTrace's constructor takes a window bank, a
feature extractor and a matcher that stage 13 assembles for it, and its `name` is reserved
(§2), so building on that template means starting inside assembly code that has nothing to do
with your method. This one has a no-argument constructor, needs no GPU and no prebuilt
artefact, and is 69 lines.

Four things in `localise` are worth copying verbatim, because each is a way to be wrong that
raises nothing:

- **Downsample `Q` by `context_mpp / query_mpp`.** Skip it and the two images are at different
  scales; everything after is measuring the wrong thing.
- **Return `Localisation(None, 0.0)` when there is no answer** — never a guessed box, never an
  exception. A null estimate is a legal answer the decision rule reads as a confident rejection
  (§7).
- **Convert the peak back to slide level-0 coordinates** through `context_bbox` and
  `ctx_bbox.w / context.width`. Your peak is a pixel offset inside the context image; what you
  return is in the slide's frame. Getting this wrong reads as "localisation is wrong everywhere
  but confident", which is hard to attribute to coordinates.
- **`reliability` increases with confidence.** The decision rule assumes it. NCC is in
  `[-1, 1]`, which is why the last line rescales it; a score where lower is better (a distance,
  say) has to be inverted by you.

Every offset, not a stride: a first version of this skeleton slid the template in steps
of a quarter of its side and, on a ten-slide freeze, localised essentially nothing (median
IoU 0.0 on test). The coordinates were right — NCC at the true source was 0.78–0.95 on
the items checked — but query origins are jittered to arbitrary pixels, and on tissue the
correlation drops to about zero 16 px away from the source, so a coarse grid never
sampled the peak. Any template-matching baseline built from this one inherits that
constraint.

```python
"""A minimal verifier: normalised cross-correlation of a downsampled Q over C."""
from __future__ import annotations

import numpy as np
from PIL import Image

from pyrafault.geom import BBoxL0
from pyrafault.verify.base import Localisation, Verifier


def _gray(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float64)


def _ncc_map(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """NCC of `template` at every valid offset of `image`, stride 1.

    Stride 1 is not an optimisation detail. A true source sits at an arbitrary pixel
    offset, and on tissue the correlation falls to about zero a few pixels away from
    it -- a coarser stride usually never samples the peak at all.
    """
    th, tw = template.shape
    t = template - template.mean()
    # Sum of (window * zero-mean template) at every offset, by FFT convolution with
    # the flipped template.
    k = np.zeros_like(image)
    k[:th, :tw] = t[::-1, ::-1]
    full = np.real(np.fft.ifft2(np.fft.fft2(image) * np.fft.fft2(k)))
    cross = full[th - 1:, tw - 1:]
    # Per-window variance from integral images.
    def window_sum(a):
        s = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        return s[th:, tw:] - s[:-th, tw:] - s[th:, :-tw] + s[:-th, :-tw]
    n = th * tw
    s1 = window_sum(image)
    var = np.maximum(window_sum(image ** 2) - s1 ** 2 / n, 0.0)
    denom = np.sqrt(var * (t ** 2).sum())
    return np.divide(cross, denom, out=np.zeros_like(cross), where=denom > 1e-6)


class NccBaseline(Verifier):
    name = "ncc_baseline"
    revision = "2"

    def localise(self, inputs) -> Localisation:
        context, query = inputs["context"], inputs["query"]
        scale = inputs["context_mpp"] / inputs["query_mpp"]  # mc / mq
        side = max(1, round(query.size[0] / scale))
        c_arr = _gray(context)
        t_arr = _gray(query.resize((side, side), Image.LANCZOS))

        if c_arr.shape[0] < side or c_arr.shape[1] < side:
            return Localisation(estimated_bbox=None, reliability=0.0)

        scores = _ncc_map(c_arr, t_arr)
        y, x = np.unravel_index(int(np.argmax(scores)), scores.shape)
        best = float(scores[y, x])

        ctx_bbox = inputs["context_bbox"]
        px_scale = ctx_bbox.w / context.width
        est = BBoxL0(
            int(round(ctx_bbox.x + x * px_scale)),
            int(round(ctx_bbox.y + y * px_scale)),
            int(round(side * px_scale)),
            int(round(side * px_scale)),
        )
        # NCC is in [-1, 1]; rescale to [0, 1] so `reliability` behaves monotonically
        # with the same sign convention the decision rule assumes (higher is better).
        return Localisation(estimated_bbox=est, reliability=(best + 1.0) / 2.0)
```

This exact code ships as `pyrafault/verify/ncc_baseline.py` — `tests/test_ncc_baseline.py`
fails if the page and the module ever differ, and runs the module through
`fit_thresholds`, `run_verifier` with the audit on, and `score_predictions`.

This is a baseline, not a competitive method: a single grayscale template with no
rotation or scale search will lose to anything feature-based on hard items. It exists to
show the minimal shape a `Verifier` must have — read `inputs`, never read
`claimed_bbox`, return a `Localisation` — and to give an author something to run end to
end before building their real method.

## 12. The caveat

**What `τ` bounds.** A *false verification* is a wrong claim **accepted** — the item
cleared `τ` and its estimate agreed with the claim within `δ`. A wrong item that clears
`τ` and whose estimate disagrees is declared `mismatched`, which is the right answer and
a success. `fit_tau` therefore takes the agreement outcome as well as the reliability,
and `δ` is fitted first so that agreement is defined when `τ` is.

**The target is 0.01, and it is the benchmark's, not yours.** `fit_tau` holds the
false-verification rate at or under `pyrafault.verify.thresholds.TARGET_FVR`. Fit your own
`τ`, but fit it against that number: two taus fitted against different targets do not
describe comparable abstention behaviour, and comparing methods is the whole reason fitting
lives in shared code.

It is 0.01 because 0.05 was measured inert. On the degraded lung freeze, 79 of 5,250 wrong
items agreed with their claim within `δ` — 1.50% at full admission, a third of the old
target. `fit_tau` therefore returned the lowest candidate score, nothing ever abstained,
coverage was 1.000, and `ρ` did not participate in a single decision. **A target no method
can breach calibrates nothing**; it only makes the three-state decision two-state while
appearing to do more.

That distinction is not academic. An earlier implementation counted every *admitted*
wrong item as a false verification, which charges a method for its own correct
rejections. Measured on the LoFTR run of a ten-slide freeze: at the ceiling of `ρ` the
admitted rate was 10.3% while the accepted rate was 0.00%, so a 5% target was
unreachable at any threshold, `fit_tau` returned a value above every score, and a
verifier that rejected faults essentially perfectly abstained on 100% of the split.

`calibration` holds two patients in the shipped freezes. A `τ`/`δ` fitted on that many
patients demonstrates that the pipeline — fitting, freezing, scoring — works end to end;
it does not calibrate a method in the sense of producing thresholds that would
generalise to a larger population. Note also that `fit_tau` raises if the calibration
split it is given contains no faulty items: a false-verification rate needs at least one
wrong claim to be measured over, and a `τ` fitted on an all-clean split would admit
everything and the verifier would never abstain. Report numbers from `test` as what they
are — a demonstration of the protocol on the shipped freezes — not as a claim about a
verifier's real-world reliability.

**Check `δ` before reading `balanced_accuracy`.** `thresholds.json` records the clean-item
residual distribution at p50/p75/p90/p95/p99 (`residual_report`,
`pyrafault/verify/thresholds.py`), and `δ` is its p99. When a method's localisation has a
heavy failure tail, that p99 is set by the tail rather than by the typical residual, and a
`δ` that wide makes the spatial test in `decide` (§7) close to vacuous: `d <= δ ·
query_side` then holds for almost any estimate, `decide` reduces to "did `ρ` clear `τ`?",
and `balanced_accuracy` stops measuring whether `ℓ̂` agrees with `ℓ`.

The two shipped matchers fall on opposite sides of this, measured on a ten-slide freeze's
calibration clean items (300 each):

| matcher | p50 | p75 | p90 | p95 | p99 = `δ` | residual > 1 query side |
|---|---|---|---|---|---|---|
| LoFTR | 0.004 | 0.006 | 0.011 | 0.014 | 0.039 | 0.0% |
| SIFT  | 0.004 | 0.911 | 4.06 | 7.97 | 10.73 | 23.7% |

With LoFTR `δ` is under a twentieth of a query side and the spatial test discriminates;
with SIFT it is more than ten query sides and it does not. So open `thresholds.json` and
compare p99 to p50 before trusting `balanced_accuracy`. `verification_score`'s AUROC and
AUPRC (§8) are computed on the continuous score, not on `decision`, and do not depend on
`δ` at all.
