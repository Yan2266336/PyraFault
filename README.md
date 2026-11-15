*English · [简体中文](README.zh-CN.md)*

# PyraFault and ZoomTrace

**Cross-Magnification Provenance Verification (CMPV).** Given a low-magnification context
`C`, a high-magnification query `Q`, and a **claimed** source location `ℓ`, decide whether
`Q`'s pixels really came from `ℓ`, and recover the true source `ℓ*`.

This repository is the code behind the paper: how the TCGA slides are fetched, how the
PyraFault benchmark is constructed from them, how ZoomTrace is built and run, and how every
baseline is run through the same protocol.

**The benchmark data is not here.** The released five-cohort benchmark — 3,057,732
coordinate-supervised examples over 2,920 TCGA diagnostic slides — is published separately
as a dataset. What this repository ships instead is the complete construction and evaluation
code, plus a **ten-slide worked example** that runs the same ten stages, with the same
parameters, on about 13 GB of source pixels. Everything below is written against that
example, so every command can be run as printed.

| | |
|---|---|
| Benchmark construction | `scripts/00_*` … `scripts/09_*`, driven by `configs/example10.yaml` |
| ZoomTrace | `pyrafault/verify/zoomtrace.py`, built by stages 12–13 |
| Baselines | `pyrafault/verify/`, one module each, run through `--verifier module:Class` |
| Third-party submissions | [`VERIFIER_PROTOCOL.md`](VERIFIER_PROTOCOL.md) — the contract, without needing this code |

**The benchmark is coordinates, not pixels.** A record is
`(slide_id, level-0 bbox, target_mpp, output_size, extraction_version)`. The PNGs under
`data/03_materialized/` and the embeddings under `data/04_features/` are derived data —
`rm -rf` either and the benchmark is unchanged. Stage 09 proves this rather than asserting
it: it re-derives a 5% sample from coordinates alone and compares SHA-256.

---

# Running

## Prerequisites, once

```bash
# 1. The environment. Either activate it, or point PYRAFAULT_PY at its python —
#    nothing in this repo hardcodes a path, so use whatever yours is.
conda activate pyrafault                      # or:
export PYRAFAULT_PY=/path/to/your/env/bin/python

"${PYRAFAULT_PY:-python}" -c "import openslide, timm, torch; print('ok')"

# 2. The source slides, wherever `source:` in your config points. For the shipped
#    example that is `data/00_source/gdc_download`; for `mode: directory`
#    it is whatever `slides_dir` names. The ten example slides are 12.87 GB:
python scripts/download_tcga.py          # see "Getting the raw slides" for gdc-client
ls -l data/00_source/gdc_download        # -> the example's GDC download directory

# 3. A HuggingFace token in the environment, for the UNI feature extractor
[ -n "$HF_TOKEN" ] && echo "HF_TOKEN present"
```

If step 1 fails, rebuild the environment — this is the riskiest dependency and worth
proving before anything else:

```bash
sbatch --wait -p day -c 4 --mem=16G -t 180 --wrap \
  "conda env create -f environment.yml -p /path/to/your/env"

# then, so the SLURM templates can find it:
export PYRAFAULT_PY=/path/to/your/env/bin/python
```

## The whole pipeline, one command

```bash
bash slurm/run_all.sh
```

A different cohort is a different config and its own data root — pass both in the
environment rather than editing anything tracked, and the example in `data/` is left alone:

```bash
PYRAFAULT_CONFIG=configs/my_cohort.yaml PYRAFAULT_DATA=data_my_cohort \
  bash slurm/run_all.sh
```

Both default to the example (`configs/example10.yaml`, `data/`) when unset. Keep them together:
pointing a new config at the old data root mixes two cohorts in one set of tables.

That submits stages 00 → 09 as a single `afterok` dependency chain: each stage starts only
if its predecessor succeeded, so a failure stops everything downstream instead of letting
the next stage build on a half-written table. Stages that read pixels (02, 03, 05, 08) run
as arrays of one task per slide; the array size comes from the slide list stage 01 writes,
so it follows the cohort automatically.

Watch it:

```bash
squeue -u $USER
tail -f logs/pf09-qc-*.out          # the last stage — the acceptance gate
```

Resume from a stage (the argument is **zero-padded**, `05` not `5`):

```bash
bash slurm/run_all.sh 05
```

## What success looks like

The chain is done when stage 09 prints:

```
invariants: 0 violation(s)
config provenance: 0 violation(s)
leakage audit (best-threshold balanced accuracy, fit in-sample):
  border_distance: 0.527
  centre_distance: 0.530
  context_mpp: 0.500
  on_grid: 0.500
  query_mpp: 0.500
  ...
leakage: 0 violation(s)
reconstruction: 14 contexts + 50 queries
after reconstruction: 0 violation(s)
QC PASSED
```

`QC PASSED` with **0 violations** is the acceptance criterion. Any violation exits 1 and
the chain stops. The full report lands at `data/05_qc/qc_report.md`.

## Running one stage, or one slide

Every stage script takes `--slides` for single-slide debugging, plus `--config` and
`--data-root`. Those two default to `configs/example10.yaml` and `data/`, so **pass them
explicitly whenever you are not working on the example** — otherwise a debugging run writes
into the frozen benchmark. **Do not use `srun`** — if your shell is already inside a SLURM
allocation, a nested `srun` hangs. Use `sbatch --wait`:

```bash
sbatch --wait -p day -c 4 --mem=32G -t 60 -o logs/%x-%j.out --wrap \
  "PYTHONPATH=$PWD ${PYRAFAULT_PY:-python} scripts/03_build_context_candidates.py \
     --config configs/my_cohort.yaml --data-root data_my_cohort \
     --slides TCGA-55-8094-01Z-00-DX1"
```

**Stages 03, 05 and 08 run in two phases**, and this catches people out. Without `--gather`
a stage does the per-slide phase only: it writes shards under `05_qc/stage03/` (etc.),
prints a normal-looking line per slide, and **exits 0 without producing the stage's table**.
The next stage then fails on a missing file, pointing at the wrong stage. Run the gather as
its own invocation once every slide is done — this is what `slurm/run_all.sh` submits as a
separate `afterok` job:

```bash
PYTHONPATH=$PWD ${PYRAFAULT_PY:-python} scripts/03_build_context_candidates.py \
  --config configs/my_cohort.yaml --data-root data_my_cohort --gather
```

`pyrafault` is not pip-installed, so a bare script run needs `PYTHONPATH=$PWD`. The SLURM
templates in `slurm/` export it for you.

Stage 05 needs a GPU:

```bash
sbatch --wait -p gpu --gpus=1 -c 4 --mem=48G -t 240 -o logs/%x-%j.out --wrap \
  "PYTHONPATH=$PWD ${PYRAFAULT_PY:-python} scripts/05_sample_visual_diversity.py \
     --config configs/my_cohort.yaml --data-root data_my_cohort --slides <slide_id>"
```

---

# Where the source data goes

`source.gdc_dir` in `configs/example10.yaml` points at **one directory that contains both the
slides and the GDC export files** — that is how `gdc-client` writes a cart download, and
stage 00 resolves the three exports relative to it. The reference layout:

```
data/00_source/
├── gdc_download/                        # INPUT — source.gdc_dir points here
│   ├── gdc_manifest.<date>.txt          #   the three GDC exports, at the top level
│   ├── gdc_sample_sheet.<date>.tsv
│   ├── metadata.cart.<date>.json
│   ├── 0c5b9279-…-5c21d73ef417/         #   one directory per file, named by its GDC uuid
│   │   └── TCGA-55-8094-01Z-00-DX1.<uuid>.svs
│   ├── 20303b02-…-b83bec2187dd/
│   │   └── TCGA-55-6986-01Z-00-DX1.<uuid>.svs
│   └── …
└── source_metadata/                     # OUTPUT — stage 00 archives the three exports here
```

`source_metadata/` is written by the pipeline, not by you. Do not put anything there.

**Two ways to arrange it, both giving the same layout:**

*Data inside the project* — download straight into place:

```bash
gdc-client download -m <manifest>.txt -d data/00_source/gdc_download
```

*Data elsewhere* — keep the slides on scratch, fast storage or a shared read-only mount, and
either symlink them into the same position:

```bash
ln -sfn /scratch/you/gdc_download_20260812_183036 data/00_source/gdc_download
```

or skip the symlink and give the config an absolute path:

```yaml
source:
  gdc_dir: /scratch/you/gdc_download_20260812_183036
```

The pipeline never writes into this directory — it opens the slides read-only and copies only
the three small export files into `source_metadata/`. So it is safe to point at a shared or
read-only location.

## If your slides are not from GDC

Stages 01–09 have no GDC dependency at all. They read only `slide_id`, `case_id` and
`source_relpath` from `slides.parquet`; the magnification, dimensions and pyramid levels come
from stage 01 opening the file with OpenSlide, so any format OpenSlide reads works — SVS,
NDPI, MRXS, TIFF, DICOM WSI.

Stage 00 is the only stage that is source-specific, and it is a pluggable **adapter**, not a
fixed foundation: its one irreplaceable job is producing those three columns of
`slides.parquet`, and `source.mode` selects how. Two adapters ship today:

| `source.mode` | needs | what it does |
|---|---|---|
| `gdc` (default) | `source.gdc_dir` + the three GDC exports | exactly what it always did: join manifest × sample sheet × cart metadata, audit every file's size and MD5 against what GDC published, abort on any mismatch |
| `directory` | `source.slides_dir`, `source.pattern`, `source.case_id_from` | walk a directory for files matching `pattern`, derive `slide_id` from each filename, resolve `case_id` by the configured rule, and record a sha256 it computes itself |

`mode` defaults to `gdc` when absent, so `configs/example10.yaml` did not need to change and its
digest is untouched — see `configs/example_directory.yaml` for a complete, working
`directory`-mode config with every field commented.

`slide_id` is the filename with its final extension stripped, and it is the identity every
later stage, filename and record uses. It is derived, not configurable — so it is worth a
glance at your filenames before the first run. Note the two adapters need not agree: run the
example's own GDC download through `directory` mode and `slide_id` comes out as
`TCGA-44-2655-01Z-00-DX1.ee255271-...` (the whole stem, UUID included) where `gdc` mode
strips the UUID. Both are stable and unique; they are simply different names for the same
slide, so a benchmark built through one adapter is not comparable slide-for-slide with one
built through the other. Pick an adapter per cohort and stay with it.

That difference is narrower than it sounds, and worth knowing exactly. Running the example's
ten slides through `directory` mode reproduces the context layer *exactly* — the same 187
contexts from the same 429 candidates, and the same case-to-split assignment — because
context selection depends only on the tissue mask and the configured geometry. The query
layer does differ (20,169 candidates versus the frozen run's 20,066): query grids are jittered
by `sha256(seed | context_id | axis)` to stop the lattice being derivable (see "Known
limitations"), and `context_id` embeds `slide_id`. So the adapter changes which query
positions get drawn, not how many contexts a slide earns or which split it lands in.

For `mode: directory`, `case_id_from` is the field that most repays care, because splits are
**case-level**: every slide from one patient must land in the same split, and a wrong
`case_id` silently lets one patient's slides appear in both `dev` and `test`, inflating the
evaluation with nothing to detect it. Three forms:

- `case_id_from: "slide_id"` — one slide per patient; `case_id` is a copy of `slide_id`. In
  in this cohort `case_id` is the TCGA patient barcode (`TCGA-44-2655`), which is also the
  first three fields of `slide_id`, so `case_id = slide_id` would *not* be correct there — but
  it is correct whenever your cohort really is one slide per patient.
- `case_id_from: "^(TCGA-[^-]+-[^-]+)"` — a regex with exactly one capture group, applied to
  `slide_id`; the group is the case. A `slide_id` the pattern does not match is a hard error
  naming that slide, never a silent fallback to `slide_id`.
- `case_id_from: "case_map.csv"` — a path to a CSV with `slide_id,case_id` columns. A
  discovered slide missing from the mapping is a hard error naming it.

**Integrity, honestly.** `directory` mode has no external source of truth to check against —
there is no manifest a publisher produced. It records `sha256_local` and `file_size` computed
from the files themselves, which establishes a *baseline* for detecting later change (re-run
stage 00 and compare), not a check against a publisher's record the way `gdc` mode's MD5 audit
is. Do not treat the two as equivalent assurance.

To use it: point `configs/example_directory.yaml` (or your own copy) at your directory, then
run the normal chain from stage 00 (`bash slurm/run_all.sh`, or `PYTHONPATH=$PWD python
scripts/00_ingest_source.py --config configs/example_directory.yaml`).

---

# Running on a new cohort

The pipeline is cohort-agnostic — the example's ten slides are data, not structure. To build
a benchmark from a fresh GDC download:

**1. Point at the new download.** Either repoint the symlink or set `source.gdc_dir`:

```bash
ln -sfn /path/to/gdc_download_YYYYMMDD_HHMMSS data/00_source/gdc_download
```

The three GDC export filenames in `configs/example10.yaml` are globs
(`gdc_manifest.*.txt`, `gdc_sample_sheet.*.tsv`, `metadata.cart.*.json`), so a new cart
download needs no edit there. Zero matches or more than one is an error naming what was
found — never a silent pick.

**2. Set the split sizes** in `configs/example10.yaml`. Either form works:

```yaml
splits:
  dev: 6            # absolute case counts — must sum to the cohort's case total
  calibration: 2
  test: 2
```
```yaml
splits:
  dev: 0.6          # or fractions summing to 1.0 — safer when the size isn't known yet
  calibration: 0.2
  test: 0.2
```

**3. Review the minimum-context floor.**

```yaml
splits:
  min_contexts_per_slide: {test: 15, calibration: 15}
```

This keeps a geometrically starved slide out of the evaluation splits. The floor of 15
suited a cohort whose largest slide yields ~200 eligible contexts; smaller specimens may
need a lower one. If no assignment satisfies it, stage 03 fails and prints the per-slide
counts — that is the signal to lower the floor, not to retry.

**4. Run it.**

```bash
bash slurm/run_all.sh
```

Everything else adapts on its own: the level-0 grid is computed per slide from that slide's
own MPP (the example already mixes several values near 0.25 µm/px, giving different grid sizes), slides
without an MPP are excluded rather than guessed at, and the SLURM array size follows the
slide list.

> **Adding slides re-freezes the whole benchmark.** Context selection, query selection and
> pair construction are global, so old `pair_id`s no longer denote the same records. Plan a
> new cohort as a new freeze, not as an increment.

---

# What you get

The benchmark itself is under `data/01_benchmark/` — under a megabyte, because it stores
coordinates:

| file | contents |
|---|---|
| `pairs.parquet` | the task records: `(C, Q, ℓ, ℓ*)` plus labels and provenance |
| `queries.parquet` | each query's true source bbox `ℓ*` and its extraction spec |
| `contexts.parquet` | each context's bbox, tissue fraction and descriptors |
| `splits.parquet` | case-level split assignment |
| `slides.parquet` | source slides with pyramid and MPP metadata; MD5-audited under `source.mode: gdc`, sha256 baseline under `directory` |
| `*_candidates.parquet` | the full eligible pools — the audit trail for selection and mining |
| `config.yaml` | the exact config that produced this freeze |
| `benchmark_manifest.json` | sha256 and row count of every table above |

Everything else under `data/` is derived and deletable: `02_cache/`, `03_materialized/`
(PNG), `04_features/` (embeddings), `05_qc/` (masks, overlays, reports).

`pairs.parquet` is the task table — one row per record, 39 columns. The ones you will
actually reach for:

| column | meaning |
|---|---|
| `pair_id`, `slide_id`, `context_id`, `query_id` | identity and what this record was built from |
| `claimed_bbox_l0_{x,y,w,h}` | `ℓ`, level-0 pixels of `slide_id` |
| `true_bbox_l0_{x,y,w,h}` | `ℓ*`, level-0 pixels of **`true_slide_id`** |
| `true_slide_id`, `same_slide` | which slide the served patch came from; equal to `slide_id` except for the out-of-slide types |
| `degradation_digest` (in `target`, not a column) | what the served `query` was degraded by; `""` when undegraded. A harness keys its localise cache on it: degradation is drawn per item, so two items can share slide, context and `true_bbox` and still be served a 5° and a 15° rotation |
| `claim_correct` | the label — `True` only for `clean` |
| `fault_type`, `fault_severity` | which of the eight, and the drift band (`small`/`medium`/`large`, null elsewhere) |
| `normalized_offset`, `offset_frac_context`, `iou_claim_true` | how far the claim is from the truth; the first two are null across slides, where a distance has no meaning |
| `true_within_context`, `local_recovery_possible` | whether `ℓ*` is inside `C` — false for the four substituted types |
| `hardness_score` | query↔claim cosine in the frozen embedding space, for stratifying by difficulty |
| `context_mpp`, `query_mpp`, `scale_ratio` | the magnification combination, so freezes concatenate |
| `degradation_band` | which imaging-degradation condition this record is in: `none` / `mild` / `moderate` / `severe` |
| `degradation_kind`, `degradation_params` | the operation replayed on the served `Q` (`none` or `geometry`) and its frozen parameters, as JSON |
| `served_pixel_sha256` | sha256 of the RGB buffer `Q` is actually served as, once a QC run has sampled that row (`""` until then) |
| `split` | `dev` / `calibration` / `test`, assigned per **case** |

**The degradation dimension.** Locating `Q` inside `C` must not be solvable by grayscale
template matching alone, so the *served* query patch carries a frozen per-row geometric
degradation: a rotation about the patch centre plus a small scale error, sampled
bilinearly, clamped at the border so no fill colour enters. `degradation_band` names the
condition — `none` (0°), `mild` (5°), `moderate` (10°), `severe` (15°) — and
`degradation_params` carries the drawn `angle_deg` and `scale` for that row.

Four things worth knowing before you build against it:

- **The parameters are frozen, never redrawn.** The loader reads them from the row and
  replays them, so the pixels are reproducible bit-for-bit on another machine; nothing in
  `pyrafault/degrade.py` delegates to a library filter or encoder whose implementation
  could differ. `served_pixel_sha256` is the check on that: stage 09 freezes it the first
  time it samples a row and compares it on later runs, so a machine that regenerates
  different pixels fails loudly instead of quietly reporting numbers computed on
  different images.
- **A `none` row is the identity transform.** Same pixels, same cache key, same hash as a
  build that never knew about degradation. `configs/example10.yaml` assigns `none` to
  every row, so the example's pixels are unchanged; a freeze that carries the four bands
  assigns them balanced within every `(split, fault_type)` cell, so the band cannot
  predict `claim_correct`.
- **The band is not in `inputs`.** A verifier is never told whether — or how much — its
  `Q` was rotated; the pixels are the only evidence. It is not in `target` either: the
  scorer reads it from `pairs.parquet`.
- **Report per band.** `score_predictions` returns `by_degradation_band`
  ([`VERIFIER_PROTOCOL.md`](VERIFIER_PROTOCOL.md) §8); a single pooled number over a
  multi-band freeze hides the dimension it was built to measure.

The other tables are the audit trail rather than the task: `contexts` and `queries` carry
each region's own extraction spec and quality measurements, `*_candidates` the full
eligible pools selection drew from, `slides` the per-slide pyramid and MPP metadata, and
`splits` the case-to-split assignment. Read any of them with
`pyrafault.schema.read_table(path, name)`, which validates the columns before returning.

A freeze's size follows from three numbers in the config and the geometry of the slides
it is given: `context.num_per_slide` contexts per slide (capped at 30, and a slide whose
tissue cannot host that many yields fewer), `query.num_per_context` queries per context,
and eight pairs per query — one per condition, `clean` plus the seven fault types. The
ten-slide example therefore produces a few thousand pairs rather than three million, and
stage 09 prints the exact counts it reached. The released benchmark's counts are in the
paper; they are not reproduced by this example and are not claimed to be.
All eight are **on** in `configs/example10.yaml`, so a freeze built from it carries the
full label vocabulary. Four of them are substitutions — the patch served is not the patch the claim
describes:

| `fault_type` | what is wrong with it | how `ℓ*` is chosen |
|---|---|---|
| `out_of_context` | the patch came from elsewhere on the slide entirely | uniformly among candidates that do not touch `C` |
| `out_of_context_hard` | ...and from the tissue that looks most like what was asked for | nearest frozen embedding to the query's own |
| `out_of_slide` | the patch is from a **different slide** — the cache/concurrency failure | uniformly, among slides in the same split |
| `out_of_slide_hard` | ...and from the closest-looking tissue anywhere in the split | nearest frozen embedding across the other slides |

`out_of_slide` is enabled separately, by `faults.out_of_slide.enabled: true`. Its
substitute is drawn **only from slides in the same split**: pulling a dev slide's pixels
into a test item would undo the case-level split as surely as putting the slide itself in
both, and no fault type is a licence to do that. A QC check asserts it per row.

For these two, `true_bbox_l0` indexes **another slide's coordinate frame**, so the record
carries `true_slide_id` (equal to `slide_id` for every other type) and the loader reads
the query image against it. `iou_claim_true` is 0 and `normalized_offset` is null by
construction — a distance between boxes in two different slides' frames is not a small
number, it is a meaningless one, and nulling it keeps it out of any average.

**Cross-slide substitution is the easiest fault type, not the hardest.** Measured on a
ten-slide 10×→40× freeze: median query↔claim cosine runs `out_of_slide` 0.155 < `out_of_context`
0.333 < `random_same_context` 0.461 < `out_of_slide_hard` 0.465 < `hard_mismatch` =
`out_of_context_hard` 0.741. It may also carry stain and scanner cues no same-slide fault
has. It measures whether a basic provenance failure is caught at all; do not pool its
accuracy with the same-slide types without saying so.

In both, the **claim stays inside `C`** — it is the location the agent actually requested —
and it is the true source that lies outside, so `true_within_context` and
`local_recovery_possible` are false. A verifier cannot reject either by checking containment;
it has to decide that similar tissue is not the *same* tissue. Enabling both adds two
conditions per query, and stage 09 reports the median query-to-claim cosine per condition
so the mining can be checked rather than assumed: slide-wide mining should place
`out_of_context_hard` above `hard_mismatch`, which should sit above
`random_same_context`, which should sit above `out_of_context`. Splits are case-level,
6 dev / 2 calibration / 2 test. Features are UNI (`MahmoodLab/UNI`, revision
`b55a5ec6cade1a39edfe6534189a9b8ca7a022f0`, 1024-dim).

---

# Using the benchmark

```python
from pyrafault.dataset import PyraFaultDataset

ds = PyraFaultDataset("data/01_benchmark", "data/00_source/gdc_download",
                      split="test", materialize=True)
#                      ^ the source root your config points at (gdc_dir or slides_dir)

item = ds[ds.item_ids[0]]

item["inputs"]    # context image, query image, claimed_bbox, context_bbox,
                  # context_mpp, query_mpp, item_id, slide_id, split, …
item["target"]    # claim_correct, true_bbox, true_slide_id, fault_type, pair_id, …
                  # — scoring only
```

`materialize=True` serves cached PNGs; without it, images are read from the source WSI at
the frozen coordinates. Both paths produce identical bytes.

## What a verifier receives

Exactly this, per item — nothing else is available, and nothing here is optional:

| field | type | meaning |
|---|---|---|
| `context` | `PIL.Image`, RGB, `context.size_px`² (2048² shipped) | the low-magnification context `C` |
| `query` | `PIL.Image`, RGB, `query.size_px`² (512² shipped) | the high-magnification patch `Q` |
| `context_bbox` | `BBoxL0(x, y, w, h)` | where `C` sits on the slide, in **level-0 pixels** |
| `claimed_bbox` | `BBoxL0(x, y, w, h)` | the claimed source `ℓ`, level-0, always inside `context_bbox` |
| `context_mpp` | `float`, µm/px | `mc` |
| `query_mpp` | `float`, µm/px | `mq`; `mc / mq` is the scale factor for aligning `C` to `Q` |
| `slide_id` | `str` | which slide `C` is on |
| `item_id` | `str`, sha256 hex | opaque handle; carries no information about the label |
| `split` | `str` | `dev` / `calibration` / `test` |
| `extraction_version` | `str` | how the pixels were produced |

All four bbox numbers are level-0 pixels of `slide_id`, so `claimed_bbox` and
`context_bbox` are directly comparable; converting a level-0 box to a position inside the
`context` image is `(box.x - context_bbox.x) / context_bbox.w * context.width`.

`target` is the answer key and must not be read while producing a decision:
`claim_correct` (bool), `true_bbox` (`ℓ*`), `true_slide_id` (the slide `true_bbox` is
expressed in — differs from `slide_id` only for the two out-of-slide fault types),
`fault_type`, `fault_severity`, `normalized_offset`, `iou_claim_true`,
`hardness_score`, `pair_id`.

`degradation_band` is in neither: the pixels a verifier is handed are the only evidence
that they were degraded, and the scorer — which holds `pairs.parquet` — is what breaks the
results out per band.

## What a verifier must return

One record per `item_id`. This is the contract every baseline and ZoomTrace itself is
scored through, so a method that returns something else cannot be compared:

```python
{
    "item_id":        str,            # echoed back from inputs
    "estimated_bbox": BBoxL0 | None,  # ℓ̂ — the visually most likely source, level-0
                                      # pixels of slide_id. None only when the method
                                      # declines to localise at all.
    "reliability":    float,          # s — higher means "this localisation is
                                      # discriminative". Any monotone scale; it is
                                      # thresholded, never compared across methods.
    "decision":       str,            # "verified" | "mismatched" | "unverifiable"
}
```

This is the conceptual shape of a `Localisation` a method's `localise()` returns — not
the file you submit for scoring. The full on-disk submission format (`predictions.parquet`,
all eleven `PREDICTION_COLUMNS`) is specified in [`VERIFIER_PROTOCOL.md`](VERIFIER_PROTOCOL.md)
§5.

`decision` follows from the other two against a reliability threshold `τ` and a spatial
tolerance `δ`: **`unverifiable`** when `reliability < τ`; otherwise **`verified`** when
`ℓ̂` agrees with the claim within `δ`, and **`mismatched`** when it does not.
`unverifiable` is the method abstaining — it is not a third ground-truth class, and no
record in the benchmark carries it.

Which field each metric consumes:

| metric | reads |
|---|---|
| AUROC / AUPRC | `reliability` together with the agreement between `estimated_bbox` and `claimed_bbox` — a continuous score, not the discretised `decision` |
| balanced accuracy, TPR at a fixed false-verification rate | `decision` |
| normalized centre error, IoU, Recall@k | `estimated_bbox`, on items whose `true_bbox` lies inside `context_bbox` |
| risk–coverage, ECE, Brier | `reliability`, with `unverifiable` as the abstention |
| source recovery accuracy | `estimated_bbox` on `mismatched` items |

Two rules make results comparable, and both are properties of the harness rather than of
the data:

1. **Score per item.** Each record's output is a function of its own
   `(context, query, claimed_bbox, …)` and nothing else. Batching for throughput is fine —
   under `eval()` with no cross-sample operation a batch of N is arithmetically identical
   to N separate passes — but a scoring rule that reads other items in the same context
   reopens the lattice residual described under "Known limitations".
2. **Fit `τ` and `δ` on `calibration`, freeze them for `test`.** They are thresholds on a
   score distribution, so estimating them on `test` uses the evaluation set to tune the
   evaluation. `dev` is for development, `calibration` for these two numbers, `test` for
   the number you report.

`inputs` carries `context_mpp` and `query_mpp` — the task's `mc` and `mq`. A
cross-magnification method needs them: aligning `C` to `Q`'s physical resolution is
arithmetic on their ratio, so without them its first step is impossible. They are
constant within one freeze and so carry no label information, but the leakage audit
scores them numerically anyway rather than exempting them — pool several magnification
combinations into one table and, if the combos differ in fault composition, magnification
becomes a pixel-free predictor. Note the pipeline works in µm/px, not in "×": these
slides are labelled 40× but measure 0.2520–0.2527 µm/px, so a 0.25 target would require
upsampling, and the config validator refuses it.

Every pair also records `context_mpp`, `query_mpp` and `scale_ratio`, so freezes built at
different magnifications concatenate into one table and stratify. Ratio alone is not a
key: 5×→20× and 10×→40× both have ratio 4.0 and differ only in absolute resolution.

**The `inputs`/`target` split is a safety property, not presentation.** `true_bbox` alone
answers the recovery task, and `normalized_offset == 0.0`, `iou_claim_true == 1.0`, a
non-null `fault_severity` and `hardness_score`'s null pattern each restate the label. They
all live in `target`, so consuming the answer requires reaching for it deliberately.

`inputs` carries an opaque `item_id = sha256(seed|pair_id)` rather than `pair_id`, because
`pair_id` is minted as `f"{query_id}__{fault_type}"` — `"__clean" in pair_id` would recover
`claim_correct` at 100% precision. `ds.item_ids` and `ds[item_id]` cover iteration and
indexing; `ds.pair_ids` and `ds[pair_id]` still work for tooling that legitimately needs the
readable form, such as keying off `pairs.parquet` or QC overlay filenames.

**The evaluation path contains no randomness** — no random crop, no colour augmentation, no
re-running tissue filtering, no re-selecting queries. Two independently constructed
datasets return byte-identical images for the same record, and a test asserts it.

## Running a verifier

Stage 12 builds the retrieval window bank a verifier matches against (needs a GPU); stage
13 runs a verifier and writes predictions; stage 14 scores them:

```bash
python scripts/12_build_window_bank.py --config configs/example10.yaml --data-root data

# calibration: fit tau and delta first, then decide and write its own predictions
python scripts/13_run_verifier.py --config configs/example10.yaml --data-root data \
    --split calibration --fit-thresholds --matcher sift --audit-independence

# test: decide against the thresholds calibration just froze
python scripts/13_run_verifier.py --config configs/example10.yaml --data-root data \
    --split test --matcher sift

python scripts/14_score_verifier.py \
    --predictions data/06_verify/zoomtrace_sift/predictions_test.parquet \
    --data-root data --split test
```

`scripts/13_run_verifier.py` calls `Thresholds.load()` at startup, so `τ` and `δ` must
exist before any split can be decided — including `calibration`, the split they are fitted
from. `--fit-thresholds` is what breaks that: on the split it is given, it localises every
item without deciding, fits both thresholds from those localisations (`fit_tau`,
`fit_delta` in `pyrafault/verify/thresholds.py`), writes `thresholds.json`, and only then
decides and writes that split's own predictions file. Pass it once, on `calibration`; later
splits (`test`) omit the flag and load the `thresholds.json` this step wrote. `--matcher
sift` is CPU-only and needs no GPU allocation for stage 13 itself; `--matcher loftr` is the
learned alternative.

A baseline runs through the same three commands with `--verifier module:Class` in place of
the ZoomTrace defaults — no window bank, no matcher, output under `06_verify/<name>/`. To
see that path end to end with no code of your own, use the shipped skeleton,
`--verifier pyrafault.verify.ncc_baseline:NccBaseline`.

The full protocol this reproduces — every `PREDICTION_COLUMNS` field, the decision rule,
submitting a predictions file without this repository at all — is
[`VERIFIER_PROTOCOL.md`](VERIFIER_PROTOCOL.md).


## ZoomTrace and every baseline

ZoomTrace is `--verifier zoomtrace` (the default) and is the only method that needs the
window bank from stage 12. Each baseline is one module under `pyrafault/verify/`, selected
by `--verifier module:Class`, and writes to `data/06_verify/<its own name>/`. The three
commands are the same for all of them: fit on `calibration`, decide `test`, score.

| method | `--verifier` | external assets |
|---|---|---|
| **ZoomTrace** | `zoomtrace` | window bank (stage 12); `--matcher sift` or `loftr` |
| Claim-conditioned LoFTR | `pyrafault.verify.plugins:ClaimConditionedPlugin` | none |
| Efficient LoFTR | `pyrafault.verify.plugins:EfficientLoFTRPlugin` | `EFFICIENT_LOFTR_SOURCE`, `EFFICIENT_LOFTR_CHECKPOINT` |
| SuperPoint + LightGlue | `pyrafault.verify.plugins:LightGluePlugin` | `LIGHTGLUE_SOURCE`, `SUPERPOINT_CHECKPOINT`, `LIGHTGLUE_CHECKPOINT` |
| MINIMA–LightGlue | `pyrafault.verify.plugins:MinimaLightGluePlugin` | `LIGHTGLUE_SOURCE`, `SUPERPOINT_CHECKPOINT`, `MINIMA_LIGHTGLUE_CHECKPOINT` |
| MatchAnything–ELoFTR | `pyrafault.verify.plugins:MatchAnythingELoFTRPlugin` | `MATCHANYTHING_ELOFTR_SOURCE`, `MATCHANYTHING_ELOFTR_CHECKPOINT` |
| RoMa | `pyrafault.verify.plugins:RoMaPlugin` | `ROMA_SOURCE`, `ROMA_CHECKPOINT`, `ROMA_DINOV2_WEIGHTS` |
| SUE | `pyrafault.verify.plugins:SUEPlugin` | `PYRAFAULT_DINOV2_SNAPSHOT` |
| Statistical Uncertainty | `pyrafault.verify.plugins:StatisticalUncertaintyPlugin` | `PYRAFAULT_DINOV2_SNAPSHOT` |
| NCC | `pyrafault.verify.ncc_baseline:NccBaseline` | none |

Each plugin reads its own paths from the environment and loads nothing at construction, so
`--from-localisations` costs nothing even for the heavy ones. `python
scripts/check_baseline_assets.py` reports which variables are set and whether each file or
directory exists, before a job is submitted rather than after it fails. Pinned upstream
revisions and checkpoint hashes are recorded in each module's `revision` string and written
into every predictions file, so a run names the weights that produced it.

`PYRAFAULT_BASELINE_DEVICE` is `auto`, `cpu` or `cuda` and applies to every plugin.

A claim-conditioned baseline is the one admitted exception to Eq. 1's independence: it reads
`inputs["claimed_bbox"]`, declares `reads_claim = True`, and the harness responds by adding
the claim to its dedupe key and refusing `--audit-independence` for it. See
`Verifier.reads_claim` in `pyrafault/verify/base.py` for why both are required.

The vision-language baselines are `VlmBaseline` and `VlmLocaliser` in
`pyrafault/verify/vlm_baseline.py`, with the API variant `AnthropicVlmLocaliser` in
`vlm_api.py`. They take a model identifier rather than no arguments, so they are driven as
library classes instead of through `--verifier`; the API variant reads `ANTHROPIC_API_KEY`
from the environment.

### A large split, sharded

Localisation and deciding are separable, which is what makes a large split affordable: the
expensive half runs as an array and the cheap half reads what it wrote.

```bash
# one array task per shard; each reads only its own contexts' banks
python scripts/13_run_verifier.py --config configs/example10.yaml --data-root data \
    --split test --localise-only --shard "$SLURM_ARRAY_TASK_ID" --n-shards 40

# then decide, reading every shard at once; no pixels, no GPU
python scripts/13_run_verifier.py --config configs/example10.yaml --data-root data \
    --split test --from-localisations
```

Shards are grouped by context, not strided by item, so the dedupe inside a run survives
sharding. The deciding pass refuses a shard set with a hole in it rather than scoring a
subset.

---

# Pipeline reference

| stage | writes | reads pixels |
|---|---|---|
| 00 ingest source | `slides.parquet` — the adapter `source.mode` selects; `gdc` audits size + MD5, `directory` walks a folder | no |
| 01 scan slides | MPP, objective power, pyramid levels; excludes slides without MPP | headers |
| 02 tissue masks | `05_qc/masks/*.npz` — summed-area tables for O(1) `tissue_fraction` | thumbnail |
| 03 contexts | `context_candidates.parquet`, **`contexts.parquet`**, `splits.parquet` | selected contexts |
| 04 query candidates | `query_candidates.parquet` — the in-context grid | **no** |
| 05 diversity sampling | **`queries.parquet`** — cluster-stratified, 5 per context | every candidate |
| 06 build pairs | `clean` / `adjacent_drift` / `random_same_context` | no |
| 07 mine hard negatives + substitutions | `hard_mismatch` (top-1 cosine inside `C`), plus the four substituted types: `out_of_context{,_hard}`, `out_of_slide{,_hard}` | no |
| 08 materialize | PNG cache (derived) + `image_sha256` written back | contexts + queries |
| 09 QC | invariants, reconstruction, leakage audit, overlays, `qc_report.md` | 5% sample |

Stage 04 performs zero pixel reads — it queries the summed-area table only, which is why it
runs as one fast task rather than an array.

Stages 03, 05 and 08 run per-slide and then gather: the per-slide phase writes shards, and a
separate `--gather` invocation turns them into the stage's table. `slurm/run_all.sh` submits
the gather as its own `afterok` job.

---

# Layout

| path | contents |
|---|---|
| `pyrafault/` | library — the unit-tested part |
| `scripts/NN_*.py` | one CLI per stage, thin shells over the library |
| `configs/` | `example10.yaml` (the ten-slide example; the run that uses it copies it verbatim into `data/01_benchmark/`) and `example_directory.yaml` (a working non-GDC config) |
| `slurm/` | one sbatch template per stage, plus `env.sh` and `run_all.sh` |
| `tests/` | 1,611 tests; the 35 marked `-m slow` need a real WSI and a compute node |
| `data/` | produced by a run, gitignored except `01_benchmark/benchmark_manifest.json`; another cohort goes in `data_*/`, also gitignored |
| `examples/` | the three GDC exports identifying the ten example slides |

---

# Environment

`environment.yml` pins Python only; everything else floats to whatever the solver resolves.
The embeddings depend on the exact stack that produced them, so that solve is frozen
separately in `environment.lock.yml`.

| file | use |
|---|---|
| `environment.yml` | loose spec; the starting point for a new solve |
| `environment.lock.yml` | the exact environment that produced the published embeddings |

The lock file pins `torch==2.13.0`, `timm==1.0.28`, `transformers==5.15.0`,
`openslide-python==1.4.6`, `openslide-bin==4.0.1.2` among others. Reproducing stage 05's
output bit-for-bit needs this exact set, not a fresh `environment.yml` solve. The export
command is recorded in the lock file's header for provenance — **do not re-run it**, or you
will overwrite the frozen record with whatever the live environment happens to be.

## Feature extractor

`configs/example10.yaml` uses `uni` (`MahmoodLab/UNI`), a gated repository whose licence must be
accepted once on the model page by the account owning `HF_TOKEN`. If it is not, stage 05
fails with a message naming the model page. The ungated fallback:

```bash
PYRAFAULT_EXTRACTOR=dinov2 bash slurm/run_all.sh 05
```

Switching afterwards means re-running stages 05 and 07 only. The extractor name, resolved HF
commit and embedding dimension are recorded per row, and the manifest is written once in the
gather step from per-slide shards that must agree — so a mixed-extractor table fails loudly
rather than being produced by accident.

---

# Tests

```bash
PY="${PYRAFAULT_PY:-python}"

$PY -m pytest -m "not slow"                      # 591 tests, a few minutes, no WSI needed

sbatch --wait -p day -c 4 --mem=32G -t 40 -o logs/%x-%j.out --wrap \
  "PYTHONPATH=$PWD $PY -m pytest -m slow"        # 10 tests, real WSIs, a compute node
```

The slow set is where the load-bearing guarantees live: byte-identical reads across
processes, reconstruction from coordinates alone, and the loader returning identical images
from two independently constructed instances.

---

# Known limitations

**The query lattice was derivable — fixed by per-context jitter (spec §15.1).**
`random_same_context` and `hard_mismatch` place `ℓ` on a query-grid cell, and `clean` places
it on `ℓ*`, also a cell; `adjacent_drift` displaces `ℓ` by a continuous radius and lands
off-grid. With a *centred* grid origin that origin is derivable from the context bbox and
the slide MPP — both of which any usable evaluation interface must expose — so one modulo
test labelled 935 pairs, 25% of the benchmark, using zero pixels. Every pair it identified
was a fault, so it handed a verifier a correct `claim_correct = False` on a quarter of the
items and inflated headline accuracy directly.

The fix offsets each context's grid origin by a deterministic, per-axis-independent jitter,
so grid membership is undecidable from any single `(C, Q, ℓ)`. Measured after the fix, the
same rule scores 0.750 precision against a 0.750 base rate — zero lift.

**A residual remains for a verifier that pools items by context** rather than scoring one
at a time. It is irreducible in this design: `clean` is by definition on the lattice and
`adjacent_drift` must be off it to reach the small severity band, so within one context the
drifted items are the minority residue class.

**The intended protocol is per-item scoring, and under it this residual is closed.** Each
record is scored from its own `(C, Q, ℓ)` alone; the modal-residue test needs several items
sharing a context and has nothing to compute from one. Note the grouping key is visible —
`inputs` carries `context_bbox`, and the ~40 pairs a context contributes all share it,
because a verifier that could not locate `C` on the slide could not do the task at all. So
what closes the residual is the protocol, not the data, and an evaluation should state that
it scores per item.

Tensor batching is not pooling: under `model.eval()` with no cross-sample operation, a batch
of N is arithmetically identical to N separate forward passes, and every extractor here is
frozen. What would break it is a scoring rule that reads other items in the group, or
thresholds re-estimated on `test` rather than frozen from `calibration`.

**`hard_same_slide.pool: slide` is refused, and the refusal is the point.** Widening
`hard_mismatch`'s search pool to the whole slide would let the *claim* land outside `C`, and
a claim outside `C` is refutable by a bbox containment test with zero pixels — the hardest
fault type would become the easiest. Slide-wide hard negatives are built the other way
round, as `out_of_context_hard` below: the claim stays put and the *served patch* is
substituted. Setting `pool: slide` fails with a message saying so.

Stage 09's §13.5 leakage audit re-measures all of this on every freeze and fails the build
if any pixel-free signal beats 0.55 balanced accuracy.

---

# Getting the raw slides

The ten whole-slide images are **not** in this repository — they come to 12.87 GB.
`examples/gdc_metadata/` holds the three GDC exports that identify them, which is enough to
fetch the **identical** set: the manifest carries every file's UUID, published size and MD5.

The ten are the first ten slides of the released cohort that pass stage 01's resolution and
readability checks, one slide per case, spanning all five of its projects: TCGA-BLCA,
TCGA-LGG, TCGA-LUAD, TCGA-LUSC and TCGA-UCEC. All are open-access diagnostic slides, so no
GDC token is involved.

Either tool fills the download directory. `gdc-client` is the official one and the better
choice for a large cart, since it resumes and parallelises:

```bash
# https://gdc.cancer.gov/access-data/gdc-data-transfer-tool
gdc-client download -m examples/gdc_metadata/gdc_manifest.2026-08-12.143046.txt \
                    -d data/00_source/gdc_download
cp examples/gdc_metadata/* data/00_source/gdc_download/
```

`scripts/download_tcga.py` needs nothing but this repository's dependencies, stages the
three exports for you, and skips files already present at the published size:

```bash
python scripts/download_tcga.py                      # all ten, 12.87 GB
python scripts/download_tcga.py --limit 2            # the first two, to try the path
python scripts/download_tcga.py --dry-run            # list what it would fetch
```

Both write `<file-uuid>/<file-name>.svs`, which is the layout `gdc-client` produces and the
path `source_relpath` records in `slides.parquet`. A flat download would make a freeze built
here unverifiable against one built with the official client.

Under `source.mode: gdc`, stage 00 audits every file's size and MD5 before anything else
runs, so a truncated or substituted download stops the pipeline rather than quietly changing
the benchmark. (`directory` mode has no publisher's record to check against and only records
a sha256 baseline it computes itself — see "If your slides are not from GDC" above.)

Nothing under `data/` is in this repository. The released benchmark tables are published as
a dataset; `data/01_benchmark/` here is whatever your own run produces, and
`benchmark_manifest.json` is the sha256 and row count that lets a second run be checked
against the first.

# What is in `examples/`

| path | contents |
|---|---|
| `gdc_metadata/gdc_manifest.*.txt` | file UUID, published name, MD5 and size for the ten slides |
| `gdc_metadata/gdc_sample_sheet.*.tsv` | project, case, sample and specimen fields, joined on the file UUID |
| `gdc_metadata/metadata.cart.*.json` | case UUID, experimental strategy and access level per file |

Stage 00 inner-joins all three on the file UUID and refuses to lose a row, so a file missing
from any one of them stops the run by name rather than silently dropping out of the cohort.

---

# What is not here

| | why |
|---|---|
| the benchmark tables | published as a dataset; this repository builds them |
| the agent tool-path harness | the paper's Section 7 experiment, not part of the benchmark or the verifiers |
| figure and table scripts | they read result files this repository does not ship |
| the design specification | kept with the authors |

Every verifier in `pyrafault/verify/` is here, including the ones that lose, and so is the
protocol a third party needs to submit predictions without running any of this code.

---

# Licence

The code is released under the MIT licence; see [LICENSE](LICENSE). The TCGA slides are not
distributed with it, and their use is governed by GDC's own terms.
