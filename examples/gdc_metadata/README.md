# The ten example slides

These are the first ten slides of the released cohort that pass stage 01's checks -- a
readable pyramid and a recorded microns-per-pixel fine enough for the 0.5 um/px query
level. One slide per case, and all five of the cohort's projects are represented, so the
example exercises cross-project variation rather than one tissue type.

| slide | project | case | GB | MPP (um/px) |
|---|---|---|---|---|
| `TCGA-B5-A1MW-01Z-00-DX1` | TCGA-UCEC | `TCGA-B5-A1MW` | 1.74 | 0.2511 |
| `TCGA-2F-A9KQ-01Z-00-DX1` | TCGA-BLCA | `TCGA-2F-A9KQ` | 1.56 | 0.2277 |
| `TCGA-56-8503-01Z-00-DX1` | TCGA-LUSC | `TCGA-56-8503` | 0.24 | 0.2520 |
| `TCGA-86-6562-01Z-00-DX1` | TCGA-LUAD | `TCGA-86-6562` | 0.49 | 0.2520 |
| `TCGA-B5-A1MR-01Z-00-DX1` | TCGA-UCEC | `TCGA-B5-A1MR` | 1.58 | 0.2456 |
| `TCGA-VM-A8CF-01Z-00-DX9` | TCGA-LGG | `TCGA-VM-A8CF` | 1.02 | 0.2527 |
| `TCGA-97-7552-01Z-00-DX1` | TCGA-LUAD | `TCGA-97-7552` | 0.46 | 0.2520 |
| `TCGA-AX-A06L-01Z-00-DX2` | TCGA-UCEC | `TCGA-AX-A06L` | 1.50 | 0.2468 |
| `TCGA-B5-A1MZ-01Z-00-DX1` | TCGA-UCEC | `TCGA-B5-A1MZ` | 2.71 | 0.2456 |
| `TCGA-E7-A6ME-01Z-00-DX1` | TCGA-BLCA | `TCGA-E7-A6ME` | 1.56 | 0.2465 |

12.87 GB in total. All are open-access TCGA diagnostic slides, FFPE
primary tumour, so `scripts/download_tcga.py` and `gdc-client` both fetch them without a
token.

## The three files

| file | what stage 00 takes from it |
|---|---|
| `gdc_manifest.*.txt` | `id`, `filename`, `md5`, `size` -- the integrity record every download is audited against, and `id + "/" + filename` is the `source_relpath` written into `slides.parquet` |
| `gdc_sample_sheet.*.tsv` | `Project ID`, `Case ID`, `Sample ID`, `Tissue Type`, `Tumor Descriptor`, `Specimen Type`, `Preservation Method` |
| `metadata.cart.*.json` | the case UUID under `associated_entities[0].case_id`, plus `experimental_strategy` and `access` |

All three are joined on the file UUID and the join refuses to lose a row, so a file present
in one export and missing from another stops the run by name. These are regenerated copies
of what the GDC data portal exports for a cart of these ten files; the timestamp in each
filename is the cart's, and the config matches them by glob rather than exact name because
GDC stamps a new one on every download.

To build a different cohort, replace all three with your own cart's exports and set the
split sizes in the config to match your case count.
