# Handoff for manuscript writing

**Prepared 2026-10-09.** Everything needed to write the NED-Net manuscript. Read this file
first, then `DRAFT.md`. Target journal: **eNeuro**, Methods/New Tools.

---

## Read in this order

| # | file | why |
|---|---|---|
| 1 | **`DRAFT.md`** | the manuscript. Abstract + Introduction written in full; Methods/Results sectioned with measured numbers in place and pending ones marked `[[ ]]`. **Write here.** |
| 2 | **`PAPER_OUTLINE.md`** | every result and discussion point in full, with the reasoning behind each. The source for anything `DRAFT.md` only sketches. |
| 3 | **`JOURNAL_SCAN.md`** | prior art (3 papers we must position against, with DOIs), the gap, journal recommendation and why high-IF is not realistic. |
| 4 | **`NEXT_STEPS.md`** | current state, what is running, what is still missing. **Contains the settled scope decision — read it.** |
| 5 | `RUN_LOG.md` (86 KB) | append-only provenance for **every number**. Searchable by date. Do not quote a figure that is not in here. |
| 6 | `MODELS.md` | every model trained, what it saw, what it scored, its operating point. |
| 7 | `PHASE1_RESULTS.md` | the source-model-fails analysis in detail. |
| 8 | `EXCLUSION_CRITERIA.md` | which animals/channels are excluded and why; the pre-registered metrics that failed. |
| 9 | `review/REVIEW_PROTOCOL.md` | how human review was done. |
| — | `archive/` | superseded versions, kept for diffing. **Do not write from these.** |
| — | `NEDNet_validation_HANDOFF.md`, `PHASE2_STRATEGY.md` | early planning docs, largely superseded by `PAPER_OUTLINE.md`. Background only. |

## Hard rules — these will embarrass us if broken

1. **Do not claim transfer has never been measured.** Fumeaux et al. 2020 *Epilepsia*
   measured it (AUROC 0.995 pooled -> 0.890 cross-dataset). Our contribution is reporting
   it **per animal**, where it reads as a subset of implants returning nothing, plus the
   remedy and the tool. See `JOURNAL_SCAN.md`.
2. **Do not present per-animal variability as novel in kind.** Kotloski 2023 *Sci Rep*
   found intra-animal > inter-animal training (0.960 vs 0.811). Cite as motivation; our
   addition is the distribution, the two failure modes, and the loop as the response.
3. **Scope is in-sample.** The paper measures what a lab gets on its own data after
   annotating it. Out-of-sample folds are supporting context only. Do **not** add
   generalisation caveats to every number — state the scope once, in Methods.
   (`NEXT_STEPS.md`, "SCOPE — SETTLED".)
4. **This is a software paper.** The detector numbers support a design claim — a model is
   maintained, not delivered. If it reads as a detection benchmark, the results look like a
   negative result instead of a justification.
5. **Never quote a number not in `RUN_LOG.md`.** Several figures in this project were
   withdrawn after correction; the log records which and why.
6. **Say in-sample or out-of-sample on every performance figure.** They differ by ~6x.
7. **UI channel N = code/DB channel N-1.** The lab always speaks in UI channels. This has
   already produced one wrong published-draft conclusion.

## What is measured vs pending

**Measured and quotable** (all in `RUN_LOG.md`):
* source model: 93.6% precision (160/171), recall 87%, event F1 0.78
* transferred to new cohort: precision 92% (12/13), recall 43% where it fires / 18% overall,
  convulsive recall 15.3% (34/222)
* per-animal recall of the transferred model: median 10%, IQR 0-27%, range 0-58%, 4 of 11
  animals at zero; within-batch spread 58 points vs between-batch 31
* two failure modes: silent (5 seizures, 0 detections) and mislocalised (50 seizures, 82
  detections, 0 correct)
* retrained, in-sample: coverage 16% -> 73% of recordings; recall 27.9% -> 47.1% (cascade),
  38.2% -> 50.0% (stage 1); median IoU 0.36 -> 0.60
* exhaustive review: 4 recordings, 48 animal-hours, 50 seizures, 92% precision where it
  fires, 0/22 where it does not
* match-rule sensitivity: result robust to removing the 5 s tolerance and to one-to-one
  matching; boundary truncation 9 s detections inside 37 s seizures
* annotation cost of the classical reference: 12,083 human adjudications -> 430 events (3.6%)
* per-channel precision within one batch: 64% / 22% / 9%
* five pitfalls, each with a measured cost (`PAPER_OUTLINE.md` R8)

**Pending — marked `[[ ]]` in `DRAFT.md`:**
* the temporal-pair model (`ramgdnf_temporal` + `conv_temporal`): jobs 3831662/3831663.
  Becomes the headline of Results 3.5. **Assume it improves on the figures above; do not
  invent values.**
* precision of the retrained detector (Results 3.6) — protocol pre-registered in
  `PAPER_OUTLINE.md` R6; tooling written (`scripts/lunarc/draw_precision_sample.py`,
  `scripts/local/score_precision_sample.py`), **not yet run**
* whether the stage-2 fix worked (pitfall 3's resolution)
* annotation-cost table's third row, which came from a model trained on a stale label set
* all figures

## Data files for figures and tables

In `review/`:

| file | contents |
|---|---|
| `frozen_per_animal_B1B2B3.csv` | **per-animal recall of the transferred model — Figure 2** |
| `frozen_vs_A3_per_animal_B1B2_cascade.csv` | **paired before/after per animal — Figure 3** (primary, cascade scope) |
| `frozen_vs_A3_per_animal_B1B2.csv` | same, stage-1 scope (secondary) |
| `frozen_per_animal_B1B2.csv` | transferred-model baseline, B1+B2 only |
| `mir_convulsive_retained.csv` | the convulsive reference set actually scored against |
| `mir_positives_matched.csv` / `_missed.csv` | which reference seizures were caught/missed |
| `precision_at_conf05_sample.csv`, `b2_retained_sample.csv` | the unbiased precision samples |
| `flood_review_events.csv` | the 113-event false-positive review |
| `nonconvulsive_exhaustive_sample.csv`, `nonconvulsive_file4.csv` | the exhaustive-review recordings |
| `groundtruth_out_of_range.csv` | the 242 defective reference rows, excluded and logged |

## Analysis scripts, if a number must be regenerated

| script | produces |
|---|---|
| `scripts/local/operating_point_table.py` | the main comparison table; `--det-conv-only` for cascade scope |
| `scripts/local/conv_threshold_sweep.py` | stage-2 threshold behaviour (free, no re-detection) |
| `scripts/local/matching_sensitivity.py` | match-rule robustness |
| `scripts/local/validate_frozen_unet_vs_mir.py` | the base scorer; `load()` is reused by the above |
| `scripts/local/score_precision_sample.py` | the pending precision estimate |

Detection databases live in `~/.eeg_seizure_analyzer/projects/` (local) and on LUNARC.

## Prior art to cite (DOIs)

* Fumeaux et al. 2020, *Epilepsia* — https://doi.org/10.1111/epi.16628
* Kotloski 2023, *Scientific Reports* — https://doi.org/10.1038/s41598-023-40628-1
* Kamintsky et al. 2025, *Epilepsia Open* — https://doi.org/10.1002/epi4.70070
* Leguia et al. 2022, *Epilepsia* — https://doi.org/10.1111/epi.17406
* Vergara-Chozas et al. 2023, *Epilepsy Research* — https://doi.org/10.1016/j.eplepsyres.2023.107151
* Chronic mouse EEG protocol, *JoVE* 2025 — https://doi.org/10.3791/69314

Retrieved from PubMed 2026-10-09. The reference list is **not** exhaustive — a full
literature search for the Introduction and Discussion still needs doing.

## Known weak points a reviewer will press

1. **Precision of the retrained model is unmeasured.** The largest gap.
2. **Single laboratory, single rig.** Whether the recovery reproduces elsewhere is untested,
   and it is the one addition that would raise the target journal a tier.
3. **n is small for per-animal claims** — 11 animals with >=5 reference events. Present as a
   distribution, never as a ranking.
4. **The reference set is convulsive-only**, so most of the detector's output has no
   reference in either direction.
5. **Exclusions were performance-informed in origin.** Every stratum is reported; do not
   claim they were independent.
