# Phase 1 — validating the frozen NED-Net pipeline on RAM_GDNF_2025

**Rewritten 2026-10-08.** The 2026-10-02 version is in
`archive/PHASE1_RESULTS_20261002_superseded.md`; it led with pooled, unthresholded numbers
that several later corrections overturned. Reproduce with
`scripts/local/consolidate_mir_annotations.py` then `validate_frozen_unet_vs_mir.py`.
Models: `MODELS.md`. Every run: `RUN_LOG.md`. Corrections: `VERIFICATION_LOG_20261001.md`.

## Scope (set 2026-10-07)

This cohort is used for **pipeline and model validation only**. No treatment-group or
seizure-burden science is claimed from it: the groups are unbalanced by the exclusions
below, and the point of the exercise is the detector, not the biology.

The primary analysis runs on **good channels**; the precision cost on poor channels is a
secondary finding, reported because "recording quality matters" is worth backing with
numbers rather than asserting.

## Inclusion criteria

**A seizure is defined as an event lasting at least 5 seconds.** This is a stated
inclusion criterion for the analysis, not a detector limitation: it is how this lab defines
a seizure when analysing a cohort, it is configured as `min_duration` in the detector, and
it applies identically to the ground truth, the automated detections and the manual review.
Shorter rhythmic discharges are not counted as seizures under this definition; NED-Net has a
separate spike detector for that class.

Consistency check: **0 of the 222 retained convulsive ground-truth events fall below 5 s**
(minimum 5.4 s, median 28.9 s), so the criterion excludes nothing from the convulsive
reference and the recall figures are unaffected by it.

## The data

4 batches x 8 animals x ~21 days, 90-minute EDFs tiling the protocol continuously:
**1,377 cohort files, 2,014 recording hours, 16,114 animal-hours** — no file over 92 min,
nothing truncated. 32 animals, all mapped.

Ground truth is Mir's independent annotation: **430 confirmed seizures** over 1,074
recordings, from an automated candidate detector (15-80 Hz envelope peaks above
median + 4 x MAD, grouped at <=0.5 s, episodes >=15 s at >=2.5 spikes/s, scored on spike
density plus low/gamma/broadband power) **adjudicated manually with video** as
Seizure / False / Normal.

**What that ground truth can and cannot support:**

* **Convulsive seizures: defensible.** These are the events video can confirm, and 8 of 8
  inspected by the reviewer are real. 222 confirmed convulsive seizures in retained
  recordings, 148 of typical duration (15-60 s).
* **Non-convulsive seizures: no usable reference.** His candidates require >=15 s episodes
  of high spike density, so brief isolated events could never be proposed. 78% of the
  U-Net's output is non-convulsive and has nothing to be scored against in either
  direction.
* **His `False` rows are not seizure-detection negatives.** `False` means "not a
  convulsive/behavioural seizure". Visual review found real activity in them — including a
  180 s block on UI Ch6 with several distinct bursts of clear high-amplitude spiking,
  rejected with the convulsive flag set. They were nonetheless used as 12,925 hard
  negatives in arm A (see `MODELS.md`).
* Bounded defect: 242 rows (2.0%), all in Batch 4, carry timestamps past the end of their
  EDF. Excluding the 37 affected recordings wholesale moves recall 9.9% -> 10.7%.

## Headline result

**The frozen pipeline does not detect convulsive seizures in this cohort.**

| frozen cascade on Batch 3 | convulsive recall |
|---|---|
| all detections | **8/153 = 5.2%** |
| Stage-2 convulsive-labelled only | 4/153 = 2.6% |
| conf >= 0.5 | 4/153 = 2.6% |
| convulsive + conf >= 0.5 | 1/153 = 0.7% |

On the **retained set** (all batches except 4, minus Batch 2 UI ch3/ch4 and 449382),
scored against convulsive ground truth only — no re-detection needed, since the U-Net runs
per channel independently so filtering output by animal is equivalent:

| filter | detections | convulsive recall |
|---|---|---|
| all detections | 2,499 | **34/222 = 15.3%** |
| Stage-2 convulsive only | 450 | 23/222 = 10.4% |
| all, conf >= 0.5 | 934 | 26/222 = 11.7% |
| convulsive + conf >= 0.5 | 294 | 17/222 = 7.7% |

| batch | convulsive recall | GT convulsive | detections |
|---|---|---|---|
| B1 | 35.7% | 56 | 867 |
| B2 | 46.2% | 13 | 1,073 |
| **B3** | **5.2%** | **153** | 559 |

**Batch 3 holds 153 of the 222 convulsive seizures and has the worst recall by sevenfold —
and it is the batch retained as clean.** So recording quality does **not** explain the
convulsive failure: the cleanest batch with the most convulsive events is where the detector
does worst. B1's and B2's better figures rest on few events (56 and 13; B2's 46% is 6 of 13).
This is what makes Batch 3 the right held-out fold for Phase 2 — it is where the headroom
is. Caught examples look like textbook convulsive seizures (UI ch8);
missed ones look atypical (UI ch7) — the sensible failure direction.

**Why**: `UNetv2_20260615` drew 640 of its 867 training positives (74%) from the
autocorrelation detector, whose criterion is rhythmic ~10 Hz spiking, and those positives
are predominantly non-convulsive. The model learned that morphology. This is a
**training-composition** explanation, not an instrument-mismatch one: the U-Net slides over
the whole recording and is not restricted to any band, so missing convulsive seizures is a
genuine sensitivity failure for that class.

## Non-convulsive recall, from exhaustive manual review (2026-10-08)

Four Batch-3 recordings reviewed exhaustively — every event >=5 s on all 8 channels,
independent of what the detector reported. Two strata, randomly drawn within stratum:
files **with** a conf>=0.5 detection (43% of full-length B3 files) and files **without**
(57%). ~48 animal-hours. Manual additions carry `source="manual"` in the sidecars.

| stratum | file | manual seizures | U-Net detections in file | caught |
|---|---|---|---|---|
| B | `B3_W2_D2_22012026(7)` | 9 | **0** | 0 |
| B | `B3_W2_D3_23012026(9)` | 13 | **0** | 0 |
| A | `B3_W3_D3_02022026(12)` | 5 | 7 | 3 |
| A | `B3_W3_D7_06022026(4)` | 18 | 8 | 5 |

**Non-convulsive recall: 15.0% (weighted 43/57)**; 8/45 unweighted; stratum A 34.8%,
stratum B **0%**. In both stratum-B files the U-Net produced **no output whatsoever** — not
merely nothing above threshold — while 22 real seizures were present.

### This collapses the class-specific explanation

**15.0% non-convulsive against 15.3% convulsive.** Two independent references (Mir's
video-adjudicated convulsive set; exhaustive manual review for all types), two seizure
classes, the same answer. So there is no instrument mismatch, no band mismatch and no
convulsive-vs-non-convulsive asymmetry: **the frozen detector finds about 15% of seizures,
of any type.** Earlier drafts of this document attributed the convulsive failure to training
composition and the apparent non-convulsive success to a reference that could not see those
events. Both were over-interpretation of class-stratified numbers; the simple reading is
correct and the earlier one is withdrawn.

### Seizure burden

45 seizures in 48 animal-hours = **~22 per animal-day**, against the reference's rate of
0.64 (430 events over 16,114 animal-hours). The true burden is roughly **35x** what the
automated-candidate-plus-video reference captured. That is why nothing was ever going to
validate cleanly against it, and it is the strongest single argument in this paper for
exhaustive manual review as the anchor for any detector validation.

**Caveats:** 45 events from 4 recordings, all Batch 3. The U-Net detections in both
stratum-A files were left `pending`, so **precision is not computable from this sample** —
recall only. Three manual events fell below the 5 s inclusion criterion and were excluded.

## Precision, on the retained set

Retained = all batches except 4, minus Batch 2 UI ch3/ch4 and the dead electrode 449382.
Unbiased samples only — two hand-picked high-confidence files that read 22/22 are excluded
from every figure here.

| batch | detections @ conf>=0.5 | share | reviewed | real | precision | 95% CI |
|---|---|---|---|---|---|---|
| B1 | 160 | 17% | 7 | 6 | 86% | 42-100% |
| B2 | 437 | 47% | 36 | 13 | 36% | 21-54% |
| B3 | 337 | 36% | 6 | 6 | 100% | 54-100% |

**Volume-weighted precision 68%** over 934 detections; **95%** if Batch 2 is excluded as
well (497 detections). Confidence is monotonically informative: 10% below 0.5, rising
through 24%, 35%, 42% to 67% above 0.8.

**These detections are overwhelmingly non-convulsive**, so this precision figure and the
convulsive recall figure above describe **different event classes** and must not be read as
a single operating point. An earlier version of this document quoted them side by side;
that was wrong.

## Secondary finding: recording quality costs precision, measurably

| | |
|---|---|
| Batch 4 | excluded entirely on expert visual review — noisy in bursts, and those bursts are where the false positives are |
| 3 animals (483552/483553/483555) | 30,669 detections = 74% of all output, **1** confirmed seizure between them. A stratified sample of 113, drawn before review across all three confidence terciles, returned **0** real events (95% upper bound 3.6%) |
| Batch 2 UI ch3/ch4 | excluded; those two channels carry 83% of the batch's detections |
| 449382 | dead electrode — demeaned `rms` 0.000, highest line noise 2.46 |

Per-channel precision within retained Batch 2: UI ch2 22%, **ch6 64%**, ch7 9%.

**The counterintuitive part, and the one worth reporting:** ch6 is animal **450096, which
has the highest spectral rhythmicity of all 32 animals** (`prominence_db` 12.9) — and the
best precision of the three. High background rhythmicity lowers precision but does **not**
make a recording unusable. Two principled automated exclusion metrics were built and both
failed their pre-registered tests (`EXCLUSION_CRITERIA.md`); the search was stopped rather
than fitting a third to the same five animals. **"Noisy" belongs in the results as a
per-channel precision figure, not as a binary verdict.**

Cohort-level context: RAM_GDNF carries a more rhythmic, lower-amplitude background than
SV2A (duty cycle 0.53 vs 0.20, p=1.7e-32; rms 0.053 vs 0.077) — a consistent shift with
moderate effect size (AUC 0.66-0.69) and heavily overlapping distributions. That is why no
subset could be excluded: the whole cohort sits further along the same axis.

## Methodological findings

These are results in their own right, and arguably the most transferable part.

1. **Operating points do not transfer between models.** Running retrained arm A at the
   frozen point (0.5 / hysteresis boundary 0.1) produced events of median **1,713 s**
   covering **72% of all recorded time**; its apparent 69.3% convulsive recall was *at
   chance*. Its own validation selected 0.9, and the retrained Stage 2 selected 0.55 against
   the frozen 0.45. Anyone reusing published weights together with published thresholds —
   the normal thing to do — would hit this, and could mistake it for high sensitivity by
   looking only at recall.
2. **A reference built for one question misleads on another.** Scoring a predominantly
   non-convulsive detector against a convulsive reference produces numbers that look like
   failure without measuring what they appear to.
3. **Reused annotations can be actively harmful as negatives.** Mir's `False` rows were
   correct for his question and wrong for ours. Removing them lowered val loss from 0.676 to
   0.595 (`MODELS.md`).
4. **Sampling discipline caught every error.** The picture moved four times — pessimistic
   from pooled unthresholded output, optimistic from files chosen for high confidence, back
   to uncertain once weighted by output volume, and a chance-level recall mistaken for a 13x
   improvement. Every correction came from a sanity check, not from new data.

## What is NOT claimed

* **No absolute recall, for any seizure type.** Every recall figure is relative to Mir's
  candidate pool. An exhaustively annotated gold set would have fixed this and was
  deliberately not done (2026-10-07).
* **No non-convulsive precision or recall at cohort scale** beyond the 49-event sample.
* **No treatment-group or burden comparison** — out of scope by design.
* Batch 3 is 36% of retained output on 6 reviewed events; that CI is wide.
