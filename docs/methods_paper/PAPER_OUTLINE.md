# Paper outline

**Written 2026-10-08.** Framing set by Marco. Numbers cross-referenced to
`PHASE1_RESULTS.md`, `MODELS.md` and `RUN_LOG.md`. Anything marked _pending_ is not yet
measured.

## What the paper is about

**NED-Net is software for building seizure-detection models from human annotations** —
annotate in the UI, train, detect, review, retrain. The paper is not "here is a good
detector". It is: does a model built this way transfer, what does it cost to keep it
working, and how many annotations does that take.

Working claim: **a trained model does not transfer to another batch of animals, let alone
another lab — but incremental retraining on each new batch makes it work, and the pipeline
is built to make that loop cheap.**

## Introduction

1. Automated rodent EEG seizure detection is necessary at scale — 16,114 animal-hours in
   this cohort alone. Published detectors report 90%+ performance; transfer to new data is
   rarely measured.
2. We built a first model on one cohort (SV2A, 867 annotated seizures) and reached
   **93.6% precision** on human spot-check (160/171 adjudicated events, 30 files).
3. We applied it unchanged to a second cohort of different animals on the same rig. It
   failed.
4. We retrained with annotations from the new animals, measured the recovery, quantified the
   annotation cost, and note that nothing in the pipeline is seizure-specific.

## Results

### R1 — The source model works
SV2A: **93.6% precision** (160/171, human spot-check); event recall 87%, event_f1 0.78 on
its validation split (0.81 at its own best threshold of 0.7).

### R2 — On a new cohort, recall collapses while precision holds
1,377 recordings, 16,114 animal-hours, same rig, different animals.

| | SV2A (source) | RAM_GDNF (new) |
|---|---|---|
| precision, human-adjudicated | **93.6%** (160/171) | **92%** (12/13) |
| recall | 87% (val split) | **43%** where it fires; **18%** overall |
| convulsive recall vs independent video-adjudicated reference | — | **15.3%** (34/222) |

**Precision is statistically indistinguishable between cohorts. Recall collapses.** The
model has not forgotten what a seizure looks like; it has stopped finding them.

### R3 — The failure is per-recording silence
Exhaustive manual review of 4 recordings — every event >=5 s on all 8 channels, independent
of detector output, then the detector's own output adjudicated on the same files.
48 animal-hours, **50 real seizures**.

| | precision | recall |
|---|---|---|
| recordings where it fires (43% of files) | **92%** (12/13) | **43%** (12/28) |
| recordings where it does not (57% of files) | undefined | **0%** (0/22) |

Zero detections against 22 real seizures across two 90-minute recordings. Within a single
recording: 4 of 6 caught on UI ch3, 1 of 3 on ch8, **0 of 9** across ch1 and ch6 — same
model, same minutes. Matches the cohort-wide asymmetry (ch8: 52 detections against 107
confirmed seizures; ch4: 15,291 against 8).

### R4 — Retraining on the new cohort's annotations recovers most of the loss
**IN-SAMPLE RESULT, 2026-10-08.** Retained B1+B2, convulsive ground truth (n=68), frozen
and retrained models scored through the identical code path so the comparison is paired.
Stage 2 tuned per model (frozen 0.45, A3 0.30) — see R6.

| | convulsive recall | fires on | median IoU | median event |
|---|---|---|---|---|
| frozen `UNetv2` | 27.9% | **16%** of recordings | 0.36 | 9 s |
| **A3 retrained** | **47.1%** | **73%** | 0.60-0.66 | 19-26 s |

Per animal-channel: **4 improved, 2 unchanged, 1 worse** (the regression is a single
seizure). Two cases carry the argument:
* animal **449387** was **wholly silent** — 0 detections across 5 seizures — and now fires.
* animal **449385** improved while firing **less**: 792 -> 729 detections, 1 -> 4 caught. So
  this is not a volume effect.

Boundaries improve too, and not because of a threshold: within A3 the median IoU barely
moves across hysteresis boundary 0.1/0.3/0.5 (0.66/0.58/0.60), so the frozen model's
9 s-fragment-inside-a-37 s-seizure behaviour was a property of the weights, which retraining
fixed.

**This is the direct analogue of how the source model's 93.6% was obtained** — measured on
the cohort it was trained on — which is what makes the comparison fair. It is **in-sample**
and must be labelled as such everywhere. The out-of-sample counterpart is held-out Batch 3
(job 3825993, pending), and the gap between the two rows is the paper's central quantity:
the difference between annotating a cohort and inheriting a model.

**Out-of-sample (held-out Batch 3, job 3825993):** recall 5.2% -> 8.5% (Stage 1),
2.6% -> 7.8% (cascade), coverage 58% -> 90%. The boundary improvement does **not** transfer
(median IoU 0.40 -> 0.39, against 0.36 -> 0.60 in-sample), so better delineation is fitted,
not learned.

**Do not read the 50.0% vs 8.5% difference as a transfer penalty.** The frozen model trained
on neither batch and still scores 7.3x better on B1+B2 (38.2%) than on B3 (5.2%), so that
spread is batch difficulty. Normalised, retraining gains **1.31x in-sample and 1.63x
out-of-sample** — larger out-of-sample, so the in-sample result is not an artefact of
memorisation. **Measuring the transfer penalty requires a LOCO fold holding out an easy
batch (B1 or B2); B3 is both held out and hard.**

**What is not yet measured: precision of the retrained model.** Every precision figure
available is scored against Mir's convulsive-only candidate set, which structurally cannot
credit a non-convulsive detection, and most of the retrained output is non-convulsive. So
"fires more" is established and "finds more" is not. A pre-registered stratified review
sample is required and is the last gap in this section.

### R5 — The annotation requirement
| training positives | source | outcome |
|---|---|---|
| 867 | SV2A alone, **one cohort** | working model, 93.6% precision |
| 184 | RAM_GDNF alone | **unusable** — f1 0.012, 12,070 predictions for 52 true events |
| 865 | SV2A + RAM_GDNF | f1 0.532 |

**The floor is in the high hundreds of annotated seizures. A single cohort can clear it** —
SV2A did. RAM_GDNF's annotated subset did not, because its seizure frequency was lower, so
the same annotation effort yielded fewer events. The requirement is a count of seizures, not
a count of cohorts.

### R6 — Two pitfalls anyone repeating this will hit
* **Operating points do not transfer.** The retrained model at the *source* model's
  published threshold flagged **72% of recorded time** as ictal (median event 1,713 s) and
  showed 69% recall — at chance for that coverage. At its own threshold: 0.01% of time,
  median 6.4 s. Reusing published weights *and* published thresholds produces a number that
  looks like high sensitivity.
* **Borrowed negatives can poison training.** Reference rows meaning "not a convulsive
  seizure" contain real non-convulsive activity. Used as hard negatives they cost
  event_f1 **0.329** against **0.532** without them.

### R7 — Recording quality costs precision, measurably
Per-channel precision within one batch: **64%, 22%, 9%**. Two pre-registered automated
quality metrics failed to predict which channels would be poor, so quality is reported as a
per-channel precision figure rather than a binary verdict. Notably the highest-rhythmicity
animal of 32 had the *best* precision of its batch.

## Discussion

**A model is maintained, not delivered.** Electrode position, impedance, noise floor and
cohort characteristics drift between batches. The workable pattern is incremental
refinement: each new batch contributes annotations, the model is retrained, performance is
re-measured. The pipeline exists to make that loop cheap — detect, review in the UI,
retrain, repeat. Over successive batches a model with good precision *and* recall is
reachable; from a single batch it is not.

**Precision transfers; recall does not.** Discrimination survives the move between cohorts;
engagement with a given recording does not. That points at calibration and per-channel
normalisation rather than at the classifier — and it explains why a better discriminator did
not help (R4).

**The pipeline is not specific to seizures.** It learns whatever humans mark as `confirmed`
or `rejected`; the same machinery already runs an interictal-spike detector. The transfer
pitfalls in R6 therefore apply to anyone training an event detector on annotated
electrophysiology.

**Validation methodology determines the conclusion.** The same detector on the same data
read as 0.12% precision (pooled, unthresholded), 100% (files chosen for high confidence),
34% (sampled), 68% (volume-weighted) and 92%-where-it-fires (exhaustive). Four
exhaustively-reviewed recordings settled what sampled review of the detector's own output
could not.

## Limitations

* Recall beyond 48 animal-hours in one batch is not measured. Precision rests on 13
  adjudicated detections (95% CI 64-100%).
* One LOCO fold completed (see below).
* The convulsive reference is automated-candidate-derived with video adjudication, and
  covers convulsive events only.
* Exclusions were performance-informed in origin; every stratum is reported rather than
  claimed as independent (`EXCLUSION_CRITERIA.md`).

## LOCO folds — definition and status

**LOCO = leave-one-cohort-out.** Train on every batch except one, test on the held-out
batch, repeat. Cross-validation at batch level, so the test animals are never seen in
training — which is what matters for a transfer claim.

| fold | test batch | retained convulsive GT | status |
|---|---|---|---|
| 1 | **Batch 3** | 153 | **done** (arms A, A2, A3 trained; A tested) |
| 2 | **Batch 1** | 56 | not started |
| 3 | **Batch 2** | 13 | not started — **underpowered**, CI would be near-uninformative |
| — | Batch 4 | excluded from analysis | not a test fold; may remain in training |

So **two usable folds (B3, B1)** plus B2 as a weak third. One fold cannot support a claim
about incremental refinement; a reviewer will ask for the others.

## What is needed to finish

1. **R4's numbers** — operating-point validation, then arm A2 on Batch 3 once.
2. **The Batch-1 fold** — train holding out B1, test on B1. Makes R4 a characterisation
   rather than an anecdote.
3. Optionally, **more exhaustive review** — the only method that gave a stable answer.
   Twelve recordings instead of four would turn wide intervals into estimates.
