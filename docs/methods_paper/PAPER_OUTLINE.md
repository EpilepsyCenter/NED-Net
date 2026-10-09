# Paper outline

**Written 2026-10-08, rewritten 2026-10-09.** Framing set by Marco. Numbers cross-referenced
to `PHASE1_RESULTS.md`, `MODELS.md` and `RUN_LOG.md`. Anything marked _pending_ is not yet
measured. Previous version in `archive/PAPER_OUTLINE_20261008.md`.

> **SCOPE.** This paper measures **in-sample** performance: what a lab gets on its own data
> after annotating it. That is how the source model's 93.6% was obtained and it is the
> question the pipeline exists to answer. Out-of-sample folds are supporting context only.
> See `NEXT_STEPS.md` — settled, not to be re-argued.

## What the paper is about

**NED-Net is software for building seizure-detection models from human annotations** —
annotate in the UI, train, detect, review, retrain. The paper is not "here is a good
detector". It is: what does a model built this way actually do on new animals, what does it
cost to keep it working, and how many annotations does that take.

Working claim: **a trained model does not carry over to a new batch of animals, let alone
another lab — you refine it as each batch arrives, and the pipeline is built to make that
loop cheap.** Electrode position, impedance and noise change between implants, so the model
is maintained, not delivered.

## Introduction

1. Automated rodent EEG seizure detection is necessary at scale — 16,114 animal-hours in
   this cohort alone. Published detectors report 90%+ performance; what happens on the next
   cohort is rarely measured.
2. We built a first model on one cohort (SV2A, 867 annotated seizures) and reached
   **93.6% precision** on human spot-check (160/171 adjudicated events, 30 files).
3. We applied it unchanged to a second cohort of different animals on the same rig. It
   failed — and the failure was not uniform, it was per-animal.
4. We annotated the new animals, retrained, measured the recovery, quantified the cost, and
   note that nothing in the pipeline is seizure-specific.

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
| convulsive recall vs video-adjudicated reference | — | **15.3%** (34/222) |

**Precision is statistically indistinguishable between cohorts. Recall collapses.** The
model has not forgotten what a seizure looks like; it has stopped finding them.

### R3 — The failure is silence, and it is per-animal not per-batch
Exhaustive manual review of 4 recordings — every event >=5 s on all 8 channels, independent
of detector output. 48 animal-hours, **50 real seizures**.

| | precision | recall |
|---|---|---|
| recordings where it fires (43% of files) | **92%** (12/13) | **43%** (12/28) |
| recordings where it does not (57% of files) | undefined | **0%** (0/22) |

**Per animal** (frozen model, retained B1-B3, animals with >=5 convulsive reference events;
it trained on none of them):

| animal | gt | caught | recall | detections | |
|---|---|---|---|---|---|
| 449388 | 26 | 15 | **58%** | 31 | |
| 450096 | 11 | 4 | 36% | 295 | |
| 449381 | 14 | 4 | 29% | 34 | |
| 459662 | 8 | 2 | 25% | 18 | |
| 459663 | 51 | 5 | 10% | 111 | |
| 449385 | 10 | 1 | 10% | 792 | |
| 459657 | 12 | 1 | 8% | 148 | |
| 459658 | 50 | **0** | **0%** | **82** | fires, never right |
| 459659 | 25 | **0** | **0%** | 177 | fires, never right |
| 459661 | 7 | **0** | **0%** | 6 | |
| 449387 | 5 | **0** | **0%** | **0** | mute |

* **Median per-animal recall 10%** (IQR 0-27%, range 0-58%). **4 of 11 animals at exactly
  zero.**
* **Within-batch spread (58 points) exceeds between-batch spread (31 points)**, so batch is
  not the meaningful unit — the implant is. Batches matter only because that is how animals
  arrive; a lab recording all animals at once would see the same spread.
* **Two distinct failure modes**, which should be reported separately:
  **mute** (449387: 5 seizures, 0 detections) and **firing but mislocalised**
  (459658: 50 seizures, 82 detections, 0 correct).

**The practical consequence is the strongest single statement in the paper:** a lab running
an inherited model unmodified gets **nothing** from roughly a third of its implants, with no
error and no warning.

### R4 — Annotating the cohort and retraining recovers most of the loss
Retained B1+B2, convulsive reference (n=68), frozen and retrained scored through the
identical code path so the comparison is paired. Stage 2 tuned per model, which is itself a
finding (R7).

| | convulsive recall | fires on | median IoU | median event |
|---|---|---|---|---|
| frozen `UNetv2`, Stage 1 | 38.2% | 51% of recordings | 0.36 | 9 s |
| frozen, full cascade | 27.9% | **16%** | 0.39 | 14 s |
| **retrained, Stage 1** | **50.0%** | **77%** | **0.60** | 19 s |
| **retrained, cascade @ tuned 0.30** | **47.1%** | **73%** | 0.66 | 26 s |

Per animal-channel: **3 improved, 4 unchanged, 0 worse** (cascade scope). Two cases carry
the argument:
* animal **449387** was **mute** — 0 detections across 5 seizures — and now fires.
* animal **449385** improved while firing **less**: 792 -> 729 detections, 1 -> 4 caught. So
  the gain is not a volume effect.

**Coverage is the unconfounded result.** Per-recording silence was the defect; the frozen
cascade fires on 16% of reviewed recordings and the retrained one on 73%, a 4.4x increase
that does not depend on the precision metric at all.

**Boundaries improve, and not via a threshold.** Within the retrained model median IoU
barely moves across hysteresis boundary 0.1/0.3/0.5 (0.66/0.58/0.60), so the frozen model's
9 s-fragment-inside-a-37 s-seizure behaviour was a property of the weights.

**This is the direct analogue of how the source model's 93.6% was obtained** — measured on
the cohort it trained on — which is what makes the comparison fair.

_Supporting context (not the headline):_ on a held-out batch the same retraining gives
5.2% -> 8.5% (Stage 1) and 2.6% -> 7.8% (cascade), coverage 58% -> 90%. Relative gain is
**1.31x in-sample and 1.63x out-of-sample**, so the in-sample figure is not an artefact of
memorisation. Batch difficulty dominates both: the frozen model, which trained on neither,
scores 7.3x better on B1+B2 than on B3.

**_pending_** — the `temporal` split result (jobs 3831662/3831663), which is the primary
design: hold out each animal's **later recordings**, every animal represented. That matches
the real workflow — annotate as the experiment starts, let the model handle the rest — and
replaces the figures above as R4's headline when it lands.

### R5 — What is still unmeasured: precision of the retrained model
Every precision figure available is scored against a convulsive-only candidate reference,
which structurally cannot credit a non-convulsive detection — and most of the retrained
output is non-convulsive. So **"fires more" is established and "finds more" is not.**
A pre-registered stratified review sample is the remaining gap, and it matters: animal
459658 produces 569 detections at the tuned operating point and catches 0 of 10 seizures.

### R6 — The annotation requirement
| training positives | source | outcome |
|---|---|---|
| 867 | SV2A alone, **one cohort** | working model, 93.6% precision |
| 184 | RAM_GDNF alone | **unusable** — f1 0.012, 12,070 predictions for 52 true events |
| 865 | SV2A + RAM_GDNF | f1 0.532 |

**The floor is in the high hundreds of annotated seizures. A single cohort can clear it** —
SV2A did. RAM_GDNF's annotated subset did not, because its seizure frequency was lower, so
the same annotation effort yielded fewer events. The requirement is a count of seizures, not
a count of cohorts. Because the unit of variability is the animal (R3), the useful form of
this question is **how many animals must be annotated**, not how many batches.

### R7 — Five pitfalls anyone repeating this will hit
Each cost us a measurable amount, and none is specific to seizures.

1. **Operating points do not transfer.** The retrained model at the *source* model's
   published threshold flagged **72% of recorded time** as ictal (median event 1,713 s) and
   showed 69% recall — at chance for that coverage. At its own threshold: 0.01% of time,
   median 6.4 s. Reusing published weights *and* published thresholds produces a number that
   looks like high sensitivity.
2. **Borrowed negatives can poison training.** Reference rows meaning "not a *convulsive*
   seizure" contain real non-convulsive activity. Used as detector hard negatives they cost
   event_f1 **0.329** against **0.532** without them. The same rows are **correct** as
   convulsive-classifier negatives — one annotation set, opposite roles, because the two
   stages answer different questions.
3. **A second-stage classifier trained only on positives silently stops working after
   retraining.** The convulsive classifier saw only confirmed seizures, yet runs on every
   detection. For the source model that failed safely — unfamiliar noise scored low and
   precision rose 33% -> 61% @0.45 -> 81% @0.80. After retraining on the new cohort the same
   noise resembles what it learned, `convulsive_confidence` median moves 0.12 -> 0.66, and
   **precision is flat at 11% at every threshold**. No threshold fixes it; it needs
   negatives. **Re-tune and re-evaluate stage 2 after every retrain; never inherit its
   threshold.**
4. **`detection_method` records who *proposed* an event, never who *labelled* it.** Model
   output written back as annotations becomes next-generation training data, and any field
   the human did not adjudicate is laundered into ground truth. A sidecar mixing human and
   model fields needs **per-field** provenance. (Here the affected count was 0.7% only
   because the review queue had barely started; at full review it would not be.)
5. **The retrain is only as current as the label snapshot.** Three successive retraining
   arms trained without 52 manual annotations and 344 adjudicated detections because the
   training tree had not been refreshed since before the review sessions. Nothing errored.
   Pipelines that re-scan a folder rather than pin a dataset definition will do this.

### R8 — Recording quality costs precision, measurably
Per-channel precision within one batch: **64%, 22%, 9%**. Two pre-registered automated
quality metrics failed to predict which channels would be poor, so quality is reported as a
per-channel precision figure rather than a binary verdict. Notably the
**highest-rhythmicity animal of 32 had the *best* precision of its batch** — which refutes
the premise of the exclusion metric directly and argues for reviewing rather than excluding.

## Discussion

**A model is maintained, not delivered.** Electrode position, impedance, noise floor and
cohort characteristics drift between implants. The workable pattern is incremental
refinement: each new batch contributes annotations, the model is retrained, performance is
re-measured. The pipeline exists to make that loop cheap — detect, review in the UI,
retrain, repeat. Over successive batches a model with good precision *and* recall is
reachable; from a single batch it is not.

**The implant is the unit of variability, not the cohort.** Per-animal recall spans 0-58%
within a single batch, a wider range than between batches, and a third of implants return
nothing. Aggregate cohort numbers hide this completely. Any report of a rodent EEG detector
should therefore be per-animal, and any deployment should include per-animal review before
the output is trusted.

**Precision transfers; engagement does not.** Discrimination survives the move between
cohorts — where the model fires it is right about as often as on its source data. What
fails is whether it fires at all on a given recording. That points at calibration and
per-channel normalisation rather than at the classifier.

**Two failure modes need different responses.** A mute channel is detectable automatically —
zero detections over hours is a flag anyone can compute. A channel that fires steadily and
is never right is not, and is the dangerous case: it produces plausible output volume while
contributing nothing.

**The pipeline is not specific to seizures.** It learns whatever humans mark as `confirmed`
or `rejected`; the same machinery already runs an interictal-spike detector. The pitfalls in
R7 therefore apply to anyone training an event detector on annotated electrophysiology.

**Validation methodology determines the conclusion.** The same detector on the same data
read as 0.12% precision (pooled, unthresholded), 100% (files chosen for high confidence),
34% (randomly sampled), 68% (volume-weighted) and 92%-where-it-fires (exhaustive). Four
exhaustively-reviewed recordings settled what sampled review of the detector's own output
could not. Separately, the retrained-vs-frozen comparison itself moved three times on
scoring choices alone — detection scope (stage 1 vs cascade), stage-2 threshold, and whether
the comparison was paired through one code path. **Every correction came from how events
were selected, not from new data.** That is the strongest available argument for
pre-registered sampling in detector validation, and it belongs in the paper.

## Limitations

* **Precision of the retrained model is unmeasured** (R5). This is the largest gap.
* Recall beyond 48 animal-hours in one batch is not exhaustively measured. The 92%
  where-it-fires precision rests on 13 adjudicated detections (95% CI 64-100%).
* The convulsive reference is automated-candidate-derived with video adjudication, and
  covers convulsive events only — so ~78% of the detector's output has no reference in
  either direction in this cohort.
* Per-animal figures rest on 11 animals with >=5 reference events; enough to separate 0%
  from 58%, not to rank adjacent animals.
* Exclusions were performance-informed in origin; every stratum is reported rather than
  claimed as independent (`EXCLUSION_CRITERIA.md`).
* Event matching is interval overlap with 5 s slack. Tested and robust — tightening the
  slack or forcing one-to-one assignment changes nothing — but boundary agreement is a
  separate, weaker claim and is reported as IoU rather than asserted from the loose rule.

## Methods notes that must appear

* **Positives capped at 100 s.** The reference's longer rows are chained seizures annotated
  as one block; SV2A's 1,136 human-reviewed events top out at **84 s**, which is the
  justification. Drops 15 of 430 reference events and 0 of SV2A's.
* **Random background negatives never drawn from annotated regions.**
* **Three split designs, reported as different questions**: by animal (transfer), by
  recording, and **temporal** (each animal's later recordings held out — the deployment
  design, and the primary one). Individual events are never split: events inside one
  90-minute recording share electrode state, ambient noise and often one seizure cluster.
* **Stage 1 is reported as detection capability; Stage 2 separately**, because Stage 2 is an
  informative type filter for the source model and not for the retrained one.
* Checkpoint selection is on `best_event_f1`; `val_loss` ranks arms differently and
  reversed one conclusion before this was noticed.

## What is needed to finish

1. **The temporal pair** — jobs 3831662 (`ramgdnf_temporal`) + 3831663 (`conv_temporal`),
   then detection over B1-B3 and its own operating-point sweep. This becomes R4's headline.
2. **R5: precision of the retrained model** — pre-registered stratified review sample.
   The single most valuable remaining measurement.
3. **Does the stage-2 fix work?** Re-detect with `conv_temporal` and sweep. Flat 11% means
   stage 2 is written up as a limitation (R7.3); rising means the cascade gains a usable
   precision filter.
4. Optionally **more exhaustive review** — the only method that gave a stable answer.
   Twelve recordings instead of four would turn wide intervals into estimates.
