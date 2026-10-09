# Paper outline

**Written 2026-10-08, rewritten 2026-10-09.** Framing set by Marco. Numbers cross-referenced
to `PHASE1_RESULTS.md`, `MODELS.md` and `RUN_LOG.md`. Anything marked _pending_ is not yet
measured. Previous version in `archive/PAPER_OUTLINE_20261008.md`.

> **SCOPE.** This paper measures **in-sample** performance: what a lab gets on its own data
> after annotating it. That is how the source model's 93.6% was obtained and it is the
> question the pipeline exists to answer. Out-of-sample folds are supporting context only.
> See `NEXT_STEPS.md` — settled, not to be re-argued.

## What the paper is about

**This is a software paper.** NED-Net is a tool for building and maintaining event-detection
models from human annotations: annotate in the UI, train, detect, review, retrain. The
contribution is the software and the workflow it implements.

The validation numbers exist to support a **design claim**, not to advertise a detector:
**a model cannot be delivered once and left alone.** It does not carry over to a new batch
of animals, let alone another lab, because electrode position, impedance and noise differ
per implant — so the software is built around continual refinement rather than one-shot
training. Every feature described in R1 exists for that reason, and the measurements in
R2-R8 are the evidence that it is needed and that it works.

What the paper is NOT: a benchmark claiming state-of-the-art detection, or a comparison
against other detectors on public data.

## Introduction

1. Automated rodent EEG event detection is necessary at scale — 16,114 animal-hours in this
   cohort alone. Published detectors report 90%+ performance; what happens on the next
   cohort, in another lab, with other electrodes, is rarely measured, and published weights
   are rarely usable as-is.
2. The practical need is therefore not a better detector but **a tool that lets a lab build
   its own and keep it current**. We present NED-Net, which implements that loop end to end:
   annotation UI, training, batch and live detection, review, retraining.
3. To show the loop is necessary we built a model on one cohort (SV2A, 867 annotated
   seizures, **93.6% precision**), applied it unchanged to a second cohort on the same rig,
   and measured what happened. It failed — and not uniformly: **per implant**.
4. We then annotated the new animals through the UI, retrained, and measured the recovery.
   We report the annotation cost, five pitfalls the exercise exposed, and the fact that
   nothing in the pipeline is seizure-specific.

## Results

### R1 — The software and the loop it implements
The unit of work is a **sidecar annotation file** (`*_ned_annotations.json`) beside each EDF,
holding per-event `confirmed` / `rejected` / `pending` labels, the channel, and a `features`
dict. Human labels and model proposals live in the same format, which is what makes the loop
closed: model output becomes a review queue, and reviewed output becomes training data.

| component | what it does |
|---|---|
| **Load / montage** | per-channel animal, cohort and group IDs (`*_ned_channels.json`); one recording = 8 animals, so everything downstream is per-animal |
| **Detection (rule-based)** | parameter-driven detectors (autocorrelation, spectral band, amplitude) to bootstrap a first annotation set with no model at all |
| **Training** | U-Net seizure detector (stage 1) and convulsive classifier (stage 2), trained from the sidecars; animal-, recording- or time-based validation splits |
| **Analysis** | trained-cascade inference: single file, **batch**, or **live** |
| **Review** | queue of `pending` events with boundary editing; confirm/reject writes straight back to the sidecar |
| **Results** | per-project SQLite DB aggregating every run, per animal and group |
| **HPC path** | the same training and detection entry points run unattended under SLURM |

**Live detection mode is the feature this paper's argument is built on**
(`analysis.py:1089`). It watches an acquisition folder, waits a configurable delay (default
30 s) so the recorder has finished writing, and runs the **full cascade** on each new file as
it lands, writing events to the project database. A `live_template` keys the montage **by
channel** rather than by filename, because live files arrive with unknown names and a session
runs a fixed montage — so events are attributed to individual animals from the first file.
Optional backlog processing covers files recorded before monitoring started.

**The software does not require machine learning at all.** The rule-based detectors
(autocorrelation, spectral band, amplitude) are a complete workflow on their own: detect,
review, filter, export, aggregate per animal in the Results tab — no model, no training, no
GPU. They exist to bootstrap a first annotation set, but a lab that never trains a model can
use NED-Net as a classical detection and review tool. The ML path is an upgrade, not a
dependency.

**Why that matters for the design claim:** it makes the refinement loop concurrent with the
experiment. A researcher can annotate during the first recording days, retrain on their own
animals, and have a model calibrated to that cohort ready before the bulk of the protocol is
recorded — instead of discovering after the fact that an inherited model returned nothing for
a third of the implants (R3). The alternative — record everything, then analyse — provides no
opportunity to correct the model while the data that would fix it is being produced.

### R2 — A model built in the UI works on the cohort it was built from

SV2A: **93.6% precision** (160/171, human spot-check); event recall 87%, event_f1 0.78 on
its validation split (0.81 at its own best threshold of 0.7).

### R3 — On a new cohort, recall collapses while precision holds
1,377 recordings, 16,114 animal-hours, same rig, different animals.

| | SV2A (source) | RAM_GDNF (new) |
|---|---|---|
| precision, human-adjudicated | **93.6%** (160/171) | **92%** (12/13) |
| recall | 87% (val split) | **43%** where it fires; **18%** overall |
| convulsive recall vs video-adjudicated reference | — | **15.3%** (34/222) |

**Precision is statistically indistinguishable between cohorts. Recall collapses.** The
model has not forgotten what a seizure looks like; it has stopped finding them.

### R4 — The failure is silence, and it is per-animal not per-batch
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

### R5 — Annotating the cohort and retraining recovers most of the loss
Retained B1+B2, convulsive reference (n=68), frozen and retrained scored through the
identical code path so the comparison is paired. Stage 2 tuned per model, which is itself a
finding (R8.3).

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
replaces the figures above as R5's headline when it lands.

### R6 — What the available references can and cannot measure
Three references exist and they measure different things. Being explicit about this is
part of the methodological contribution.

| reference | n | covers non-convulsive? | independent of detector output? |
|---|---|---|---|
| external candidate set, video-adjudicated | 430 | **no** — convulsive/behavioural only | yes |
| **exhaustive review**, 4 recordings, every event >=5 s on all 8 channels | **50** | **yes** | **yes** |
| **reviewer's manual additions in the UI** | **52** | **yes — all 52 are non-convulsive** | **no** — added while reviewing detector proposals |

So it is **not** true that the non-convulsive arm has no reference. The exhaustive review is
small but independent and is what produced the only stable numbers in the project
(**92% precision where it fires, 43% recall, 0/22 where it does not**). The 52 manual
additions confirm the cohort's real events are predominantly non-convulsive — which is
itself a finding, since the external reference contains none of them — but recall measured
against them would be biased upward, because they were added while looking at the detector's
own proposals.

**What remains unmeasured is precision of the *retrained* model.** The convulsive reference
cannot credit its non-convulsive detections, and the exhaustive review predates it. So
"fires more" is established and "finds more" is not. It matters: animal 459658 produces 569
detections at the tuned operating point and catches 0 of 10 reference seizures.

**The measurement, pre-registered here before the sample is drawn.** Spot-check review of the
retrained model's own detections, in the UI, confirm/reject per event:
* **Sampling**: random, stratified by **animal** (not by batch — R4) and by confidence
  tercile, drawn from retained channels at the chosen operating point. Seed recorded in
  `RUN_LOG.md` before any event is opened.
* **Target n**: ~150-200 events. At 70% true, 150 events gives a 95% CI of roughly +-7
  points, which is enough to separate "usable" from "not" and to compare against the
  source model's 93.6%.
* **Two numbers come out of one review pass**: overall precision, and **convulsive
  precision** on the subset Stage 2 labelled convulsive — which is also the only honest test
  of whether the Stage-2 fix worked on real output rather than on validation windows.
* **Fixed in advance**: strata, n, seed, and that every drawn event is adjudicated (no
  skipping hard cases). Sampling discipline has reversed this project's conclusions three
  times; the protocol is stated before the data is seen for exactly that reason.

### R7 — The annotation requirement
| training positives | source | outcome |
|---|---|---|
| 867 | SV2A alone, **one cohort** | working model, 93.6% precision |
| 184 | RAM_GDNF alone | **unusable** — f1 0.012, 12,070 predictions for 52 true events |
| 865 | SV2A + RAM_GDNF | f1 0.532 |

**The floor is in the high hundreds of annotated seizures. A single cohort can clear it** —
SV2A did. RAM_GDNF's annotated subset did not, because its seizure frequency was lower, so
the same annotation effort yielded fewer events. The requirement is a count of seizures, not
a count of cohorts. Because the unit of variability is the animal (R4), the useful form of
this question is **how many animals must be annotated**, not how many batches.

### R8 — Five pitfalls anyone repeating this will hit
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

### R9 — Recording quality costs precision, measurably
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

**Live detection changes WHEN the loop can run, which is the point of the software.**
These measurements were made retrospectively, on 1,377 recordings already acquired — and
retrospectively there is nothing to be done about the finding that a third of implants
returned nothing. Live mode moves the loop inside the experiment: detect as files land,
review the first days, retrain on those animals, deploy a cohort-calibrated model for the
rest of the protocol. The per-animal result (R4) is what makes this more than a convenience
— the thing a researcher most needs to know early is *which of my implants is this model
useless on*, and live mode can answer it from the first days because it attributes events
per channel from the first file.

**ML is optional.** The rule-based path is a usable product by itself, and the paper should
say so plainly: a lab with no GPU, no annotations and no interest in training can still use
NED-Net to detect, review, filter and aggregate. That also makes the upgrade path
incremental — the classical detectors generate the first annotation set, which trains the
first model, whose output becomes the next review queue.

**The pipeline is not specific to seizures.** It learns whatever humans mark as `confirmed`
or `rejected`; the same machinery already runs an interictal-spike detector. The pitfalls in
R8 therefore apply to anyone training an event detector on annotated electrophysiology.

**Validation methodology determines the conclusion.** The same detector on the same data
read as 0.12% precision (pooled, unthresholded), 100% (files chosen for high confidence),
34% (randomly sampled), 68% (volume-weighted) and 92%-where-it-fires (exhaustive). Four
exhaustively-reviewed recordings settled what sampled review of the detector's own output
could not. Separately, the retrained-vs-frozen comparison itself moved three times on
scoring choices alone — detection scope (stage 1 vs cascade), stage-2 threshold, and whether
the comparison was paired through one code path. **Every correction came from how events
were selected, not from new data.** That is the strongest available argument for
pre-registered sampling in detector validation, and it belongs in the paper.

## Classical vs learned detection — a deliberate non-goal, and what we can already say

**Not attempted, on purpose.** A fair head-to-head would require tuning the rule-based
detectors and their post-detection filters as hard as the model was tuned — parameter sweeps
per cohort, per channel, plus the local-baseline and amplitude filters — which is a project
of its own and would not change this paper's claim. Stated here so it is visible as a choice
rather than an omission. If a reviewer asks, the tractable version is a small matched subset
with both pipelines swept over their own parameters.

**What the data already shows, without that work.** The external reference set *is* a
classical detector applied to this cohort: 15-80 Hz envelope peaks above median + 4 x MAD,
grouped into episodes of >=15 s at >=2.5 spikes/s, then adjudicated on video.

| | classical candidate generator | the learned cascade |
|---|---|---|
| human adjudications required | **12,083** | 1,377 files' output, reviewed in samples |
| events confirmed | **430** | — |
| yield per adjudication | **3.6%** | — |
| event population found | 15-80 Hz episodes >=15 s | ~2-14 Hz rhythmicity, any duration >=5 s |

Two instruments, **low overlap in both directions**: the classical generator cannot propose
the short non-convulsive events the model is best at (no >=15 s high-density episode around
them), and the model misses most of the classical generator's episodes. Neither is a superset
of the other. The honest statement is therefore not "learned beats classical" but **"they
find different things, and the classical route cost 12,083 human adjudications to yield 430
events"** — which is an annotation-economics argument, and the one that actually matters to a
lab choosing between them.

## Limitations

* **Precision of the retrained model is unmeasured** (R6). This is the largest gap.
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
   then detection over B1-B3 and its own operating-point sweep. This becomes R5's headline.
2. **R6: precision of the retrained model** — pre-registered stratified review sample.
   The single most valuable remaining measurement.
3. **Does the stage-2 fix work?** Re-detect with `conv_temporal` and sweep. Flat 11% means
   stage 2 is written up as a limitation (R8.3); rising means the cascade gains a usable
   precision filter.
4. **R7's third row needs its number swapped** (not the whole section): `865 -> f1 0.532`
   comes from arm A, trained on the stale label tree (R8.5), so it understates what the
   pipeline delivers. The 867 and 184 rows are unaffected. Replace with the temporal pair's
   figure when it lands.
5. **A figure for R1.** A software paper has to *show* the UI and the loop, not only
   describe it: annotation view, review queue, live-monitoring panel, and a loop diagram
   (rule-based bootstrap -> annotate -> train -> detect -> review -> retrain).
6. Optionally **more exhaustive review** — the only method that gave a stable answer.
   Twelve recordings instead of four would turn wide intervals into estimates.

## Author-facing note on framing

Keep the detector numbers subordinate. Every table in R3-R9 answers "is the refinement loop
necessary, and does it work?" — not "how good is this detector?". Read as a detection
benchmark, R3 and R4 look like a negative result; read as a software paper they are the
justification for the design. The ordering — software first, evidence second — is deliberate
and should survive revision.
