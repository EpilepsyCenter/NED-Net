# DRAFT — NED-Net manuscript

**Started 2026-10-09.** Target: **eNeuro**, Methods/New Tools (see `JOURNAL_SCAN.md`).
Numbers in `[[ ]]` are placeholders pending the temporal-pair result and the precision
sample. Everything outside `[[ ]]` is measured — see `RUN_LOG.md` for provenance.
Structure follows `PAPER_OUTLINE.md`; this file is the prose.

---

## Title (candidates)

1. **NED-Net: closing the annotation–training loop for rodent EEG event detection**
2. **A seizure detector is maintained, not delivered: software for per-cohort model
   refinement in rodent EEG**
3. **Per-animal failure of transferred seizure detectors, and a workflow that corrects it**

_Preference: (1) for a Methods/New Tools submission — names the tool and the contribution.
(2) is the better title if the result leads; keep as a fallback._

## Abstract (~250 words)

Long-term rodent EEG is central to epilepsy research, and manual review does not scale:
the cohort analysed here comprises 16,114 animal-hours. Automated detectors are available
and report high accuracy, but they are distributed as fixed models, and a laboratory that
adopts one has no way to adapt it to its own electrodes, animals and noise.

We present **NED-Net**, software that treats a detector as something to be maintained
rather than delivered. Human annotations, rule-based detections and model output share one
file format, so model proposals become a review queue and reviewed events become training
data. The same interface supports rule-based detection with no machine learning at all,
training of a U-Net event detector and a secondary classifier, batch inference, and
detection **during acquisition**, so a model can be refined while the experiment is still
running.

To test whether such refinement is necessary, we trained a detector on one cohort
(867 annotated seizures; 93.6% precision on human spot-check) and applied it unchanged to a
second cohort recorded on the same rig. Precision was preserved (92%) but recall collapsed,
and the collapse was **per implant**: median per-animal recall 10% (range 0–58%), with 4 of
11 animals yielding **no correct detections at all** — either silent or firing steadily in
the wrong places. Annotating the new cohort and retraining raised the fraction of
recordings on which the detector fired from 16% to 73% [[and recall from X% to Y%]].

A detector therefore cannot be inherited and trusted per animal. We report the annotation
cost of refinement, five reproducible pitfalls, and the software that makes the loop
routine.

## 1. Introduction

Chronic EEG is the primary read-out in most rodent epilepsy models, and recording volume
has outgrown manual review: the dataset analysed here contains 16,114 animal-hours across
1,377 recordings, which no laboratory can score by eye. Automated detection is therefore
not optional.

Capable rodent detectors exist. Feature-based classifiers reach sensitivity and positive
predictive value above 98% on the data they were developed on (Kamintsky et al., 2025),
and a generalised linear model over 141 features achieves pooled AUROC 0.995 on 1,012
spontaneous seizures (Fumeaux et al., 2020). Crucially, that same study also did what most
do not: it validated externally, on an independently constructed cohort of 96 rats, and
performance fell to AUROC 0.890. Separately, a deep convolutional classifier trained per
animal outperformed the same architecture trained across animals (0.960 vs 0.811;
Kotloski, 2023). The ingredients of the problem are thus already on record — detectors
degrade off their development data, and the degradation is at least partly animal-specific.

What follows from this has not been addressed. If a detector's performance depends on the
individual implant, then distributing a fixed model is the wrong delivery mechanism, and
the useful artefact is not a set of weights but **a workflow for producing and maintaining
them locally**. That reframing has three consequences this paper addresses: the failure has
to be characterised at the level a researcher acts on (the animal, not the pooled cohort);
the remedy has to be demonstrated, not assumed; and the cost of the remedy, in human
annotation effort, has to be stated honestly.

We present NED-Net, software built around that loop, and use it on two cohorts to measure
each of the three. We show that pooled metrics conceal the practically important failure —
that a transferred detector can return **nothing at all** for a substantial minority of
implants, with no error and no warning — that annotating the new cohort and retraining
restores detection across most recordings, and that the loop is cheap enough to run during
an experiment rather than after it. We also report five pitfalls that cost us measurable
performance and which anyone repeating this will meet, including one that reversed our own
conclusions three times: the choice of validation sampling.

Nothing in the pipeline is specific to seizures, and nothing requires machine learning;
the rule-based path is a complete workflow on its own.

## 2. Materials and Methods

### 2.1 Software architecture
_[Write from `PAPER_OUTLINE.md` R1. Cover: sidecar annotation format as the shared
currency; per-channel animal/cohort/group identity; rule-based detectors; U-Net stage 1 and
convulsive classifier stage 2; batch and live inference; review queue; per-project database;
SLURM entry points. Figure 1 = the loop + UI panels.]_

### 2.2 Live detection during acquisition
_[Watch folder, write-completion delay, full cascade per file, channel-keyed montage
template, backlog option, events to project DB. The design point: a fixed montage per
session means events are attributed to animals from the first file, so per-animal
performance is visible within days.]_

### 2.3 Animals, recordings and reference annotations
_[Two cohorts. SV2A: 867 annotated seizures, 165 EDFs. RAM_GDNF: 1,377 recordings,
16,114 animal-hours, 4 batches of 8 animals. Reference sets: (i) external
video-adjudicated candidate set, 12,083 adjudications yielding 430 convulsive/behavioural
seizures; (ii) exhaustive review of 4 recordings, every event >=5 s on all 8 channels,
48 animal-hours, 50 seizures; (iii) 52 manual additions, all non-convulsive. State plainly
what each can and cannot measure.]_

### 2.4 Model training
_[U-Net, 60 s windows, 250 Hz, per-channel z-scoring. Negative sourcing (rejected events as
hard negatives; random background that never lands on an annotated region). Positive
duration cap at 100 s, justified by the source cohort's 84 s maximum. Checkpoint selection
on best_event_f1. Three validation split designs — by animal, by recording, and temporal —
and why individual events are never split.]_

### 2.5 Evaluation
_[Event matching: interval overlap on the same file and channel with 5 s tolerance;
sensitivity analysis showing the result is robust to removing the tolerance and to forcing
one-to-one assignment; IoU reported separately for boundary agreement. Per-animal reporting
as the primary unit. Wilson intervals. Pre-registered sampling for the precision estimate,
with seed recorded before review.]_

## 3. Results

### 3.1 The software and the loop
_[Descriptive; Figure 1. No statistics.]_

### 3.2 A model built in the interface performs well on its own cohort
93.6% precision (160/171 adjudicated events across 30 recordings); event recall 87%;
event F1 0.78 on its validation split.

### 3.3 Applied unchanged to a second cohort, recall collapses while precision holds
Precision 93.6% -> 92% (12/13); recall 87% -> 43% where the detector fires, 18% overall;
convulsive recall against the external reference 15.3% (34/222). **The model has not
forgotten what a seizure looks like; it has stopped finding them.**

### 3.4 The failure is per implant, and takes two forms
Exhaustive review of 4 recordings (48 animal-hours, 50 seizures): where the detector fires
it is 92% precise and catches 43%; on the 57% of recordings where it does not fire, 0 of 22
seizures. Per animal (>=5 reference events, n=11 animals): median recall **10%**
(IQR 0–27%, range 0–58%), **4 animals at exactly zero**. Within-batch spread (58 points)
exceeds between-batch spread (31 points), so the implant and not the cohort is the unit of
variability. Two distinct modes: **silent** (one animal, 5 seizures, 0 detections) and
**firing but mislocalised** (one animal, 50 seizures, 82 detections, 0 correct).
_Figure 2 = per-animal recall distribution with the two failure modes marked._

### 3.5 Annotating the new cohort and retraining restores detection
Fraction of recordings with any detection: 16% -> 73%. [[Recall X% -> Y%.]] Event boundaries
improve from a median 9 s fragment inside a 37 s seizure (IoU 0.36) to [[IoU Z]].
[[Per-animal: A improved, B unchanged, C worse.]] Two cases are diagnostic: the silent
animal now fires, and one animal improved while firing *fewer* times (792 -> 729
detections, 1 -> 4 seizures caught), so the gain is not a volume effect.
_Figure 3 = paired per-animal before/after._

### 3.6 Precision of the retrained detector
[[Pre-registered stratified sample, n~180, seed recorded before review. Overall precision
P% (95% CI), convulsive precision Q%. Per animal and per confidence tercile.]]

### 3.7 Annotation cost
[[Training-set size vs outcome table, re-derived from the temporal pair.]] The requirement
is a count of annotated seizures, not of cohorts, and because the unit of variability is the
animal, the operative question is how many **animals** must be annotated.

### 3.8 Five pitfalls
_[From `PAPER_OUTLINE.md` R8: operating points do not transfer; borrowed negatives poison
one stage while being correct for the other; a second-stage classifier trained only on
positives silently stops discriminating after retraining; `detection_method` records who
proposed an event, not who labelled it; a retrain is only as current as its label snapshot.
Each with its measured cost.]_

### 3.9 Recording quality is a per-channel precision statement, not a verdict
Per-channel precision within one batch: 64%, 22%, 9%. Two pre-registered automated quality
metrics failed to predict which channels would be poor, and the highest-rhythmicity animal
of 32 had the best precision of its batch.

## 4. Discussion

_[From `PAPER_OUTLINE.md`. Order: (i) a detector is maintained, not delivered; (ii) the
implant is the unit of variability — report per animal, and note Kotloski's independent
intra- vs inter-animal result; (iii) precision transfers but engagement does not; (iv) the
two failure modes need different responses, and only one is auto-detectable; (v) live
detection changes *when* the loop can run, which is the practical contribution; (vi) ML is
optional and the pipeline is not seizure-specific; (vii) validation methodology determined
the conclusion — the same data read as 0.12%, 100%, 34%, 68% and 92% precision depending on
how events were sampled, and our own retrained-vs-source comparison moved three times on
scoring choices alone.]_

### Relation to prior work
Fumeaux et al. (2020) established that a rodent detector degrades on an independent cohort
(AUROC 0.995 -> 0.890); we report the same phenomenon at the unit a researcher acts on,
where it reads not as a tolerable loss of discrimination but as a subset of implants
returning nothing. Kotloski (2023) found per-animal training superior to across-animal
training; we quantify the distribution behind that mean and show that the scalable response
is a retraining loop rather than one model per animal. Kamintsky et al. (2025) provide a
rodent detector with a graphical interface; NED-Net differs in closing the loop from review
back into training, in learning from a user's own annotations rather than fixed engineered
features, and in detecting during acquisition.

### Limitations
[[Precision of the retrained detector — fill from 3.6.]] Exhaustive recall measurement
covers 48 animal-hours in one batch; the 92% where-it-fires precision rests on 13
adjudicated detections (95% CI 64–100%). The external reference covers convulsive and
behavioural seizures only, so most of the detector's output has no reference in either
direction in this cohort. Per-animal figures rest on 11 animals with >=5 reference events —
enough to separate 0% from 58%, not to rank adjacent animals. Exclusions were
performance-informed in origin and every stratum is reported rather than claimed as
independent. All recordings come from one laboratory and one acquisition rig; whether the
recovery reproduces elsewhere is untested and is the obvious next study.

## Figures

1. **The loop.** Rule-based bootstrap -> annotate -> train -> detect (batch/live) ->
   review -> retrain, with UI panels: annotation view, review queue, live-monitoring panel.
2. **Per-animal recall of the transferred model**, with silent and mislocalised animals
   marked.
3. **Paired per-animal recall and coverage**, source model vs retrained.
4. **Operating-point and boundary behaviour** — why reusing a published threshold is unsafe.
5. _(supplementary)_ Validation-methodology panel: the same detector read five ways.

## Data and code availability
Code, trained weights, annotation sidecars and the dataset definitions used for every model
reported. [[Repository DOI, licence, archived release tag.]]
