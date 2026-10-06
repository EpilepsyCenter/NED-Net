# Annotation review protocol

Written 2026-10-06, before any review, so the sampling scheme is pre-registered rather
than reconstructed afterwards. **Record deviations rather than silently changing the plan.**

## Why sampling at all

The U-Net produced 41,408 detections. Exhaustive adjudication is not feasible (at the
classical detectors' rates the union would exceed 60,000 events). All quantitative claims
therefore come from **defined samples with known selection probabilities**, never from
opportunistic clicking.

## UI settings — mandatory

- **"min Amp (xBL)" and "min local BL" must be 0.** They filter on
  `features["max_amplitude_x_baseline"]` and `quality_metrics["local_baseline_ratio"]`,
  which classical detectors produce and the U-Net does not. `min_amp > 0` *drops* every
  annotation lacking the field, i.e. the entire U-Net queue disappears.
- Triage by **confidence** (`detector_confidence`, populated from `cnn_confidence`) and
  **duration** only.
- Do not change the detection operating point. Review adjudicates the frozen run as it is.

## Review 1 — the unadjudicated flood (precision)

**Question.** 30,669 detections (74% of all output) come from animals 483552, 483553 and
483555, which have one confirmed seizure between them and on which Mir's detector proposed
only 6, 3 and 22 candidates. Nothing adjudicates them. If they are artefact, full-cohort
precision is nearer 0.1% than the measured 13.7% and they become an abundant hard-negative
source; if they are real seizures, the ground truth is incomplete and the 9.1% recall
figure is wrong in the other direction.

**Design.** Cluster sample of whole files — open a file, clear its entire queue. Clusters,
not individual events, because it matches the UI and gives clean weights.

- Population: 351 files containing those three animals' detections.
- Stratification: files split into confidence terciles by mean `cnn_confidence`.
- Sample: **3 files per tercile, 9 files, 832 events**, drawn with seed 42.
- File list: `flood_review_sample.csv`. Every event: `flood_review_events.csv`.
- **Review every event in a sampled file.** Partial files break the weighting.

**Weights.** 9 of 351 files; estimate per tercile and combine, so a tercile with fewer
files is not over-weighted.

**Record for each event:** confirmed / rejected, convulsive flag, and a free-text note on
what it actually is (movement artefact, electrical noise, chewing, genuine ictal, …). The
note matters more than the label here — the paper needs to say *what* the model fires on.

## Review 2 — Batch 3 pending (recall)

**Question.** Batch 3 is quiet and blind: 1.7 detections/file and 5.2% recall on 154
confirmed seizures. Does it fire on anything meaningful at all?

**Design.** All **401 pending** U-Net events across Batch 3's 338 files (complete, no
sampling — Mir's 154 confirmed and 5,422 rejected are already adjudicated in the same
sidecars and need no re-review).

## Review 3 — round-1 hard negatives (later)

After round-0 training, review a sample of the new model's own false positives and add them
as hard negatives. Sampling design to be written then, following the same rules.

## Logging

Append to `../RUN_LOG.md` when each review completes: date, which sample, how many events,
the label split, and any deviation from this protocol.
