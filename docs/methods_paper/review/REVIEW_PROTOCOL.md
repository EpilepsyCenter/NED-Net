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

**Record for each event:** confirmed / rejected, plus the convulsive flag. That is the
quantitative output and it is all that is required for every event.

**Do NOT attempt behavioural attribution.** The behaviour videos are not on LUNARC, so
"movement artefact" versus "genuine event" is not decidable at review time. Judge the EEG
trace only. This is a stated limitation of the precision estimate: it tells us the share of
detections that are not seizures, not what each one physically was.

**Characterisation (secondary, ~10 events, not all 832).** For a handful of representative
rejected events, use the Training page's notes box (`tr-notes`) to describe the *EEG
morphology* only — e.g. high-amplitude broadband noise, flat or clipped signal, rhythmic
low-frequency activity, electrode pop. The note saves when you navigate to another event,
so move off the event before closing the file. Ten of these is enough to say what the model
fires on; per-event notes across the whole sample are not worth the time.

If behavioural attribution turns out to matter for the paper, the Batch_N_Behavior folders
on the research share would need syncing for the sampled files only — a separate decision,
not a blocker here.

## Review 2 — Batch 3 pending (recall)

**Question.** Batch 3 is quiet and blind: 1.7 detections/file and 5.2% recall on 154
confirmed seizures. Does it fire on anything meaningful at all?

**Design.** All **401 pending** U-Net events across Batch 3's 338 files (complete, no
sampling — Mir's 154 confirmed and 5,422 rejected are already adjudicated in the same
sidecars and need no re-review).

## Review 3 — round-1 hard negatives (later)

After round-0 training, review a sample of the new model's own false positives and add them
as hard negatives. Sampling design to be written then, following the same rules.

## UI issue: editing boundaries then confirming can leave a duplicate

**Reported by the reviewer, 2026-10-08, in the Training-tab review mode.** Changing an
event's boundaries and confirming can leave a second event at the *original* duration.

Dragging is the intended way to create a manual event in that mode, so the issue is in the
confirm path, not the add path.

**Likely mechanism (code inspection, NOT reproduced):**
`_sync_boundary_to_seizure_events` (`training.py:238`) pushes a boundary change back to
`state.seizure_events`, matching on channel plus onset within **0.01 s** of the original
onset. On a failed match it returns silently, leaving `seizure_events` holding the pre-edit
boundaries while the annotation holds the new ones. Annotations are re-derived from
`seizure_events` via `detections_to_annotations` when loaded (`training.py:1027`), so a
stale entry reappears as an annotation at the original duration. `_save_detection_file`
also writes the detection JSON back on every edit, so the stale copy persists on disk.

Observed twice in the four exhaustively-reviewed files, both on ch2:

```
B3_W3_D3_02022026(12): detector 1693.9-1703.5 (orig_onset 1693.896) + manual 1694.2-1700.6
B3_W3_D7_06022026(4):  detector 5266.2-5286.7 (orig_onset 5267.324) + manual 5266.1-5276.8
```

**Effect on the measurements: none.** The miss count includes only manual events overlapping
*no* detection, and a duplicate overlaps one by definition — so recall 12/28 and precision
12/13 stand. Only the raw unique-seizure total shifts, 50 -> ~48.

**Deferred.** Not investigated further mid-annotation with jobs running against the tree. If
pursued: widen the match tolerance, match on event_id rather than onset, and make a failed
match loud rather than silent.

## Logging

Append to `../RUN_LOG.md` when each review completes: date, which sample, how many events,
the label split, and any deviation from this protocol.
