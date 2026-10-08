# Where we are, and what happens next

**Updated 2026-10-08, end of day.** Single source of truth for project state. Every run is
in `RUN_LOG.md`; commands in `LUNARC_CHEATSHEET.md`; models in `MODELS.md`. The
2026-10-06 version is in `archive/NEXT_STEPS_20261006.md` — its Phase-2 plan is superseded.

## Running overnight (2026-10-08)

| job | what | where | read it with |
|---|---|---|---|
| **3825989** | `ramgdnf_all_prod` — U-Net on ALL retained batches, no hold-out | gpua100 | `grep -E "Best event_f1|background sampling" logs/unet_train_3825989.out` |
| **3825993** | A3 on **held-out Batch 3** at thr 0.9 / bnd 0.5 | lu48 | `operating_point_table.py --batches 3` |
| **(submitted)** | `conv_rejneg_holdB3` — Stage 2 **with rejected negatives** | gpua100 | `grep "convulsive negatives" logs/conv_train_*.out` |

## The story, as it now stands

Three rows, and they answer three different questions. Keep them apart.

| | trained on | measured on | convulsive recall | fires on |
|---|---|---|---|---|
| frozen `UNetv2` | SV2A only | RAM_GDNF B1+B2 | **27.9%** | **16%** of recordings |
| **A3 retrained** | SV2A + Mir's B1/B2/B4 labels | RAM_GDNF B1+B2 (**in-sample**) | **47.1%** | **73%** |
| A3 retrained | as above, B3 held out | **Batch 3 (out-of-sample)** | _job 3825993_ | _pending_ |

(Stage 2 tuned per model: frozen @ 0.45, A3 @ 0.30. Retained channels, n=68 convulsive
ground-truth events. In-sample numbers are optimistic by construction — that is the point
of the row, not a flaw in it.)

1. **A model trained on one cohort does not transfer to another.** The frozen detector was
   93.6% precise on SV2A and manages 27.9% convulsive recall here while firing on only
   **16% of recordings** — half the data got no detector at all. That silence, not the
   false positives, is the real failure.
2. **Annotating the new cohort and retraining fixes most of it.** 27.9% -> 47.1% recall,
   16% -> 73% coverage, and event boundaries go from a 9 s fragment inside a 37 s seizure
   (median IoU 0.36) to median IoU 0.60-0.66. Same UI, same pipeline, no code changes.
   This is the direct analogue of how SV2A's 93.6% was obtained.
3. **The retrained model's precision is not yet measured.** Everything we can compute is
   against Mir's convulsive-only reference, which cannot credit a non-convulsive detection,
   and most of A3's output is non-convulsive. **This is the biggest hole in the paper.**

## Tomorrow, in order

1. **Read the three overnight jobs.** B3 (3825993) is the headline — the gap between
   "annotate this cohort" and "transfer to the next batch".
2. **Test whether the Stage-2 fix worked.** Re-detect B1+B2 with
   `CONV_MODEL=conv_rejneg_holdB3` (cheap, lu48), then
   `conv_threshold_sweep.py`. The symptom was precision **flat at 11%** across every
   threshold; the fix is a curve that **rises**. If it is still flat, the negatives were
   not the problem and the whole Stage-2 line should be reported as a limitation rather
   than pursued further.
3. **Sweep an operating point for `ramgdnf_all_prod`** on B1+B2, exactly as for A3. Do
   **not** inherit 0.9/0.5 — A3 trained on a different (stale) label set.
4. **Measure true precision on retrained output.** A stratified random review sample of
   `ramgdnf_all_prod` detections, drawn before looking at them, seed recorded. This is the
   one number that converts "fires more" into "finds more", and sampling discipline has
   already corrected this picture three times (see `RUN_LOG.md` 2026-10-07/08).
5. 569 detections catching **0 of 10** seizures on animal 449385 needs explaining — either
   a flood or real non-convulsive activity. Step 4 answers it.

## Open weaknesses, stated plainly

* **True precision of any retrained model: unmeasured.** Step 4.
* **In-sample vs out-of-sample** must be labelled on every number. The 47.1% is in-sample.
* **n is small**: 68 convulsive events on 7 channels for the in-sample comparison. Enough
  to separate 16% coverage from 73%, not to rank adjacent animals.
* **`ramgdnf_all_prod` has no held-out fold** by design, so it supports only in-sample
  claims.
* **It also changes four things at once** vs A3 (B3 included, review work included, 100 s
  cap, background excluded from annotated regions). If it beats A3 we will not know which
  change did it. Fine for a production number, not an ablation.
* **Mir's reference is convulsive-only**, so the non-convulsive arm — most of the output —
  has no ground truth in this cohort in either direction.

## Decisions on record

* Batch 4 **excluded** from the retained set (as are B2 UI ch3/ch4 and animal 449382).
  The out-of-range-timestamp defect is localised to B4 and immaterial to B1-B3.
* Exclusions are **signal-based, never performance-based**. The rhythmicity metric failed
  its pre-registered test (`EXCLUSION_CRITERIA.md`) and animal 450096 — the highest
  `prominence_db` of all 32 — has the *best* precision of B2's retained channels. "Noisy"
  is a per-channel precision statement, not a verdict.
* **Positives are capped at 100 s** (`MAX_POSITIVE_SEC=100`). Mir's longer rows are chained
  seizures; SV2A's 1,136 human-reviewed events top out at **84 s**, which is the
  justification. Drops 15 of Mir's 430 and 0 of SV2A's.
* **Mir's rejected rows are excluded from the detector and trained on by Stage 2.** Same
  rows, opposite role: video-adjudicated "not a convulsive/behavioural seizure" is
  contaminated as a detection negative and exactly right as a Stage-2 negative
  (`build_merged_sidecars.py:76`).
* **Random background never lands on an annotated region** (`BG_AVOID_REJECTED=1`).
* **Report Stage 1 as detection capability for both models**, Stage 2 separately — because
  Stage 2 is informative for the frozen model and not for A3, so filtering A3 by it does
  not type-match, it deletes detections near-randomly.
* **Event matching is interval overlap** on the same file and channel with 5 s slack. Tested
  and robust: tightening the slack or forcing one-to-one changes nothing. Boundary quality
  is a separate, weaker claim and must not be asserted from the loose rule.
* **Re-tune `CONV_THRESHOLD` after every Stage-2 retrain; never inherit it.** It is free to
  sweep post-hoc because `convulsive_confidence` is stored per event.

## Two process failures worth reporting in the paper

Both were caught by provenance already written in the repo, not by new data — which is the
argument for the logging discipline itself.

1. **A stale label tree.** `refresh_training_tree.py` had not run since before the review
   sessions, so arms A/A2/A3 all trained without 52 manual additions and 344 adjudicated
   detections. Its own docstring described the loop it was not completing.
2. **An unused note.** `build_merged_sidecars.py:76` recorded a week earlier that Mir's
   rejections are "valid as convulsive-CLASSIFIER negatives, contaminated as DETECTION
   negatives". Acting on it is tonight's Stage-2 job.

Also: the picture has moved **four** times on sampling discipline alone — pessimistic
(pooled, unthresholded), optimistic (hand-picked high-confidence files), uncertain (random
sample, re-weighted by output volume), and now positive (paired, same code path, tuned
per model). Every correction came from how events were selected, not from new data. That
is the strongest available argument for pre-registered sampling in detector validation,
and it belongs in the discussion.

## Key artifacts

| What | Where |
|---|---|
| Ground truth (14,740 rows, 430 seizures) | `~/ground_truth/mir_ramgdnf_annotations.csv` (local) |
| Frozen detection DB | `~/.eeg_seizure_analyzer/projects/ram_gdnf_unet_v0.db` |
| A3 operating-point sweep DBs | `~/.eeg_seizure_analyzer/projects/armA3_tune_t*_b*.db` |
| Paired per-animal tables | `review/frozen_vs_A3_per_animal_B1B2{,_cascade}.csv` |
| Training tree (symlinks + sidecars) | `~/train_nomirneg` on LUNARC |
| EDFs | `/lunarc/nobackup/projects/lu2026-2-60/RAM_GDNF_2025` (**not** `lu2026-12-29`) |
| SV2A EDFs + sidecars | `/lunarc/nobackup/projects/lu2026-2-60/edf_data` |
| Annotation backup | `~/Dropbox/NED-Net_backups/sv2a_annotations_20261001.tar.gz` (2026-10-01; **re-snapshot, it predates the reviews**) |

### Scripts, and what each is for

| Script | Use |
|---|---|
| `local/operating_point_table.py` | select / compare an operating point; `--det-conv-only` for the cascade |
| `local/conv_threshold_sweep.py` | Stage-2 threshold, free, no re-detection |
| `local/matching_sensitivity.py` | is a result an artefact of the match rule? |
| `local/validate_frozen_unet_vs_mir.py` | the base scorer; `load()` is reused by the above |
| `lunarc/refresh_training_tree.py` | **run with `--keep-mir-rejections` before every retrain** |
| `lunarc/train_unet.sh` / `train_convulsive.sh` | Stage 1 / Stage 2 training |
| `lunarc/detect_ramgdnf_unet.sbatch` | detection; every setting is an env override |

## Open questions

- **For Mir:** was any video reviewed **blind to the EEG candidates**, even for a subset?
  That is the only route to absolute convulsive recall; everything we report is recall
  relative to his 15-80 Hz candidate detector.
- **For Mir:** is `start_s` file-relative or session-relative for the 242 out-of-range rows?
  10,740 s is about two files, so a multi-file time base is the likely explanation.
- Whether the UI edit path preserves custom `features` keys through a reviewer edit.
- The UI duplicate-event bug on the confirm path (commit `c6ac87d`) — not blocking.
