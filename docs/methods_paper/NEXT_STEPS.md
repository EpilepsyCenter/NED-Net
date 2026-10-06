# Where we are, and what happens next

**Updated 2026-10-06.** Single source of truth for project state. Update it when a step
completes. Every run is in `RUN_LOG.md`; every correction in
`VERIFICATION_LOG_20261001.md`.

## State

| | |
|---|---|
| Phase 0 | **done** |
| Phase 1 detection | **done** — job 3770118, 41,408 events over 1,377 EDFs |
| Phase 1 scoring | **done** — frozen model fails: recall ~10.6%, convulsive precision 27.6%, full-cohort precision 0.12% |
| Phase 1 mechanism | **done** — RAM_GDNF has a more rhythmic background than SV2A; one cause for both failure modes |
| Sidecars on LUNARC | **done** — 1,364 merged, plus 1,377 `_ned_channels.json` |
| Flood review | **done enough** — 0-1 real events in 113, all three confidence terciles |
| Signal-quality sweep | **done** — jobs 3798487 (v1) and 3799553 (v2); both metrics failed their pre-registered test, see `EXCLUSION_CRITERIA.md` |
| LUNARC repo | **consolidated** at `4786254`+ |
| Round-0 arm B | **done** — job 3795251, overfit from epoch 1, val event_f1 0.056 |
| Round-0 arm A | **RUNNING** — job 3800378 (SV2A + RAM_GDNF, hold out Batch 3) |
| Batch-3 review | **not started** — 401 pending; the only lead left on recall |

## The three headline results so far

1. **The frozen model does not transfer.** Recall ~10.6% against 392 evaluable confirmed
   seizures; convulsive precision 27.6%; full-cohort precision 0.12%. Worst on short
   events (3% under 10 s).
2. **Why**: RAM_GDNF carries a more rhythmic, more periodic background at lower amplitude
   than SV2A (duty cycle 0.53 vs 0.20, p=1.7e-32). The model's boundary — learned from
   autocorrelation-proposed positives — is "rhythmic activity against non-rhythmic
   background", and that premise fails here. Explains the false positives, the misses, and
   why no exclusion rule could be built.
3. **Phase 2 is the test of that account.** If a rhythmic background is the cause,
   retraining on RAM_GDNF should recover performance. Arm A is therefore discriminating,
   not merely better-powered.

## Decisions on record

* Batch 4 **kept**, minus the 37 recordings with out-of-range ground-truth timestamps.
* Exclusions must be **signal-based, never performance-based**. Only an amplitude rule
  survived (dead/saturated channels; catches 449382). The three flood animals are handled
  by **reporting both with and without them**, with their performance-driven origin stated.
* Both Phase-2 arms use `--stable-val-split`, or the comparison confounds data with split.
* Ground truth covers **convulsive and behavioural seizures only** — 78% of the model's
  output is non-convulsive and has no reference in this cohort. Arm B therefore cannot
  learn that class at all; only SV2A supplies it.

## Key artifacts


| What | Where |
|---|---|
| Ground truth (14,740 rows, 430 seizures) | `~/ground_truth/mir_ramgdnf_annotations.csv` (local) |
| Detection DB | `~/.eeg_seizure_analyzer/projects/ram_gdnf_unet_v0.db` (local + LUNARC) |
| EDFs | `/lunarc/nobackup/projects/lu2026-2-60/RAM_GDNF_2025` (**not** `lu2026-12-29`) |
| SV2A EDFs + sidecars | `/lunarc/nobackup/projects/lu2026-2-60/edf_data` |
| Scoring script | `scripts/local/validate_frozen_unet_vs_mir.py` |
| Ground-truth consolidation | `scripts/local/consolidate_mir_annotations.py` |
| DB -> sidecars | `scripts/lunarc/db_to_sidecars.py` (**needs merge mode, see step 1b**) |
| Annotation backup | `~/Dropbox/NED-Net_backups/sv2a_annotations_20261001.tar.gz` |

## Steps

### 0. ~~EDF header comparison~~ — DONE, and the batch effect was three animals
Not acquisition drift: excluding 483552/483553/483555 drops Batch 4 from 94.7 to 8.8
detections/file. Recordings are uniform 90-minute files tiling the 21-day protocol
(2,014 h x 8 ch = 16,114 animal-hours, no file over 92 min). B1-B3 are 16-channel
(8 EEG + 8 activity); **Batch 4 is mixed 8- and 16-channel**, so its acquisition is the
one genuine heterogeneity.

### 1. Merged-sidecar converter — to build
`db_to_sidecars.py` currently writes U-Net detections as `pending` and **skips files that
already have a sidecar**. That is unsafe once Mir's labels are on disk: on 578 recordings
both sources apply and share a filename. Needs a merge mode:
- Mir's adjudicated rows -> `confirmed` / `rejected`
- U-Net detections -> `pending`, deduplicated against Mir's events by overlap
- `features["detectors"]` records which source(s) proposed each event
- channel mapping: **Mir `k` -> NED-Net `k-1`** (confirmed at a 6x margin)
- Batch 4's `session_name` carries a `.edf` suffix the other batches lack — strip it, or
  192 of the 430 seizures silently vanish

Generate and verify locally, then `scp` the sidecars into the LUNARC EDF folders. No tool
needs write access near the live tree.

### 2. Batch 3 review in ThinLinc — needs the lab PC
Convert B3 first (559 detections, 338 files — small, and the interesting question is why
it is *silent* on 154 seizures).

```bash
python scripts/lunarc/db_to_sidecars.py \
    --db ~/.eeg_seizure_analyzer/projects/ram_gdnf_unet_v0.db \
    --path-include 'Batch_3_Recordings' --dry-run
```

**In the UI, leave "min Amp (xBL)" and "min local BL" at 0.** They filter on classical
detector features that U-Net events do not have, and `min_amp > 0` *drops* every
annotation lacking the field — the whole queue would disappear. Triage by **confidence and
duration** instead; the converter populates `detector_confidence` from `cnn_confidence`.

### 3. Round-0 retraining — can start before the review
No new annotation needed: Mir's 430 positives + 11,653 hard negatives are already
adjudicated. Write them as sidecars beside the EDFs (step 1), then:

```bash
# feasibility probe: hold out Batch 3 (154 test seizures, worst recall -> most headroom)
EXCL="459657 459658 459659 459660 459661 459662 459663 459664"   # B3, all 8
# animal lists come from RAM_GDNF_2025_batch_metadata.csv (file-derived), NOT cohort_key.csv:
# B4 is 483550/551/552/553/554/555/557/559 — the key lists 12 but 4 are EXCLUDED/not implanted

# Arm A — with SV2A: point at a dir containing both trees, or run from the shared parent
MODEL_NAME=ramgdnf_armA_holdB3 EXCLUDE_ANIMALS="$EXCL" bash scripts/lunarc/train_unet.sh
# Arm B — RAM_GDNF only: --data-dir restricted to RAM_GDNF_2025
```

Arm A vs Arm B answers *does a new lab need our data, or just their own annotations?*

Evaluate by detecting over Batch 3 with the new model, then scoring with
`validate_frozen_unet_vs_mir.py --db <new_db>` — same metric, directly comparable to the
9.1% / 5.2% frozen baseline.

**If the B3 fold does not move recall substantially, stop and rethink** before running the
full 2 arms x 4 folds x seeds grid.

### 4. Then
Round-1 annotation (the model's own false positives as hard negatives), learning curve vs
cumulative reviewed events, catastrophic-forgetting check on SV2A, classical-detector arms
for the four-method benchmark. See `PHASE2_STRATEGY.md`.

## Open questions

- **Q1 (ask Mir):** what generated his candidates? His export has `candidate_spike_freq`,
  `candidate_peak_score`, `candidate_mean_low/gamma/broad` — spike-frequency *and*
  spectral features, so his detector overlaps both classical arms we planned to use as
  "independent" sources. Also worth asking: **was any video reviewed blind to the EEG
  candidates, even for a subset?** That is the only route to absolute convulsive recall.
- Why B4 floods and B3 is silent (step 0 may answer it).
- Whether the UI edit path preserves custom `features` keys through a reviewer edit.
