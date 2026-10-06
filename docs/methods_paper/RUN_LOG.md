# Run log — NED-Net methods paper

**Append-only.** One entry per compute run or analysis that produces a number we might
quote. Add the entry when you launch, complete it when it finishes. The project DBs record
no model name (only `source` and `mode`), so **this log plus the script's git commit is the
only provenance** linking a result to the weights and settings that produced it.

Template:

```
### <date> — <what> — job <id>
script:   <path> @ <git commit>
config:   <the settings that matter>
inputs:   <data / DB / ground truth>
outputs:  <DB / model / figure path>
result:   <the headline number, or FAILED + why>
```

---

### 2026-10-01 — Phase-1 frozen detection, FAILED RUN — job 3769954
script:   `scripts/lunarc/detect_ramgdnf_unet.sbatch` (pre-fix)
config:   frozen operating point; `EDF_DIR=.../lu2026-12-29/RAM_GDNF_2025`
result:   **FAILED** — 0 EDFs matched, exited after 9 s. Wrong storage path
          (`lu2026-12-29` is the compute allocation; the symlink exists only in the LUNARC
          home dir). The sanity check used `2>/dev/null` so a missing dir reported `0` and
          the job printed "done". Both fixed; the check now `exit 1`s on zero matches.

### 2026-10-01 — Phase-1 frozen detection — job 3770118
script:   `scripts/lunarc/detect_ramgdnf_unet.sbatch` @ `74e6e47`
config:   `UNetv2_20260615` @ 0.5 + boundary 0.1 -> `Convulsive_v4LUNARC_20260616` @ 0.45,
          no re-ranker, min_dur 5, merge_gap 2, `--path-include 'Batch_[1-4]_Recordings'`
inputs:   1,377 cohort EDFs, `/lunarc/nobackup/projects/lu2026-2-60/RAM_GDNF_2025`
outputs:  `~/.eeg_seizure_analyzer/projects/ram_gdnf_unet_v0.db` (10 MB)
run:      lu48 node cn096, 14:27-18:36 CEST (4 h 09 m), 48 workers, 0 errors
result:   **41,408 events** over 2,014 recording hours (~16,100 animal-hours);
          78.1% non-convulsive / 21.9% convulsive

### 2026-10-02 — ground-truth consolidation
script:   `scripts/local/consolidate_mir_annotations.py` @ `74e6e47`
inputs:   69 `Day_N.xlsx` workbooks on the LU research share
outputs:  `~/ground_truth/mir_ramgdnf_annotations.csv` (4.2 MB)
result:   14,740 reviewed rows over 1,074 recordings; **430 confirmed seizures**
          (368 convulsive, 51 behaviour, 11 untyped), 11,653 `False`, 2,657 `Normal`

### 2026-10-02 — Phase-1 scoring, frozen model vs ground truth
script:   `scripts/local/validate_frozen_unet_vs_mir.py` @ `74e6e47`
config:   match tolerance 5 s; Mir channel `k` -> NED-Net `k-1`
result:   **recall 9.1%** (39/430; convulsive 9.8%, behaviour 5.9%);
          **precision 13.7%** (51/372 adjudicated overlaps).
          Convulsive-only: 6.8% recall / 27.6% precision.
          conf>=0.5: 7.1% / 17.3%. Convulsive+conf>=0.5: 4.6% / 29.6%.
          Controls: channel `k-1` confirmed at 6x margin over `k`/`k+1`; time bases
          aligned (median gap -2.2 s); all 1,074 GT recordings processed.
          **74% of detections are unadjudicated** — Mir reviewed only 6/3/22 candidates on
          animals 483552/483553/483555, which produce 30,669 detections (74.1%) and have
          1 confirmed seizure between them. Excluding them leaves precision and recall
          unchanged (only the count falls, 41,408 -> 10,739).

### 2026-10-02 — merged sidecars built
script:   `scripts/local/build_merged_sidecars.py` @ `ee72f24`
config:   dedup tolerance 5 s; `Normal` rows dropped; `event_id` int (schema requirement)
outputs:  `~/staged_sidecars/` — 1,364 sidecars, 44 MB
result:   430 confirmed + 11,653 rejected + 41,045 pending; 282 events proposed by both
          Mir and the U-Net. All 5,977 Batch-3 events verified to round-trip through
          `AnnotatedEvent.from_dict`.

### 2026-10-06 — sidecars transferred to LUNARC
result:   1,364 sidecars in place beside the EDFs; project at 6,635/50,000 files after the
          quota increase (was 5,000, and directories count). Grace cleared.

### 2026-10-06 — Round-0 arm B, hold out Batch 3, CANCELLED — job 3795249
script:   `scripts/lunarc/train_unet.sh` @ `f7736d1`
result:   **CANCELLED before running.** Submitted with `exclude=none` despite
          `EXCLUDE_ANIMALS` being set: `POS_WEIGHT` and `EXCLUDE_ANIMALS` used a bare
          `read`, which sets the variable to empty on Enter and destroyed the preset.
          Would have trained the hold-out-B3 fold on its own test set. Fixed in `a3ce36f`
          (both fields now use the preserving `ask()` helper, which also displays the
          current value). **Always check the summary line reads `exclude=<ids>`.**

### 2026-10-06 — Round-0 arm B, hold out Batch 3 — job 3795251
script:   `scripts/lunarc/train_unet.sh` @ `a3ce36f`
config:   `EDF_DIR=.../RAM_GDNF_2025` (RAM_GDNF only, no SV2A); exclude 459657-459664;
          neg_pos_ratio 10, pos_weight auto (=10), epochs 50, batch 32, lr 3e-4,
          patience 10, base_filters 32, depth 4, dropout 0.2, neg-source hard
inputs:   Mir's labels as merged sidecars + `*_ned_channels.json` (32 animals).
          After exclusion: 24 animal groups, **276 positives**, 6,231 hard negatives;
          default split train 184 pos / val 92 pos
outputs:  `~/.eeg_seizure_analyzer/models/ramgdnf_armB_holdB3/`
result:   _pending_
notes:    Prediction on record — this should improve recall but NOT the over-detection,
          because `_balance_negatives` prefers rejected events and only tops up with
          random background on a shortfall; 6,231 rejected exceeds the 2,760 target, so
          no background is drawn and the unlabelled flood regions are never seen.

### 2026-10-06 — channel-ID files written and transferred
script:   `scripts/local/build_channel_ids.py` @ `f7736d1`
outputs:  1,377 `*_ned_channels.json` beside the EDFs (project ~8,012/50,000 files)
result:   `--analyze` now reports **32 animal groups** with real IDs (was 8, pooled by
          channel index). Batch 3's eight animals total exactly 154 positives, matching
          Mir's count. Without this, `--exclude-animals` matched nothing.

---

## Entries to add as you go

- Flood review (sample defined in `review/flood_review_sample.csv`)
- Batch-3 review
- Round-0 arm A (SV2A + RAM_GDNF), hold out Batch 3
- Round-0 remaining folds
- Post-training detection + scoring per fold
- Classical-detector sweeps
