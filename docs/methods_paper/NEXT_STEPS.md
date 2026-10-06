# Where we are, and what happens next

**Updated 2026-10-02.** Single source of truth for project state. Update it when a step
completes.

## State

| | |
|---|---|
| Phase 0 | **done** — ground truth consolidated, leakage audit clean |
| Phase 1 detection | **done** — job 3770118, `ram_gdnf_unet_v0.db`, 41,408 events, copied back |
| Phase 1 scoring | **done** — frozen model fails: 9.1% recall. See `PHASE1_RESULTS_20261002.md` |
| Sidecars on LUNARC | **done** 2026-10-06 — 1,364 merged sidecars beside the EDFs |
| Phase 1 review | **not started** — needs the lab PC / ThinLinc. Priority: the 3 flood animals, then B3 |
| Phase 2 round 0 | **unblocked** — Mir's labels are in place, needs no lab PC |
| Classical detector arms | **not started** |

**Blocked on hardware:** annotation needs the lab PC. Everything in steps 0, 1 and 3 below
can proceed without it.

## RESOLVED: LUNARC file-count quota (2026-10-06)

Quota raised to **50,000 files / 55,000 hard** (was 5,000/5,500). The merged sidecars
transferred on 2026-10-06: **1,364 files** now beside the EDFs under `RAM_GDNF_2025`,
project at 6,635/50,000, grace `none`.

Historical note, in case it recurs: directories count toward the file quota (the project
holds ~4,300 files + ~1,180 directories), and space is never the constraint here — 517 GB
of 4.883 TB. A home tree of symlinked EDFs plus real sidecars was considered as a
workaround and verified technically sound (`annotation_json_path` is purely lexical;
nothing in the dash app calls `resolve()`/`realpath()`), but rejected as fragile — opening
an EDF by its project-storage path makes the UI write an *empty* sidecar there and silently
lose the review queue.

## Also worth running (no project-storage inode cost)

**The classical-detector sweeps can run now.** `detect_autocorr_batch.py` only *reads* EDFs
from project storage; per-worker part-DBs go to `${SNIC_TMP:-/tmp}` (node-local) and the
merged DB lands in `~/.eeg_seizure_analyzer/projects/` — home, which has ~349,000 free
inodes and 78 GB free. Zero project-storage inode cost.

Worth doing now because it is on the critical path for the strongest version of the paper:
if the published methods from other labs (White 2006 autocorrelation, Casillas-Espinosa
2019 spectral, Twele 2017 spike-train) fail on this data the same way our U-Net did, the
generalization claim becomes a four-method benchmark rather than one model's anecdote.
Scoring is free afterwards — `validate_frozen_unet_vs_mir.py` works against any DB.

Also unblocked: asking Mir the Q1 questions (below), and all local analysis.

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

### 0. EDF header comparison across batches — no compute, do first
Batches span ~9 months (B1 Sep 2025, B2 Oct 2025, B3 Jan 2026, B4 Jun 2026) and behave
completely differently (2.6 vs 69.3 detections/file). Compare sampling rate, physical
min/max, dimension and prefilter across a few files per batch. If acquisition drifted,
that partly explains Phase 1 and is itself a result.

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
