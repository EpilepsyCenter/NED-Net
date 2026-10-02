# Verification log — 2026-10-01

Every factual correction applied to `NEDNet_validation_HANDOFF.md` (v2), with the
evidence and the command that produced it, so each can be re-checked independently.
Source of record for v1 is `archive/HANDOFF_v1_20261001_as_received.md`.

Machine: Marco's Mac. Package `eeg_seizure_analyzer` at repo HEAD `27b79fe`.

---

## V1 — LUNARC storage path (v1 §2, §6b, §10, §11 were inverted)

**v1 said:** data storage is `lu2026-12-29/RAM_GDNF_2025`; compute account `lu2026-2-60`.
**Actual:** EDFs live at `/lunarc/nobackup/projects/lu2026-2-60/RAM_GDNF_2025`.

```
for R in /lunarc/nobackup/projects/lu2026-2-60/RAM_GDNF_2025 \
         /lunarc/nobackup/projects/lu2026-2-60/edf_data/RAM_GDNF_2025; do
  echo "$R -> $(find "$R" -type f -iname '*.edf' 2>/dev/null | wc -l)"; done
```
→ `lu2026-2-60/RAM_GDNF_2025` = **1385 EDFs**; the `edf_data/` variant = 0.

`lu2026-2-60` is the **storage** project (group `sto_lu2026-2-60`) *and* the working
SLURM account (`#SBATCH -A lu2026-2-60`, job 3770118 ran on it 2026-10-01).
`lu2026-12-29` is the compute/SLURM allocation ID with a convenience symlink
`~/lu2026-12-29 -> /lunarc/nobackup/projects/lu2026-2-60` **in the LUNARC home dir only** —
so the absolute path `/lunarc/nobackup/projects/lu2026-12-29/...` does not resolve.

**Cost of the v1 error:** job 3769954 (2026-10-01 14:20) exited after 9 s with
`cohort EDFs matched: 0`. Fixed path → job 3770118 matched **1377/1377**.

Batch subfolders confirmed present and matching the filter `Batch_[1-4]_Recordings`:
`Batch_1..4_Recordings` (+ `Batch_N_Behavior`, `483580_baseline data`,
`Hamza Files to test`, `Imaging Data`, `V2_All_EDF_Results`). 1385 − 8 = **1377**.

---

## V2 — `events.source` is `seizure_unet`, not `seizure_cnn` (v1 §3.1)

**v1 said:** the pipeline "Writes `source="seizure_cnn"`".
**Actual:** `analysis.py:509` passes `source="seizure_cnn"` to `db.write_events`, but each
event dict already carries `"source": _arch_source(feat.get("detection_method"), "seizure")`,
which wins. Confirmed against the production SV2A DB:

```
sqlite3 ~/.eeg_seizure_analyzer/projects/lunarc_detect_wk1-6_final.db \
  "SELECT source, type, COUNT(*) FROM events GROUP BY 1,2;"
```
→ `seizure_unet | non_convulsive | 18384` and `seizure_unet | convulsive | 6455`.
**24839/24839 rows are `seizure_unet`; zero are `seizure_cnn`.**

`seizure_cnn` survives only as the column DEFAULT (`db.py:63`, `db.py:113`) and as a
legacy UI label (`results.py:80` → "CNN (legacy)").

**Consequence:** the combined converter (v1 §7) must filter on `seizure_unet`, and
`features["detectors"]` should use `"seizure_unet"`, not `"seizure_cnn"`.

---

## V3 — the SV2A autocorrelation operating point IS recoverable (v1 §6b, §10)

**v1 said:** "The exact SV2A settings could not be recovered — those sidecars were
overwritten by a later U-Net run."

Both halves are false. The sidecars were never overwritten (see V4), and the settings are
reconstructible from the 8,722 surviving `detection_method: autocorrelation` events, each of
which stores `baseline_mean`, `baseline_std`, `threshold`, `acorr_threshold`,
`mean_spike_frequency_hz`, `max_amplitude_x_baseline`.

| Quantity | Measurement | Conclusion |
|---|---|---|
| `(threshold − baseline_mean)/baseline_std` | **3.0000** exactly, stdev 0.0000, n=8722 | spike threshold = mean + 3σ |
| `baseline_mean` within each (file, channel) | **constant in 663/663** groups with >2 events | baseline is global per file+channel → **`baseline_method="percentile"`**, NOT `rolling` |
| `mean_spike_frequency_hz` | observed minimum exactly **2.0** | `min_spike_freq_hz = 2.0` |
| `max_amplitude_x_baseline` | hard floor at **6.04** (lowest 10: 6.04, 6.42, 6.44, 6.47, 6.51, …) | candidate pool was gated at ≈6× baseline, not 3× |

`autocorrelation_seizure.py:128` shows `spike_amplitude_x_baseline` is a **z-multiplier**
when `baseline_std > 0` (`threshold = bl_mean + param × bl_std`), falling back to a plain
ratio only when std is 0 (line 130). So `spike_amplitude_x_baseline = 3.0` is the correct
SV2A value, and its meaning is 3σ.

**On "local BL over 3"** (user, 2026-10-01): that is a *secondary, post-detection* filter in
the NED-Net UI, distinct from the detector's internal 3σ spike threshold. The training-page
control (`training.py:154-158`, filtering `max_amplitude_x_baseline >= min_amp`) defaults to
`min_amp=0`, so it binds only when set by hand — and at 3 it would not bind at all, since the
whole SV2A candidate pool already sits ≥6.04.

Distribution of `max_amplitude_x_baseline` by label (SV2A, autocorrelation events):

| label | n | min | p5 | median |
|---|---|---|---|---|
| confirmed | 640 | 15.59 | 22.07 | 39.82 |
| rejected | 310 | 7.83 | 11.36 | 35.69 |
| pending | 7772 | 6.04 | 8.04 | 17.41 |

**Open (inferred, not proven):** the 6.0 floor is most likely a
`spike_prominence_x_baseline` gate — `spikes.py:1338` carries a default of `6.0`. Not
confirmed for the seizure path; treat the floor as an empirical fact and the mechanism as a
hypothesis.

---

## V4 — SV2A sidecars were NOT overwritten by U-Net output

Checked because v1 §6b asserted they were.

- All **165/165** files in dataset definition `SV2A_20260615` resolve to an existing EDF
  **and** sidecar.
- Confirmed/rejected totals match the stored definition **exactly**: 867 / 1325.
- Every confirmed/rejected event has `annotated_at ≤ 2026-06-15`; **zero** are dated after
  `UNetv2_20260615`'s creation (`2026-06-15T18:46Z`) — so no post-training label drift.
- Only `n_pending` has drifted (21 files); `pending` never enters training
  (`ml/dataset.py`, confirmed/rejected only).
- Each event retains `original_onset_sec` / `original_offset_sec` (raw detector boundaries
  before human adjustment).

The mixed `detection_method` is a deliberate two-round active-learning set:

| Round | Annotated | Events | `detection_method` |
|---|---|---|---|
| 1 | 2026-06-10/11 | 640 confirmed + 310 rejected | `autocorrelation` |
| 2 | 2026-06-15 | 227 confirmed + 1015 rejected | `ml_unet` (reviewed proposals from a predecessor U-Net no longer on disk) |

Round 1's 640/310 is exactly the total in `SV2A_training_dataset_20260611.json`.

Backed up 2026-10-01 as `sv2a_annotations_20261001.tar.gz` (21 MB, 227 files,
SHA256-verified) in `~/Dropbox/NED-Net_backups/` and the matching iCloud Drive folder.

---

## V5 — FROC/PR cannot be swept below the frozen core threshold (v1 §4)

v1 asks to "report at both operating points (FROC/PR curves + the fixed point)" from the
single frozen run. Partially impossible.

`cnn_confidence` is the event's **mean** probability (`predict.py:386`:
`confidence = float(np.mean(seg_probs))`), while the detection gate is on the per-sample
probability crossing `THRESHOLD=0.5` with hysteresis down to `BOUNDARY_THRESHOLD=0.1` for
boundaries. So the stored confidences do spread below 0.5:

```
sqlite3 lunarc_detect_wk1-6_final.db \
  "SELECT MIN(cnn_confidence), MAX(cnn_confidence), COUNT(*), SUM(cnn_confidence<0.5) FROM events;"
```
→ min **0.174**, max 0.964, n 24839, **10697 (43%) below 0.5**.

A PR curve over `cnn_confidence` is therefore drawable, but its high-sensitivity end is hard
-gated: events whose *peak* probability never reached 0.5 do not exist in the DB and cannot
be recovered post hoc. A complete FROC needs a second detection run at a low core threshold.

Secondary confound: because boundary hysteresis (0.1) grows event extent, the mean
probability depends on how far the boundary grew — so sweeping mean probability mixes
"confidence" with "event extent". Relevant to the calibration figure specifically.

---

## V6 — LEAKAGE HAZARD: `import_mir_annotations.py --inplace`

`scripts/import_mir_annotations.py` converts Mir's Excel export into
`<edf_stem>_ned_annotations.json` files. Its docstring line 2 states the purpose as
"for direct training", and `--inplace` writes each JSON **next to the EDF**.

`ml/train_unet.py` builds its dataset by recursively scanning a `--data-dir` for exactly
those sidecars (`dataset_store.scan_annotation_files`). Therefore importing Mir's set
`--inplace` into the LUNARC tree would (a) make the evaluation ground truth into training
data, violating v1 §4's "never evaluate on events used in training", and (b) collide by
filename with the pending review sidecars the combined converter writes.

The offline default (`-o ned_annotations_out`) is safe. See v2 §12 for the required
separation.

**Also noted from the import schema:** Mir's export carries `candidate_spike_freq` and
`candidate_peak_score` columns, and `candidate_type` ∈ {`convulsive`, `behavior`}. Those two
feature names are shaped like an autocorrelation-family detector's outputs (spike frequency
+ peak autocorrelation score). This is the specific thing to ask Mir about — see v2 §10 Q1.

---

## V7 — `features` survives the sidecar round-trip

Needed for `features["detectors"]` (capture–recapture membership) to survive human review.
`io/annotation_store.py` serialises as `"features": dict(self.features)` (line 103) and
reads back `features=dict(d.get("features", {}))` (line 134) — a plain copy, no whitelist or
key filtering. Custom keys persist at the store layer.

**Still unverified:** that the UI's edit path does not rebuild `features` from scratch when a
reviewer adjusts an event. Check before relying on it.

---

## V8 — operational notes

- **`--exclusive` is expensive on lu48.** Job 3769954/3770118 sat in `(Resources)` waiting
  for a wholly empty node; `StartTime` advanced a few seconds on every query, which is
  backfill saying "no node free", not a reservation. The proven SV2A scripts
  (`detect_all.sh`, `detect_spikes_all.sh`) do not use `--exclusive`.
- **`WORKERS` resolution works.** `WORKERS="${SLURM_CPUS_PER_TASK:-$(nproc)}"` resolved to
  **48** under `--exclusive` (job 3770118, node cn096). `scontrol` reporting `NumCPUs=1`
  while PENDING is requested-minimum bookkeeping, not the allocation.
- **LUNARC can be scripted within a session.** `~/.ssh/config` has `ControlMaster auto` with
  a `ControlPath`, so after one 2FA login the socket is reused and further `ssh`/`scp` need
  no re-auth. v1 §9 overstated this as "cannot be driven as an automated SSH target".
- **Silent-zero bug fixed.** `detect_ramgdnf_unet.sbatch`'s sanity check used
  `find ... 2>/dev/null`, so a missing `EDF_DIR` reported `0` and the job exited "done".
  It now omits the redirect and `exit 1`s on zero matches, so `--mail-type=FAIL` fires.
- **No model name in the DBs.** `events` records `source` and `chunks.mode='batch'` but no
  model name/version. Provenance for a figure DB = the git commit of the launching script at
  run time. Hence `detect_ramgdnf_unet.sbatch` is now version-controlled in
  `scripts/lunarc/`.
